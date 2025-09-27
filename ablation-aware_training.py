import argparse
import os
import time
import random
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2, resnet50
from tqdm import tqdm
import csv


# Utilities: device, AMP

def pick_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_autocast_ctx(device):
    # return a callable context manager factory: with autocast_ctx(): ...
    # Try new unified API, otherwise fall back to cuda autopcast, else nullcontext
    try:
        # prefer unified API if available
        return lambda **kw: torch.amp.autocast(device_type=device.type, **kw)
    except Exception:
        if device.type == "cuda" and hasattr(torch.cuda.amp, "autocast"):
            return lambda **kw: torch.cuda.amp.autocast(**kw)
        # no autocast available (MPS / CPU fallback)
        return lambda **kw: nullcontext()

def make_gradscaler(device):
    # return a GradScaler instance if AMP is usable on this device, else None
    if device.type == "cuda":
        # try new API then fall back to legacy
        try:
            return torch.amp.GradScaler(device_type="cuda")
        except Exception:
            return torch.cuda.amp.GradScaler()
    # MPS / CPU: GradScaler support is limited; return None (no AMP)
    return None


# Mixup / CutMix utils

def mixup_data(x, y, alpha=1.0, device=None):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device) if device is not None else torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def rand_bbox(size, lam):
    # size: (B,C,H,W)
    W = size[3]
    H = size[2]
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # uniform center
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=1.0, device=None):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    batch_size = x.size(0)
    rand_index = torch.randperm(batch_size, device=device) if device is not None else torch.randperm(batch_size)
    x_new = x.clone()
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    # swap patch
    x_new[:, :, y1:y2, x1:x2] = x[rand_index, :, y1:y2, x1:x2]
    # adjust lambda to actual area
    area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - area / (x.size(2) * x.size(3))
    return x_new, y, y[rand_index], lam

def mix_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)


# Dataset / Dataloader

def get_dataloaders(data_dir, batch_size, num_workers, pin_memory=True, persistent_workers=False,
                    prefetch_factor=2, use_randaugment=False):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf_list = [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip()
    ]
    if use_randaugment:
        # torchvision.transforms.RandAugment exists in modern torchvision
        try:
            train_tf_list.append(T.RandAugment())
        except Exception:
            print("Warning: RandAugment not available in this torchvision. Skipping.")
    train_tf_list.extend([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    train_tf = T.Compose(train_tf_list)

    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    # DataLoader kwargs
    dl_kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers,
                     pin_memory=pin_memory if torch.cuda.is_available() else False,
                     persistent_workers=persistent_workers,
                     prefetch_factor=prefetch_factor if num_workers > 0 else 2)

    val_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers//2),
                      pin_memory=pin_memory if torch.cuda.is_available() else False,
                      persistent_workers=persistent_workers)

    train_loader = DataLoader(train_dataset, **dl_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    return train_loader, val_loader, len(train_dataset.classes), train_dataset.classes


# Distillation loss helper

def distillation_loss_fn(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    # CE with labels
    ce = F.cross_entropy(student_logits, labels)
    # KL divergence between softened outputs
    p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(p_student, p_teacher, reduction="batchmean") * (T * T)
    return alpha * ce + (1.0 - alpha) * kd


# Training & Validation

def train_one_epoch(model, train_loader, optimizer, scaler, autocast_ctx, device, args, teacher=None, classes=None):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    use_mix = args.use_mixup
    use_cut = args.use_cutmix
    accum_steps = max(1, args.accum_steps)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train", leave=False)
    for step, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Optionally mixup / cutmix
        if use_mix:
            images, y_a, y_b, lam = mixup_data(images, labels, args.alpha, device=device)
        elif use_cut:
            images, y_a, y_b, lam = cutmix_data(images, labels, args.alpha, device=device)
        else:
            y_a = y_b = labels
            lam = 1.0

        with autocast_ctx():
            outputs = model(images)
            if use_mix or use_cut:
                ce_loss = mix_criterion(outputs, y_a, y_b, lam)
            else:
                ce_loss = F.cross_entropy(outputs, labels)

            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(images)
                kd = F.kl_div(
                    F.log_softmax(outputs / args.distill_T, dim=1),
                    F.softmax(t_out / args.distill_T, dim=1),
                    reduction="batchmean"
                ) * (args.distill_T ** 2)
                loss = args.distill_alpha * ce_loss + (1.0 - args.distill_alpha) * kd
            else:
                loss = ce_loss

            loss = loss / accum_steps

        # Backprop (AMP-aware)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step if accumulation boundary
        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Metrics for training when top-1 is only meaningful when not using mixup/cutmix
        batch_size = images.size(0)
        running_loss += (loss.item() * accum_steps) * batch_size  # unscale accumulation
        total += batch_size
        if not (use_mix or use_cut):
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()

        pbar.set_postfix(loss=running_loss/total, acc=(correct/total if total > 0 else 0.0))

    if (step + 1) % accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    train_loss = running_loss / total if total > 0 else 0.0
    train_acc = (correct / total) if not (use_mix or use_cut) else None
    return train_loss, train_acc

def validate(model, val_loader, device, autocast_ctx):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_ctx():
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc


# Checkpoint helpers

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


# Argparse

def get_args():
    p = argparse.ArgumentParser(description="Train MobileNetV2 with ablations (mixup, cutmix, randaugment, distill)")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--use-mixup", action="store_true")
    p.add_argument("--use-cutmix", action="store_true")
    p.add_argument("--use-randaugment", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0, help="Mixup/CutMix alpha")
    p.add_argument("--teacher-checkpoint", type=str, default=None)
    p.add_argument("--distill-alpha", type=float, default=0.5, help="alpha weight for CE vs KD")
    p.add_argument("--distill-T", type=float, default=4.0, help="temperature for distillation")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=2)
    return p.parse_args()


# Main

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device()
    print(f"Using device: {device}")

    # AMP utilities
    autocast_ctx_factory = make_autocast_ctx(device)
    scaler = make_gradscaler(device)

    # Dataloaders
    train_loader, val_loader, num_classes, class_ids = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers,
        pin_memory=args.pin_memory, persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor, use_randaugment=args.use_randaugment
    )
    print(f"Found {num_classes} classes")

    # Model
    model = mobilenet_v2(weights="IMAGENET1K_V1" if args.pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.to(device)


    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = 0.0

    # Resume checkpoint?
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        # ckpt may be a raw state_dict or a dict with keys
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_acc = ckpt.get("best_acc", 0.0)
        else:
            # assume raw state_dict
            model.load_state_dict(ckpt)
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    # CSV logger
    csv_path = os.path.join(args.save_dir, "train_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # Main loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        # autocast_ctx = autocast_ctx_factory()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, autocast_ctx_factory, device, args, teacher, class_ids)
        val_loss, val_acc = validate(model, val_loader, device, autocast_ctx_factory)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}  time={elapsed:.1f}s  train_loss={train_loss:.4f}  train_acc={train_acc}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={lr:.3g}")

        # CSV log
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, (train_acc if train_acc is not None else ""), val_loss, val_acc, lr])

        # checkpoint (save full state)
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "args": vars(args)
            }
            save_path = os.path.join(args.save_dir, "mobilenetv2_best_full.pth")
            save_checkpoint(ckpt, save_path)
            print(f"Saved best checkpoint to {save_path} (val_acc={best_acc:.4f})")
        # also save last epoch
        last_path = os.path.join(args.save_dir, "mobilenetv2_last.pth")
        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc
        }, last_path)

if __name__ == "__main__":
    main()
