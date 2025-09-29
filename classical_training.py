# train_amp.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2
from tqdm import tqdm


# Parse arguments

def get_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on ImageNet-100 (AMP + scheduler)")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


# Data pipeline

def get_dataloaders(data_dir, batch_size, num_workers):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_dataset   = datasets.ImageFolder(f"{data_dir}/val", transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(train_dataset.classes)


# Training & validation

def train_one_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def validate(model, dataloader, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# Main

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, num_classes = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    # Model
    model = mobilenet_v2(weights="IMAGENET1K_V1" if args.pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    # Optimizer + Scheduler + AMP
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc     = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        scheduler.step()  # Update learning rate

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "mobilenetv2_best.pth")
            print(f" Saved new best model (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()

