# Fine-Tuning MobileNetV2 on ImageNet-100 (Classical vs Ablation-Aware Training)

This repository contains experiments comparing **classical fine-tuning** against **ablation-aware fine-tuning** (Mixup, CutMix, RandAugment) for MobileNetV2 on an ImageNet-100 subset. The goal is to evaluate the effects of augmentation strategies on accuracy and convergence when targeting lightweight models for edge inference.

---

## File Overview

* `classical_training.py` – Standard fine-tuning with random resized crops and horizontal flips.
* `ablation-aware_training.py` – Fine-tuning with additional augmentations (Mixup, CutMix, RandAugment).

---

## Training Parameters

* **Model:** MobileNetV2
* **Optimizer:** Adam
* **Learning Rate:** `1e-3`
* **Scheduler:** Cosine Annealing
* **Batch Size:** 128
* **Precision:** Mixed precision training (`torch.cuda.amp`)
* **Epochs:** 30
* **Workers:** 4 (`--num-workers`)
* **Data format:** Folders structured as `data/train/<class>` and `data/val/<class>`

---

## Usage

### Classical Fine-Tuning

```bash
python classical_training.py \
  --data-dir ./data \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --num-workers 4
```

### Ablation-Aware Fine-Tuning

```bash
python ablation-aware_training.py \
  --data-dir ./data \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --num-workers 4
```

---

## Results

Validation accuracy and loss are logged per epoch to the console and can be saved to log files.
Performance comparisons (classical vs ablation-aware) are discussed in the accompanying paper.
