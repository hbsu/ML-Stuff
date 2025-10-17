
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — MobileNetV1 binary classifier with cross‑validation and optional quantization.

Follows the ECE 720 "train.py pseudocode summary":
1) Initialize dataset & epochs
2) For each cross‑validation split:
      Initialize training run
      If epochs > 0:
         Execute training loop
         Save & evaluate the model
         Plot validation curve
      else:
         Evaluate the model
3) Plot cross‑validation error summary

HOW TO USE (examples):
----------------------
# Basic training with 5 CV splits, 20 epochs, AdamW wd=1e-4
python train.py --train_dir "C:\\Users\\hbsu\\Desktop\\dataset\\training" \
                --epochs 20 --splits 5 --weight_decay 1e-4

# Use a separate fixed validation set (no CV) and just evaluate a pretrained checkpoint:
python train.py --train_dir "C:\\path\\to\\training" --val_dir "C:\\path\\to\\validation" \
                --epochs 0 --resume checkpoints/mnv1_split0.pt

# Enable INT8 static quantization (CPU) after training (PTQ); save quantized model:
python train.py --train_dir "C:\\...\\training" --epochs 10 --quantize int8

# Train/validate with fp16 mixed precision (useful on NVIDIA GPUs):
python train.py --train_dir "C:\\...\\training" --epochs 10 --quantize fp16

# Other useful flags:
--batch_size 64 --lr 1e-3 --splits 8 --seed 0 --save_dir runs/mnv1 --device cuda

NOTES:
- Expects directory layout compatible with torchvision.datasets.ImageFolder:
  training/
    negatives/
      img1.jpg, ...
    positives/
      img999.jpg, ...
  validation/ (optional, same structure)
- MobileNetV1 backbone via `timm` if available (mobilenetv1_100). If `timm` is not
  installed, a minimal MobileNetV1 implementation is used as a fallback.
- "AdamW optimizer regularization" is the weight decay argument: tune via --weight_decay.
- INT8 path uses torch.ao.quantization (static PTQ) with simple calibration on
  a handful of batches from the validation loader (or training if no val set).
"""

import argparse
import os
import math
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
USER_PLOT_DIR = "runs/mobilenetv1/plots"

# -----------------------------
# Model: MobileNetV1 (prefer TIMM; fallback to local minimal impl)
# -----------------------------
def create_mobilenet_v1(num_classes: int = 2, pretrained: bool = False):
    try:
        import timm
        model = timm.create_model("mobilenetv1_100", pretrained=pretrained, num_classes=num_classes)
        return model
    except Exception:
        # Minimal MobileNetV1 fallback (depthwise separable conv stacks).
        def dw_sep(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        class MiniMobileNetV1(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    dw_sep(32, 64, 1),
                    dw_sep(64, 128, 2),
                    dw_sep(128, 128, 1),
                    dw_sep(128, 256, 2),
                    dw_sep(256, 256, 1),
                    dw_sep(256, 512, 2),
                    *[dw_sep(512, 512, 1) for _ in range(5)],
                    dw_sep(512, 1024, 2),
                    dw_sep(1024, 1024, 1),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Linear(1024, num_classes)
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        return MiniMobileNetV1(num_classes=num_classes)

# -----------------------------
# Data utilities
# -----------------------------
def build_transforms(img_size: int = 160) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def get_imagefolder_with_labels(root: str, eval_tf):
    ds = datasets.ImageFolder(root=root, transform=eval_tf)
    labels = [label for _, label in ds.samples]
    return ds, np.array(labels)

# -----------------------------
# Train / Validate helpers
# -----------------------------
def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def run_one_epoch(loader, model, criterion, optimizer=None, device="cpu", use_amp=False):
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss, epoch_acc, n = 0.0, 0.0, 0

    # Use new autocast API to avoid the deprecation warning
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            # autocast for CUDA only when use_amp is True
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            # ✅ correct AMP order:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, targets)

        bs = images.size(0)
        epoch_loss += loss.item() * bs
        epoch_acc  += (logits.argmax(dim=1) == targets).float().sum().item()
        n += bs

    return epoch_loss / max(n, 1), epoch_acc / max(n, 1)

# -----------------------------
# Quantization
# -----------------------------
def post_training_int8_quantize(model: nn.Module, calib_loader: DataLoader):
    model.eval()
    model.cpu()
    # fuse typical Conv-BN-ReLU where possible (works best on fallback impl; timm handles internally)
    def fuse_sequential(m):
        for module_name, m_child in m.named_children():
            fuse_sequential(m_child)
        # best-effort: no exact names guaranteed for timm; skip if unsupported
    try:
        torch.ao.quantization.fuse_modules_qat  # check availability
        # Prepare for static PTQ
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model_prepared = torch.ao.quantization.prepare(model, inplace=False)
        # Calibration
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_loader):
                model_prepared(images)
                if i >= 10:  # a few batches suffice for simple calibration
                    break
        model_int8 = torch.ao.quantization.convert(model_prepared, inplace=False)
        return model_int8
    except Exception as e:
        print(f"[warn] INT8 quantization not available: {e}")
        return model

# -----------------------------
# Main training flow (ECE 720 style)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="MobileNetV1 binary classifier with CV + quantization")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset (ImageFolder).")
    parser.add_argument("--val_dir", type=str, default=None, help="Optional fixed validation dataset.")
    parser.add_argument("--epochs", type=int, default=0, help="Epochs per split. If 0, evaluate only.")
    parser.add_argument("--splits", type=int, default=5, help="Number of CV splits (ignored if val_dir is set).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4, help="AdamW weight decay (L2 regularization).")
    parser.add_argument("--step_size", type=int, default=50, help="LR scheduler step size (epochs).")
    parser.add_argument("--gamma", type=float, default=0.7, help="LR scheduler decay factor.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="runs/mobilenetv1")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to evaluate/resume.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights if available (timm).")
    parser.add_argument("--quantize", type=str, default="none", choices=["none","int8","fp16"], help="Quantization mode.")
    args = parser.parse_args()

    # Use the user-editable directory if set, else fall back to CLI save_dir
    plot_dir = Path(USER_PLOT_DIR) if USER_PLOT_DIR else Path(args.save_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)


    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size)
    device = torch.device(args.device)

    # Load datasets
    train_ds_eval, y_full = get_imagefolder_with_labels(args.train_dir, eval_tf)
    train_ds = datasets.ImageFolder(root=args.train_dir, transform=train_tf)  # for training augmentations

    fixed_val = None
    if args.val_dir:
        fixed_val = datasets.ImageFolder(root=args.val_dir, transform=eval_tf)

    # Containers for CV summary
    split_accs, split_losses = [], []

    # Cross-validation setup (only if fixed_val is not provided)
    if fixed_val is None:
        skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.arange(len(train_ds_eval)), y_full))
    else:
        # Single split: all training indices for train, fixed val as val
        splits = [ (np.arange(len(train_ds)), None) ]

    for split_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n=== Split {split_idx} ===")
        # Initialize training run
        if fixed_val is None:
            train_subset = Subset(train_ds, train_idx.tolist())
            val_subset   = Subset(train_ds_eval, val_idx.tolist())
        else:
            train_subset = train_ds
            val_subset   = fixed_val

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = create_mobilenet_v1(num_classes=2, pretrained=args.pretrained)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        # Resume?
        ckpt_path = os.path.join(args.save_dir, f"mnv1_split{split_idx}.pt")
        if args.resume and os.path.isfile(args.resume):
            print(f"[info] Loading checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        model = model.to(device)

        # If epochs > 0: training loop
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        if args.epochs > 0:
            print(f"Training for {args.epochs} epochs on device: {device}")
            use_amp = (args.quantize == "fp16") and device.type == "cuda"
            for epoch in range(1, args.epochs + 1):
                tl, ta = run_one_epoch(train_loader, model, criterion, optimizer=optimizer, device=device, use_amp=use_amp)
                vl, va = run_one_epoch(val_loader,   model, criterion, optimizer=None,    device=device, use_amp=False)
                train_losses.append(tl); val_losses.append(vl); val_accs.append(va); train_accs.append(ta)
                scheduler.step()
                if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
                    print(f"Epoch {epoch:03d}: train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={va*100:.2f}%")

            # Save the last epoch's model
            torch.save(model.state_dict(), ckpt_path)
            saved_train_loss, saved_val_loss = train_losses[-1], val_losses[-1]
        else:
            print("Epochs == 0, evaluating only.")
            saved_train_loss, saved_val_loss = math.nan, math.nan

        # Optional post-training quantization
        if args.quantize == "int8":
            print("[info] Performing post‑training INT8 static quantization (CPU).")
            calib_loader = val_loader if val_loader is not None else train_loader
            model_q = post_training_int8_quantize(model.to("cpu"), calib_loader)
            q_ckpt_path = os.path.join(args.save_dir, f"mnv1_split{split_idx}_int8.pt")
            torch.save(model_q.state_dict(), q_ckpt_path)
            print(f"[info] Saved INT8 model to: {q_ckpt_path}")
            model = model_q.to(device)
        elif args.quantize == "fp16":
            # Already used AMP during training; for eval we can cast weights to half for speed
            if device.type == "cuda":
                model.half()

        # Evaluate the model (validation metrics)
        vl, va = run_one_epoch(val_loader, model, criterion, optimizer=None, device=device, use_amp=False)
        print(f"Split {split_idx} — Val loss: {vl:.4f}  Val acc: {va*100:.2f}%")
        split_losses.append(vl); split_accs.append(va)

        # Plot validation curve
        if args.epochs > 0:
            #loss curve
            fig = plt.figure()
            plt.plot(range(1, len(val_losses)+1), val_losses, label="val_loss")
            plt.plot(range(1, len(train_losses)+1), train_losses, label="train_loss", alpha=0.7)
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Validation Curve — split {split_idx}")
            plt.legend(); plt.grid(True, linestyle=":")
            out_png = os.path.join(args.save_dir, f"val_curve_split{split_idx}.png")
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[info] Saved validation curve: {out_png}")

            #accuracy curve
            fig = plt.figure()
            plt.plot(range(1, len(val_accs)+1),  [a*100.0 for a in val_accs],  label="val_acc (%)")
            plt.plot(range(1, len(train_accs)+1),[a*100.0 for a in train_accs],label="train_acc (%)", alpha=0.7)
            plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title(f"Accuracy Curve — split {split_idx}")
            plt.legend(); plt.grid(True, linestyle=":")
            out_png = plot_dir / f"acc_curve_split{split_idx}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[info] Saved accuracy curve: {out_png}")

    # Plot cross‑validation error summary
    fig = plt.figure()
    plt.boxplot([np.array(split_accs)*100.0], labels=["Val Acc (%)"], vert=True, showmeans=True)
    plt.title("Cross‑Validation Accuracy Summary")
    out_png = os.path.join(args.save_dir, "cv_error_summary.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nOverall mean val acc: {np.mean(split_accs)*100:.2f}%  (+/- {np.std(split_accs)*100:.2f} std)")
    print(f"[info] Saved CV summary: {out_png}")

if __name__ == "__main__":
    main()
