# train_xs_mobilenet_coco.py
# XS-MobileNet (grayscale) for "person present?" on COCO.
# Follows the PyTorch tutorial flow: dataset/dataloader -> train loop -> val loop -> checkpoints.
# See: your uploaded tutorials for dataset/dataloader patterns and train/test loops.
# (Henry Su, senior design bring-up)

import os, time, argparse, random, math, csv
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# COCO API
# pip install pycocotools
from pycocotools.coco import COCO

# ------------------------------------------------------------
# Model blocks: Stem, Depthwise (DW), Pointwise (PW), GAP, FC
# ------------------------------------------------------------

class Stem(nn.Module):
    """3x3 s2 conv -> BN -> ReLU6 (1 -> C)""" 
    def __init__(self, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(1, out_ch, 3, 2, 1, bias=False)     # grayscale input
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act  = nn.ReLU6(inplace=True)
    def forward(self, x):  # (N,1,480,640) -> (N,out_ch,240,320)
        return self.act(self.bn(self.conv(x)))

class DW(nn.Module):
    """Depthwise 3x3 conv -> BN -> ReLU6"""
    def __init__(self, ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride, 1, groups=ch, bias=False)
        self.bn   = nn.BatchNorm2d(ch, eps=1e-3, momentum=0.03)
        self.act  = nn.ReLU6(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class PW(nn.Module):
    """Pointwise 1x1 conv -> BN -> ReLU6 (C_in -> C_out)"""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.bn   = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act  = nn.ReLU6(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class GAP(nn.Module):
    def forward(self, x): return F.adaptive_avg_pool2d(x, 1)

class FC(nn.Module):
    def __init__(self, c_in: int, c_out: int = 1):
        super().__init__()
        self.fc = nn.Linear(c_in, c_out)
    def forward(self, x): return self.fc(torch.flatten(x, 1))

def round_ch(c: int, mult: int = 8) -> int:
    return int(math.ceil(c / mult) * mult)

class XSMobileNet(nn.Module):
    """
    6 DW/PW blocks + GAP + FC (1 logit).
    Downsample path: stem(s2) -> block2(s2) -> block3(s2) -> block4(s2)
    Keep spatial in blocks 1,5,6.
    """
    def __init__(self, stem_c=16, c2=24, c3=32, c4=48, c5=64, c6=96, c7=128, round_to=8, dropout=0.0):
        super().__init__()
        stem_c = round_ch(stem_c, round_to)
        c2, c3, c4, c5, c6, c7 = [round_ch(x, round_to) for x in (c2, c3, c4, c5, c6, c7)]

        self.stem = Stem(stem_c)                 # 480x640 -> 240x320
        self.dw1  = DW(stem_c, 1); self.pw1 = PW(stem_c, c2)    # 240x320
        self.dw2  = DW(c2, 2);     self.pw2 = PW(c2, c3)        # 120x160
        self.dw3  = DW(c3, 2);     self.pw3 = PW(c3, c4)        # 60x80
        self.dw4  = DW(c4, 2);     self.pw4 = PW(c4, c5)        # 30x40
        self.dw5  = DW(c5, 1);     self.pw5 = PW(c5, c6)        # 30x40
        self.dw6  = DW(c6, 1);     self.pw6 = PW(c6, c7)        # 30x40

        self.gap  = GAP()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc   = FC(c7, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.pw1(self.dw1(x))
        x = self.pw2(self.dw2(x))
        x = self.pw3(self.dw3(x))
        x = self.pw4(self.dw4(x))
        x = self.pw5(self.dw5(x))
        x = self.pw6(self.dw6(x))
        x = self.gap(x)
        x = self.drop(x)
        return self.fc(x)  # logits

# ------------------------------------------------------------
# COCO "person present?" dataset (binary, no boxes required)
# ------------------------------------------------------------

class CocoPersonPresence(Dataset):
    """
    Label = 1 if the image has any 'person' annotation, else 0.
    Images are converted to grayscale and resized to 640x480 (WxH = 640x480).
    """
    def __init__(self, images_dir: str, ann_path: str, size_hw=(480, 640), max_images: int = None):
        super().__init__()
        self.images_dir = images_dir
        self.coco = COCO(ann_path)
        self.size_hw = size_hw  # (H, W)

        cat_ids = self.coco.getCatIds(catNms=['person'])
        assert len(cat_ids) == 1, "COCO 'person' category not found."
        self.person_cat = cat_ids[0]

        all_img_ids = self.coco.getImgIds()
        if max_images is not None:
            random.shuffle(all_img_ids)
            all_img_ids = all_img_ids[:max_images]

        self.items = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[self.person_cat], iscrowd=None)
            label = 1 if len(ann_ids) > 0 else 0
            info = self.coco.loadImgs([img_id])[0]
            path = os.path.join(images_dir, info["file_name"])
            if os.path.isfile(path):
                self.items.append((path, label))

        # Grayscale + fixed 640x480 + [0,1] normalization (ToTensor)
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.size_hw, antialias=True),  # (H, W) = (480, 640)
            transforms.ToTensor(),  # uint8 -> float32 in [0,1]
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)                         # (1, 480, 640)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y

# ------------------------------------------------------------
# Metrics / logging
# ------------------------------------------------------------

def bin_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == labels).float().mean().item()

def save_checkpoint(model, path: str, extra: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if extra: payload.update(extra)
    torch.save(payload, path)

def log_csv(path: str, row: dict, fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: w.writeheader()
        w.writerow(row)

# ------------------------------------------------------------
# Train / validate (matches PyTorch tutorial structure)
# ------------------------------------------------------------

def train_one_epoch(dl, model, loss_fn, opt, device):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for X, y in dl:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        acc_sum  += bin_accuracy_from_logits(logits.detach(), y)
        n += 1
    return (loss_sum / max(1, n)), (acc_sum / max(1, n))

@torch.no_grad()
def validate(dl, model, loss_fn, device):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for X, y in dl:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss_sum += loss.item()
        acc_sum  += bin_accuracy_from_logits(logits, y)
        n += 1
    return (loss_sum / max(1, n)), (acc_sum / max(1, n))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(
    coco_root: str,
    split_train: str = "train2017",
    split_val: str   = "val2017",
    epochs: int = 5,
    batch: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_train_images: int = None,
    max_val_images: int = None,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ckpt_dir: str = "runs/checkpoints",
    log_csv_path: str = "runs/logs/train_log.csv",
    use_plateau_scheduler: bool = True
):
    # Repro
    torch.manual_seed(42)
    random.seed(42)

    train_imgs = os.path.join(coco_root, split_train)
    val_imgs   = os.path.join(coco_root, split_val)
    train_ann  = os.path.join(coco_root, "annotations", f"instances_{split_train}.json")
    val_ann    = os.path.join(coco_root, "annotations", f"instances_{split_val}.json")

    # Datasets / DataLoaders (tutorial style)  :contentReference[oaicite:3]{index=3}
    ds_train = CocoPersonPresence(train_imgs, train_ann, size_hw=(480, 640), max_images=max_train_images)
    ds_val   = CocoPersonPresence(val_imgs,   val_ann,   size_hw=(480, 640), max_images=max_val_images)

    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model / loss / optimizer (tutorial train/test structure)  :contentReference[oaicite:4]{index=4}
    model = XSMobileNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Validation-aware LR scheduler (step at end of epoch after val)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, verbose=True) \
            if use_plateau_scheduler else None

    print(f"Train images: {len(ds_train)} | Val images: {len(ds_val)} | Device: {device}")
    best_val_acc = 0.0
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(dl_train, model, loss_fn, opt, device)
        va_loss, va_acc = validate(dl_val, model, loss_fn, device)

        if sched is not None:
            # ReduceLROnPlateau expects a metric; we use validation accuracy
            sched.step(va_acc)

        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"epoch {epoch:02d} | train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val_loss {va_loss:.4f} acc {va_acc:.3f} | lr {lr_now:.2e} | {dt:.1f}s")

        log_csv(log_csv_path, {
            "epoch": epoch,
            "train_loss": f"{tr_loss:.6f}",
            "train_acc": f"{tr_acc:.6f}",
            "val_loss": f"{va_loss:.6f}",
            "val_acc": f"{va_acc:.6f}",
            "lr": f"{lr_now:.6e}",
            "epoch_time_s": f"{dt:.2f}"
        }, fields)

        # Checkpointing: best + last (tutorial best practice)  :contentReference[oaicite:5]{index=5}
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(model, os.path.join(ckpt_dir, "best.pt"),
                            {"epoch": epoch, "val_acc": va_acc, "train_acc": tr_acc})
        save_checkpoint(model, os.path.join(ckpt_dir, "last.pt"),
                        {"epoch": epoch, "val_acc": va_acc, "train_acc": tr_acc})

    print(f"Done. Best val_acc = {best_val_acc:.3f}. "
          f"Checkpoints in {ckpt_dir}/best.pt and {ckpt_dir}/last.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="XS-MobileNet (grayscale 640x480) - COCO person presence")
    ap.add_argument("--coco_root", type=str, required=True,
                    help="COCO root with train2017/, val2017/, annotations/")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_train_images", type=int, default=None)
    ap.add_argument("--max_val_images", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt_dir", type=str, default="runs/checkpoints")
    ap.add_argument("--log_csv", type=str, default="runs/logs/train_log.csv")
    ap.add_argument("--no_plateau_scheduler", action="store_true",
                    help="Disable ReduceLROnPlateau (validation-aware) scheduler")
    args = ap.parse_args()

    main(
        coco_root=args.coco_root,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        num_workers=args.num_workers,
        device=args.device,
        ckpt_dir=args.ckpt_dir,
        log_csv_path=args.log_csv,
        use_plateau_scheduler=not args.no_plateau_scheduler
    )


#to run:

#pip install torch torchvision pillow pycocotools

# #make sure coco2017 is laid out like:
# <COCO_ROOT>/
#   train2017/
#   val2017/
#   annotations/
#     instances_train2017.json
#     instances_val2017.json

# python train_xs_mobilenet_coco.py \
#   --coco_root <COCO_ROOT> \
#   --epochs 5 \
#   --batch 64 \
#   --max_train_images 20000 \
#   --max_val_images 5000

#this code does not have int8 quantization yet
#Native Pytorch QAT can be inserted right after constructing the model
#After training is done, switch to eval() and call convert(model, inplace = true) to get
#a quantized int8 model for inference / export.
#Pytorch QAT targets int8