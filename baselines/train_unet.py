"""
baselines/train_unet.py
========================
UNet baseline: random-init ResNet34 + SMP UNet, end-to-end training.

Two-stage transfer pipeline:
  Stage 1: TASK="syntax",  PRETRAINED_FROM=None -> train vessel seg
  Stage 2: TASK="stenoza", PRETRAINED_FROM=vessel.pt -> fine-tune stenosis

Checkpoint name = RUN_NAME (or auto: {TASK}_seed{SEED}).
Set RUN_NAME at top to identify each run and avoid overwrites.

MLflow logs all params + per-epoch metrics + final test F1.
Primary metric: F1 on positive class (vessel / stenosis pixels).
Loss: BCE(pos_weight) + Dice.
"""
import sys
sys.path.append(".")

import os
import random
import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image

from utils.helpers import set_seed
from data.data import finetune_dataset


# === CONFIG ===================================================
TASK = "syntax"               # "syntax" (vessel) | "stenoza" (stenosis)
PRETRAINED_FROM = None        # path to .pt for stage 2 (None = from scratch)

SEED = 0
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 50
WEIGHT_DECAY = 0.01
USE_AUGMENTATION = True
RUN_NAME = None              # None = auto "{TASK}_seed{SEED}"; set string for custom name

POS_WEIGHT_BY_TASK = {"syntax": 30.0, "stenoza": 100.0}
ENCODER_NAME = "resnet34"
IMAGE_SIZE = 224

CHECKPOINT_DIR = "checkpoints/baselines"
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================================


# === DATASET ==================================================

class UnetDataset(Dataset):
    def __init__(self, task, split, augment=False):
        self.ds = finetune_dataset(split=split, task=task, transform=None)
        self.augment = augment

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]
        image = image.convert("L")
        label = label.convert("L")

        image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        label = TF.resize(label, [IMAGE_SIZE, IMAGE_SIZE], interpolation=Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image); label = TF.hflip(label)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, fill=[0])
                label = TF.rotate(label, angle, fill=[0])

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5], std=[0.5])
        label = TF.to_tensor(label)
        label = (label > 0.5).float()
        return image, label


def build_loader(task, split, batch_size, augment=False):
    ds = UnetDataset(task, split, augment=augment)
    shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)


# === METRICS ==================================================
# F1 positive = Dice. F1 macro = (F1_pos + F1_neg) / 2.

def f1_metrics(pred_logits, target, eps=1e-7):
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    target = target.float()
    tp = (pred_bin * target).sum(dim=(2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(2, 3))
    tn = ((1 - pred_bin) * (1 - target)).sum(dim=(2, 3))

    p_pos = tp / (tp + fp + eps)
    r_pos = tp / (tp + fn + eps)
    f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos + eps)

    p_neg = tn / (tn + fn + eps)
    r_neg = tn / (tn + fp + eps)
    f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg + eps)

    f1_macro = (f1_pos + f1_neg) / 2
    return f1_pos.squeeze(1), f1_macro.squeeze(1)


# === LOSS =====================================================

def dice_loss(pred_logits, target, eps=1e-7):
    pred_s = torch.sigmoid(pred_logits)
    inter = (pred_s * target).sum(dim=(2, 3))
    union = pred_s.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def loss_fn(pred_logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target,
        pos_weight=torch.tensor(pos_weight, device=pred_logits.device)
    )
    return bce + dice_loss(pred_logits, target)


# === EVAL =====================================================

def evaluate(model, loader, pos_weight):
    model.eval()
    f1_pos_list, f1_macro_list, losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            pred = model(x)
            losses.append(loss_fn(pred, y, pos_weight).item())
            f1p, f1m = f1_metrics(pred, y)
            f1_pos_list.extend(f1p.tolist())
            f1_macro_list.extend(f1m.tolist())
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "f1_positive": sum(f1_pos_list) / max(len(f1_pos_list), 1),
        "f1_macro": sum(f1_macro_list) / max(len(f1_macro_list), 1),
    }


# === TRAIN ====================================================

def train():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pos_weight = POS_WEIGHT_BY_TASK[TASK]

    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=1,
        classes=1,
    ).to(DEVICE)

    if PRETRAINED_FROM is not None:
        ckpt = torch.load(PRETRAINED_FROM, map_location=DEVICE)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Loaded pretrained UNet from {PRETRAINED_FROM}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNet ({ENCODER_NAME}, random init) — {total_params:,} params")

    train_loader = build_loader(TASK, "train", BATCH_SIZE, augment=USE_AUGMENTATION)
    val_loader   = build_loader(TASK, "validation", BATCH_SIZE, augment=False)
    test_loader  = build_loader(TASK, "test", BATCH_SIZE, augment=False)
    print(f"Splits | train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)} | test: {len(test_loader.dataset)}")
    print(f"Loss: BCE(pos_weight={pos_weight}) + Dice   |   Augment: {USE_AUGMENTATION}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    run_name = RUN_NAME if RUN_NAME else f"{TASK}_seed{SEED}"

    base = run_name
    counter = 1
    while os.path.exists(os.path.join(CHECKPOINT_DIR, f"{run_name}.pt")):
        counter += 1
        run_name = f"{base}_v{counter}"

    exp_name = f"unet_baseline_{TASK}" + ("_from_vessel" if PRETRAINED_FROM else "")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "task": TASK,
            "pretrained_from": PRETRAINED_FROM or "scratch",
            "run_name": run_name,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "pos_weight": pos_weight,
            "augmentation": USE_AUGMENTATION,
            "encoder": ENCODER_NAME,
            "encoder_weights": "random",
            "image_size": IMAGE_SIZE,
            "total_params": total_params,
            "device": DEVICE,
        })

        print(f"\nMLflow | experiment={exp_name} | run={run_name} | run_id={run_id}")
        print(f"Tracking URI: {MLFLOW_TRACKING_URI}\n")

        best_val_f1 = 0.0
        best_epoch = -1
        save_path = os.path.join(CHECKPOINT_DIR, f"{run_name}.pt")

        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for x, y in train_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                pred = model(x)
                loss = loss_fn(pred, y, pos_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()
            train_loss = sum(train_losses) / max(len(train_losses), 1)

            val_m = evaluate(model, val_loader, pos_weight)

            mlflow.log_metric("train_loss",     train_loss,           step=epoch)
            mlflow.log_metric("val_loss",       val_m["loss"],        step=epoch)
            mlflow.log_metric("val_f1_positive", val_m["f1_positive"], step=epoch)
            mlflow.log_metric("val_f1_macro",   val_m["f1_macro"],    step=epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0],      step=epoch)

            print(f"ep {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  "
                  f"val_f1_pos={val_m['f1_positive']:.4f}  val_f1_macro={val_m['f1_macro']:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

            if val_m["f1_positive"] > best_val_f1:
                best_val_f1 = val_m["f1_positive"]
                best_epoch = epoch
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_f1_positive": best_val_f1,
                    "config": {"task": TASK, "lr": LR, "epochs": EPOCHS, "seed": SEED},
                }, save_path)

        best = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(best["model"])
        test_m = evaluate(model, test_loader, pos_weight)

        mlflow.log_metrics({
            "test_f1_positive": test_m["f1_positive"],
            "test_f1_macro":    test_m["f1_macro"],
            "test_loss":        test_m["loss"],
            "best_val_f1_positive": best_val_f1,
            "best_epoch": best_epoch,
        })

        print(f"\n=== TEST (best epoch {best_epoch}) ===")
        print(f"  F1 positive : {test_m['f1_positive']:.4f}")
        print(f"  F1 macro    : {test_m['f1_macro']:.4f}")
        print(f"  checkpoint  : {save_path}")
        print(f"  mlflow run  : {run_id}")


if __name__ == "__main__":
    train()