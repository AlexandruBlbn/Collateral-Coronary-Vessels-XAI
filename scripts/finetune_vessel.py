"""
finetune_vessel.py
==================
Fine-tunes VesselNetV2EfficientEncoder (best_model.pth) on the 126 all-vessel
annotated images in data/Extra/ so the model learns to segment ALL vessels
(collaterals, branches) instead of only LCA/RCA.

Run:
    python scripts/finetune_vessel.py
"""

import sys, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add engine to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from trainv2 import VesselNetV2EfficientEncoder

# ── Config ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
CHECKPOINT  = BASE / "Demo" / "best_model.pth"
SAVE_PATH   = BASE / "Demo" / "best_model_allvessel.pth"
IMG_DIR     = BASE / "data" / "Extra" / "images"
MASK_DIR    = BASE / "data" / "Extra" / "masks"

IMG_SIZE    = 512
BATCH_SIZE  = 4       # safe for 8GB VRAM with efficientnetv2_s
EPOCHS      = 50
LR          = 1e-4
VAL_SPLIT   = 0.15    # 15% validation (~19 images)
SEED        = 42

# ── Preprocessing (identical to InferencePreprocessor) ──────────────────────
_CLAHE      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_MORPH_KERN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

def make_4ch(arr: np.ndarray) -> np.ndarray:
    """Convert grayscale uint8 (H,W) to 4-channel float32 (4,H,W) in [0,1]."""
    c1 = _CLAHE.apply(arr)
    c2 = cv2.morphologyEx(arr, cv2.MORPH_TOPHAT,   _MORPH_KERN)
    c3 = cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, _MORPH_KERN)
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=10)
    c4   = np.clip(cv2.addWeighted(arr, 4.0, blur, -4.0, 128), 0, 255).astype(np.uint8)
    stack = np.stack([c1, c2, c3, c4], axis=0).astype(np.float32) / 255.0
    return stack


# ── Augmentations (minimal, geometry-safe) ──────────────────────────────────
def augment(img4: np.ndarray, mask: np.ndarray):
    """img4: (4,H,W) float32  |  mask: (H,W) float32 {0,1}"""
    # Random horizontal flip
    if random.random() > 0.5:
        img4 = img4[:, :, ::-1].copy()
        mask = mask[:, ::-1].copy()
    # Random vertical flip
    if random.random() > 0.5:
        img4 = img4[:, ::-1, :].copy()
        mask = mask[::-1, :].copy()
    # Random 90-degree rotation
    k = random.randint(0, 3)
    if k > 0:
        img4 = np.rot90(img4, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k).copy()
    return img4, mask


# ── Dataset ──────────────────────────────────────────────────────────────────
class VesselDataset(Dataset):
    def __init__(self, pairs, augment_fn=None):
        self.pairs      = pairs
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        img4 = make_4ch(img)                          # (4,H,W) float32 [0,1]
        mask_f = (mask > 127).astype(np.float32)      # (H,W) {0,1}

        if self.augment_fn is not None:
            img4, mask_f = self.augment_fn(img4, mask_f)

        return (
            torch.from_numpy(img4),
            torch.from_numpy(mask_f).unsqueeze(0)    # (1,H,W)
        )


# ── Loss ─────────────────────────────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    flat_p = pred.view(-1)
    flat_t = target.view(-1)
    inter  = (flat_p * flat_t).sum()
    return 1.0 - (2.0 * inter + smooth) / (flat_p.sum() + flat_t.sum() + smooth)

def combined_loss(pred_dict, target):
    seg = pred_dict["seg_logits"]
    bce = F.binary_cross_entropy_with_logits(seg, target)
    dc  = dice_loss(seg, target)
    loss = 0.6 * bce + 0.4 * dc

    # Deep supervision on intermediate heads
    for dl in pred_dict.get("deep_logits", []):
        dl_up = F.interpolate(dl, size=target.shape[2:], mode="bilinear", align_corners=False)
        loss  = loss + 0.2 * (0.6 * F.binary_cross_entropy_with_logits(dl_up, target)
                              + 0.4 * dice_loss(dl_up, target))
    return loss


# ── Metric ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def dice_score(pred_dict, target, threshold=0.5):
    prob = torch.sigmoid(pred_dict["seg_logits"])
    pred = (prob > threshold).float()
    inter = (pred * target).sum()
    return (2.0 * inter / (pred.sum() + target.sum() + 1e-6)).item()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Collect pairs ─────────────────────────────────────────────────────
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_files  = sorted([p for p in IMG_DIR.rglob("*")  if p.suffix.lower() in exts])
    mask_files = sorted([p for p in MASK_DIR.rglob("*") if p.suffix.lower() in exts])

    # Match by stem name
    mask_by_stem = {p.stem: p for p in mask_files}
    pairs = [(img, mask_by_stem[img.stem])
             for img in img_files if img.stem in mask_by_stem]
    print(f"Matched pairs: {len(pairs)}")

    # ── Train/Val split ───────────────────────────────────────────────────
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * VAL_SPLIT))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    train_ds = VesselDataset(train_pairs, augment_fn=augment)
    val_ds   = VesselDataset(val_pairs,   augment_fn=None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=2,          shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = VesselNetV2EfficientEncoder(
        in_chans=4, num_classes=1,
        encoder_name="efficientnetv2_s",
        encoder_pretrained=False,
    )

    # Load pretrained weights (strict=False: ignores size mismatches if any)
    state = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"[INFO] Missing keys  ({len(missing)}): {missing[:3]} ...")
    if unexpected: print(f"[INFO] Unexpected    ({len(unexpected)}): {unexpected[:3]} ...")
    print(f"Loaded pretrained weights from: {CHECKPOINT.name}")

    model.to(device)

    # ── Optimizer: lower LR for encoder, higher for decoder ───────────────
    enc_params = list(model.encoder.parameters())
    dec_params = [p for p in model.parameters()
                  if not any(p is ep for ep in enc_params)]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR * 0.1},   # encoder: fine-tune gently
        {"params": dec_params, "lr": LR},          # decoder: learn freely
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────
    best_dice  = 0.0
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [train]", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred  = model(imgs)
                loss  = combined_loss(pred, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= len(train_loader)

        # -- Validate --
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(imgs)
                val_dice += dice_score(pred, masks)
        val_dice /= len(val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={train_loss:.4f}  val_dice={val_dice:.4f}",
              "  [BEST]" if val_dice > best_dice else "")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), str(SAVE_PATH))

    print(f"\nDone! Best val Dice: {best_dice:.4f}")
    print(f"Checkpoint saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
