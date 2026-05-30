import os
import sys
import random
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision

try:
    import timm
except ImportError:
    timm = None

try:
    import torchvision.ops as ops
    HAS_DEFORM_CONV = hasattr(ops, "DeformConv2d")
except Exception:
    ops = None
    HAS_DEFORM_CONV = False

# --- UTILS ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _signed_distance_map(mask01: np.ndarray) -> np.ndarray:
    mask_u8 = (mask01 > 0).astype(np.uint8)
    if mask_u8.max() == 0:
        return np.full(mask_u8.shape, -1.0, dtype=np.float32)
    pos_dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    neg_dist = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 5)
    sdm = (pos_dist - neg_dist)
    max_abs = float(np.max(np.abs(sdm)))
    if max_abs > 0: sdm /= max_abs
    return sdm.astype(np.float32)

def _disconnect_tensor(mask_tensor: torch.Tensor) -> torch.Tensor:
    """Adds random breaks to the tensor mask to simulate disconnected vessels."""
    B, C, H, W = mask_tensor.shape
    device = mask_tensor.device
    mask = mask_tensor.clone()
    for b in range(B):
        num_breaks = random.randint(4, 10)
        indices = torch.nonzero(mask[b, 0] > 0.5)
        if len(indices) > 0:
            for _ in range(num_breaks):
                idx = random.randint(0, len(indices) - 1)
                y, x = indices[idx]
                r = random.randint(3, 8)
                # Create coordinate grids
                Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                dist_sq = (Y - y)**2 + (X - x)**2
                mask[b, 0][dist_sq <= r**2] = 0
    return mask

def _tp_fp_fn(preds: torch.Tensor, masks: torch.Tensor):
    p = preds.int()
    m = masks.int()
    tp = torch.logical_and(p == 1, m == 1).sum().item()
    fp = torch.logical_and(p == 1, m == 0).sum().item()
    fn = torch.logical_and(p == 0, m == 1).sum().item()
    return float(tp), float(fp), float(fn)

def _f1_iou_dice_from_counts(tp: float, fp: float, fn: float):
    f1 = (2.0 * tp) / max(1e-8, (2.0 * tp + fp + fn))
    iou = tp / max(1e-8, (tp + fp + fn))
    dice = (2.0 * tp) / max(1e-8, (2.0 * tp + fp + fn))
    return float(f1), float(iou), float(dice)

def _log_prediction_grid(tb_writer: SummaryWriter, tag: str, step: int, images, masks, probs, threshold: float = 0.5):
    num_samples = min(4, images.size(0))
    grid_images = []
    preds = (probs > threshold).float()

    for i in range(num_samples):
        # We take the first channel of the original image (CLAHE) for visualization
        img_vis = images[i, 0:1].detach().cpu().repeat(3, 1, 1)
        pred_vis = preds[i, 0:1].detach().cpu().repeat(3, 1, 1)
        mask_vis = masks[i, 0:1].detach().cpu().repeat(3, 1, 1)
        grid_images.extend([img_vis, pred_vis, mask_vis])

    grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
    tb_writer.add_image(tag, grid, step)

# --- DATASET ---
class VesselDatasetV3(Dataset):
    def __init__(
        self,
        json_path: str,
        split: str = "train",
        img_size: int = 512,
        mode: str = "train",
        root_dir: str = ".",
    ):
        self.img_size = img_size
        self.mode = mode
        self.root_dir = Path(root_dir)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        with open(json_path, "r") as f:
            data = yaml.safe_load(f)
        
        split_data = data.get(split, data.get("validation" if split=="val" else split, {}))
        
        # We ONLY want the syntax dataset for now (main model training for vessels)
        self.samples = []
        if "syntax" in split_data and isinstance(split_data["syntax"], dict):
            for s_id, s_info in split_data["syntax"].items():
                img_path = s_info.get("data")
                lbl_path = s_info.get("label")
                if img_path and lbl_path and isinstance(img_path, str) and isinstance(lbl_path, str):
                    self.samples.append({"img": img_path, "lbl": lbl_path})
                    
        print(f"[INFO] Dataset ({split}): {len(self.samples)} syntax samples loaded.")

    def _inject_artifacts(self, img: np.ndarray):
        if self.mode == "train":
            if random.random() < 0.3:
                for _ in range(random.randint(1, 3)):
                    x1, y1 = random.randint(0, self.img_size), random.randint(0, self.img_size)
                    x2, y2 = x1 + random.randint(-150, 150), y1 + random.randint(-150, 150)
                    cv2.line(img, (x1, y1), (x2, y2), random.randint(10, 80), random.randint(2, 5))
            if random.random() < 0.2:
                cx, cy = random.randint(0, self.img_size), random.randint(0, self.img_size)
                cv2.circle(img, (cx, cy), random.randint(15, 60), random.randint(30, 100), -1)
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(self.root_dir / s["img"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        if self.mode == "train":
            img = self._inject_artifacts(img)
            if random.random() > 0.65: 
                sigma = random.uniform(0.5, 1.5)
                img = cv2.GaussianBlur(img, (5, 5), sigma)
            if random.random() > 0.5: 
                img = np.clip(random.uniform(0.8, 1.2) * img + random.randint(-10, 10), 0, 255).astype(np.uint8)

        # 1-Channel Vessel Mask
        mask = cv2.imread(str(self.root_dir / s["lbl"]), 0) if s["lbl"] else np.zeros_like(img)
        mask = (cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

        # 4-Channel Prep
        c1 = self.clahe.apply(img)
        c2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, self.morph_kernel)
        c3 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, self.morph_kernel)
        c4 = cv2.addWeighted(img, 4.0, cv2.GaussianBlur(img, (0, 0), 10), -4.0, 128)
        channels = [c1, c2, c3, c4]

        img_t = torch.from_numpy(np.stack(channels, -1).astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        sdm_t = torch.from_numpy(_signed_distance_map(mask)).unsqueeze(0).float()

        return img_t, mask_t, sdm_t, s["img"]

# --- MODEL ---
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.act = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]: g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        return x * self.psi(self.act(g1 + x1))

class DCNBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        if HAS_DEFORM_CONV:
            self.offset = nn.Conv2d(out_c, 18, 3, padding=1)
            self.conv = ops.DeformConv2d(out_c, out_c, 3, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        res = self.proj(x)
        out = self.conv(res, self.offset(res)) if HAS_DEFORM_CONV else self.conv(res)
        return self.act(self.norm(out) + res)

class SubPixelUp(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(in_c, out_c * 4, 3, padding=1), nn.PixelShuffle(2), nn.GELU())
        self.att = AttentionGate(out_c, skip_c, out_c // 2)
        self.fuse = DCNBlock(out_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]: x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, self.att(x, skip)], 1))

class VesselNetV3(nn.Module):
    def __init__(self, in_chans=4, num_classes=1, encoder_name="efficientnetv2_s", pretrained=True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for VesselNetV3 but is not installed.")

        try:
            self.encoder = timm.create_model(
                encoder_name,
                features_only=True,
                in_chans=in_chans,
                pretrained=pretrained,
            )
        except RuntimeError as e:
            if pretrained and "No pretrained weights exist" in str(e):
                print(f"[WARN] {e} Falling back to pretrained=False for {encoder_name}.")
                self.encoder = timm.create_model(encoder_name, features_only=True, in_chans=in_chans, pretrained=False)
            else:
                raise
        ch = self.encoder.feature_info.channels()
        self.up4 = SubPixelUp(ch[4], ch[3], ch[3])
        self.up3 = SubPixelUp(ch[3], ch[2], ch[2])
        self.up2 = SubPixelUp(ch[2], ch[1], ch[1])
        self.up1 = SubPixelUp(ch[1], ch[0], ch[0])
        self.up0 = nn.Sequential(nn.Conv2d(ch[0], ch[0]*4, 3, padding=1), nn.PixelShuffle(2), nn.BatchNorm2d(ch[0]), nn.GELU(),
                                 nn.Conv2d(ch[0], 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.seg_head = nn.Conv2d(32, num_classes, 1)
        self.sdm_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        f = self.encoder(x)
        d = self.up1(self.up2(self.up3(self.up4(f[4], f[3]), f[2]), f[1]), f[0])
        d = self.up0(d)
        seg, sdm = self.seg_head(d), self.sdm_head(d)
        if seg.shape[2:] != (h, w):
            seg = F.interpolate(seg, (h, w), mode="bilinear", align_corners=False)
            sdm = F.interpolate(sdm, (h, w), mode="bilinear", align_corners=False)
        return {"seg": seg, "sdm": sdm}

# --- LOSS & TRAIN ---
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.SmoothL1Loss()

    def soft_cldice(self, p, t):
        def erode(x): return -F.max_pool2d(-x, 3, 1, 1)
        def dilate(x): return F.max_pool2d(x, 3, 1, 1)
        def skel(x):
            s = F.relu(x - dilate(erode(x)))
            for _ in range(5):
                x = erode(x)
                delta = F.relu(x - dilate(erode(x)))
                s = s + F.relu(delta - s * delta)
            return s
        losses = []
        for i in range(p.shape[1]):
            pi, ti = p[:, i:i+1], t[:, i:i+1]
            if ti.sum() == 0: continue
            sp, st = skel(pi), skel(ti)
            prec = (sp * ti).sum() / (sp.sum() + 1e-8)
            sens = (st * pi).sum() / (st.sum() + 1e-8)
            losses.append(1.0 - 2.0 * (prec * sens) / (prec + sens + 1e-8))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=p.device)

    def forward(self, pred, target, sdm_target):
        return self.bce(pred["seg"], target) + 0.2 * self.soft_cldice(torch.sigmoid(pred["seg"]), target) + 0.1 * self.l1(torch.tanh(pred["sdm"]), sdm_target)

def train_epoch(model, loader, opt, crit, dev, epoch, writer, tag, base_model=None):
    model.train()
    if base_model: base_model.eval()
    
    l_sum = 0
    total_tp, total_fp, total_fn = 0.0, 0.0, 0.0
    
    pbar = tqdm(loader, desc=f"Train {tag}")
    
    last_batch = None
    last_preds = None
    
    for img, m, sdm, _ in pbar:
        img, m, sdm = img.to(dev), m.to(dev), sdm.to(dev)
        
        if base_model:
            with torch.no_grad():
                base_out = base_model(img)
                base_mask = torch.sigmoid(base_out["seg"])
                base_mask = _disconnect_tensor(base_mask)
                img = torch.cat([img, base_mask], dim=1)
                
        opt.zero_grad()
        out = model(img)
        loss = crit(out, m, sdm)
        loss.backward()
        opt.step()
        l_sum += loss.item()
        
        # Metrics calculation
        preds_prob = torch.sigmoid(out["seg"])
        preds_bin = (preds_prob > 0.5).float()
        
        tp, fp, fn = _tp_fp_fn(preds_bin, m)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        f1, _, _ = _f1_iou_dice_from_counts(tp, fp, fn)
        pbar.set_postfix({"loss": loss.item(), "batch_f1": f1})
        
        last_batch = (img, m)
        last_preds = preds_prob
        
    f1_epoch, iou_epoch, dice_epoch = _f1_iou_dice_from_counts(total_tp, total_fp, total_fn)
    
    if writer and last_batch is not None:
        _log_prediction_grid(writer, f"{tag}/Train_Preds", epoch, last_batch[0], last_batch[1], last_preds)
        writer.add_scalar(f"Loss/{tag}_Train", l_sum / len(loader), epoch)
        writer.add_scalar(f"F1/{tag}_Train", f1_epoch, epoch)
        
    return l_sum / len(loader), f1_epoch

def val_epoch(model, loader, crit, dev, epoch, writer, tag, base_model=None):
    model.eval()
    if base_model: base_model.eval()
    
    l_sum = 0
    total_tp, total_fp, total_fn = 0.0, 0.0, 0.0
    
    last_batch = None
    last_preds = None

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Val {tag}")
        for img, m, sdm, _ in pbar:
            img, m, sdm = img.to(dev), m.to(dev), sdm.to(dev)
            if base_model:
                base_out = base_model(img)
                base_mask = torch.sigmoid(base_out["seg"])
                base_mask = _disconnect_tensor(base_mask) 
                img = torch.cat([img, base_mask], dim=1)
                
            out = model(img)
            loss = crit(out, m, sdm)
            l_sum += loss.item()
            
            # Metrics calculation
            preds_prob = torch.sigmoid(out["seg"])
            preds_bin = (preds_prob > 0.5).float()
            
            tp, fp, fn = _tp_fp_fn(preds_bin, m)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            last_batch = (img, m)
            last_preds = preds_prob
            
    f1_epoch, iou_epoch, dice_epoch = _f1_iou_dice_from_counts(total_tp, total_fp, total_fn)
    
    if writer and last_batch is not None:
        _log_prediction_grid(writer, f"{tag}/Val_Preds", epoch, last_batch[0], last_batch[1], last_preds)
        writer.add_scalar(f"Loss/{tag}_Val", l_sum / len(loader), epoch)
        writer.add_scalar(f"F1/{tag}_Val", f1_epoch, epoch)
        
    return l_sum / len(loader), f1_epoch

def load_vessel_model(ckpt, in_c=4, nc=1, dev="cpu"):
    m = VesselNetV3(in_chans=in_c, num_classes=nc).to(dev)
    if os.path.isfile(ckpt):
        d = torch.load(ckpt, map_location=dev)
        m.load_state_dict(d["model_state_dict"] if "model_state_dict" in d else d)
        print(f"[INFO] Loaded: {ckpt}")
    return m.eval()

if __name__ == "__main__":
    # TASK choices: "train_base", "train_refiner"
    # Step 1: Set TASK = "train_base" (Trains the 4-channel model on SYNTAX)
    # Step 2: Set TASK = "train_refiner" (Trains the 5-channel model to reconnect vessels)
    TASK = "train_base" 
    
    EP, BS, LR, DEV = 100, 8, 2e-4, "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    if TASK.startswith("train"):
        IS_R = (TASK == "train_refiner")
        
        # We increase num_workers and enable pin_memory to speed up the data loading
        ds_t = VesselDatasetV3("data/ARCADE/processed/dataset.json", split="train", mode="train")
        ds_v = VesselDatasetV3("data/ARCADE/processed/dataset.json", split="val", mode="val")
        ds_test = VesselDatasetV3("data/ARCADE/processed/dataset.json", split="test", mode="val")
        
        ld_t = DataLoader(ds_t, BS, shuffle=True, num_workers=8, pin_memory=True)
        ld_v = DataLoader(ds_v, BS, shuffle=False, num_workers=8, pin_memory=True)
        ld_test = DataLoader(ds_test, BS, shuffle=False, num_workers=8, pin_memory=True)
        
        base_model = None
        if IS_R:
            print("[INFO] Loading Main Model for Refiner training...")
            base_model = load_vessel_model("checkpoints/v3_base.pth", in_c=4, nc=1, dev=DEV)
            
        model = VesselNetV3(in_chans=5 if IS_R else 4, num_classes=1).to(DEV)
        opt = optim.AdamW(model.parameters(), lr=LR)
        crit = HybridLoss()
        sch = CosineAnnealingLR(opt, T_max=EP)
        writer = SummaryWriter(f"runs/v3_{TASK}")

        for e in range(EP):
            tl, t_f1 = train_epoch(model, ld_t, opt, crit, DEV, e, writer, tag=TASK, base_model=base_model)
            vl, v_f1 = val_epoch(model, ld_v, crit, DEV, e, writer, tag=TASK, base_model=base_model)
            sch.step()
            print(f"Ep {e+1}/{EP} | Train Loss: {tl:.4f} F1: {t_f1:.4f} | Val Loss: {vl:.4f} F1: {v_f1:.4f}")
            if (e+1) % 10 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/v3_{'refiner' if IS_R else 'base'}.pth")
        
        # Test Phase
        print(f"\n[INFO] Starting Test Phase for {TASK}...")
        model.load_state_dict(torch.load(f"checkpoints/v3_{'refiner' if IS_R else 'base'}.pth", map_location=DEV))
        test_l, test_f1 = val_epoch(model, ld_test, crit, DEV, EP, writer, tag=f"{TASK}_Test", base_model=base_model)
        print(f"[INFO] Test Complete | Loss: {test_l:.4f} | F1: {test_f1:.4f}")
        
        writer.close()
