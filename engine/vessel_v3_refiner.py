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

# --- DATASET ---
class VesselDatasetV3(Dataset):
    def __init__(
        self,
        json_path: str,
        split: str = "train",
        img_size: int = 512,
        mode: str = "train",
        root_dir: str = ".",
        refiner_mode: bool = False,
    ):
        self.img_size = img_size
        self.mode = mode
        self.root_dir = Path(root_dir)
        self.refiner_mode = refiner_mode
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        with open(json_path, "r") as f:
            data = yaml.safe_load(f)
        
        split_data = data.get(split, data.get("validation" if split=="val" else split, {}))
        
        # Collect and pair labels across all sources (syntax, stenoza, extra, cadica, etc.)
        path_to_labels = {} 
        for src_name, src_data in split_data.items():
            if not isinstance(src_data, dict): continue
            for s_id, s_info in src_data.items():
                img_path = s_info.get("data")
                if not img_path: continue
                if img_path not in path_to_labels:
                    path_to_labels[img_path] = {"v": None, "s": None}
                lbl = s_info.get("label")
                if src_name == "stenoza":
                    path_to_labels[img_path]["s"] = lbl
                else:
                    if path_to_labels[img_path]["v"] is None or src_name == "syntax":
                        path_to_labels[img_path]["v"] = lbl
        
        self.samples = [{"img": p, "v_lbl": l["v"], "s_lbl": l["s"]} for p, l in path_to_labels.items()]
        print(f"[INFO] Dataset ({split}): {len(self.samples)} unique samples.")

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

    def _disconnect_vessels(self, mask: np.ndarray) -> np.ndarray:
        distorted = mask.copy()
        if self.mode == "train" and np.sum(distorted) > 0:
            num_breaks = random.randint(2, 6)
            coords = np.argwhere(distorted[..., 0] > 0)
            if len(coords) > 0:
                for _ in range(num_breaks):
                    idx = random.randint(0, len(coords) - 1)
                    y, x = coords[idx]
                    radius = random.randint(2, 5)
                    cv2.circle(distorted[..., 0], (x, y), radius, 0, -1)
        return distorted

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(self.root_dir / s["img"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        if self.mode == "train":
            img = self._inject_artifacts(img)
            if random.random() > 0.65: img = cv2.GaussianBlur(img, (5, 5), random.uniform(0.5, 1.5))
            if random.random() > 0.5: img = np.clip(random.uniform(0.8, 1.2) * img + random.randint(-10, 10), 0, 255).astype(np.uint8)

        # Labels
        v_mask = cv2.imread(str(self.root_dir / s["v_lbl"]), 0) if s["v_lbl"] else np.zeros_like(img)
        s_mask = cv2.imread(str(self.root_dir / s["s_lbl"]), 0) if s["s_lbl"] else np.zeros_like(img)
        v_mask = (cv2.resize(v_mask, (self.img_size, self.img_size), interpolation=0) > 127).astype(np.uint8)
        s_mask = (cv2.resize(s_mask, (self.img_size, self.img_size), interpolation=0) > 127).astype(np.uint8)
        
        mask_2ch = np.stack([v_mask, s_mask], axis=-1)

        # 4-Channel Prep
        c1 = self.clahe.apply(img)
        c2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, self.morph_kernel)
        c3 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, self.morph_kernel)
        c4 = cv2.addWeighted(img, 4.0, cv2.GaussianBlur(img, (0, 0), 10), -4.0, 128)
        channels = [c1, c2, c3, c4]
        
        if self.refiner_mode:
            channels.append(self._disconnect_vessels(mask_2ch)[..., 0] * 255)

        img_t = torch.from_numpy(np.stack(channels, -1).astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_2ch).permute(2, 0, 1).float()
        sdm_t = torch.from_numpy(np.stack([_signed_distance_map(v_mask), _signed_distance_map(s_mask)], 0)).float()

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
        if g1.shape[2:] != x1.shape[2:]: g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear")
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
        if x.shape[2:] != skip.shape[2:]: x = F.interpolate(x, size=skip.shape[2:], mode="bilinear")
        return self.fuse(torch.cat([x, self.att(x, skip)], 1))

class VesselNetV3(nn.Module):
    def __init__(self, in_chans=4, num_classes=2, encoder_name="efficientnetv2_s"):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, features_only=True, in_chans=in_chans, pretrained=True)
        ch = self.encoder.feature_info.channels()
        self.up4 = SubPixelUp(ch[4], ch[3], ch[3])
        self.up3 = SubPixelUp(ch[3], ch[2], ch[2])
        self.up2 = SubPixelUp(ch[2], ch[1], ch[1])
        self.up1 = SubPixelUp(ch[1], ch[0], ch[0])
        self.up0 = nn.Sequential(nn.Conv2d(ch[0], ch[0]*4, 3, padding=1), nn.PixelShuffle(2), nn.BatchNorm2d(ch[0]), nn.GELU(),
                                 nn.Conv2d(ch[0], 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.seg_head = nn.Conv2d(32, num_classes, 1)
        self.sdm_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        f = self.encoder(x)
        d = self.up1(self.up2(self.up3(self.up4(f[4], f[3]), f[2]), f[1]), f[0])
        d = self.up0(d)
        seg, sdm = self.seg_head(d), self.sdm_head(d)
        if seg.shape[2:] != (h, w):
            seg = F.interpolate(seg, (h, w), mode="bilinear")
            sdm = F.interpolate(sdm, (h, w), mode="bilinear")
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

def train_epoch(model, loader, opt, crit, dev):
    model.train()
    l_sum = 0
    for img, m, sdm, _ in tqdm(loader, desc="Train"):
        img, m, sdm = img.to(dev), m.to(dev), sdm.to(dev)
        opt.zero_grad()
        loss = crit(model(img), m, sdm)
        loss.backward()
        opt.step()
        l_sum += loss.item()
    return l_sum / len(loader)

def val_epoch(model, loader, crit, dev):
    model.eval()
    l_sum = 0
    with torch.no_grad():
        for img, m, sdm, _ in tqdm(loader, desc="Val"):
            img, m, sdm = img.to(dev), m.to(dev), sdm.to(dev)
            l_sum += crit(model(img), m, sdm).item()
    return l_sum / len(loader)

def load_vessel_model(ckpt, in_c=4, nc=2, dev="cpu"):
    m = VesselNetV3(in_chans=in_c, num_classes=nc).to(dev)
    if os.path.isfile(ckpt):
        d = torch.load(ckpt, map_location=dev)
        m.load_state_dict(d["model_state_dict"] if "model_state_dict" in d else d)
        print(f"[INFO] Loaded: {ckpt}")
    return m.eval()

@torch.no_grad()
def inference(path, b_mod, r_mod, dev, sz=512):
    img = cv2.resize(cv2.imread(str(path), 0), (sz, sz))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c1, c2, c3 = clahe.apply(img), cv2.morphologyEx(img, 1, cv2.getStructuringElement(2, (15,15))), cv2.morphologyEx(img, 2, cv2.getStructuringElement(2, (15,15)))
    c4 = cv2.addWeighted(img, 4.0, cv2.GaussianBlur(img, (0,0), 10), -4.0, 128)
    
    in4 = torch.from_numpy(np.stack([c1,c2,c3,c4], -1).astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    b_p = torch.sigmoid(b_mod(in4)["seg"]).squeeze().cpu().numpy()
    
    in5 = torch.from_numpy(np.stack([c1,c2,c3,c4,((b_p[0]>0.5)*255).astype(np.uint8)], -1).astype(np.float32)/255.).permute(2,0,1).unsqueeze(0).to(dev)
    r_p = torch.sigmoid(r_mod(in5)["seg"]).squeeze().cpu().numpy()
    return b_p, r_p

if __name__ == "__main__":
    TASK, EP, BS, LR, DEV = "train_base", 50, 8, 2e-4, "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    if TASK.startswith("train"):
        IS_R = (TASK == "train_refiner")
        ds_t = VesselDatasetV3("data/ARCADE/processed/dataset.json", "train", refiner_mode=IS_R)
        ds_v = VesselDatasetV3("data/ARCADE/processed/dataset.json", "val", refiner_mode=IS_R)
        ld_t, ld_v = DataLoader(ds_t, BS, True, num_workers=4), DataLoader(ds_v, BS, False, num_workers=4)
        
        model = VesselNetV3(in_chans=5 if IS_R else 4, num_classes=2).to(DEV)
        opt = optim.AdamW(model.parameters(), lr=LR)
        crit, sch = HybridLoss(), CosineAnnealingLR(opt, T_max=EP)
        writer = SummaryWriter(f"runs/v3_{TASK}")

        for e in range(EP):
            tl, vl = train_epoch(model, ld_t, opt, crit, DEV), val_epoch(model, ld_v, crit, DEV)
            sch.step()
            writer.add_scalar("Loss/train", tl, e); writer.add_scalar("Loss/val", vl, e)
            print(f"Ep {e+1}/{EP} | Train: {tl:.4f} | Val: {vl:.4f}")
            if (e+1)%10==0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/v3_{'refiner' if IS_R else 'base'}.pth")
        writer.close()
