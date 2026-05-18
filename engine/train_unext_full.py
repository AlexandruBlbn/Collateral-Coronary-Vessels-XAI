"""
Train UNeXt on all available data: 126 original + curated pseudo-labels.

Sources:
  - data/extra/images/  + data/extra/masks/    (126 original labelled pairs)
  - data/frangi_pseudolabels/accepted/          (curated Frangi masks + JSON)

Usage:
    python engine/train_unext_full.py --epochs 200
"""
import os, sys, json, random, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'XA-SSL-REPO', 'segmodel'))
from unext.model import UNext_S

CHECKPOINT_DIR = Path("checkpoints/unext_full")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def load_pairs():
    pairs = []
    # 1. Original labelled data (126 images)
    img_dir = Path("data/extra/images")
    msk_dir = Path("data/extra/masks")
    for f in sorted(img_dir.glob("*.png")):
        m = msk_dir / f.name
        if m.exists():
            pairs.append((str(f), str(m), "original"))
    print(f"Original labelled: {sum(1 for p in pairs if p[2]=='original')}")

    # 2. Curated pseudo-labels (51+ images)
    pseudo_dir = Path("data/frangi_pseudolabels/accepted")
    for f in sorted(pseudo_dir.glob("*.png")):
        jp = Path(str(f) + ".json")
        if not jp.exists(): continue
        with open(jp) as jf: meta = json.load(jf)
        ip, lp = meta.get("data"), meta.get("label")
        if ip and lp and os.path.exists(ip) and os.path.exists(lp):
            # Filter empty masks
            mask = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
            if mask is not None and (mask > 0).sum() >= 50:
                pairs.append((ip, lp, "curated"))
    print(f"Curated pseudo: {sum(1 for p in pairs if p[2]=='curated')}")

    print(f"Total: {len(pairs)} pairs")
    return pairs

class MixedDataset(Dataset):
    def __init__(self, pairs, img_size=224, augment=True):
        self.pairs = pairs; self.img_size = img_size; self.augment = augment
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip, lp, src = self.pairs[idx]
        img = cv2.imread(ip)
        if img is None: return self.__getitem__((idx+1)%len(self.pairs))
        mask = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if mask is None: return self.__getitem__((idx+1)%len(self.pairs))
        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        if self.augment:
            if random.random() > 0.5: img=cv2.flip(img,1); mask=cv2.flip(mask,1)
            if random.random() > 0.5: img=cv2.flip(img,0); mask=cv2.flip(mask,0)
            if random.random() > 0.3:
                ang = random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((self.img_size//2,self.img_size//2),ang,1.0)
                img=cv2.warpAffine(img,M,(self.img_size,self.img_size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
                mask=cv2.warpAffine(mask,M,(self.img_size,self.img_size),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485,0.456,0.406],dtype=np.float32)
        std  = np.array([0.229,0.224,0.225],dtype=np.float32)
        img = (img-mean)/std
        return torch.from_numpy(img).permute(2,0,1).float(), torch.from_numpy(mask).unsqueeze(0).float()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, smooth=1e-5):
        super().__init__(); self.bce_w=bce_w; self.smooth=smooth
    def forward(self, p, t):
        p = torch.sigmoid(p); bce = F.binary_cross_entropy(p, t)
        pf = p.view(-1); tf = t.view(-1); inter = (pf*tf).sum()
        dice = 1-(2*inter+self.smooth)/(pf.sum()+tf.sum()+self.smooth)
        return self.bce_w*bce+(1-self.bce_w)*dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    pairs = load_pairs()
    if not pairs: print("ERROR: no data"); return

    random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    train_ds = MixedDataset(pairs[:split], args.img_size, augment=True)
    val_ds   = MixedDataset(pairs[split:], args.img_size, augment=False)
    tl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4,
                    pin_memory=torch.cuda.is_available())
    vl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4,
                    pin_memory=torch.cuda.is_available())

    model = UNext_S(num_classes=1, img_size=args.img_size).to(DEVICE)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    crit = BCEDiceLoss()
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = LinearLR(optim, 0.1, 1.0, total_iters=min(10, args.epochs))
    cosine = CosineAnnealingLR(optim, T_max=max(1, args.epochs-10))
    sched = SequentialLR(optim, [warmup, cosine], [min(10, args.epochs)])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best = float('inf')
    for ep in range(args.epochs):
        model.train(); tloss = 0
        pbar = tqdm(tl, desc=f"E{ep+1}/{args.epochs}")
        for im, ma in pbar:
            im, ma = im.to(DEVICE), ma.to(DEVICE)
            optim.zero_grad()
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    pr = model(im); lo = crit(pr, ma)
                scaler.scale(lo).backward(); scaler.step(optim); scaler.update()
            else:
                pr = model(im); lo = crit(pr, ma); lo.backward(); optim.step()
            tloss += lo.item(); pbar.set_postfix(loss=lo.item())

        model.eval(); vloss = 0
        with torch.no_grad():
            for im, ma in vl:
                im, ma = im.to(DEVICE), ma.to(DEVICE)
                vloss += crit(model(im), ma).item()
        sched.step()
        tloss /= len(tl); vloss /= len(vl)
        lr = optim.param_groups[0]['lr']
        print(f"  Train: {tloss:.4f} | Val: {vloss:.4f} | LR: {lr:.2e}")
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "unext_full_best.pth")
            print(f"  *** Saved best ({best:.4f})")

    torch.save(model.state_dict(), CHECKPOINT_DIR / "unext_full_final.pth")
    print(f"\nDone. Best val loss: {best:.4f}")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")

if __name__ == "__main__":
    main()
