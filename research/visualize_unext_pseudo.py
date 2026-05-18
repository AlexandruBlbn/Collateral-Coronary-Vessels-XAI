"""Visualize UNeXt predictions vs Frangi pseudo-labels."""
import os, sys, json
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'XA-SSL-REPO', 'segmodel'))
from unext.model import UNext_S

CKPT = Path("checkpoints/unext_pseudo/unext_pseudo_best.pth")
PSEUDO_DIR = Path("data/frangi_pseudolabels/accepted")
OUTPUT = "research/unext_pseudo_preview.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNext_S(num_classes=1, img_size=224)
state = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()
print(f"Loaded {CKPT}")

pairs = []
for f in sorted(PSEUDO_DIR.glob("*.png")):
    jp = Path(str(f) + ".json")
    if not jp.exists(): continue
    with open(jp) as jf: meta = json.load(jf)
    ip, lp = meta.get("data"), meta.get("label")
    if ip and lp and os.path.exists(ip) and os.path.exists(lp):
        pairs.append((ip, lp, f))

pairs = pairs[:8]
n = len(pairs)
fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))

for i, (ip, lp, _) in enumerate(pairs):
    img_bgr = cv2.imread(ip)
    if img_bgr is None: continue
    h, w = img_bgr.shape[:2]
    gt = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
    if gt.shape[0] != h or gt.shape[1] != w:
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    gt_bin = (gt > 127).astype(np.uint8) * 255

    img_224 = cv2.resize(img_bgr, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_t = torch.from_numpy((img_224 - mean) / std).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(img_t)).cpu().numpy().squeeze()
    pred_rs = cv2.resize(pred, (w, h))
    pred_bin = (pred_rs > 0.5).astype(np.uint8) * 255
    pred_prob = (pred_rs * 255).astype(np.uint8)

    axes[i,0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[i,0].set_title(f"Input", fontsize=10); axes[i,0].axis("off")
    axes[i,1].imshow(gt_bin, cmap="gray")
    axes[i,1].set_title("Pseudo-label", fontsize=10); axes[i,1].axis("off")
    axes[i,2].imshow(pred_prob, cmap="hot", vmin=0, vmax=255)
    axes[i,2].set_title("UNeXt prob", fontsize=10); axes[i,2].axis("off")
    axes[i,3].imshow(pred_bin, cmap="gray")
    axes[i,3].set_title("UNeXt bin", fontsize=10); axes[i,3].axis("off")

    gt_p = (gt_bin>0).sum()/gt_bin.size*100
    pr_p = (pred_bin>0).sum()/pred_bin.size*100
    inter = ((gt_bin>0)&(pred_bin>0)).sum()
    dice = 2*inter/max((gt_bin>0).sum()+(pred_bin>0).sum(),1)
    axes[i,0].set_ylabel(f"DSC={dice:.3f}\nG={gt_p:.0f}% P={pr_p:.0f}%", fontsize=8)

plt.tight_layout(); plt.savefig(OUTPUT, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {OUTPUT}")
