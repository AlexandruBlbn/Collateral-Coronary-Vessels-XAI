"""UNeXt soft mask visualization on random XA-170K pretraining images."""
import os, sys, random
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'XA-SSL-REPO', 'segmodel'))
from unext.model import UNext_S

CKPT = Path("checkpoints/unext_full/unext_full_best.pth")
if not CKPT.exists():
    CKPT = Path("checkpoints/unext_pseudo/unext_pseudo_best.pth")
OUTPUT = "research/unext_softmasks_pretrain.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM = 8

model = UNext_S(num_classes=1, img_size=224)
state = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()
print(f"Loaded: {CKPT}")

candidates = [str(p) for p in Path("XA-170K/dataset").rglob("*")
              if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
random.shuffle(candidates)
selected = candidates[:NUM]

fig, axes = plt.subplots(2, NUM, figsize=(4*NUM, 8))

for i, ip in enumerate(selected):
    img_bgr = cv2.imread(ip)
    if img_bgr is None: continue
    h, w = img_bgr.shape[:2]
    img_224 = cv2.resize(img_bgr, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    img_t = torch.from_numpy((img_224-mean)/std).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(img_t)).cpu().numpy().squeeze()
    prob_rs = cv2.resize(prob, (w, h))

    axes[0,i].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0,i].set_title(Path(ip).name[:15], fontsize=8); axes[0,i].axis("off")
    axes[1,i].imshow(prob_rs, cmap="hot", vmin=0, vmax=1)
    axes[1,i].set_title(f"Soft ({prob.mean():.2f})", fontsize=8); axes[1,i].axis("off")

plt.suptitle("UNeXt Soft Masks on XA-170K", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT}")
