from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_JSON = PROJECT_ROOT / "data" / "ARCADE" / "processed" / "dataset.json"
EXTRA_ROOT = PROJECT_ROOT / "data" / "Extra"

# Phase 1 architecture — do not change
PHASE1_CONFIG = dict(in_channels=4, num_classes=1, base_channels=64,
                     depths=[2, 2, 2], mlp_ratio=4, drop_rate=0.2, attention=True)


# ============================================================================
# Inference-only preprocessing (no augmentation)
# ============================================================================

class InferencePreprocess:
    """4-channel preprocessing: CLAHE, white-hat, black-hat, high-pass, no aug."""

    def __init__(self, image_size: int = 512):
        self.image_size = image_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def __call__(self, image: Image.Image, label: Optional[Image.Image] = None):
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        if label is not None:
            label = TF.resize(label, [self.image_size, self.image_size], InterpolationMode.NEAREST)

        image_np = np.array(image, dtype=np.uint8)

        c1 = self.clahe.apply(image_np)
        c2 = cv2.morphologyEx(image_np, cv2.MORPH_TOPHAT, self.morph_kernel)
        c3 = cv2.morphologyEx(image_np, cv2.MORPH_BLACKHAT, self.morph_kernel)
        blurred = cv2.GaussianBlur(image_np, (0, 0), sigmaX=10)
        c4 = cv2.addWeighted(image_np, 4.0, blurred, -4.0, 128)

        stacked = np.stack([c1, c2, c3, c4], axis=-1)
        img_t = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0

        if label is not None:
            label_t = TF.pil_to_tensor(label).float()
            if label_t.max() > 1.0:
                label_t = label_t / 255.0
            label_t = (label_t > 0.5).float()
            return img_t, label_t

        return img_t


# ============================================================================
# Datasets
# ============================================================================

class SyntaxInferenceDataset(Dataset):
    def __init__(self, json_path: str = str(DATASET_JSON), transform=None):
        self.transform = transform
        self.samples: List[dict] = []
        with open(json_path, "r") as f:
            data = json.load(f)
        source = data.get("test", {}).get("syntax", {})
        for sid in sorted(source.keys(), key=_natural_key):
            self.samples.append(source[sid])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["data"]).convert("L")
        label = Image.open(item["label"]).convert("L")
        if self.transform is not None:
            return self.transform(image, label)
        return TF.to_tensor(image), label


class ExtraInferenceDataset(Dataset):
    def __init__(self, root: str = str(EXTRA_ROOT), transform=None):
        self.transform = transform
        root_p = Path(root)
        self.image_dir = root_p / "images"
        self.mask_dir = root_p / "masks"
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in exts],
            key=lambda p: p.name,
        )
        self.paths = [p for p in self.paths if (self.mask_dir / p.name).exists()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        mask_path = self.mask_dir / img_path.name
        image = Image.open(img_path).convert("L")
        label = Image.open(mask_path).convert("L")
        if self.transform is not None:
            return self.transform(image, label)
        return TF.to_tensor(image), label


def _natural_key(s: str) -> tuple:
    try:
        return (0, int(s))
    except (TypeError, ValueError):
        return (1, s)


# ============================================================================
# Visualization
# ============================================================================

def visualize(
    model_path: str,
    data_source: str = "syntax",
    output_dir: str = "runs/unext_vis",
    num_samples: int = 12,
    image_size: int = 512,
    device: torch.device = torch.device("cpu"),
):
    from zoo.unext import UNeXt_S

    model = UNeXt_S(**PHASE1_CONFIG)

    state = torch.load(model_path, map_location="cpu")
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {len(missing)} — {missing[:5]}...")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {len(unexpected)}")
    if not missing and not unexpected:
        print("  [OK] All weights loaded cleanly.")

    model = model.to(device)
    model.eval()

    preproc = InferencePreprocess(image_size=image_size)

    if data_source == "syntax":
        ds = SyntaxInferenceDataset(json_path=str(DATASET_JSON), transform=preproc)
        tag = "syntax"
        print(f"\nInference on Syntax test ({len(ds)} samples)")
    else:
        ds = ExtraInferenceDataset(root=str(EXTRA_ROOT), transform=preproc)
        tag = "extra"
        print(f"\nInference on Extra ({len(ds)} samples)")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    count = 0

    for batch in loader:
        if count >= num_samples:
            break

        img_t = batch[0].to(device)
        label_t = batch[1]

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(img_t)

        probs = torch.sigmoid(logits).float()  # [1, 1, H, W]

        inp_np = img_t[0, 0].cpu().numpy()
        lbl_np = label_t[0, 0].cpu().numpy()
        prd_np = probs[0, 0].cpu().numpy()

        print(f"  [{count:03d}] pred: mean={prd_np.mean():.4f} "
              f"min={prd_np.min():.4f} max={prd_np.max():.4f} | "
              f"vessel_px={lbl_np.sum():.0f}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(inp_np, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Data (CLAHE)", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(lbl_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Label", fontsize=14)
        axes[1].axis("off")

        im = axes[2].imshow(prd_np, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("Prediction (raw sigmoid)", fontsize=14)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_sample_{count:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        count += 1

    print(f"\nSaved {count} figures to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UNeXt inference visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="syntax", choices=["syntax", "extra"])
    parser.add_argument("--output_dir", type=str, default="runs/unext_vis")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    visualize(
        model_path=args.checkpoint,
        data_source=args.data,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        device=device,
    )
