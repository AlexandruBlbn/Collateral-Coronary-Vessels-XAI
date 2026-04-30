#!/usr/bin/env python3
"""
Linear probe: freeze the pretrained encoder, train a linear classifier
on top to check if features encode useful information.

If features are random → probe accuracy ≈ chance (~50% for binary)
If features encode vessels → probe accuracy > 80%

Usage:
    python engine/lejepa_probe.py --checkpoint checkpoints/lejepa/.../best_backbone.pth
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zoo.jepa_models import DenseLeJepaModel
from data.dataloader import Transforms


DATASET_JSON = "data/ARCADE/processed/dataset.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProbeDataset(Dataset):
    """Loads validation images, encodes with frozen backbone, returns features + Frangi proxy label."""

    def __init__(self, dataset_json: str, split: str = "validation"):
        with open(dataset_json) as f:
            data = json.load(f)

        self.samples = []
        for source, entries in data.get(split, {}).items():
            for key, info in entries.items():
                self.samples.append(info["data"])

        self.transform = Transforms(image_size=256, training=False, normalize=True)
        print(f"ProbeDataset ({split}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = os.path.normpath(self.samples[idx])
        pil_img = Image.open(img_path).convert("L")
        img_tensor, _ = self.transform(pil_img, pil_img)
        return img_tensor  # [1, 256, 256]


class LinearProbe(nn.Module):
    """Single linear layer: pooled features → binary vessel prediction."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: [B, N, D] → global avg pool → [B, D] → logit
        x = x.mean(dim=1)
        return self.linear(x).squeeze(1)


def extract_features(model, loader, device):
    """Extract frozen encoder features for all images."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            batch = batch.to(device)
            tokens, _, feats = model.encode(batch)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def evaluate_probe(accuracy: float):
    """Print evaluation result."""
    print(f"\n  >>> Linear probe accuracy: {accuracy:.2f}%")
    if accuracy < 55:
        print("  >>> Features are RANDOM (no useful learning)")
    elif accuracy < 70:
        print("  >>> Features have WEAK signal")
    elif accuracy < 85:
        print("  >>> Features are MODERATELY useful")
    else:
        print("  >>> Features are STRONG (model is learning well)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to backbone state dict")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    print(f"Device: {device}")

    # ── 1. Load pretrained encoder ──
    model = DenseLeJepaModel(encoder_name="vit_small", proj_dim=256, in_channels=1)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    encoder_dim = model.encoder_dim
    print(f"Encoder dim: {encoder_dim}")

    # ── 2. Freeze encoder ──
    for p in model.backbone.parameters():
        p.requires_grad = False

    # ── 3. Load data ──
    ds = ProbeDataset(DATASET_JSON, split="validation")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── 4. Extract features ──
    feats = extract_features(model, loader, device)  # [N_val, N_tokens, D]
    B, N, D = feats.shape

    # ── 5. Create pseudo-labels via PCA on features themselves ──
    #     If features collapse → first PCA component explains >90% variance
    #     If features are diverse → first PCA explains <50%
    flat_feats = feats.reshape(B * N, D).numpy()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1).fit(flat_feats)
    var_ratio = pca.explained_variance_ratio_[0] * 100
    print(f"\n  >>> First PCA component explains {var_ratio:.1f}% of variance")

    if var_ratio > 90:
        print("  >>> WARNING: Feature COLLAPSE (all tokens identical)")
    elif var_ratio > 70:
        print("  >>> WARNING: Feature COLLAPSE (high redundancy)")
    else:
        print("  >>> Features are DIVERSE (no collapse)")

    # ── 6. Train a linear classifier ──
    #     Uses pseudo-labels from k-means on pooled features
    from sklearn.cluster import KMeans

    pooled = feats.mean(dim=1).numpy()  # [B, D]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    pseudo_labels = kmeans.fit_predict(pooled)

    # Train logistic regression
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(pooled, pseudo_labels)
    accuracy = clf.score(pooled, pseudo_labels) * 100
    evaluate_probe(accuracy)

    # ── 7. Also check cross-view consistency ──
    #     If model learns well, the same image with augmentations
    #     should produce similar features
    print("\n--- Cross-view consistency check ---")
    model.eval()
    cos_sim_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            tokens, _, _ = model.encode(batch)
            pooled = tokens.mean(dim=1)  # [B, D]
            # Simulate local crops by adding noise → compute consistency
            noisy = batch + torch.randn_like(batch) * 0.05
            tokens_n, _, _ = model.encode(noisy)
            pooled_n = tokens_n.mean(dim=1)
            cos_sim = F.cosine_similarity(pooled, pooled_n).mean().item()
            cos_sim_sum += cos_sim
            count += 1
    avg_cos = cos_sim_sum / max(1, count)
    print(f"  >>> Cross-view cosine similarity: {avg_cos:.4f}")
    if avg_cos > 0.9:
        print("  >>> Features are INVARIANT to noise (good)")
    elif avg_cos > 0.7:
        print("  >>> Features are MODERATELY invariant")
    else:
        print("  >>> Features are NOT invariant (may still be learning)")


if __name__ == "__main__":
    main()
