"""
LeJEPA Feature Validation via k-NN Pixel Correspondence.

This module evaluates the quality of dense features learned by a pretrained
LeJEPA backbone using k-Nearest Neighbours retrieval at the pixel (token) level.
The protocol is inspired by DINOv3 / DINOv2 k-NN evaluation.

Key idea:
  If the backbone learns semantically meaningful dense representations, then
  a vessel pixel's nearest-neighbour tokens across the dataset should also
  correspond to vessel regions (and vice versa for background).  Frangi
  vesselness maps serve as a dense "label" proxy.

Metrics reported:
  - Retrieval mAP@k (mean Average Precision at k neighbours)
  - Vessel Precision@k — fraction of neighbours that are vessel pixels
  - Background Precision@k — fraction of neighbours that are background pixels
  - Correspondence visualisation: top-k matches overlaid on query images

Usage:
    python engine/lejepa_validate.py --config config/lejepa_config.yaml \\
        --checkpoint checkpoints/lejepa/denselejepa_pretrain_swinv2_tiny_window8_256/best_backbone.pth \\
        --output results/knn_validation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Project path setup ──────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from zoo.jepa_models import DenseLeJepaModel
from zoo.backbones import get_backbone
from data.frangi_cache import FrangiCache
from data.dataloader import LeJepaDenseDataset


# ── Config & helpers ────────────────────────────────────────────────────────


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_three_channels(x: torch.Tensor) -> torch.Tensor:
    """Convert 1- or 2-channel tensor to 3-channel for visualisation."""
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 2:
        return torch.cat([x[:, :1], x[:, 1:2], x[:, :1]], dim=1)
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :3]


# ── Feature extractor ───────────────────────────────────────────────────────


class LeJepaFeatureExtractor(nn.Module):
    """Wraps a DenseLeJepaModel to extract projected dense tokens from the encoder.

    Returns:
        tokens:  (B, L, D) projected spatial tokens
        spatial: (H', W')  grid dimensions
    """

    def __init__(self, backbone: nn.Module, proj: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.proj = proj

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        feats = self.backbone(x)
        feat_map = feats[-1] if isinstance(feats, (list, tuple)) else feats
        _, _, h_p, w_p = feat_map.shape
        tokens = feat_map.flatten(2).permute(0, 2, 1)  # (B, L, C)
        proj_tokens = self.proj(tokens)  # (B, L, D)
        return proj_tokens, (h_p, w_p)


# ── Dataset wrapper for validation images ───────────────────────────────────


class ValidationImageDataset(Dataset):
    """Loads full-resolution grayscale images + Frangi vesselness from a sample list.

    Each sample returns:
        ``{"image": [C, H, W] tensor, "frangi": [H, W] tensor, "path": str}``
    """

    def __init__(
        self,
        base_dataset_json: str,
        root_dir: str = ".",
        frangi_cache: Optional[FrangiCache] = None,
        max_samples: int | str = "all",
        split: str = "train",
    ):
        with open(base_dataset_json, "r") as f:
            base_data = json.load(f)

        split_data = base_data.get(split, {})
        self.entries: List[dict] = []
        for source, items in split_data.items():
            if not isinstance(items, dict):
                continue
            for s_id, s_info in items.items():
                img_path = s_info.get("data")
                if isinstance(img_path, str) and img_path:
                    self.entries.append({
                        "path": img_path,
                        "source": source,
                        "id": s_id,
                    })

        if isinstance(max_samples, int) and 0 < max_samples < len(self.entries):
            self.entries = self.entries[:max_samples]

        self.root_dir = root_dir
        self.frangi_cache = frangi_cache
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        abs_path = os.path.join(self.root_dir, entry["path"])
        img = np.array(Image.open(abs_path).convert("L"))
        img = self.clahe.apply(img)
        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        img_t = img_t * 2.0 - 1.0  # → [-1, 1]

        # Build 2-channel input: grayscale + Frangi
        if self.frangi_cache is not None:
            frangi_map = self.frangi_cache.get(entry["path"])
            frangi_map = frangi_map.unsqueeze(0).unsqueeze(0)
            frangi_map = F.interpolate(
                frangi_map, size=img.shape, mode="bilinear", align_corners=False
            )
            frangi_map = frangi_map.squeeze(0).squeeze(0)
            img_t = torch.cat([img_t, frangi_map.unsqueeze(0)], dim=0)  # [2, H, W]
        else:
            img_t = img_t.repeat(2, 1, 1)  # [2, H, W]

        return {
            "image": img_t,  # [2, H, W]
            "frangi": frangi_map if self.frangi_cache is not None else None,
            "path": entry["path"],
        }


# ── Feature bank ────────────────────────────────────────────────────────────


class FeatureBank:
    """On-device feature bank for k-NN retrieval.

    Stores token-level features along with spatial and image-level metadata
    so that retrieved neighbours can be mapped back to image locations.
    """

    def __init__(self, max_size: int = 200_000):
        self.features: Optional[torch.Tensor] = None     # [N, D]
        self.labels: Optional[torch.Tensor] = None       # [N] binary vessel/non-vessel
        self.image_indices: List[int] = []                # per-token: which image
        self.spatial_positions: List[Tuple[int, int]] = []  # per-token: (y, x)
        self.image_paths: List[str] = []                  # image index → path
        self._max_size = max_size

    @property
    def size(self) -> int:
        return len(self.image_indices) if self.features is None else self.features.shape[0]

    def add(
        self,
        tokens: torch.Tensor,
        spatial_dims: Tuple[int, int],
        image_idx: int,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """Add tokens from one image to the bank.

        Args:
            tokens:       [L, D] projected token features.
            spatial_dims: (H', W') spatial grid dimensions.
            image_idx:    Index in ``self.image_paths`` (must already be appended).
            labels:       [L] optional binary vessel/non-vessel labels (0/1) for
                          each token. If provided, they are stored alongside features
                          and used during k-NN evaluation.
        """
        L, D = tokens.shape
        h_p, w_p = spatial_dims
        if h_p * w_p != L:
            # Fallback: treat as 1D sequence
            ys = torch.arange(L).tolist()
            xs = [0] * L
        else:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h_p), torch.arange(w_p), indexing="ij"
            )
            ys = grid_y.flatten().tolist()
            xs = grid_x.flatten().tolist()

        # Subsample if bank is near capacity
        tokens_cpu = tokens.cpu()
        labels_cpu = labels.cpu() if labels is not None else None
        if self.size + L > self._max_size:
            keep_frac = self._max_size / (self.size + L)
            n_keep = max(1, int(L * keep_frac))
            perm = torch.randperm(L)
            indices = perm[:n_keep]
            tokens_cpu = tokens_cpu[indices]
            if labels_cpu is not None:
                labels_cpu = labels_cpu[indices]
            ys = [ys[i] for i in indices.tolist()]
            xs = [xs[i] for i in indices.tolist()]
            L = n_keep

        if self.features is None:
            self.features = tokens_cpu
        else:
            self.features = torch.cat([self.features, tokens_cpu], dim=0)

        if labels_cpu is not None:
            if self.labels is None:
                self.labels = labels_cpu
            else:
                self.labels = torch.cat([self.labels, labels_cpu], dim=0)

        self.image_indices.extend([image_idx] * L)
        self.spatial_positions.extend(zip(ys, xs))

    def knn(
        self,
        query: torch.Tensor,
        k: int = 5,
        device: Optional[torch.device] = None,
        chunk_size: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find k nearest neighbours for each query token.

        Performs computation **on CPU** to preserve GPU memory for the backbone
        forward pass.  If a GPU device is passed, it is used as a secondary
        accelerator but the bank features are never permanently held on GPU.

        Args:
            query:  [Q, D] query features (on any device).
            k:      Number of nearest neighbours.
            device: Ignored — computation is always on CPU to save GPU memory.
            chunk_size: Number of query rows to process at once on CPU.

        Returns:
            distances: [Q, k] cosine distances (1 - cosine_sim).
            indices:   [Q, k] indices into the feature bank.
        """
        if self.features is None or self.features.shape[0] == 0:
            raise RuntimeError("Feature bank is empty — call add() first.")

        q = F.normalize(query.float(), dim=-1).cpu()
        db = F.normalize(self.features.float(), dim=-1).cpu()
        Q = q.shape[0]

        all_distances = []
        all_indices = []

        for start in range(0, Q, chunk_size):
            end = min(start + chunk_size, Q)
            chunk = q[start:end]  # [C, D]
            sim = chunk @ db.T    # [C, N]
            vals, idx = sim.topk(k=k, dim=1, largest=True, sorted=True)
            all_distances.append(1.0 - vals)
            all_indices.append(idx)

        distances = torch.cat(all_distances, dim=0)  # [Q, k]
        indices = torch.cat(all_indices, dim=0)      # [Q, k]
        return distances, indices


# ── k-NN evaluation ─────────────────────────────────────────────────────────


def compute_map_at_k(
    query_labels: torch.Tensor,
    neighbour_labels: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    """Compute mean Average Precision @ k for each query.

    Args:
        query_labels:      [Q] binary labels (0/1) for each query token.
        neighbour_labels:  [Q, k] binary labels for each neighbour.

    Returns:
        mean_ap:   Scalar mAP@k across all queries.
        per_query: [Q] AP for each query.
    """
    k = neighbour_labels.shape[1]
    correct = neighbour_labels == query_labels.unsqueeze(1)  # [Q, k]

    # Cumulative precision at each rank
    cumsum = correct.cumsum(dim=1).float()
    positions = torch.arange(1, k + 1, device=cumsum.device).float().unsqueeze(0)
    precision_at_rank = cumsum / positions  # [Q, k]

    # Average precision = mean of precision at ranks where neighbour is correct
    ap = (precision_at_rank * correct.float()).sum(dim=1) / max(k, 1)
    return float(ap.mean().item()), ap


@torch.no_grad()
def validate(
    extractor: LeJepaFeatureExtractor,
    bank: FeatureBank,
    query_loader: DataLoader,
    device: torch.device,
    k: int = 10,
    vessel_threshold: float = 0.05,
    max_queries: int = 5000,
) -> dict:
    """Run k-NN retrieval evaluation on query images.

    For each spatial token in each query image:
      1. Find its k nearest neighbours in the feature bank.
      2. Use Frangi vesselness as a proxy label: a pixel is "vessel" if its
         Frangi response exceeds ``vessel_threshold``.
      3. Compute vessel-Precision@k and background-Precision@k.
      4. Compute mAP@k (binary classification: vessel vs background).

    Returns:
        A dict of scalar metrics.
    """
    extractor.eval()
    all_query_labels: List[torch.Tensor] = []
    all_neighbour_labels: List[torch.Tensor] = []
    vessel_precisions: List[float] = []
    bg_precisions: List[float] = []

    total_queries = 0
    for batch in tqdm(query_loader, desc="k-NN evaluation"):
        if total_queries >= max_queries:
            break

        images = batch["image"].to(device, non_blocking=True)  # [B, 2, H, W]
        frangi_maps = batch["frangi"]  # list/None per sample
        B = images.shape[0]

        # Extract dense tokens
        tokens, (h_p, w_p) = extractor(images)  # [B, L, D]
        L = h_p * w_p

        for b in range(B):
            if total_queries >= max_queries:
                break

            frangi = frangi_maps[b]  # [H, W] or None
            if frangi is None:
                continue

            # Resize Frangi to token grid resolution
            frangi_grid = F.interpolate(
                frangi.unsqueeze(0).unsqueeze(0),
                size=(h_p, w_p),
                mode="bilinear",
                align_corners=False,
            ).squeeze()  # [h_p, w_p]
            labels = (frangi_grid > vessel_threshold).float().flatten()  # [L]

            # Query features
            q_tokens = tokens[b:b + 1]  # [1, L, D]
            q_flat = q_tokens.squeeze(0)  # [L, D]

            # k-NN retrieval
            _, nn_indices = bank.knn(q_flat, k=k, device=device)

            # Map neighbours to their ACTUAL labels stored in the feature bank
            bank_labels = bank.labels.to(device) if bank.labels is not None else None
            nn_labels = torch.zeros(
                nn_indices.shape[0], k, device=device, dtype=torch.float32
            )
            if bank_labels is not None:
                for row in range(nn_indices.shape[0]):
                    for col in range(k):
                        idx = int(nn_indices[row, col].item())
                        nn_labels[row, col] = bank_labels[idx]
            else:
                # Fallback: use query label (approximate, only for diagnostics)
                for row in range(nn_indices.shape[0]):
                    for col in range(k):
                        idx = int(nn_indices[row, col].item())
                        nn_labels[row, col] = labels[row] if idx < len(labels) else 0.0
            all_query_labels.append(labels.cpu())
            all_neighbour_labels.append(nn_labels.cpu())

            # Compute vessel precision@k
            vessel_mask = labels > 0.5
            bg_mask = ~vessel_mask
            if vessel_mask.any():
                vp = (nn_labels[vessel_mask].float().mean(dim=1)).mean().item()
                vessel_precisions.append(vp)
            if bg_mask.any():
                bp = (1.0 - nn_labels[bg_mask].float().mean(dim=1)).mean().item()
                bg_precisions.append(bp)

            total_queries += 1

    if not all_query_labels:
        return {"error": "No queries processed — check Frangi availability."}

    query_labels_all = torch.cat(all_query_labels, dim=0)
    neighbour_labels_all = torch.cat(all_neighbour_labels, dim=0)

    mean_ap, _ = compute_map_at_k(query_labels_all, neighbour_labels_all)
    mean_vp = float(np.mean(vessel_precisions)) if vessel_precisions else 0.0
    mean_bp = float(np.mean(bg_precisions)) if bg_precisions else 0.0

    return {
        "k": k,
        "vessel_threshold": vessel_threshold,
        "num_queries": total_queries,
        "bank_size": bank.size,
        "mAP@k": mean_ap,
        "vessel_Precision@k": mean_vp,
        "background_Precision@k": mean_bp,
    }


# ── Correspondence visualisation ────────────────────────────────────────────


@torch.no_grad()
def visualize_correspondences(
    extractor: LeJepaFeatureExtractor,
    bank: FeatureBank,
    query_loader: DataLoader,
    output_dir: str,
    device: torch.device,
    num_queries: int = 4,
    k: int = 5,
    grid_size: int = 256,
) -> List[str]:
    """Generate correspondence visualisation figures.

    For each query image, selects a set of evenly-spaced token positions and
    shows their k nearest neighbours overlaid on the Frangi map.

    Returns:
        List of saved figure paths.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib required for visualisation — skipping.")
        return []

    extractor.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    sample_count = 0
    for batch in tqdm(query_loader, desc="Correspondence visualisation"):
        if sample_count >= num_queries:
            break

        images = batch["image"].to(device, non_blocking=True)
        frangi_maps = batch["frangi"]
        B = images.shape[0]

        tokens, (h_p, w_p) = extractor(images)

        for b in range(B):
            if sample_count >= num_queries:
                break
            frangi = frangi_maps[b]
            if frangi is None:
                continue

            # Frangi at full resolution for display
            frangi_np = frangi.cpu().numpy()

            q_tokens = tokens[b:b + 1]  # [1, L, D]
            q_flat = q_tokens.squeeze(0)

            # Sample query positions evenly from the spatial grid
            n_queries_per_image = min(12, h_p * w_p)
            sq = int(np.sqrt(n_queries_per_image))
            if sq * sq < n_queries_per_image:
                sq += 1
            step_y = max(1, h_p // sq)
            step_x = max(1, w_p // sq)
            query_positions = []
            query_indices = []
            for gy in range(0, h_p, step_y):
                for gx in range(0, w_p, step_x):
                    idx = gy * w_p + gx
                    if idx < q_flat.shape[0] and len(query_positions) < n_queries_per_image:
                        query_positions.append((gy, gx))
                        query_indices.append(idx)

            if not query_indices:
                continue

            # k-NN retrieval
            q_selected = q_flat[query_indices]  # [Q, D]
            _, nn_indices = bank.knn(q_selected, k=k, device=device)

            # Build figure
            fig, axes = plt.subplots(
                len(query_positions), k + 1,
                figsize=(2.5 * (k + 1), 2.5 * len(query_positions)),
            )
            if len(query_positions) == 1:
                axes = axes.reshape(1, -1)

            for qi, (gy, gx) in enumerate(query_positions):
                # Query patch: highlight on Frangi map
                q_ax = axes[qi, 0]
                q_ax.imshow(frangi_np, cmap="gray", vmin=0.0, vmax=1.0)
                # Mark query position (scale from token grid to image grid)
                qy_px = int(gy * grid_size / h_p)
                qx_px = int(gx * grid_size / w_p)
                patch_size = max(1, grid_size // max(h_p, w_p))
                rect = Rectangle(
                    (qx_px, qy_px), patch_size, patch_size,
                    linewidth=1.5, edgecolor="red", facecolor="none",
                )
                q_ax.add_patch(rect)
                q_ax.set_title(f"Query ({gy},{gx})", fontsize=8)
                q_ax.axis("off")

                for ni in range(k):
                    nn_idx = int(nn_indices[qi, ni].item())
                    img_idx = bank.image_indices[nn_idx]
                    sy, sx = bank.spatial_positions[nn_idx]

                    n_ax = axes[qi, ni + 1]
                    # Retrieve and show the Frangi map for this neighbour's image
                    n_path = bank.image_paths[img_idx]
                    n_img = np.array(Image.open(n_path).convert("L"))
                    n_img_resized = cv2.resize(
                        n_img, (grid_size, grid_size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    # Simple Frangi approximation for visualisation
                    n_frangi = n_img_resized.astype(np.float32)
                    n_frangi = (n_frangi - n_frangi.min()) / (
                        n_frangi.max() - n_frangi.min() + 1e-8
                    )

                    n_ax.imshow(n_frangi, cmap="gray", vmin=0.0, vmax=1.0)
                    n_patch_size = max(1, grid_size // max(h_p, w_p))
                    n_px = int(sx * grid_size / w_p)
                    n_py = int(sy * grid_size / h_p)
                    n_rect = Rectangle(
                        (n_px, n_py), n_patch_size, n_patch_size,
                        linewidth=1.5, edgecolor="lime", facecolor="none",
                    )
                    n_ax.add_patch(n_rect)
                    n_ax.set_title(f"NN {ni + 1} ({sy},{sx})", fontsize=8)
                    n_ax.axis("off")

            fig.suptitle(
                f"k-NN Correspondence — Query: {Path(batch['path'][b]).name}",
                fontsize=10,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.97])

            save_name = f"correspondence_sample_{sample_count:04d}.png"
            fig.savefig(output_path / save_name, dpi=150)
            plt.close(fig)
            saved_paths.append(str(output_path / save_name))
            sample_count += 1

    return saved_paths


# ── Main entry point ────────────────────────────────────────────────────────


def build_model_from_checkpoint(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
) -> LeJepaFeatureExtractor:
    """Load a DenseLeJEPA model checkpoint and return a feature extractor.

    Supports both full-model checkpoints (containing ``model_state_dict``)
    and backbone-only checkpoints (raw backbone weights).

    Avoids constructing the full DenseLeJepaModel (predictor, transformer etc.)
    to minimise GPU memory footprint.
    """
    import gc

    model_cfg = cfg["model"]
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Determine hidden dimension on CPU without holding the backbone on GPU
    dummy_cpu = torch.randn(1, model_cfg["in_channels"], 256, 256)
    backbone_cpu = get_backbone(
        model_name=model_cfg["backbone"],
        in_channels=model_cfg["in_channels"],
        pretrained=False,
    )
    with torch.no_grad():
        feats = backbone_cpu(dummy_cpu)
        feat_list = list(feats) if isinstance(feats, (list, tuple)) else [feats]
        enc_c = feat_list[-1].shape[1]
    del backbone_cpu, feats, dummy_cpu
    gc.collect()

    # Build the projection head (same as in DenseLeJepaModel)
    proj = nn.Sequential(
        nn.Linear(enc_c, 512),
        nn.LayerNorm(512),
        nn.Linear(512, model_cfg["proj_dim"]),
    )

    # Build backbone on CPU, load weights, then move both to device
    backbone = get_backbone(
        model_name=model_cfg["backbone"],
        in_channels=model_cfg["in_channels"],
        pretrained=False,
    )

    if "model_state_dict" in ckpt:
        # Full LeJEPA checkpoint: extract only backbone.* and proj.* keys
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("backbone.")
        }
        proj_state = {
            k[len("proj."):]: v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("proj.")
        }
        if backbone_state:
            backbone.load_state_dict(backbone_state)
        else:
            print("  Warning: no backbone keys found in checkpoint state_dict.")
        if proj_state:
            proj.load_state_dict(proj_state, strict=False)
    else:
        # Backbone-only checkpoint — load directly
        backbone.load_state_dict(ckpt, strict=False)

    extractor = LeJepaFeatureExtractor(backbone, proj).to(device)
    # Clear leftover CPU checkpoint data
    del ckpt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return extractor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeJEPA k-NN feature validation (DINOv3-style)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/lejepa_config.yaml",
        help="Path to the training configuration YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained backbone .pth or full-model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/knn_validation",
        help="Directory for validation output (metrics JSON + figures).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbours for retrieval.",
    )
    parser.add_argument(
        "--vessel_threshold",
        type=float,
        default=0.05,
        help="Frangi response threshold for vessel/background labelling.",
    )
    parser.add_argument(
        "--bank_size",
        type=int,
        default=50000,
        help="Maximum number of tokens to store in the feature bank.",
    )
    parser.add_argument(
        "--bank_samples",
        type=int,
        default=50,
        help="Number of images used to build the feature bank.",
    )
    parser.add_argument(
        "--query_samples",
        type=int,
        default=20,
        help="Number of images used for queries (separate set).",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=5000,
        help="Maximum query tokens to evaluate.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate correspondence visualisation figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on.",
    )
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    cfg = _load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    extractor = build_model_from_checkpoint(cfg, args.checkpoint, device)

    # ── Feature bank images (first N samples from train set) ───────────────
    print(f"Building feature bank from {args.bank_samples} images ...")
    frangi_cache = FrangiCache(
        cache_dir=cfg["data"]["frangi_cache_dir"],
        image_size=cfg["model"]["input_size"],
    )
    bank_dataset = ValidationImageDataset(
        base_dataset_json=cfg["data"]["base_dataset_json"],
        root_dir=cfg["data"]["root_dir"],
        frangi_cache=frangi_cache,
        max_samples=args.bank_samples,
        split="train",
    )
    bank_loader = DataLoader(
        bank_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    bank = FeatureBank(max_size=args.bank_size)
    for idx, batch in enumerate(tqdm(bank_loader, desc="Extracting bank features")):
        # --- Forward pass on GPU, immediately pull tokens back to CPU ---
        image = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            tokens, (h_p, w_p) = extractor(image)
            tokens_cpu = tokens.squeeze(0).cpu()
        # Release GPU tensors immediately
        del image, tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --- Frangi labels at token-grid resolution (all on CPU) ---
        frangi = batch["frangi"][0]  # [H, W] on CPU
        if frangi is not None:
            frangi_grid = F.interpolate(
                frangi.unsqueeze(0).unsqueeze(0),
                size=(h_p, w_p),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            token_labels = (frangi_grid > args.vessel_threshold).float().flatten()
        else:
            token_labels = None

        bank.add(
            tokens_cpu,
            (h_p, w_p),
            idx,
            labels=token_labels,
        )
        bank.image_paths.append(batch["path"][0])

    print(f"  Feature bank has {bank.size} tokens from {len(bank.image_paths)} images.")

    # ── Query images (separate subset, overlapping allowed for limited data) ─
    query_dataset = ValidationImageDataset(
        base_dataset_json=cfg["data"]["base_dataset_json"],
        root_dir=cfg["data"]["root_dir"],
        frangi_cache=frangi_cache,
        max_samples=args.bank_samples + args.query_samples,
        split="train",
    )
    # Offset to use images after the bank set for queries
    query_dataset.entries = query_dataset.entries[args.bank_samples:]
    if len(query_dataset.entries) == 0:
        print("No query images available — falling back to first N images.")
        query_dataset.entries = bank_dataset.entries[:args.query_samples]
    else:
        query_dataset.entries = query_dataset.entries[:args.query_samples]

    query_loader = DataLoader(
        query_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    print(f"Prepared {len(query_dataset)} query images.")

    # ── Run evaluation ─────────────────────────────────────────────────────
    print("Running k-NN validation ...")
    metrics = validate(
        extractor=extractor,
        bank=bank,
        query_loader=query_loader,
        device=device,
        k=args.k,
        vessel_threshold=args.vessel_threshold,
        max_queries=args.max_queries,
    )

    print("\n── Validation Results ──")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # ── Correspondence visualisation ───────────────────────────────────────
    if args.visualize:
        print("Generating correspondence visualisations ...")
        saved = visualize_correspondences(
            extractor=extractor,
            bank=bank,
            query_loader=query_loader,
            output_dir=str(output_dir / "figures"),
            device=device,
            num_queries=min(8, len(query_dataset)),
            k=min(5, args.k),
        )
        print(f"  Saved {len(saved)} figures.")

    print("Validation complete.")


if __name__ == "__main__":
    main()
