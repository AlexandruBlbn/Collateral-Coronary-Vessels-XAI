"""
Diagnostic tools for evaluating LeJEPA pretraining quality.

Three complementary tests:
  1. Representation Invariance  — the backbone's ability to produce similar embeddings
                                  for different augmented views of the same image.
                                  This is the direct measurable output of the LeJEPA objective.
  2. GradCAM                    — visual proof that frozen backbone feature maps focus on
                                  vessel structures, not background noise.
  3. t-SNE                      — whether backbone embeddings cluster by vessel content.
                                  A random backbone should show NO structure; a good pretrained
                                  backbone should show a gradient from vessel-heavy to
                                  vessel-sparse images.

Usage (from project root):
    python utils/saliency.py \
        --backbone_path checkpoints/convnextv2_tiny_lejepa_strict_probe/best_backbone.pth \
        --encoder_name  convnextv2_tiny \
        --output_dir    runs/diagnostics/convnextv2_tiny_lejepa
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timm
from tqdm import tqdm
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Compose

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.helpers import set_seed

set_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fix_channels(t: torch.Tensor) -> torch.Tensor:
    """
    timm SwinV2 with features_only=True returns (B, H, W, C). Fix to (B, C, H, W).
    ConvNeXt / ResNet already return (B, C, H, W) — this is a no-op for them.
    """
    if t.dim() == 4 and t.shape[-1] > t.shape[1] and t.shape[-1] > t.shape[-2]:
        return t.permute(0, 3, 1, 2).contiguous()
    return t


def _pool_embed(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Run backbone, global-average-pool the last feature map, L2-normalize.
    Returns (B, D) float32 on CPU.
    """
    pool = nn.AdaptiveAvgPool2d(1)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        feats = backbone(images)
    last = _fix_channels(feats[-1] if isinstance(feats, (list, tuple)) else feats)
    emb = pool(last).flatten(1).float()
    return F.normalize(emb, dim=-1).cpu()


def _build_backbone(encoder_name: str, weights_path: str | None) -> nn.Module:
    """Load a timm backbone with features_only=True. Optionally load pretrained weights."""
    m = timm.create_model(encoder_name, pretrained=False, in_chans=1, features_only=True)
    if weights_path and os.path.exists(weights_path):
        sd = torch.load(weights_path, map_location='cpu')
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"  Loaded: {weights_path}")
        print(f"  Missing keys: {len(missing)}  |  Unexpected keys: {len(unexpected)}")
    else:
        if weights_path:
            print(f"  WARNING: {weights_path} not found — using random init as LeJEPA model.")
    return m.cuda().eval()


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRESENTATION INVARIANCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_representation_invariance(
    backbone: nn.Module,
    dataloader,
    num_batches: int = 30,
) -> dict:
    """
    Core LeJEPA diagnostic. Creates two independent augmented views of every image in
    the batch and measures:
      - intra-image cosine similarity  (same image, different augmentations)
      - inter-image cosine similarity  (two different images)
      - invariance_ratio = intra / |inter|

    A well-pretrained backbone should have:
      invariance_ratio >> 1.0  (it learned to be invariant to augmentation)
      mean_inter_cosine ≈ 0    (representations are spread, not collapsed)

    A random or collapsed backbone will have invariance_ratio ≈ 1.0.
    """
    augment = Compose([
        RandomResizedCrop(256, scale=(0.5, 1.0)),
        RandomHorizontalFlip(),
    ])

    backbone.eval()
    intra_sims, inter_sims = [], []

    for i, (images, _) in enumerate(tqdm(dataloader, total=num_batches, desc="  Invariance")):
        if i >= num_batches:
            break
        images = images.cuda()
        B = images.size(0)

        # Two independently augmented views
        v1 = torch.stack([augment(images[j]) for j in range(B)]).cuda()
        v2 = torch.stack([augment(images[j]) for j in range(B)]).cuda()

        e1 = _pool_embed(backbone, v1)
        e2 = _pool_embed(backbone, v2)

        # Diagonal = same-image similarity
        intra_sims.append((e1 * e2).sum(dim=-1))

        # Off-diagonal = cross-image similarity
        sim_mat = e1 @ e2.T
        mask = ~torch.eye(B, dtype=torch.bool)
        inter_sims.append(sim_mat[mask].flatten())

    intra = torch.cat(intra_sims)
    inter = torch.cat(inter_sims)

    return {
        'mean_intra_cosine': intra.mean().item(),
        'std_intra_cosine':  intra.std().item(),
        'mean_inter_cosine': inter.mean().item(),
        'std_inter_cosine':  inter.std().item(),
        'invariance_ratio':  (intra.mean() / (inter.mean().abs() + 1e-8)).item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRADCAM  —  vessel-guided variant
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam(
    backbone: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Vessel-guided GradCAM when masks are provided.

    Without masks (old behaviour):
      score = mean of last feature map  →  highlights high-activation regions in general
      (tends to fire on bright cardiac chambers, NOT vessels)

    With masks (vessel-guided):
      score = mean activation UNDER vessel pixels only
      → asks the backbone "which regions in the image drove the features that
        respond to vessel locations?"  This will highlight vessels if the backbone
        has learned anything vessel-specific.

    images : (B, 1, H, W) CUDA [-1,1]
    masks  : (B, 1, H, W) CPU  [ 0,1]  (GT segmentation masks)
    Returns: (B, H, W) numpy [0,1]
    """
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(True)
    backbone.zero_grad()

    with torch.enable_grad():
        feats = backbone(images)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        last = _fix_channels(feats[-1])  # (B, C, h, w)
        last.retain_grad()

        if masks is not None:
            # Downsample GT mask to feature-map resolution and use as spatial selector
            h, w = last.shape[2:]
            mask_feat = F.interpolate(
                masks.float().cuda(), size=(h, w), mode='bilinear', align_corners=False
            )  # (B, 1, h, w)
            vessel_px = mask_feat.sum()
            if vessel_px > 0:
                score = (last * mask_feat).sum() / vessel_px
            else:
                score = last.mean()   # fallback: no vessel pixels in this batch
        else:
            score = last.mean()

        score.backward()

    grad = last.grad
    act  = last.detach()

    if grad is None:
        raise RuntimeError(
            "Gradients are None. The backbone may have detach() calls inside it, "
            "or features_only=True is not propagating gradients correctly."
        )

    weights = grad.mean(dim=(2, 3), keepdim=True)     # (B, C, 1, 1)
    cam = F.relu((weights * act).sum(dim=1))          # (B, h, w)

    B = cam.shape[0]
    cam_min = cam.view(B, -1).min(1).values.view(B, 1, 1)
    cam_max = cam.view(B, -1).max(1).values.view(B, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    H, W = images.shape[2:]
    cam = F.interpolate(cam.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

    for p in backbone.parameters():
        p.requires_grad_(False)

    return cam.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 2b. CHANNEL SELECTIVITY  —  which feature channels correlate with vessels?
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_channel_selectivity(
    backbone: nn.Module,
    dataloader,
    num_batches: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each channel C in the last feature map, compute the Pearson correlation
    between its spatial activations and the downsampled GT vessel mask — averaged
    over all images in the dataset.

    Returns:
        correlations : (C,) numpy array — per-channel correlation with vessel mask
        top_k_idx    : indices sorted by |correlation| descending

    What to look for:
      - Random backbone: all correlations near 0 — no channel specifically
        responds to vessel presence
      - Good LeJEPA backbone: several channels with |correlation| > 0.3,
        meaning those channels learned to activate on vessel-like structures
    """
    backbone.eval()
    sum_corr = None
    count = 0

    for i, (images, masks) in enumerate(tqdm(dataloader, total=num_batches, desc="  Channel selectivity")):
        if i >= num_batches:
            break
        images = images.cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feats = backbone(images)
        last = _fix_channels(feats[-1] if isinstance(feats, (list, tuple)) else feats).float()  # (B, C, h, w)

        B, C, h, w = last.shape
        mask_feat = F.interpolate(
            masks.float(), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(1)  # (B, h, w)

        # Flatten spatial dims: (B, C, N) and (B, N)
        act_flat  = last.cpu().view(B, C, -1)            # (B, C, N)
        mask_flat = mask_feat.view(B, 1, -1).expand_as(act_flat)  # (B, C, N)

        # Pearson correlation per image per channel
        act_z  = act_flat  - act_flat.mean(dim=-1, keepdim=True)
        mask_z = mask_flat - mask_flat.mean(dim=-1, keepdim=True)
        num    = (act_z * mask_z).sum(dim=-1)            # (B, C)
        denom  = act_z.norm(dim=-1) * mask_z.norm(dim=-1) + 1e-8
        corr   = (num / denom).mean(dim=0)               # (C,) averaged over batch

        sum_corr = corr if sum_corr is None else sum_corr + corr
        count += 1

    avg_corr = (sum_corr / count).numpy()                # (C,)
    top_k    = np.argsort(np.abs(avg_corr))[::-1]        # descending by |corr|
    return avg_corr, top_k


def save_channel_selectivity_plot(
    correlations: np.ndarray,
    top_k_idx: np.ndarray,
    save_path: str,
    title: str = "",
    top_k: int = 20,
):
    """Bar chart of the top-K channels most correlated with vessel mask."""
    idxs = top_k_idx[:top_k]
    vals = correlations[idxs]
    colors = ['steelblue' if v >= 0 else 'tomato' for v in vals]

    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.6), 4))
    ax.bar(range(top_k), vals, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline( 0.3, color='green',  linewidth=1, linestyle='--', label='|r|=0.3 (moderate)')
    ax.axhline(-0.3, color='green',  linewidth=1, linestyle='--')
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f'Ch {i}' for i in idxs], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Pearson r with vessel mask')
    ax.set_xlabel(f'Top-{top_k} channels by |r|')
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def save_gradcam_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    cams_guided: np.ndarray,
    cams_unguided: np.ndarray,
    save_path: str,
    title: str = "",
    n_show: int = 6,
):
    """
    4-column grid: [Input | Vessel-guided GradCAM | Unguided GradCAM | GT Mask]
    Expects images in [-1,1], masks in [0,1].
    """
    B = min(images.shape[0], n_show)
    fig, axes = plt.subplots(B, 4, figsize=(13, 3 * B))
    if B == 1:
        axes = axes[np.newaxis, :]

    cmap_heat = matplotlib.colormaps['jet']

    col_titles = [
        "Input",
        "Vessel-guided GradCAM",
        "Unguided GradCAM\n(fires on bright chambers)",
        "GT Mask",
    ]

    for i in range(B):
        img_np  = ((images[i, 0].cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        mask_np = (masks[i, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_rgb = np.stack([img_np] * 3, axis=-1)

        def overlay(cam_np):
            heat_rgb = (cmap_heat(cam_np)[:, :, :3] * 255).astype(np.uint8)
            alpha    = (cam_np[:, :, None] * 0.6).clip(0, 0.6)
            return (img_rgb * (1 - alpha) + heat_rgb * alpha).clip(0, 255).astype(np.uint8)

        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 1].imshow(overlay(cams_guided[i]))
        axes[i, 2].imshow(overlay(cams_unguided[i]))
        axes[i, 3].imshow(mask_np, cmap='gray')

        for j in range(4):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. t-SNE OF BACKBONE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    dataloader,
    max_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts global-average-pooled, L2-normalized embeddings from the last backbone stage.
    Also records the vessel pixel ratio per image (fraction of mask pixels == 1).

    Returns:
        embeddings   : (N, D) float32 numpy
        vessel_ratio : (N,)   float32 numpy — used for coloring t-SNE scatter
    """
    backbone.eval()
    all_emb, all_ratio = [], []

    for images, masks in tqdm(dataloader, desc="  Embeddings"):
        emb = _pool_embed(backbone, images.cuda())
        ratio = masks.float().mean(dim=(1, 2, 3)).cpu()
        all_emb.append(emb)
        all_ratio.append(ratio)
        if sum(e.shape[0] for e in all_emb) >= max_samples:
            break

    embeddings   = torch.cat(all_emb,   dim=0)[:max_samples].numpy()
    vessel_ratio = torch.cat(all_ratio, dim=0)[:max_samples].numpy()
    return embeddings, vessel_ratio


def plot_tsne(
    embeddings: np.ndarray,
    vessel_ratios: np.ndarray,
    title: str,
    save_path: str,
):
    """
    2D t-SNE scatter colored by vessel content.

    What to look for:
      - Random/bad backbone:   uniform scatter, no spatial structure
      - Good pretrained:       images with similar vessel density cluster together,
                               forming a gradient from vessel-sparse (red) to vessel-heavy (green)
    """
    from sklearn.manifold import TSNE

    print("  Running t-SNE (this may take ~30s)...")
    coords = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=vessel_ratios, cmap='RdYlGn',
        s=14, alpha=0.75,
        vmin=0, vmax=max(vessel_ratios.max(), 1e-6)
    )
    plt.colorbar(sc, ax=ax, label='Vessel pixel ratio (0=no vessel, 1=all vessel)')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DIAGNOSTIC RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(
    backbone_path: str,
    encoder_name: str,
    output_dir: str,
    img_size: int = 256,
    batch_size: int = 8,
):
    os.makedirs(output_dir, exist_ok=True)

    from data.dataloader import ArcadeDataset
    from data.transformWrapper import TransformsWrapper
    from torch.utils.data import DataLoader

    base_ds = ArcadeDataset(
        split='validation', transform=None, root_dir='.',
        json_path='data/ARCADE/processed/dataset.json'
    )
    ds = TransformsWrapper(base_ds, input_size=img_size, mode='validation')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"\n{'='*60}")
    print(f"  Diagnostics for: {encoder_name}")
    print(f"  Backbone:        {backbone_path}")
    print(f"  Output:          {output_dir}")
    print(f"{'='*60}\n")

    print("[Building backbones]")
    bb_lejepa  = _build_backbone(encoder_name, backbone_path)
    bb_random  = _build_backbone(encoder_name, None)   # fresh random init for comparison

    # ── 1. Representation Invariance ──────────────────────────────────────────
    print("\n[1/3] Representation Invariance")
    stats_lejepa = measure_representation_invariance(bb_lejepa, loader, num_batches=30)
    stats_random = measure_representation_invariance(bb_random, loader, num_batches=30)

    print(f"\n  {'Metric':<30} {'LeJEPA':>12} {'Random':>12}")
    print("  " + "-" * 56)
    for k in stats_lejepa:
        print(f"  {k:<30} {stats_lejepa[k]:>12.4f} {stats_random[k]:>12.4f}")

    inv_path = os.path.join(output_dir, 'invariance_stats.txt')
    with open(inv_path, 'w') as f:
        f.write(f"Encoder : {encoder_name}\n")
        f.write(f"Backbone: {backbone_path}\n\n")
        f.write(f"{'Metric':<30} {'LeJEPA':>12} {'Random':>12}\n")
        f.write("-" * 56 + "\n")
        for k in stats_lejepa:
            f.write(f"{k:<30} {stats_lejepa[k]:>12.4f} {stats_random[k]:>12.4f}\n")
        f.write("\nInterpretation\n")
        f.write("-" * 56 + "\n")
        f.write("invariance_ratio >> 1.0  → backbone learned view-invariant features (GOOD)\n")
        f.write("invariance_ratio ≈  1.0  → backbone ignores augmentation context   (BAD)\n")
        f.write("mean_inter_cosine → 0    → representations are diverse, not collapsed (GOOD)\n")
        f.write("mean_inter_cosine → 1    → representational collapse (all embeddings same) (BAD)\n")
    print(f"  Saved: {inv_path}")

    # ── 2. GradCAM + Channel Selectivity ─────────────────────────────────────
    print("\n[2/3] GradCAM + Channel Selectivity")
    images, masks = next(iter(loader))
    images_cuda = images.cuda()

    for tag, bb in [('lejepa', bb_lejepa), ('random', bb_random)]:
        print(f"  Computing GradCAM for [{tag}]...")
        try:
            cams_guided   = compute_gradcam(bb, images_cuda, masks=masks)
            cams_unguided = compute_gradcam(bb, images_cuda, masks=None)
            save_gradcam_grid(
                images=images_cuda,
                masks=masks,
                cams_guided=cams_guided,
                cams_unguided=cams_unguided,
                save_path=os.path.join(output_dir, f'gradcam_{tag}.png'),
                title=f"GradCAM — {tag.upper()} backbone ({encoder_name})",
            )
        except RuntimeError as e:
            print(f"  GradCAM failed for {tag}: {e}")

        print(f"  Computing channel selectivity for [{tag}]...")
        corr, top_idx = compute_channel_selectivity(bb, loader, num_batches=20)
        save_channel_selectivity_plot(
            correlations=corr,
            top_k_idx=top_idx,
            save_path=os.path.join(output_dir, f'channel_selectivity_{tag}.png'),
            title=f"Channel–vessel correlation — {tag.upper()} ({encoder_name})\n"
                  f"Channels with |r|>0.3 respond specifically to vessel structures",
        )
        top5_corr = corr[top_idx[:5]]
        print(f"  Top-5 |r| values: {np.abs(top5_corr).round(3).tolist()}")

    # ── 3. t-SNE ──────────────────────────────────────────────────────────────
    print("\n[3/3] t-SNE")
    for tag, bb in [('lejepa', bb_lejepa), ('random', bb_random)]:
        print(f"  Extracting embeddings [{tag}]...")
        emb, ratios = extract_embeddings(bb, loader, max_samples=500)
        plot_tsne(
            embeddings=emb,
            vessel_ratios=ratios,
            title=f"t-SNE — {tag.upper()} backbone ({encoder_name})\n"
                  f"Color = vessel pixel ratio (green = more vessels)",
            save_path=os.path.join(output_dir, f'tsne_{tag}.png'),
        )

    print(f"\n✅  All diagnostics saved to: {output_dir}")
    print("\n[How to read the results]")
    print("  invariance_stats.txt              → numerical proof of LeJEPA objective learned")
    print("  gradcam_lejepa.png col 1          → vessel-guided GradCAM (uses GT mask to direct gradient)")
    print("  gradcam_lejepa.png col 2          → unguided GradCAM (fires on bright chambers — expected)")
    print("  channel_selectivity_lejepa.png    → bars above 0.3 = channels that specifically encode vessels")
    print("  tsne_lejepa.png                   → should show vessel-density gradient structure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeJEPA pretraining quality diagnostics")
    parser.add_argument('--backbone_path', type=str, required=True,
                        help="Path to best_backbone.pth from LeJEPA training")
    parser.add_argument('--encoder_name',  type=str, required=True,
                        choices=['convnextv2_tiny', 'swinv2_tiny_window8_256', 'resnet50'],
                        help="Encoder architecture name (timm name)")
    parser.add_argument('--output_dir',    type=str, default='runs/diagnostics',
                        help="Directory to save diagnostic outputs")
    parser.add_argument('--img_size',      type=int, default=256)
    parser.add_argument('--batch_size',    type=int, default=8)
    args = parser.parse_args()

    run_diagnostics(
        backbone_path=args.backbone_path,
        encoder_name=args.encoder_name,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )
