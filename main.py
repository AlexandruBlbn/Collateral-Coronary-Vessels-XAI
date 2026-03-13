"""
Visualise what the model receives during LeJEPA pretraining.

Layout
------
For each image in the batch (rows) the figure shows:

  Col 0    : RAW input  (straight from the dataloader, before any augmentation)
  Col 1-2  : Global crop 1 & 2  (70-100 % scale, after BorderJitter + stochastic aug)
  Col 3-5  : Local  crop 1-3   (40-80 % scale, after BorderJitter + stochastic aug)

All images are displayed in their normalised [-1, 1] range remapped to [0, 1] for
display so you can directly see what the backbone receives (not a pretty de-normalised
version — the actual tensor values).

Run: python main.py
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')          # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

# ── make project imports available ──────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from engine.LeJepa_Train import loader as lejepa_loader, augmentariLeJepa
from engine.SparK_Train import SparKModel
from utils.helpers import set_seed

# ── config ──────────────────────────────────────────────────────────────────
IMG_SIZE   = 256
BATCH_SIZE = 8          # how many images to show (one row each)
SEED       = 42

set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM comparison: UNetPlusPlus encoder  vs  SparK backbone
# ─────────────────────────────────────────────────────────────────────────────

def _build_smp_encoder(ckpt_path: str) -> nn.Module:
    """
    Reconstruct the SMP UnetPlusPlus used in train.py, load the full checkpoint,
    then return the SMP encoder submodule.

    smp_model.encoder(x) returns a list of multi-scale feature maps (EncoderMixin),
    identical interface to a timm features_only backbone.
    """
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        encoder_depth=5,
        decoder_channels=(512, 256, 128, 64, 32),
        decoder_use_batchnorm=True,
        decoder_attention_type='scse',
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    encoder = model.encoder
    encoder.eval()
    return encoder


def _build_spark_backbone(ckpt_path: str) -> nn.Module:
    """
    Build a bare timm resnet50 (features_only=True, in_chans=1) and load the
    SparK best_backbone.pth weights saved by SparK_Train.py.
    """
    backbone = timm.create_model(
        'resnet50',
        pretrained=False,
        in_chans=1,
        features_only=True,
    )
    state = torch.load(ckpt_path, map_location='cpu')
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    return backbone


def _build_convnext_spark_backbone(ckpt_path: str) -> nn.Module:
    """
    Build a bare timm convnextv2_tiny (features_only=True, in_chans=1) and load
    the SparK best_backbone.pth weights saved by SparK_Train.py.
    ConvNeXtV2-tiny strides: [4, 8, 16, 32] → stride-8 feature is index 1.
    """
    backbone = timm.create_model(
        'convnextv2_tiny',
        pretrained=False,
        in_chans=1,
        features_only=True,
    )
    state = torch.load(ckpt_path, map_location='cpu')
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    return backbone


def _build_spark_model(encoder_name: str, ckpt_path: str) -> SparKModel:
    """
    Rebuild the full SparK model so the trained FrangiHead can drive the CAM.
    """
    model = SparKModel(
        encoder_name=encoder_name,
        img_size=IMG_SIZE,
        patch_size=16,
        mask_ratio=0.65,
        vessel_bias=3.0,
        frangi_weight=1.0,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _gradcam(backbone: nn.Module, imgs: torch.Tensor,
             feat_idx: int | None = None) -> np.ndarray:
    """
    Compute GradCAM for a batch.

    feat_idx : which feature level to use (None = last/coarsest). Use stride-8
               for SparK backbones (ResNet50: idx 2; ConvNeXtV2-tiny: idx 1).

    Works for any backbone whose forward() returns List[Tensor] (both the SMP
    encoder and a timm features_only backbone satisfy this contract).

    Returns (B, H, W) float32 numpy array in [0, 1].
    """
    with torch.enable_grad():
        feats = backbone(imgs.detach().float())
        idx   = feat_idx if feat_idx is not None else -1
        last  = feats[idx].float()         # (B, C, h, w)
        last.retain_grad()
        last.mean().backward()

    with torch.no_grad():
        grad  = last.grad                                              # (B, C, h, w)
        alpha = grad.mean(dim=(2, 3), keepdim=True)                    # (B, C, 1, 1)
        cam   = F.relu((alpha * last).sum(dim=1, keepdim=True))        # (B, 1, h, w)
        cam   = F.interpolate(cam, size=imgs.shape[2:],
                              mode='bilinear', align_corners=False)    # (B, 1, H, W)
        cam   = cam / (cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        cam_np = cam.squeeze(1).cpu().numpy()                          # (B, H, W)

    return cam_np


def _spark_vessel_cam(model: SparKModel, imgs: torch.Tensor) -> np.ndarray:
    """
    Compute a CAM by backpropagating the trained FrangiHead's mean vesselness
    score to the stride-4 backbone feature it actually supervises.

    Returns (B, H, W) float32 numpy array in [0, 1].
    """
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        feats = model.backbone(imgs.detach().float())
        target_feat = feats[model.frangi_feat_idx].float()
        target_feat.retain_grad()
        vessel_pred = model.frangi_head(target_feat)
        vessel_pred.mean().backward()

    with torch.no_grad():
        grad = target_feat.grad
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * target_feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        cam = cam / (cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    return cam.squeeze(1).cpu().numpy()


def _jet_blend(img_np: np.ndarray, cam_np: np.ndarray,
               alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured CAM over a greyscale image. Returns (H, W, 3) uint8."""
    rgb  = np.stack([img_np] * 3, axis=-1)                # (H, W, 3)
    jet  = cm.jet(cam_np)[:, :, :3]                        # (H, W, 3)
    out  = (1 - alpha) * rgb + alpha * jet
    return np.clip(out, 0, 1)


def run_gradcam_comparison():
    """
    Load one validation batch, run GradCAM through both the UnetPlusPlus-trained
    resnet50 encoder and the SparK-pretrained resnet50, and save a side-by-side
    figure.

    Columns per image row:
        [original | UNet++ GradCAM | SparK GradCAM | GT mask]
    """
    UNET_CKPT  = 'checkpoints/resnet50_unetplusplus/best_model.pth'
    SPARK_CKPT = 'checkpoints/resnet50_spark/best_model.pth'
    OUT_PATH   = 'gradcam_comparison.png'
    N_VIS      = 8

    print("Loading UNet++ encoder …")
    bb_unet  = _build_smp_encoder(UNET_CKPT)

    print("Loading SparK model …")
    spark_model = _build_spark_model('resnet50', SPARK_CKPT)

    # Validation loader — returns (img, mask) in the 'syntax' split
    val_loader = lejepa_loader(IMG_SIZE, N_VIS, split='validation', mode='validation')
    batch      = next(iter(val_loader))
    imgs       = batch[0]          # (B, 1, H, W) in [-1, 1]
    gt_masks   = batch[1]          # (B, 1, H, W)
    B          = imgs.size(0)

    print(f"Running GradCAM on {B} images …")
    # ResNet50: strides [2,4,8,16,32] → stride-8 is index 2
    cam_unet  = _gradcam(bb_unet,  imgs)
    cam_spark = _spark_vessel_cam(spark_model, imgs)

    # ── Plot ────────────────────────────────────────────────────────────────
    n_cols = 4   # original | unet++ cam | spark cam | GT mask
    fig, axes = plt.subplots(B, n_cols, figsize=(3.5 * n_cols, 3.2 * B),
                              squeeze=False)
    fig.suptitle(
        "GradCAM comparison — same validation batch\n"
        "Col: Original  |  UNet++ encoder  |  SparK vessel CAM  |  GT mask",
        fontsize=11, y=1.005
    )

    col_titles  = ["Original", "UNet++ GradCAM", "SparK VesselCAM", "GT Mask"]
    col_borders = ["#555555", "#1f77b4", "#d62728", "#555555"]

    for row in range(B):
        img_np  = (imgs[row, 0].numpy() * 0.5 + 0.5).clip(0, 1)   # (H, W) in [0,1]
        mask_np = gt_masks[row, 0].numpy().clip(0, 1)

        panels = [
            img_np,
            _jet_blend(img_np, cam_unet[row]),
            _jet_blend(img_np, cam_spark[row]),
            mask_np,
        ]
        cmaps  = ['gray', None, None, 'gray']

        for col, (panel, cmap, title, border) in enumerate(
                zip(panels, cmaps, col_titles, col_borders)):
            ax = axes[row, col]
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2.5)
            if row == 0:
                ax.set_title(title, fontsize=9, pad=3, color=border)
        axes[row, 0].set_ylabel(f"img {row}", fontsize=8, rotation=0,
                                labelpad=30, va='center')

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130, bbox_inches='tight')
    print(f"Saved → {OUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# LeJEPA augmentation visualisation (original main)
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).float().cpu().numpy()
    arr = (arr * 0.5 + 0.5).clip(0.0, 1.0)
    return arr


def run_lejepa_vis():
    OUT_PATH = "pretrain_batch_vis.png"
    print(f"Loading one pretrain batch  (batch_size={BATCH_SIZE}, img_size={IMG_SIZE}) …")
    train_loader = lejepa_loader(IMG_SIZE, BATCH_SIZE, split='train', mode='lejepa')

    raw_imgs, _, _ = next(iter(train_loader))
    augment = augmentariLeJepa(img_size=IMG_SIZE)

    with torch.no_grad():
        crops = augment(raw_imgs)

    crop_labels = ["Global 1", "Global 2", "Local 1", "Local 2", "Local 3"]
    B      = raw_imgs.size(0)
    n_cols = 1 + len(crops)
    col_labels = ["RAW"] + crop_labels

    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * B))
    fig.suptitle(
        "LeJEPA pretrain — what the backbone receives\n"
        "Rows = images in batch   |   Cols = RAW · Global×2 · Local×3",
        fontsize=12, y=1.01
    )
    gs = gridspec.GridSpec(B, n_cols, figure=fig, hspace=0.05, wspace=0.05)

    for row in range(B):
        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            if col == 0:
                img_np = tensor_to_display(raw_imgs[row])
                border_color = "#444444"
            else:
                img_np = tensor_to_display(crops[col - 1][row])
                border_color = "#2ca02c" if col <= 2 else "#ff7f0e"

            ax.imshow(img_np, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9, pad=3,
                             color=border_color if col > 0 else "black")
            if col == 0:
                ax.set_ylabel(f"img {row}", fontsize=8, rotation=0,
                              labelpad=28, va='center')

    plt.savefig(OUT_PATH, dpi=120, bbox_inches='tight')
    print(f"Saved → {OUT_PATH}")


def run_convnext_gradcam():
    """
    Load ConvNeXtV2-tiny SparK backbone and ResNet50 SparK backbone, run
    GradCAM on a validation batch, and save a side-by-side figure.

    Columns per image row:
        [Original | ResNet50 SparK | ConvNeXtV2-tiny SparK | GT mask]
    """
    RESNET_CKPT  = 'checkpoints/resnet50_spark/best_model.pth'
    CONVNEXT_CKPT = 'checkpoints/convnextv2_tiny_spark/best_model.pth'
    OUT_PATH     = 'gradcam_convnext_vs_resnet.png'
    N_VIS        = 8

    print("Loading ResNet50 SparK model …")
    spark_resnet = _build_spark_model('resnet50', RESNET_CKPT)

    print("Loading ConvNeXtV2-tiny SparK model …")
    spark_convnext = _build_spark_model('convnextv2_tiny', CONVNEXT_CKPT)

    val_loader = lejepa_loader(IMG_SIZE, N_VIS, split='validation', mode='validation')
    batch      = next(iter(val_loader))
    imgs       = batch[0]          # (B, 1, H, W) in [-1, 1]
    gt_masks   = batch[1]          # (B, 1, H, W)
    B          = imgs.size(0)

    print(f"Running vessel-aware CAM on {B} images …")
    cam_resnet   = _spark_vessel_cam(spark_resnet, imgs)
    cam_convnext = _spark_vessel_cam(spark_convnext, imgs)

    n_cols = 4
    fig, axes = plt.subplots(B, n_cols, figsize=(3.5 * n_cols, 3.2 * B),
                              squeeze=False)
    fig.suptitle(
        "Vessel CAM comparison — SparK pretrained backbones\n"
        "Col: Original  |  ResNet50 SparK  |  ConvNeXtV2-tiny SparK  |  GT mask",
        fontsize=11, y=1.005
    )

    col_titles  = ["Original", "ResNet50 VesselCAM", "ConvNeXtV2 VesselCAM", "GT Mask"]
    col_borders = ["#555555", "#d62728", "#2ca02c", "#555555"]

    for row in range(B):
        img_np  = (imgs[row, 0].numpy() * 0.5 + 0.5).clip(0, 1)
        mask_np = gt_masks[row, 0].numpy().clip(0, 1)

        panels = [
            img_np,
            _jet_blend(img_np, cam_resnet[row]),
            _jet_blend(img_np, cam_convnext[row]),
            mask_np,
        ]
        cmaps  = ['gray', None, None, 'gray']

        for col, (panel, cmap, title, border) in enumerate(
                zip(panels, cmaps, col_titles, col_borders)):
            ax = axes[row, col]
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(2.5)
            if row == 0:
                ax.set_title(title, fontsize=9, pad=3, color=border)
        axes[row, 0].set_ylabel(f"img {row}", fontsize=8, rotation=0,
                                labelpad=30, va='center')

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130, bbox_inches='tight')
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    run_convnext_gradcam()

