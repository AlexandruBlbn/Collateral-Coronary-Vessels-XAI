"""
SparK-style pretraining for hierarchical ConvNet backbones.

Reference: "Designing BERT for Convolutional Networks: Sparse and Hierarchical
Masked Modeling" (Tian et al., 2023) — arxiv.org/abs/2301.03580

HOW THIS DIFFERS FROM THE PAPER
---------------------------------
The original SparK uses true *sparse convolutions* (from MinkowskiEngine / SparseConv)
so that masked positions genuinely receive no gradient from their neighbours during
the forward pass.  Sparse conv backends are not available in this environment.

Instead we use the standard "masked-input" approach (also used by SimMIM):
  1. Divide the image into non-overlapping 32×32 patches.
  2. Randomly mask ~60 % of patches by replacing them with a learned mask token.
  3. Feed the corrupted image through the backbone unchanged.
  4. Decode hierarchically from the multi-scale feature pyramid.
  5. Supervise reconstruction ONLY on masked patches with normalised-pixel loss.

This is functionally equivalent for downstream segmentation transfer: the backbone
must infer what was in the masked regions from context — learning semantic, spatially-
local features (vessel branches, texture) — because the loss only rewards correct
reconstruction of the masked patches.

Supported backbones (CNN hierarchical, features_only compatible):
  • resnet50
  • convnextv2_tiny

SwinV2 is deliberately excluded: it is a Transformer backbone and is already covered
by SimMIM_Train.py which handles it correctly.
"""

import os
import sys
import gc
import yaml
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper
from utils.helpers import set_seed

set_seed(42)
scaler = torch.amp.GradScaler()


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable Frangi vesselness filter (GPU, batched)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_second_derivatives(
    x: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Scale-normalised Hessian components via separable Gaussian convolution.
    Returns (Dxx, Dyy, Dxy), each (B, 1, H, W), multiplied by σ² for scale
    normalisation so responses are comparable across scales.
    """
    ksize = max(3, 2 * int(4 * sigma + 0.5) + 1)
    pad   = ksize // 2
    t     = torch.arange(ksize, device=x.device, dtype=torch.float32) - pad

    g   = torch.exp(-0.5 * (t / sigma) ** 2)
    g   = g / g.sum()
    dg  = -(t / sigma ** 2) * g
    d2g = ((t ** 2 / sigma ** 2 - 1) / sigma ** 2) * g
    d2g = d2g - d2g.mean()          # zero-sum correction (removes DC offset)

    g_row   = g.view  (1, 1, 1, ksize)
    g_col   = g.view  (1, 1, ksize, 1)
    d2g_row = d2g.view(1, 1, 1, ksize)
    d2g_col = d2g.view(1, 1, ksize, 1)
    dg_row  = dg.view (1, 1, 1, ksize)
    dg_col  = dg.view (1, 1, ksize, 1)

    # Reflect padding avoids the artificial intensity discontinuity that
    # zero-padding creates at image borders.  With sigma=11 the kernel is
    # 89px wide — zero-padding would produce a strong Hessian response at
    # the border that the filter mistakes for a tubular structure.
    def _cv(inp, k, ph, pw):   # reflect-pad then convolve (no implicit padding)
        return F.conv2d(F.pad(inp, (pw, pw, ph, ph), mode='reflect'), k)

    Dxx = _cv(_cv(x, g_col,   pad, 0), d2g_row, 0, pad)
    Dyy = _cv(_cv(x, d2g_col, pad, 0), g_row,   0, pad)
    Dxy = _cv(_cv(x, dg_col,  pad, 0), dg_row,  0, pad)

    s2 = sigma ** 2
    return Dxx * s2, Dyy * s2, Dxy * s2


@torch.no_grad()
def frangi_torch(
    x:           torch.Tensor,
    sigmas:      tuple = (1, 3, 5, 7, 9, 11),
    beta:        float = 1.0,
    black_ridges: bool = True,
) -> torch.Tensor:
    """
    2-D Frangi tubeness matching the scikit-image implementation.

    Args:
        x            : (B, 1, H, W) float tensor, any value range
        sigmas       : Gaussian scales in pixels — matches frangiPreproces.py
                       range(1, 16, 2) = (1,3,5,7,9,11,13,15); we stop at 11
                       since vessels thinner than 1px and wider than ~22px are
                       not meaningful at 256×256 angio resolution.
        beta         : blobness penalty (Frangi β).  1.0 matches the working
                       config in frangiPreproces.py (scikit-image default is 0.5
                       but that config uses 1.0).
        black_ridges : True = dark tubes on bright background (angiograms).
                       CRITICAL: without this sign gate the filter also fires on
                       metal guidewires / catheter edges (strong positive
                       curvature) at the same strength as vessel walls.

    Returns:
        vesselness : (B, 1, H, W) in [0, 1], per-sample min-max normalised
    """
    x = x.detach().float()
    best  = torch.zeros_like(x)
    pairs = []

    for sigma in sigmas:
        Dxx, Dyy, Dxy = _gaussian_second_derivatives(x, sigma)
        disc = ((Dxx - Dyy).pow(2) / 4 + Dxy.pow(2)).clamp(min=0).sqrt()
        l1   = (Dxx + Dyy) / 2 - disc   # algebraically smaller eigenvalue
        l2   = (Dxx + Dyy) / 2 + disc   # algebraically larger eigenvalue
        # Sort by absolute value: lA = minor (≈0 along vessel axis),
        #                         lB = principal (large curvature across vessel)
        lA = torch.where(l1.abs() <= l2.abs(), l1, l2)
        lB = torch.where(l1.abs() <= l2.abs(), l2, l1)
        pairs.append((lA, lB))

    # Dynamic c² normalisation (same as original Frangi paper)
    c_sq = max(
        (lA.pow(2) + lB.pow(2)).amax().item()
        for lA, lB in pairs
    ) / 4 + 1e-8

    for lA, lB in pairs:
        # Sign gate — only respond to the correct ridge polarity.
        # For dark ridges (black_ridges=True): principal curvature lB must be
        # negative (bowl-shaped dip). Guidewires/catheter edges have lB > 0
        # (ridge-shaped peak) and are suppressed here.
        if black_ridges:
            valid = (lB < 0).float()
        else:
            valid = (lB > 0).float()

        Rb_sq = (lA / (lB.abs() + 1e-8)).pow(2)    # blobness (0 = tube)
        S_sq  = lA.pow(2) + lB.pow(2)              # structureness
        v = (torch.exp(-Rb_sq / (2 * beta ** 2))
             * (1 - torch.exp(-S_sq / (2 * c_sq)))
             * valid)
        best = torch.maximum(best, v)

    return best / (best.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def loader(img_size: int, batch_size: int, split: str = 'train'):
    def seed_worker(wid):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    base = ArcadeDataset(
        split=split, mode='pretrain', transform=None,
        root_dir='.', json_path='data/ARCADE/processed/dataset.json'
    )
    ds = TransformsWrapper(base, input_size=img_size, mode='lejepa')   # no label needed
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=4, persistent_workers=True,
        worker_init_fn=seed_worker, generator=g
    )


# ─────────────────────────────────────────────────────────────────────────────
# Patch masking
# ─────────────────────────────────────────────────────────────────────────────

class PatchMasker(nn.Module):
    """
    Divides input images into non-overlapping square patches and replaces a
    random fraction of them with a learned scalar mask token.

    Design choices
    --------------
    patch_size = 16 : With 256×256 images and vessels that are 1-5 px wide,
        a 32×32 patch leaves only 8×8 = 64 total patches (38 masked at 60%).
        That is too few context patches: the backbone has to reconstruct large
        dark holes from just 26 visible tokens, which is too easy by predicting
        the background mean.  16×16 gives 16×16 = 256 patches instead, each
        erasing a short vessel segment that is inferable from direction/continuity
        in neighbouring patches.  This decouples the mask grid from the backbone's
        stride-32 feature map — fine, because our masked-input formulation does
        not require alignment (unlike the original sparse-conv SparK).

    mask_ratio = 0.65 : Slightly above the original 60% to compensate for having
        more total patches (256 vs 64). At 65%, 166 of 256 patches are masked,
        leaving 90 context patches — still challenging but not trivially solvable
        by background prediction.

    Scalar mask token (shape (1,1,1,1)) + expand: identical concept to SimMIM's
        mask token — simple, avoids the complexity of a (1, C, patch, patch)
        learnable block which would inject spatial bias into every masked position.
    """

    def __init__(self, img_size: int = 256, patch_size: int = 16,
                 mask_ratio: float = 0.65, vessel_bias: float = 3.0):
        """
        vessel_bias : how strongly to oversample vessel-containing patches.
            With vessel_bias=3, a fully-vessel patch (frangi score=1.0) is
            ~4× more likely to be masked than a pure-background patch (score=0).
            Set to 0.0 to revert to uniform random masking.
        """
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size   = patch_size
        self.mask_ratio   = mask_ratio
        self.vessel_bias  = vessel_bias
        self.n_patches_side = img_size // patch_size
        n = self.n_patches_side ** 2
        self.n_masked = int(n * mask_ratio)
        # Learnable mask value broadcast over the entire patch area
        self.mask_token = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor,
                vessel_hint: torch.Tensor | None = None):
        """
        Args
        ----
        x            : (B, 1, H, W) — input image
        vessel_hint  : (B, N*N) float in [0, 1] — per-patch vessel importance
                       (typically Frangi pooled to patch level).  If None,
                       falls back to uniform random masking.

        Returns
        -------
        x_masked : (B, 1, H, W)   — image with masked patches replaced
        mask_map : (B, 1, H, W)   — 1 where the patch is masked, 0 elsewhere
        """
        B, C, H, W = x.shape
        P = self.patch_size
        N = self.n_patches_side

        # Sample a fresh random mask for every image independently.
        # Vessel-biased masking: add vessel_bias × vessel_score to each patch's
        # random value, then sort descending.  Higher-scoring patches rise to
        # the top and are selected as the n_masked targets first.
        noise = torch.rand(B, N * N, device=x.device)
        if vessel_hint is not None and self.vessel_bias > 0:
            noise = noise * (1.0 + self.vessel_bias * vessel_hint)
        ids      = noise.argsort(dim=1, descending=True)  # highest score first
        mask_ids = ids[:, :self.n_masked]                  # (B, n_masked)

        # Build a binary patch-level mask: 1 = masked
        patch_mask = torch.zeros(B, N * N, device=x.device)
        patch_mask.scatter_(1, mask_ids, 1.0)
        patch_mask = patch_mask.view(B, 1, N, N)           # (B, 1, N, N)

        # Upsample to pixel resolution
        mask_map = F.interpolate(patch_mask, scale_factor=P, mode='nearest')  # (B, 1, H, W)

        # Replace masked pixels with the learned scalar token
        x_masked = x * (1.0 - mask_map) + self.mask_token * mask_map
        return x_masked, mask_map


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical decoder
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Single decoder stage: 2× bilinear upsample → lateral add (if skip provided)
    → 3×3 conv → BN → GELU.

    WHY BN+GELU and not LayerNorm: the decoder is tiny (no weight sharing needed)
    and processes spatial feature maps in standard (B, C, H, W) format. BN is
    faster and more stable at small batch sizes than LN for conv features.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        fused = in_ch + skip_ch if skip_ch > 0 else in_ch
        self.conv = nn.Sequential(
            nn.Conv2d(fused, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SparKDecoder(nn.Module):
    """
    Lightweight hierarchical decoder.

    The backbone produces N feature maps at strides [s0, …, sN-1] depending on
    the architecture (ResNet50: 5 maps at strides 2/4/8/16/32; ConvNeXtV2-tiny:
    4 maps at strides 4/8/16/32).  The decoder starts from the coarsest feature,
    upsamples 2× at each stage while adding the lateral skip from the next-finer
    encoder stage, then applies log2(finest_stride) extra no-skip upsample blocks
    to reach full resolution.

    Output channels are derived dynamically: 128 → 64 → 32 → … keeping the
    decoder narrow so the backbone must do the heavy lifting.
    """

    def __init__(self, encoder_channels: list[int], finest_stride: int = 4,
                 out_channels: int = 1):
        super().__init__()
        import math
        N = len(encoder_channels)          # number of encoder stages

        # One dec_ch entry per encoder stage plus one for the extra-ups start
        dec_ch = [max(16, 128 >> i) for i in range(N + 1)]

        # Stage 0: project coarsest feature to dec_ch[0]
        self.input_proj = nn.Conv2d(encoder_channels[0], dec_ch[0], 1, bias=False)

        # N-1 decoder blocks with encoder skips (coarsest → finest)
        self.blocks = nn.ModuleList()
        for i in range(N - 1):
            skip_ch = encoder_channels[i + 1]
            self.blocks.append(DecoderBlock(dec_ch[i], skip_ch, dec_ch[i + 1]))

        # After N-1 blocks the spatial resolution is at finest_stride.
        # Apply log2(finest_stride) additional no-skip 2× blocks to reach stride-1.
        n_extra = int(math.log2(finest_stride))
        ch = dec_ch[N - 1]
        self.extra_ups = nn.ModuleList(
            [DecoderBlock(ch, 0, ch) for _ in range(n_extra)]
        )

        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        features : list ordered coarsest → finest, each (B, C, h, w)
        Returns  : (B, out_channels, H, W) — full-resolution reconstruction
        """
        x = self.input_proj(features[0])
        for i, block in enumerate(self.blocks):
            skip = features[i + 1]
            x = block(x, skip)
        for block in self.extra_ups:
            x = block(x, None)
        return self.head(x)


class FrangiHead(nn.Module):
    """
    Auxiliary vessel-detection head branching off the coarsest backbone feature.

    During pretraining the backbone sees only masked inputs and must still predict
    where vessels are — producing a direct vessel-structure gradient every step
    regardless of which patches happen to be masked.

    Architecture: coarsest feature (B, C, h, w)
        → bilinear upsample to full resolution
        → 1×1 conv projection (C → 64)
        → BN + GELU
        → 3×3 conv (64 → 1)
        → sigmoid

    Loss: MSE against frangi_torch(x) target (per-sample [0,1] normalised).
    MSE (vs BCE) because the Frangi target is a soft continuous vesselness map,
    not a binary label — most pixels are intermediate (~0.1–0.4) near vessels.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Predict at feature-map resolution (e.g. 8×8 for ResNet50 stride-32).
        # The caller is responsible for downsampling the target to match.
        return self.net(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Full SparK model
# ─────────────────────────────────────────────────────────────────────────────

class SparKModel(nn.Module):
    """
    Masked image modelling for hierarchical ConvNets.

    backbone (timm, features_only=True)
        → multi-scale feature list (coarsest last internally, reversed here)
    SparKDecoder
        → full-res reconstruction

    Loss: normalised pixel MSE on masked patches only.

    Normalised pixel target (Eq. 1 of the SparK / MAE papers):
        target = (x - mean(x)) / (std(x) + 1e-6)
    Mean and std are computed per patch so the loss is not dominated by
    globally bright or dark images.  Without normalisation the model quickly
    learns the overall mean brightness of angiograms (trivially consistent across
    all masked patches) and the loss plateaus without learning structure.
    """

    def __init__(self, encoder_name: str = 'resnet50',
                 img_size: int = 256, patch_size: int = 32,
                 mask_ratio: float = 0.60,
                 vessel_bias: float = 3.0,
                 frangi_weight: float = 1.0):
        """
        vessel_bias   : passed to PatchMasker — oversample vessel patches by this
                        factor (0 = uniform random masking)
        frangi_weight : weight of the auxiliary Frangi vesselness prediction loss
                        relative to the reconstruction MSE loss
        """
        super().__init__()

        self.backbone = timm.create_model(
            encoder_name,
            pretrained=False,
            in_chans=1,
            features_only=True,
        )
        self.channels_list = self.backbone.feature_info.channels()   # finest → coarsest
        self.masker = PatchMasker(img_size=img_size, patch_size=patch_size,
                                  mask_ratio=mask_ratio, vessel_bias=vessel_bias)
        # Determine the spatial stride of the finest feature map so the decoder
        # knows how many extra 2× upsample blocks are needed to reach stride-1.
        strides      = list(self.backbone.feature_info.reduction())   # e.g. [2,4,8,16,32]
        finest_stride = int(min(strides))
        # Decoder expects coarsest → finest
        dec_channels = list(reversed(self.channels_list))
        self.decoder = SparKDecoder(encoder_channels=dec_channels,
                                    finest_stride=finest_stride, out_channels=1)
        self.patch_size   = patch_size
        self.img_size     = img_size
        self.frangi_weight = frangi_weight

        # Feature-level indices (timm returns features finest→coarsest).
        #
        # FrangiHead at stride-4 (64×64 for 256px input):
        #   A 4px vessel is 1 cell wide at this resolution — actually spatially
        #   localisable.  At stride-32 (8×8) it's 0.125 cells — invisible to MSE.
        # GradCAM at stride-8 (32×32): good semantic+spatial balance for
        #   diagnostic visualisation; fine enough to show individual vessel segments.
        self.frangi_feat_idx  = min(range(len(strides)), key=lambda i: abs(strides[i] - 4))
        self.gradcam_feat_idx = min(range(len(strides)), key=lambda i: abs(strides[i] - 8))

        # Auxiliary vessel-detection head (branches off stride-4 backbone feature)
        frangi_ch = self.channels_list[self.frangi_feat_idx]
        self.frangi_head = FrangiHead(frangi_ch)

    def _norm_pixel_target(self, x: torch.Tensor) -> torch.Tensor:
        """Per-patch normalised pixel target, same resolution as x."""
        P = self.patch_size
        # Unfold into patches, compute per-patch mean/std, fold back
        # Using F.unfold for efficiency, then fold the stats back
        B, C, H, W = x.shape
        N = H // P
        patches = x.unfold(2, P, P).unfold(3, P, P)  # (B,1,N,N,P,P)
        # mean / std per patch
        flat = patches.reshape(B, 1, N, N, P * P)    # (B,1,N,N,P*P)
        mean = flat.mean(-1, keepdim=True)            # (B,1,N,N,1)
        std  = flat.std(-1, keepdim=True).clamp(min=1e-6)
        flat_norm = (flat - mean) / std
        # Fold back to pixel space
        out = flat_norm.view(B, 1, N, N, P, P)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(B, 1, H, W)
        return out

    def forward(self, x: torch.Tensor):
        # ── Vessel-biased masking ──────────────────────────────────────────────
        # Compute Frangi vesselness, pool to patch grid, use to bias which
        # patches get masked (vessel patches are ~4× more likely to be masked
        # when vessel_bias=3, putting direct reconstruction pressure on vessels).
        vessel_hint = None
        if self.masker.vessel_bias > 0:
            with torch.no_grad():
                fmap = frangi_torch(x, sigmas=(1, 3, 5, 7), beta=1.0)  # (B,1,H,W)
                P = self.masker.patch_size
                vessel_hint = F.avg_pool2d(fmap, kernel_size=P)     # (B,1,N,N)
                vessel_hint = vessel_hint.view(x.size(0), -1)        # (B,N*N)
                vessel_hint = vessel_hint / (vessel_hint.amax(dim=1, keepdim=True) + 1e-8)

        x_masked, mask_map = self.masker(x, vessel_hint=vessel_hint)

        # Batch masked+clean images through the backbone in one pass.
        # FrangiHead gets clean features so it cannot shortcut via mask-token
        # detection: biased masking puts the mask token preferentially over
        # vessel patches, so without this the head trivially inverts the mask.
        B = x.size(0)
        combined   = torch.cat([x_masked, x], dim=0)        # (2B, 1, H, W)
        feats_all  = self.backbone(combined)                 # list of (2B, C, h, w)
        feats       = [f[:B] for f in feats_all]             # masked -> reconstruction
        feats_clean = [f[B:] for f in feats_all]             # clean  -> FrangiHead
        feats_dec = list(reversed(feats))                    # coarsest -> finest

        # ── Reconstruction loss ────────────────────────────────────────────────
        recon = self.decoder(feats_dec)         # (B, 1, H, W)

        target     = self._norm_pixel_target(x)
        recon_norm = self._norm_pixel_target(recon)

        # Exclude the outermost patch ring from the reconstruction loss.
        # Border patches straddle the film/collimator edge — a sharp intensity
        # discontinuity that is genuinely hard to reconstruct and thus produces
        # a persistent gradient that teaches the backbone to attend to borders.
        N  = self.img_size // self.patch_size          # patches per side
        P  = self.patch_size
        inner = torch.ones(B, 1, N, N, device=x.device)
        inner[:, :,  0, :] = 0   # top row
        inner[:, :, -1, :] = 0   # bottom row
        inner[:, :, :,  0] = 0   # left col
        inner[:, :, :, -1] = 0   # right col
        inner_map = F.interpolate(inner, scale_factor=P, mode='nearest')

        loss_mask  = mask_map * inner_map
        recon_loss = ((recon_norm - target) ** 2 * loss_mask).sum() / (loss_mask.sum() + 1e-6)

        # ── Auxiliary Frangi prediction loss ──────────────────────────────────
        # The FrangiHead predicts vessel structure from the coarsest (most semantic)
        # backbone feature, providing explicit vessel-structure gradient every step
        # regardless of the masking pattern.
        frangi_loss = torch.tensor(0.0, device=x.device)
        if self.frangi_weight > 0:
            # Use stride-4 clean feature (64x64): vessels are ~1 cell wide here,
            # so the head can actually learn spatial vessel locations.
            # feats_clean is indexed finest→coarsest (timm ordering).
            mid_clean = feats_clean[self.frangi_feat_idx].float()   # (B, C, 64, 64)
            with torch.no_grad():
                frangi_target = frangi_torch(x, sigmas=(1, 3, 5, 7, 9, 11), beta=1.0)
                frangi_target_ds = F.adaptive_avg_pool2d(
                    frangi_target, mid_clean.shape[2:])
            frangi_pred = self.frangi_head(mid_clean)
            frangi_loss = F.mse_loss(frangi_pred, frangi_target_ds)

        loss = recon_loss + self.frangi_weight * frangi_loss
        return loss, recon, mask_map, feats


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_reconstructions(imgs: torch.Tensor, recon: torch.Tensor,
                          mask_map: torch.Tensor, epoch: int,
                          writer: SummaryWriter, n: int = 4):
    """
    TensorBoard image grid: [original | masked input | reconstruction | inpainted]

    inpainted = original where unmasked, reconstruction where masked — the
    cleanest visual test: if pretraining works, inpainted vessels should look
    like the original.
    """
    n = min(n, imgs.size(0))
    with torch.no_grad():
        orig  = (imgs[:n] * 0.5 + 0.5).clamp(0, 1).float().cpu()
        msk   = mask_map[:n].float().cpu()

        # The decoder predicts per-patch normalised values (mean=0, unbounded range).
        # sigmoid() maps 0→0.5 making everything grey. Instead, min-max normalise
        # per sample so the full dynamic range of the prediction is visible.
        rec = recon[:n].float().cpu()
        rec_min = rec.amin(dim=(1, 2, 3), keepdim=True)
        rec_max = rec.amax(dim=(1, 2, 3), keepdim=True)
        rec = (rec - rec_min) / (rec_max - rec_min + 1e-8)

        inp_m = orig * (1 - msk)          # original with masked patches zeroed
        inp_f = orig * (1 - msk) + rec * msk  # inpainted: original + predicted patches

        items = []
        for i in range(n):
            items += [orig[i].repeat(3, 1, 1),
                      inp_m[i].repeat(3, 1, 1),
                      rec[i].repeat(3, 1, 1),
                      inp_f[i].repeat(3, 1, 1)]

        grid = torchvision.utils.make_grid(items, nrow=4, padding=2, normalize=False)
        writer.add_image("Val/Recon [orig|masked|recon|inpainted]", grid, epoch)


def _log_gradcam(model: nn.Module, imgs: torch.Tensor, masks: torch.Tensor,
                 epoch: int, writer: SummaryWriter, n: int = 4):
    """
    Vessel-conditioned CAM using the trained FrangiHead.

    Interpretation:
      GOOD — hot regions (red/yellow) overlap vessel tree in the GT mask column.
      BAD  — activation concentrated on image borders or catheter artifact.

    Grid (nrow=3): [input | GradCAM overlay | GT mask]
    """
    m = model.module if hasattr(model, 'module') else model
    m.eval()
    n = min(n, imgs.size(0))
    imgs_f = imgs[:n].detach().float()

    with torch.enable_grad():
        m.zero_grad(set_to_none=True)
        feats = m.backbone(imgs_f)
        target_feat = feats[m.frangi_feat_idx].float()
        target_feat.retain_grad()
        vessel_pred = m.frangi_head(target_feat)
        vessel_pred.mean().backward()

    with torch.no_grad():
        grad  = target_feat.grad
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        cam   = F.relu((alpha * target_feat).sum(dim=1, keepdim=True))
        cam   = F.interpolate(cam, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        cam   = cam / (cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

    img_v = (imgs[:n] * 0.5 + 0.5).float().clamp(0, 1).cpu()
    msk_v = masks[:n].float().cpu()
    cam_np = cam.cpu().numpy()

    items = []
    for i in range(n):
        inp_rgb = img_v[i].repeat(3, 1, 1)
        jet = torch.from_numpy(cm.jet(cam_np[i, 0])[:, :, :3]).float().permute(2, 0, 1)
        blend = 0.55 * inp_rgb + 0.45 * jet
        items += [inp_rgb, blend, msk_v[i].repeat(3, 1, 1)]

    grid = torchvision.utils.make_grid(items, nrow=3, padding=2, normalize=False)
    writer.add_image("Val/VesselCAM [input|cam|gt]", grid, epoch)
    m.train()


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate epochs
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model: SparKModel, dataloader: DataLoader, optimiser: torch.optim.Optimizer,
                scheduler, epoch: int, writer: SummaryWriter) -> float:
    model.train()
    running = 0.0
    epoch_total = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")
    for step, batch in pbar:
        # Dataloader may return (img, mask, is_syntax) or (img, mask)
        img = batch[0].cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, _, _, _ = model(img)

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        v = loss.item()
        running     += v
        epoch_total += v

        pbar.set_postfix({'loss': running / (step + 1)})
        global_step = epoch * len(dataloader) + step
        writer.add_scalar("Train/Step_Loss", v, global_step)

    avg = epoch_total / len(dataloader)
    writer.add_scalar("Train/Epoch_Loss", avg, epoch)
    return avg


@torch.no_grad()
def validate_epoch(model: SparKModel, dataloader: DataLoader,
                   epoch: int, writer: SummaryWriter) -> float:
    model.eval()
    running = 0.0
    first_img = first_mask = first_recon = first_mmap = None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Val {epoch + 1}")
    for step, batch in pbar:
        img  = batch[0].cuda()
        gt   = batch[1].cuda() if len(batch) > 1 else img   # GT mask if available

        loss, recon, mask_map, _ = model(img)
        running += loss.item()
        pbar.set_postfix({'val_loss': running / (step + 1)})

        if step == 0:
            first_img   = img
            first_mask  = gt
            first_recon = recon
            first_mmap  = mask_map

    avg = running / len(dataloader)
    writer.add_scalar("Val/Loss", avg, epoch)
    print(f"  Val loss: {avg:.4f}")

    # Visual diagnostics (need grad for GradCAM → called outside no_grad)
    if first_img is not None:
        _log_reconstructions(first_img, first_recon, first_mmap, epoch, writer)

    return avg, first_img, first_mask


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(path: str, model: SparKModel, optimiser, scheduler,
                     epoch: int, best_loss: float):
    m = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     m.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'best_loss':            best_loss,
    }, path)


def _load_checkpoint(path: str, model: SparKModel, optimiser, scheduler):
    if not os.path.isfile(path):
        print(f"  No checkpoint at '{path}' — starting from scratch.")
        return 0, float('inf')

    ckpt = torch.load(path, map_location='cuda')
    m = model.module if hasattr(model, 'module') else model
    missing, unexpected = m.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  [ckpt] New keys initialised from scratch: {missing}")
    if unexpected:
        print(f"  [ckpt] Unexpected keys ignored: {unexpected}")
    try:
        optimiser.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
    except (ValueError, KeyError) as e:
        print(f"  [ckpt] Optimizer/scheduler state incompatible ({e}); "
              "model weights loaded but optimizer restarted from scratch.")

    start  = ckpt['epoch'] + 1
    best   = ckpt.get('best_loss', float('inf'))
    print(f"  Resumed from epoch {start}  (best_loss={best:.4f})")
    return start, best


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_encoder(config: dict, writer: SummaryWriter):
    img_size   = config['training']['img_size']
    batch_size = config['training']['batch_size']
    epochs     = config['training']['epochs']
    ckpt_dir   = config['logging']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    last_ckpt = os.path.join(ckpt_dir, 'last_model.pth')
    best_ckpt = os.path.join(ckpt_dir, 'best_model.pth')
    done_file = os.path.join(ckpt_dir, 'DONE')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = loader(img_size, batch_size, split='train')
    val_loader   = loader(img_size, batch_size // 2, split='validation')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SparKModel(
        encoder_name=config['model']['encoder_name'],
        img_size=img_size,
        patch_size=config['model']['patch_size'],
        mask_ratio=config['model']['mask_ratio'],
        vessel_bias=config['model'].get('vessel_bias', 3.0),
        frangi_weight=config['model'].get('frangi_weight', 0.3),
    ).cuda()

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    total_iters   = epochs * len(train_loader)
    warmup_iters  = config['training']['warmup_epochs'] * len(train_loader)
    scheduler1    = LinearLR(optimiser, start_factor=0.01, end_factor=1.0,
                             total_iters=warmup_iters)
    scheduler2    = CosineAnnealingLR(optimiser, T_max=total_iters - warmup_iters,
                                      eta_min=1e-6)
    scheduler     = SequentialLR(optimiser, [scheduler1, scheduler2],
                                 milestones=[warmup_iters])

    start_epoch, best_loss = _load_checkpoint(last_ckpt, model, optimiser, scheduler)

    patience       = config['training'].get('patience', 30)
    no_improve     = 0
    save_every     = config['training'].get('save_every', 10)

    for epoch in range(start_epoch, epochs):
        avg_train = train_epoch(model, train_loader, optimiser, scheduler, epoch, writer)
        avg_val, first_img, first_mask = validate_epoch(model, val_loader, epoch, writer)

        # GradCAM — needs grad, called after no_grad block
        if first_img is not None:
            _log_gradcam(model, first_img, first_mask, epoch, writer)

        writer.add_scalar("Train/LR", optimiser.param_groups[0]['lr'], epoch)

        # ── Save last ──────────────────────────────────────────────────────────
        _save_checkpoint(last_ckpt, model, optimiser, scheduler, epoch, best_loss)

        # ── Save best (by val loss) ────────────────────────────────────────────
        if avg_val < best_loss:
            best_loss = avg_val
            no_improve = 0
            _save_checkpoint(best_ckpt, model, optimiser, scheduler, epoch, best_loss)
            # Save bare backbone weights for fine-tuning (compatible with finetune.py)
            backbone = (model.module.backbone if hasattr(model, 'module')
                        else model.backbone)
            torch.save(backbone.state_dict(),
                       os.path.join(ckpt_dir, 'best_backbone.pth'))
            print(f"  ✓ Best saved  (val_loss={best_loss:.4f})  epoch {epoch + 1}")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{patience}  "
                  f"(val={avg_val:.4f}, best={best_loss:.4f})")

        # ── Periodic backbone snapshot ─────────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            backbone = (model.module.backbone if hasattr(model, 'module')
                        else model.backbone)
            snap = os.path.join(ckpt_dir, f'backbone_ep{epoch + 1}.pth')
            torch.save(backbone.state_dict(), snap)
            print(f"  [Snapshot] backbone_ep{epoch + 1}.pth")

        if no_improve >= patience:
            print(f"  Early stopping after {epoch + 1} epochs.")
            break

    with open(done_file, 'w') as f:
        f.write('Training completed.')
    print(f"\n✅ Done: {config['model']['encoder_name']}  best_val_loss={best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # SwinV2 is a transformer — use SimMIM_Train.py for it.
    ENCODERS = ['resnet50', 'convnextv2_tiny']

    for encoder in ENCODERS:
        experiment_name = f'{encoder}_spark'
        checkpoint_dir  = f'checkpoints/{experiment_name}'
        done_file       = os.path.join(checkpoint_dir, 'DONE')

        if os.path.exists(done_file):
            print(f"\n{'='*60}\n⏭  {encoder} already trained (DONE found). Skipping.\n{'='*60}")
            continue

        print(f"\n{'='*60}\n🚀  Starting SparK pretraining: {encoder}\n{'='*60}")

        config = {
            'experiment_name': experiment_name,
            'logging': {
                'log_dir':        f'runs/{experiment_name}',
                'checkpoint_dir': checkpoint_dir,
            },
            'model': {
                'encoder_name': encoder,
                # 16-pixel patches: 256×256 → 16×16 = 256 patch grid.
                # 32×32 leaves only 64 patches total — too few context tokens for
                # the backbone to infer thin vessel trajectories through large masked
                # holes. 16×16 masks shorter vessel segments, inferable from direction
                # in adjacent patches, and keeps 90 context patches at mask_ratio=0.65.
                'patch_size':   16,
                # 65% mask ratio — slightly above the SparK 60% to keep the task
                # challenging with the larger 256-patch grid.
                'mask_ratio':   0.65,
                # Vessel-biased masking: oversample vessel-containing patches by
                # this factor. With 3.0 a fully-vessel patch is ~4× more likely
                # to be masked. Set 0.0 to disable (pure uniform masking).
                'vessel_bias':  3.0,
                # Weight of the auxiliary Frangi vesselness prediction loss.
                # 1.0 balances it against the reconstruction loss (Frangi MSE
                # is ~0.05-0.1, recon loss is ~1-2, so 1.0 gives ~5-10% weight).
                'frangi_weight': 1.0,
            },
            'training': {
                'img_size':       256,
                # Larger batch than LeJEPA: no 5-view augmentation overhead,
                # each forward pass is a single image with a mask.
                'batch_size':     32,
                'epochs':         150,
                'lr':             1e-4,
                'weight_decay':   0.05,
                'warmup_epochs':  10,
                # Early stopping: reconstruction loss plateaus faster than
                # contrastive losses; 30 epochs without improvement is generous.
                'patience':       30,
                'save_every':     10,
            },
        }

        writer = SummaryWriter(log_dir=config['logging']['log_dir'])

        # Dump config next to TensorBoard logs for reproducibility
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
        with open(os.path.join(config['logging']['log_dir'], 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)

        try:
            train_encoder(config, writer)
        finally:
            writer.close()
            # Release GPU memory before starting the next backbone
            torch.cuda.empty_cache()
            gc.collect()

    print("\n🎉 All SparK encoders processed.")
