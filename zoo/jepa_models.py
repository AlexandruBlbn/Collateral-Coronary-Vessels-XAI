"""
DenseLeJEPA model with V-JEPA 2.1 innovations:
1. Deep self-supervision (multi-level losses)
2. Distance-weighted context loss coefficients

Combined with LeJEPA SIGReg for collapse prevention.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zoo.backbones import get_backbone


class ProjectionHead(nn.Module):
    """MLP projector: encoder_dim → proj_dim, with BN + L2 normalize."""

    def __init__(self, in_dim: int, proj_dim: int = 256, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or proj_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )
        self.proj_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
            x = self.mlp(x)
            x = x.reshape(B, N, self.proj_dim)
        else:
            x = self.mlp(x)
        return F.normalize(x, dim=-1, p=2)


class PredictionHead(nn.Module):
    """
    Dense predictor: context features → target token predictions.
    Uses attention pooling over context tokens based on target bounding box.
    """

    def __init__(self, proj_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        assert proj_dim % num_heads == 0

        self.q_proj = nn.Linear(proj_dim, proj_dim)
        self.k_proj = nn.Linear(proj_dim, proj_dim)
        self.v_proj = nn.Linear(proj_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, proj_dim)
        self.scale = self.head_dim ** -0.5

        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 4),
            nn.GELU(),
            nn.Linear(proj_dim * 4, proj_dim),
        )

        self.box_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, proj_dim),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, Dh = x.shape
        return x.transpose(1, 2).reshape(B, N, H * Dh)

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        ctx_boxes: torch.Tensor,
        tgt_box: torch.Tensor,
        num_tgt_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_ctx, D = ctx_tokens.shape

        box_rel = tgt_box - ctx_boxes
        q = self.box_embed(box_rel).unsqueeze(1)

        Q = self._split_heads(self.q_proj(q))
        K = self._split_heads(self.k_proj(ctx_tokens))
        V = self._split_heads(self.v_proj(ctx_tokens))

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        pooled = attn_weights @ V
        pooled = self._merge_heads(pooled)

        pooled = pooled.expand(-1, num_tgt_tokens, -1)
        pred = self.out_proj(pooled)
        pred = self.mlp(pred)

        return pred, attn_weights.squeeze(2)


def compute_distance_weights(
    ctx_boxes: torch.Tensor,
    tgt_boxes: torch.Tensor,
    num_patches_h: int = 16,
    num_patches_w: int = 16,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    V-JEPA 2.1 distance-weighted loss coefficients.

    For each target crop, compute λ_i = λ / sqrt(d_min(i, M))
    where d_min is the minimum spatial distance from target patch i
    to any context patch, normalized by patch grid dimensions.

    Args:
        ctx_boxes: [B, 4] normalized context boxes [y, x, h, w]
        tgt_boxes: [B, nl, 4] normalized target boxes per crop
        num_patches_h: height of patch grid (e.g. 16 for 256/16)
        num_patches_w: width of patch grid

    Returns:
        weights: [B, nl, N_tgt] distance weights per target token
    """
    B, nl, _ = tgt_boxes.shape
    N_tgt = num_patches_h * num_patches_w

    # Convert normalized box coordinates to patch grid indices
    # [y, x, h, w] ∈ [0, 1] → patch indices
    ctx_y = ctx_boxes[:, 0:1]  # [B, 1]
    ctx_x = ctx_boxes[:, 1:2]
    ctx_h = ctx_boxes[:, 2:3]
    ctx_w = ctx_boxes[:, 3:4]

    # Center of context box in patch coords
    ctx_cy = (ctx_y + ctx_h / 2) * num_patches_h
    ctx_cx = (ctx_x + ctx_w / 2) * num_patches_w

    # Target grid patch positions
    patch_y = torch.arange(num_patches_h, device=tgt_boxes.device).float() + 0.5
    patch_x = torch.arange(num_patches_w, device=tgt_boxes.device).float() + 0.5
    gy, gx = torch.meshgrid(patch_y, patch_x, indexing='ij')
    grid = torch.stack([gy, gx], dim=-1)  # [H, W, 2]

    weights = torch.zeros(B, nl, N_tgt, device=tgt_boxes.device)
    for bi in range(B):
        for li in range(nl):
            # Target box center in patch coords
            tgt_box = tgt_boxes[bi, li]  # [4]
            tgt_cy = (tgt_box[0] + tgt_box[2] / 2) * num_patches_h
            tgt_cx = (tgt_box[1] + tgt_box[3] / 2) * num_patches_w

            # Distance from each target patch to context center
            dist_y = (grid[..., 0] - ctx_cy[bi, 0]) / num_patches_h
            dist_x = (grid[..., 1] - ctx_cx[bi, 0]) / num_patches_w
            dist = torch.sqrt(dist_y ** 2 + dist_x ** 2 + eps)

            # V-JEPA 2.1: λ_i = 1 / sqrt(d_min)
            w = 1.0 / torch.sqrt(dist + eps)
            weights[bi, li] = w.view(-1)

    return weights  # [B, nl, N_tgt]


class DenseLeJepaModel(nn.Module):
    """
    Dense LeJEPA model with V-JEPA 2.1 deep supervision.

    Architecture:
        1. ViT/Swin Encoder (grayscale in, patch tokens out)
        2. Multi-level Projection Heads (one per supervised layer)
        3. Prediction Heads (one per level) with cross-attention
        4. SIGReg applied externally on projected tokens

    When deep_supervision=True:
        - Backbone returns features from multiple layers (out_indices)
        - Each level has its own projection + prediction head
        - Loss is summed across levels with distance weights
    """

    def __init__(
        self,
        encoder_name: str = "vit_small",
        proj_dim: int = 256,
        in_channels: int = 1,
        deep_supervision: bool = False,
        deep_supervision_out_indices: Tuple[int, ...] = (3, 6, 9, 11),
        predictor_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.proj_dim = proj_dim
        self.deep_supervision = deep_supervision

        # 1. Encoder
        self.backbone = get_backbone(
            encoder_name,
            in_channels=in_channels,
            return_intermediates=deep_supervision,
            out_indices=deep_supervision_out_indices if deep_supervision else None,
        )

        # Infer encoder dimension for each level
        self.encoder_dims = self._get_encoder_dims()
        if deep_supervision:
            self.num_levels = len(self.encoder_dims)
        else:
            self.num_levels = 1

        # 2. Projection Heads (one per level for deep supervision)
        self.projections = nn.ModuleList([
            ProjectionHead(dim, proj_dim)
            for dim in self.encoder_dims
        ])

        # 3. Prediction Heads (one per level)
        self.predictors = nn.ModuleList([
            PredictionHead(proj_dim, num_heads=predictor_heads)
            for _ in range(self.num_levels)
        ])

    def _get_encoder_dims(self) -> List[int]:
        """Determine encoder output dimension(s) via forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 224, 224)
            out = self.backbone(dummy)

            if isinstance(out, (list, tuple)):
                dims = []
                for o in out:
                    if o.dim() == 4:
                        # Feature map [B, C, H, W] → flatten tokens → dim = C
                        dims.append(o.shape[1])
                    else:
                        # Token [B, N, D]
                        dims.append(o.shape[-1])
                return dims
            else:
                if out.dim() == 4:
                    return [out.shape[1]]
                return [out.shape[-1]]

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[
        List[torch.Tensor],
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
    ]:
        """
        Encode input and project to dense tokens at each level.

        Args:
            x: [B, 1, H, W]

        Returns:
            tokens_list: list of [B, N, proj_dim] per level
            spatial: None
            feats_list: list of raw features per level
        """
        feats = self.backbone(x)  # single tensor or list

        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        # Reshape feature maps to tokens if needed (conv backbones)
        token_feats = []
        for f in feats:
            if f.dim() == 4:
                B, C, Hf, Wf = f.shape
                f = f.reshape(B, C, Hf * Wf).transpose(1, 2)  # [B, N, C]
            token_feats.append(f)

        # Project each level
        tokens_list = []
        for i, f in enumerate(token_feats):
            tokens_list.append(self.projections[i](f))

        return tokens_list, None, token_feats

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        tokens_list, _, _ = self.encode(x)
        return tokens_list  # list of [B, N, proj_dim]
