"""
DenseLeJEPA model: encoder + projection + predictor.

Pure JEPA + SIGReg on raw grayscale images. No Frangi, no teacher distillation.
"""

from typing import Optional, Tuple

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

    Uses attention pooling over context tokens based on target bounding box,
    then MLP to predict target tokens.
    """

    def __init__(self, proj_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        assert proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"

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


class DenseLeJepaModel(nn.Module):
    """
    Dense LeJEPA model.

    Architecture:
        1. ViT Encoder (grayscale in, patch tokens out)
        2. Projection Head (d_encoder → proj_dim, L2-normalized)
        3. Prediction Head (cross-attention over context → predict target)
        4. SIGReg applied externally on projected tokens

    Pure JEPA + SIGReg — no Frangi, no teacher distillation.
    """

    def __init__(
        self,
        encoder_name: str = "vit_small",
        proj_dim: int = 256,
        in_channels: int = 1,
        deep_supervision: bool = False,
        deep_supervision_out_indices: Tuple[int, ...] = (2, 3),
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
            out_indices=deep_supervision_out_indices if deep_supervision else None,
        )
        self.encoder_dim = self._get_encoder_dim()

        # 2. Projection Head
        self.projection = ProjectionHead(self.encoder_dim, proj_dim)

        # 3. Prediction Head
        self.predictor = PredictionHead(proj_dim, num_heads=predictor_heads)

    def _get_encoder_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 224, 224)
            if self.deep_supervision:
                out = self.backbone(dummy)
                if isinstance(out, (list, tuple)):
                    return out[-1].shape[-1]
            out = self.backbone(dummy)
            if isinstance(out, (list, tuple)):
                return out[0].shape[-1] if out[0].dim() == 3 else out[-1].shape[-1]
            if out.dim() == 4:
                return out.shape[1]
            return out.shape[-1]

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode input and project to dense token space.

        Args:
            x: [B, 1, H, W] — grayscale input

        Returns:
            tokens:  [B, N, proj_dim] — projected patch tokens
            spatial: None
            feats:   [B, N, d_encoder] — raw encoder features
        """
        feats = self.backbone(x)

        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        tokens = self.projection(feats)

        return tokens, None, feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _, _ = self.encode(x)
        return tokens
