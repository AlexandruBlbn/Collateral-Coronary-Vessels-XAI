import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from typing import Dict, List, Sequence, Tuple

from zoo.backbones import get_backbone


class DualJepaPredictor(nn.Module):
    def __init__(
        self,
        proj_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        max_pos_tokens: int = 256,
    ):
        super().__init__()
        # Two separate MLPs: context anchor vs target query (different semantic roles)
        self.ctx_coord_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, proj_dim),
        )
        self.tgt_coord_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, proj_dim),
        )
        
        # Spatial positional embeddings to differentiate tokens within the target bounding box
        self.local_pos_emb = nn.Parameter(torch.randn(1, max_pos_tokens, proj_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            activation='gelu', batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # SimSiam-style predictor head (hidden BN, no BN on output) applied token-wise
        self.simsiam_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim, bias=False),
            # Token-wise BatchNorm requires reshaping or using LayerNorm. Standard SimSiam uses BN.
            # We'll transpose dimensions in the forward pass to apply BN1d.
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )

    def _target_positional_tokens(self, num_target_tokens: int) -> torch.Tensor:
        if num_target_tokens <= self.local_pos_emb.shape[1]:
            return self.local_pos_emb[:, :num_target_tokens, :]

        # Handle higher-resolution token grids via 1D interpolation on position sequence.
        pos = F.interpolate(
            self.local_pos_emb.transpose(1, 2),
            size=num_target_tokens,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return pos

    def forward(self, context_tokens: torch.Tensor, context_boxes: torch.Tensor,
                target_boxes: torch.Tensor, num_target_tokens: int) -> tuple:
        """
        context_tokens: (B, L_c, D)
        context_boxes:  (B, 4) normalised [y, x, h, w] of the context crop
        target_boxes:   (B, 4) normalised [y, x, h, w] of the target crop
        num_target_tokens: number of spatial tokens the target resolves to (L_t)
        """
        B, L_c, D = context_tokens.shape

        # Anchor context tokens to their origin on the global canvas
        c_emb = self.ctx_coord_mlp(context_boxes)          # (B, D)
        context_tokens = context_tokens + c_emb.unsqueeze(1)

        # Build target queries: box embedding + learned spatial positions
        t_emb = self.tgt_coord_mlp(target_boxes)            # (B, D)
        queries = t_emb.unsqueeze(1).expand(B, num_target_tokens, D).clone()
        queries = queries + self._target_positional_tokens(num_target_tokens)

        seq = torch.cat([queries, context_tokens], dim=1)   # (B, L_t + L_c, D)
        out_seq = self.transformer(seq)

        pred_dense  = out_seq[:, :num_target_tokens, :]     # (B, L_t, D)
        pred_pooled = pred_dense.mean(dim=1)                # (B, D)

        # Apply SimSiam MLP with hidden BatchNorm
        # Flatten tokens to apply BN1d
        B, L_t, D = pred_dense.shape
        # Flatten: (B*L_t, D) -> Linear -> BN1d -> ReLU -> Linear -> (B, L_t, D)
        def apply_simsiam(x_in):
            x_flat = x_in.reshape(-1, D)
            x_h = self.simsiam_head[0](x_flat)
            x_h = self.simsiam_head[1](x_h) # BN
            x_h = self.simsiam_head[2](x_h) # ReLU
            x_out = self.simsiam_head[3](x_h) # Linear
            return x_out.reshape(x_in.shape)

        pred_dense = apply_simsiam(pred_dense)
        pred_pooled = apply_simsiam(pred_pooled)
        return pred_dense, pred_pooled

class DenseLeJepaModel(nn.Module):
    """
    True LeJEPA architecture: a SINGLE shared encoder for both context and target views.
    There is no EMA teacher, no stop-gradient, no momentum.
    Collapse is prevented entirely by SIGReg (Balestriero & LeCun, 2025).
    """
    def __init__(
        self,
        encoder_name='swinv2_tiny_window8_256',
        proj_dim=384,
        deep_supervision: bool = False,
        deep_supervision_out_indices: Sequence[int] = (2, 3),
    ):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.deep_supervision_out_indices = tuple(int(i) for i in deep_supervision_out_indices)

        if self.deep_supervision:
            if len(self.deep_supervision_out_indices) == 0:
                raise ValueError("deep_supervision_out_indices cannot be empty")
            self.backbone = get_backbone(
                model_name=encoder_name,
                in_channels=1,
                pretrained=False,
                return_intermediates=True,
                out_indices=self.deep_supervision_out_indices,
            )
        else:
            self.backbone = get_backbone(encoder_name)

        # Dynamically resolve backbone output channels for any architecture
        with torch.no_grad():
            dummy = torch.randn(1, 1, 256, 256)
            feats = self.backbone(dummy)
            feat_list = list(feats) if isinstance(feats, (list, tuple)) else [feats]
            enc_c = feat_list[-1].shape[1]

        self.proj      = MLP(enc_c, [512, proj_dim], norm_layer=nn.LayerNorm)
        self.predictor = DualJepaPredictor(proj_dim=proj_dim)

        self.stage_keys: List[str] = []
        self.stage_projs = nn.ModuleDict()
        self.stage_predictors = nn.ModuleDict()
        if self.deep_supervision:
            if len(feat_list) != len(self.deep_supervision_out_indices):
                raise RuntimeError(
                    "Backbone returned a different number of stages than requested "
                    f"({len(feat_list)} vs {len(self.deep_supervision_out_indices)})."
                )
            for out_idx, feat in zip(self.deep_supervision_out_indices, feat_list):
                key = str(out_idx)
                self.stage_keys.append(key)
                self.stage_projs[key] = MLP(feat.shape[1], [512, proj_dim], norm_layer=nn.LayerNorm)
                self.stage_predictors[key] = DualJepaPredictor(proj_dim=proj_dim)

    @staticmethod
    def _tokens_from_feature_map(feat_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, h_p, w_p = feat_map.shape
        tokens = feat_map.flatten(2).permute(0, 2, 1)
        return tokens, (h_p, w_p)

    def encode(self, x: torch.Tensor):
        """
        Encode any crop through the shared backbone + projection head.
        Called for BOTH context and target crops — full gradients flow through.

        Returns:
            proj_tokens: (B, L, proj_dim)  projected spatial tokens
            spatial_dims: (H', W')          grid size before flattening
            raw_feat:   (B, C, H', W')     pre-projection feature map (for probe)
        """
        # Resize to 256x256 for SwinV2 window-attention alignment
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        features = self.backbone(x)
        feat_map = features[-1] if isinstance(features, (list, tuple)) else features
        tokens, spatial_dims = self._tokens_from_feature_map(feat_map)
        proj_tokens = self.proj(tokens)                      # (B, H'*W', proj_dim)
        return proj_tokens, spatial_dims, feat_map

    def encode_multiscale(self, x: torch.Tensor) -> Dict[str, object]:
        """
        Return stage-wise projected token maps for deep self-supervision.
        In non-deep mode, this returns a single "final" stage for compatibility.
        """
        if not self.deep_supervision:
            proj_tokens, spatial_dims, feat_map = self.encode(x)
            stage = {
                "stage_key": "final",
                "proj_tokens": proj_tokens,
                "spatial_dims": spatial_dims,
                "raw_feat": feat_map,
            }
            return {
                "stages": [stage],
                "proj_tokens": proj_tokens,
                "spatial_dims": spatial_dims,
                "raw_feat": feat_map,
            }

        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        features = self.backbone(x)
        if not isinstance(features, (list, tuple)):
            raise RuntimeError("Deep supervision requires a backbone that returns intermediate stages")
        if len(features) != len(self.stage_keys):
            raise RuntimeError(
                f"Mismatch between backbone stages ({len(features)}) and configured stage heads ({len(self.stage_keys)})"
            )

        stages: List[Dict[str, object]] = []
        for stage_key, feat_map in zip(self.stage_keys, features):
            tokens, spatial_dims = self._tokens_from_feature_map(feat_map)
            proj_tokens = self.stage_projs[stage_key](tokens)
            stages.append(
                {
                    "stage_key": stage_key,
                    "proj_tokens": proj_tokens,
                    "spatial_dims": spatial_dims,
                    "raw_feat": feat_map,
                }
            )

        final_stage = stages[-1]
        return {
            "stages": stages,
            "proj_tokens": final_stage["proj_tokens"],
            "spatial_dims": final_stage["spatial_dims"],
            "raw_feat": final_stage["raw_feat"],
        }

    def get_stage_predictor(self, stage_key: str) -> nn.Module:
        if self.deep_supervision:
            return self.stage_predictors[stage_key]
        return self.predictor

    def forward(self, x: torch.Tensor):
        return self.encode(x)


class LinearClsProbe(nn.Module):
    """Lightweight classification head probing frozen backbone representations."""
    def __init__(self, encoder_name='swinv2_tiny_window8_256', num_classes=26):
        super().__init__()
        with torch.no_grad():
            m = get_backbone(encoder_name)
            dummy = torch.randn(1, 1, 256, 256)
            feats = m(dummy)
            enc_c = feats[0].shape[1] if isinstance(feats, (list, tuple)) else feats.shape[1]
        self.probe = nn.Linear(enc_c, num_classes)

    def forward(self, features: torch.Tensor):
        """features: (B, C, H, W) or (B, L, C) — GAP then linear."""
        if features.dim() == 4:
            x = features.mean(dim=(2, 3))
        elif features.dim() == 3:
            x = features.mean(dim=1)
        else:
            x = features
        return self.probe(x)
