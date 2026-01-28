
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import copy
import math

try:
    from zoo.backbones import get_backbone
except ImportError:
    from backbones import get_backbone


class DINOHead(nn.Module):
    """
    DINO projection head with prototypes.
    Maps backbone features to a high-dimensional space for self-distillation.
    """
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int = 2048, 
        bottleneck_dim: int = 256,
        out_dim: int = 8192,
        use_bn: bool = True,
        norm_last_layer: bool = True
    ):
        super().__init__()
        
        # MLP layers
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Last layer (prototypes)
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.data.fill_(1)
            self.last_layer.parametrizations.weight.original0.requires_grad = False
            
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class GramAnchor(nn.Module):
    """
    Gram Anchoring module from DINOv3.
    Stabilizes feature learning by anchoring to a running Gram matrix.
    """
    def __init__(self, feature_dim: int, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.feature_dim = feature_dim
        
        # Register anchor as buffer (not a parameter, but saved in state_dict)
        self.register_buffer('gram_anchor', torch.zeros(feature_dim, feature_dim))
        self.register_buffer('initialized', torch.tensor(False))
        
    @torch.no_grad()
    def update_anchor(self, features: torch.Tensor):
        """
        Update the Gram anchor using EMA.
        features: (B, D) tensor of normalized features
        """
        # Compute current batch Gram matrix
        # G = F^T @ F / B
        batch_gram = torch.mm(features.T, features) / features.shape[0]
        
        if not self.initialized:
            self.gram_anchor.copy_(batch_gram)
            self.initialized.fill_(True)
        else:
            # EMA update
            self.gram_anchor.mul_(self.momentum).add_(batch_gram, alpha=1 - self.momentum)
    
    def compute_gram_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram anchoring loss.
        Encourages current batch Gram matrix to match the anchor.
        """
        # Current batch Gram matrix
        batch_gram = torch.mm(features.T, features) / features.shape[0]
        
        # Frobenius norm difference
        loss = F.mse_loss(batch_gram, self.gram_anchor.detach())
        
        return loss


class MultiCropWrapper(nn.Module):
    """
    Wrapper to process multiple crops through backbone.
    Handles gradient checkpointing for memory efficiency.
    Processes crops of different sizes separately.
    """
    def __init__(self, backbone, head, use_checkpoint: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.use_checkpoint = use_checkpoint
        
    def forward(self, x):
        # x can be a list of tensors (multi-crop) or single tensor
        if isinstance(x, list):
            # Group crops by size to process together
            # Assume first 2 are global crops (same size), rest are local crops (same size)
            outputs = []
            
            # Process global crops (256x256)
            global_crops = torch.cat(x[:2], dim=0)
            if self.use_checkpoint and self.training:
                global_features = checkpoint(self._forward_backbone, global_crops, use_reentrant=False)
            else:
                global_features = self._forward_backbone(global_crops)
            global_output = self.head(global_features)
            
            # Split global outputs back
            batch_size = x[0].shape[0]
            outputs.extend(torch.split(global_output, batch_size, dim=0))
            
            # Process local crops (96x96) if any
            if len(x) > 2:
                local_crops = torch.cat(x[2:], dim=0)
                if self.use_checkpoint and self.training:
                    local_features = checkpoint(self._forward_backbone, local_crops, use_reentrant=False)
                else:
                    local_features = self._forward_backbone(local_crops)
                local_output = self.head(local_features)
                
                # Split local outputs back
                outputs.extend(torch.split(local_output, batch_size, dim=0))
            
            return outputs
        else:
            if self.use_checkpoint and self.training:
                features = checkpoint(self._forward_backbone, x, use_reentrant=False)
            else:
                features = self._forward_backbone(x)
            return self.head(features)
    
    def _forward_backbone(self, x):
        # Resize to 256x256 if needed (for local crops with fixed-size backbones like Swin)
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        features = self.backbone(x)
        # Global average pooling if needed (for CNN backbones or ViT with spatial output)
        if len(features.shape) == 4:  # (B, C, H, W)
            features = features.mean(dim=[2, 3])
        elif len(features.shape) == 3:  # (B, N, D) - ViT tokens
            features = features.mean(dim=1)
        return features
    
    def get_intermediate_features(self, x):
        """Get features before projection head for Gram computation."""
        # Resize to 256x256 if needed
        if x.shape[-1] != 256 or x.shape[-2] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = features.mean(dim=[2, 3])
        elif len(features.shape) == 3:
            features = features.mean(dim=1)
        return features


class DINOv3Loss(nn.Module):
    """
    DINOv3 Loss with Gram Anchoring.
    Combines:
    - Cross-entropy distillation loss (student vs teacher)
    - Center regularization
    - Gram anchoring loss
    """
    def __init__(
        self,
        out_dim: int,
        num_local_crops: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        num_epochs: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_local_crops = num_local_crops
        self.num_global_crops = 2
        
        # Teacher temperature schedule
        self.teacher_temp_schedule = self._build_temp_schedule(
            warmup_teacher_temp, 
            teacher_temp, 
            warmup_teacher_temp_epochs, 
            num_epochs
        )
        
        # Center for teacher outputs (prevents collapse)
        self.register_buffer("center", torch.zeros(1, out_dim))
        
    def _build_temp_schedule(self, warmup_temp, final_temp, warmup_epochs, total_epochs):
        """Build teacher temperature schedule."""
        warmup = torch.linspace(warmup_temp, final_temp, warmup_epochs)
        constant = torch.ones(total_epochs - warmup_epochs) * final_temp
        return torch.cat([warmup, constant])
    
    def forward(self, student_output, teacher_output, epoch):
        """
        Compute DINO loss.
        
        student_output: list of tensors, one per crop
        teacher_output: list of tensors, only global crops
        epoch: current epoch for temperature scheduling
        """
        student_out = torch.cat(student_output, dim=0)
        teacher_out = torch.cat(teacher_output, dim=0)
        
        # Get temperatures
        student_temp = self.student_temp
        teacher_temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        
        # Student: softmax with temperature
        student_probs = F.log_softmax(student_out / student_temp, dim=-1)
        
        # Teacher: softmax with temperature and centering
        teacher_probs = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)
        teacher_probs = teacher_probs.detach()
        
        # Cross-entropy loss
        # For each global teacher output, compute loss against all student outputs
        total_loss = 0
        n_loss_terms = 0
        
        batch_size = student_output[0].shape[0]
        
        for t_idx in range(self.num_global_crops):
            for s_idx in range(len(student_output)):
                if t_idx == s_idx:
                    # Skip matching views
                    continue
                
                t_start = t_idx * batch_size
                t_end = (t_idx + 1) * batch_size
                s_start = s_idx * batch_size
                s_end = (s_idx + 1) * batch_size
                
                loss = -torch.sum(
                    teacher_probs[t_start:t_end] * student_probs[s_start:s_end],
                    dim=-1
                ).mean()
                
                total_loss += loss
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        
        # Update center
        self._update_center(teacher_out)
        
        return total_loss
    
    @torch.no_grad()
    def _update_center(self, teacher_output):
        """Update center with EMA of teacher outputs."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOv3(nn.Module):
    """
    Complete DINOv3 model with student, teacher, and Gram anchoring.
    """
    def __init__(
        self,
        backbone_name: str = 'vit_small_patch16_224',
        in_channels: int = 1,
        embed_dim: int = None,  # Auto-detect if None
        projection_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        use_gram_anchoring: bool = True,
        gram_momentum: float = 0.99,
        use_checkpoint: bool = True
    ):
        super().__init__()
        
        self.use_gram_anchoring = use_gram_anchoring
        
        # Create student backbone
        student_backbone = get_backbone(
            model_name=backbone_name,
            in_channels=in_channels,
            pretrained=False
        )
        
        # Auto-detect embed_dim from backbone output
        if embed_dim is None:
            with torch.no_grad():
                dummy = torch.randn(1, in_channels, 256, 256)
                feats = student_backbone(dummy)
                if len(feats.shape) == 4:  # (B, C, H, W)
                    embed_dim = feats.shape[1]
                elif len(feats.shape) == 3:  # (B, N, D)
                    embed_dim = feats.shape[2]
                else:
                    embed_dim = feats.shape[-1]
            print(f"--> Auto-detected embed_dim: {embed_dim}")
        
        # Create projection head
        student_head = DINOHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            out_dim=projection_dim
        )
        
        # Wrap student
        self.student = MultiCropWrapper(
            student_backbone,
            student_head,
            use_checkpoint=use_checkpoint
        )
        
        # Create teacher as EMA copy of student
        self.teacher = copy.deepcopy(self.student)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Gram anchoring module
        if use_gram_anchoring:
            self.gram_anchor = GramAnchor(
                feature_dim=embed_dim,
                momentum=gram_momentum
            )
        
        self.embed_dim = embed_dim
        
    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """Update teacher with EMA of student weights."""
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)
    
    def forward(self, crops, update_gram: bool = False):
        """
        Forward pass for DINOv3.
        
        crops: list of crop tensors [global1, global2, local1, local2, ...]
        update_gram: whether to update Gram anchor this step
        
        Returns:
            student_output: list of projection outputs
            teacher_output: list of projection outputs (only global crops)
            gram_loss: Gram anchoring loss (if enabled)
        """
        # Student forward on all crops
        student_output = self.student(crops)
        
        # Teacher forward only on global crops
        with torch.no_grad():
            teacher_output = self.teacher(crops[:2])  # Only global crops
        
        # Gram anchoring
        gram_loss = torch.tensor(0.0, device=crops[0].device)
        if self.use_gram_anchoring:
            # Get intermediate features from student (before projection)
            with torch.no_grad():
                # Only use global crops for Gram computation
                global_crops = torch.cat(crops[:2], dim=0)
                features = self.student.get_intermediate_features(global_crops)
                features = F.normalize(features, dim=-1, p=2)
                
                if update_gram:
                    self.gram_anchor.update_anchor(features)
            
            # Compute Gram loss (with gradients)
            global_crops_grad = torch.cat(crops[:2], dim=0)
            features_grad = self.student.get_intermediate_features(global_crops_grad)
            features_grad = F.normalize(features_grad, dim=-1, p=2)
            gram_loss = self.gram_anchor.compute_gram_loss(features_grad)
        
        return student_output, teacher_output, gram_loss
    
    def get_student_backbone(self):
        """Extract student backbone for downstream tasks."""
        return self.student.backbone


def get_attention_maps(model, image, device):
    """
    Extract attention maps from ViT/Swin for visualization.
    Uses feature gradient-based attention for backbone-agnostic visualization.
    
    Args:
        model: DINOv3 model or backbone
        image: (1, C, H, W) tensor
        device: torch device
        
    Returns:
        attention_maps: (B, 1, H, W) tensor - feature importance heatmap
    """
    model.eval()
    
    # Get the backbone
    if hasattr(model, 'student'):
        backbone = model.student.backbone
    else:
        backbone = model
    
    # Get the inner model if wrapped
    if hasattr(backbone, 'model'):
        inner_model = backbone.model
    else:
        inner_model = backbone
    
    with torch.no_grad():
        image = image.to(device)
        
        # Resize if needed
        if image.shape[-1] != 256 or image.shape[-2] != 256:
            image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Try to get attention maps based on architecture
        
        # Check for ViT with blocks
        if hasattr(inner_model, 'blocks'):
            try:
                x = inner_model.patch_embed(image)
                if hasattr(inner_model, 'cls_token'):
                    cls_token = inner_model.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([cls_token, x], dim=1)
                if hasattr(inner_model, 'pos_embed'):
                    x = x + inner_model.pos_embed
                x = inner_model.pos_drop(x) if hasattr(inner_model, 'pos_drop') else x
                
                # Get attention from last block
                for i, block in enumerate(inner_model.blocks):
                    if i == len(inner_model.blocks) - 1:
                        B, N, C = x.shape
                        qkv = block.attn.qkv(block.norm1(x) if hasattr(block, 'norm1') else x)
                        qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
                        qkv = qkv.permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]
                        
                        attn = (q @ k.transpose(-2, -1)) * (C // block.attn.num_heads) ** -0.5
                        attn = attn.softmax(dim=-1)
                        
                        if hasattr(inner_model, 'cls_token'):
                            attn_map = attn[:, :, 0, 1:]
                        else:
                            attn_map = attn.mean(dim=2)
                        
                        num_patches = int(attn_map.shape[-1] ** 0.5)
                        attn_map = attn_map.mean(dim=1)  # Average across heads
                        attn_map = attn_map.reshape(B, 1, num_patches, num_patches)
                        return attn_map
                    else:
                        x = block(x)
            except Exception as e:
                print(f"ViT attention extraction failed: {e}")
        
        # For Swin or other architectures: use feature magnitude as importance
        # This is a simple but effective visualization
        try:
            features = backbone(image)
            
            if len(features.shape) == 4:  # (B, C, H, W)
                # Use channel-wise mean of feature magnitudes
                attn_map = features.abs().mean(dim=1, keepdim=True)
            elif len(features.shape) == 3:  # (B, N, D)
                # Reshape tokens to spatial
                B, N, D = features.shape
                H = W = int(N ** 0.5)
                features_spatial = features.transpose(1, 2).reshape(B, D, H, W)
                attn_map = features_spatial.abs().mean(dim=1, keepdim=True)
            else:
                return None
            
            # Normalize to [0, 1]
            attn_map = attn_map - attn_map.min()
            attn_map = attn_map / (attn_map.max() + 1e-8)
            
            return attn_map
            
        except Exception as e:
            print(f"Feature-based attention extraction failed: {e}")
            return None
    
    return None