import os
import sys
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
import timm
from torchvision.ops import MLP, roi_align
from monai.losses import DiceCELoss
from segmentation_models_pytorch.losses import TverskyLoss, SoftBCEWithLogitsLoss
from torchmetrics.classification import BinaryF1Score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import gc
import matplotlib.cm as cm
import cv2
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper
from utils.helpers import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_amp_dtype(dtype_name):
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {dtype_name}")


def create_grad_scaler(device_type, amp_dtype):
    use_scaler = device_type == 'cuda' and amp_dtype == torch.float16
    return torch.amp.GradScaler(device=device_type, enabled=use_scaler)

def loader(img_size, batch_size, split='train', mode='train', frangi_preview_dir='results/frangi_all_patients'):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    ds_mode = 'pretrain' if mode == 'lejepa' else 'syntax'
    frangi_dir = 'data/ARCADE/processed/frangi_masks' if mode == 'lejepa' else None
    frangi_preview_dir = frangi_preview_dir if mode == 'lejepa' else None
    base = ArcadeDataset(
        split=split,
        mode=ds_mode,
        transform=None,
        root_dir='.',
        json_path='data/ARCADE/processed/dataset.json',
        frangi_dir=frangi_dir,
        frangi_preview_dir=frangi_preview_dir,
    )
    ds = TransformsWrapper(base, input_size=img_size, mode=mode)
    g = torch.Generator()
    g.manual_seed(42)
    
    return DataLoader(
        ds,
        batch_size=batch_size, 
        shuffle=(split=='train'),
        num_workers=20, 
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

def configCreate(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)

# ==========================================
# 1. GPU-Accelerated Frangi Filter (PyTorch)
# ==========================================
def frangi_2d_torch(image: torch.Tensor, sigmas=[1.0, 3.0, 5.0], alpha=0.5, beta=1.0, gamma=10.0, black_ridges=True):
    """
    Native PyTorch implementation of 2D Frangi Vesselness Filter.
    Computes analytical eigenvalues of the Hessian matrix for extreme speed on GPU.
    image: (B, 1, H, W) float tensor
    """
    B, C, H, W = image.shape
    vesselness = torch.zeros_like(image)
    
    for sigma in sigmas:
        # Create Gaussian derivative kernels
        size = int(2 * round(3 * sigma) + 1)
        x = torch.arange(size, dtype=torch.float32, device=image.device) - size // 2
        y = x.view(-1, 1)
        x = x.view(1, -1)
        
        # Gaussian formula
        g = torch.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * math.pi * sigma**2)
        
        # Derivatives
        g_x = -x / (sigma**2) * g
        g_y = -y / (sigma**2) * g
        g_xx = (x**2 / sigma**4 - 1 / sigma**2) * g
        g_yy = (y**2 / sigma**4 - 1 / sigma**2) * g
        g_xy = (x * y / sigma**4) * g
        
        # Reshape for conv2d
        k_xx = g_xx.view(1, 1, size, size)
        k_yy = g_yy.view(1, 1, size, size)
        k_xy = g_xy.view(1, 1, size, size)
        
        # Calculate Hessian components (Dxx, Dyy, Dxy)
        pad = size // 2
        Dxx = F.conv2d(image, k_xx, padding=pad)
        Dyy = F.conv2d(image, k_yy, padding=pad)
        Dxy = F.conv2d(image, k_xy, padding=pad)
        
        # Analytical Eigenvalues of 2x2 matrix
        # Matrix: [[Dxx, Dxy], [Dxy, Dyy]]
        trace = Dxx + Dyy
        det = Dxx * Dyy - Dxy**2
        
        # Lambda1 and Lambda2 calculation
        sqrt_term = torch.sqrt((trace**2) / 4 - det + 1e-8)
        L1 = trace / 2 + sqrt_term
        L2 = trace / 2 - sqrt_term
        
        # Sort eigenvalues by absolute magnitude: |L1| <= |L2|
        mask_sort = torch.abs(L1) > torch.abs(L2)
        lambda1 = torch.where(mask_sort, L2, L1)
        lambda2 = torch.where(mask_sort, L1, L2)
        
        # Frangi features
        Rb = torch.abs(lambda1) / (torch.abs(lambda2) + 1e-8)
        S = torch.sqrt(lambda1**2 + lambda2**2)
        
        # Vesselness equation
        exp_Rb = torch.exp(-(Rb**2) / (2 * alpha**2))
        exp_S = 1.0 - torch.exp(-(S**2) / (2 * gamma**2))
        
        v_sigma = exp_Rb * exp_S
        
        # Black ridges (blood vessels in X-ray are dark) -> we look for positive lambda2
        if black_ridges:
            v_sigma = torch.where(lambda2 > 0, v_sigma, torch.zeros_like(v_sigma))
        else:
            v_sigma = torch.where(lambda2 < 0, v_sigma, torch.zeros_like(v_sigma))
            
        vesselness = torch.max(vesselness, v_sigma)
        
    return vesselness

# ==========================================
# 2. Hierarchical Augmentation Pipeline
# ==========================================
class augmentariLeJepa(nn.Module):
    def __init__(
        self,
        img_size=256,
        num_global_crops=2,
        num_local_crops=4,
        global_scale=(0.7, 1.0),
        local_scale=(0.10, 0.25), # Adjusted to true JEPA local scales
        global_vessel_threshold=0.015,
        local_vessel_threshold=0.02,
        local_background_threshold=0.01,
        max_global_retries=12,
        max_local_retries=20,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.local_vessel_fraction = 0.75
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.crop_ratio = (3.0 / 4.0, 4.0 / 3.0)
        self.global_vessel_threshold = global_vessel_threshold
        self.local_vessel_threshold = local_vessel_threshold
        self.local_background_threshold = local_background_threshold
        self.max_global_retries = max_global_retries
        self.max_local_retries = max_local_retries

        self.BorderJitterSize = int(img_size * 0.88)
        self.elastic = transforms.ElasticTransform(alpha=60.0, sigma=6.0)

    def _apply_stochastic_aug(self, crop: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            crop = TF.hflip(crop)
        if torch.rand(1).item() < 0.5:
            crop = TF.vflip(crop)
        if torch.rand(1).item() < 0.3:
            crop = self.elastic(crop)
        return crop

    def _get_random_crop_params_inside(self, parent_h, parent_w, scale):
        target_area = parent_h * parent_w * random.uniform(scale[0], scale[1])
        aspect_ratio = random.uniform(self.crop_ratio[0], self.crop_ratio[1])
        
        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))
        h, w = min(h, parent_h), min(w, parent_w)
        
        i = random.randint(0, parent_h - h)
        j = random.randint(0, parent_w - w)
        return i, j, h, w

    def __call__(self, batch_img: torch.Tensor):
        # Assumes batch_img is on GPU: (B, 1, H, W)
        B, C, H, W = batch_img.shape
        
        # 1. Fast GPU Frangi Response
        with torch.no_grad():
            normalized_img = (batch_img * 0.5 + 0.5).clamp(0, 1)
            vessel_batch = frangi_2d_torch(normalized_img, sigmas=[1.0, 3.0, 5.0], gamma=5.0)
            
            # Simple normalization of vesselness
            v_max = vessel_batch.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            vessel_batch = vessel_batch / (v_max + 1e-8)

        global_views = [[] for _ in range(self.num_global_crops)]
        local_views = [[] for _ in range(self.num_local_crops)]
        local_parent_idx = [[] for _ in range(self.num_local_crops)]
        local_boxes = [[] for _ in range(self.num_local_crops)]

        for b in range(B):
            img = batch_img[b]
            vessel = vessel_batch[b]

            parents = []
            # --- GLOBAL CROPS ---
            for g in range(self.num_global_crops):
                best_g_params = (0, 0, H, W)
                for _ in range(self.max_global_retries):
                    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.global_scale, ratio=self.crop_ratio)
                    score = vessel[..., i:i+h, j:j+w].mean().item()
                    if score >= self.global_vessel_threshold:
                        best_g_params = (i, j, h, w)
                        break
                
                g_i, g_j, g_h, g_w = best_g_params
                g_img = TF.crop(img, g_i, g_j, g_h, g_w)
                g_img_resized = TF.resize(g_img, [self.img_size, self.img_size], antialias=True)
                g_img_aug = self._apply_stochastic_aug(g_img_resized)
                
                global_views[g].append(g_img_aug)
                parents.append((g_i, g_j, g_h, g_w))

            # --- LOCAL CROPS (Sampled strictly inside global crops) ---
            n_vessel = int(round(self.num_local_crops * self.local_vessel_fraction))
            modes = ['vessel'] * n_vessel + ['background'] * (self.num_local_crops - n_vessel)
            random.shuffle(modes)

            for l, mode in enumerate(modes):
                p_idx = random.randrange(len(parents))
                g_i, g_j, g_h, g_w = parents[p_idx]
                
                best_l_params = (0, 0, g_h, g_w)
                best_score = -1.0 if mode == 'vessel' else float('inf')
                
                for _ in range(self.max_local_retries):
                    l_i, l_j, l_h, l_w = self._get_random_crop_params_inside(g_h, g_w, self.local_scale)
                    abs_i, abs_j = g_i + l_i, g_j + l_j
                    score = vessel[..., abs_i:abs_i+l_h, abs_j:abs_j+l_w].mean().item()
                    
                    if mode == 'vessel':
                        if score > best_score:
                            best_score, best_l_params = score, (abs_i, abs_j, l_h, l_w)
                        if score >= self.local_vessel_threshold:
                            break
                    else:
                        if score < best_score:
                            best_score, best_l_params = score, (abs_i, abs_j, l_h, l_w)
                        if score <= self.local_background_threshold:
                            break

                abs_i, abs_j, l_h, l_w = best_l_params
                c_img = TF.crop(img, abs_i, abs_j, l_h, l_w)
                c_img_resized = TF.resize(c_img, [self.img_size, self.img_size], antialias=True)
                c_img_aug = self._apply_stochastic_aug(c_img_resized)
                
                # Calculate bounding box coordinates RELATIVE to the parent global crop (Normalized 0-1)
                rel_y = (abs_i - g_i) / g_h
                rel_x = (abs_j - g_j) / g_w
                rel_h = l_h / g_h
                rel_w = l_w / g_w

                local_views[l].append(c_img_aug)
                local_parent_idx[l].append(p_idx)
                local_boxes[l].append([rel_y, rel_x, rel_h, rel_w])

        global_views = [torch.stack(v, dim=0) for v in global_views]
        local_views = [torch.stack(v, dim=0) for v in local_views]
        parent_tensor = torch.tensor(local_parent_idx, dtype=torch.long, device=batch_img.device)
        box_tensor = torch.tensor(local_boxes, dtype=torch.float32, device=batch_img.device)
        
        return {
            'global_crops': global_views,
            'local_crops': local_views,
            'local_parent_idx': parent_tensor,
            'local_boxes': box_tensor,
        }

# ==========================================
# 3. Model Architecture & Spatial Predictor
# ==========================================
class JepaPredictor(nn.Module):
    def __init__(self, proj_dim: int, spatial_tokens_total: int, hidden_dim: int = 512):
        super().__init__()
        # Flattening the context map: spatial_tokens_total is S*S (e.g., 4x4=16)
        in_features = (spatial_tokens_total * proj_dim) + hidden_dim
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pred = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, context_tokens: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        # context_tokens: (B, S*S, D) -> Flatten to preserve spatial grid mapping
        B = context_tokens.size(0)
        ctx_flat = context_tokens.reshape(B, -1) # (B, S*S*D)
        
        box_feat = self.coord_mlp(boxes)         # (B, hidden_dim)
        return self.pred(torch.cat([ctx_flat, box_feat], dim=-1)) # Output: (B, D)

        out = crops.clone()
        n = out.size(0)
        device = out.device

        hflip_mask = torch.rand(n, device=device) < 0.5
        if hflip_mask.any():
            out[hflip_mask] = torch.flip(out[hflip_mask], dims=(-1,))

        vflip_mask = torch.rand(n, device=device) < 0.5
        if vflip_mask.any():
            out[vflip_mask] = torch.flip(out[vflip_mask], dims=(-2,))

        if self.elastic_prob > 0.0:
            elastic_mask = torch.rand(n, device=device) < self.elastic_prob
            if elastic_mask.any():
                out[elastic_mask] = self.elastic(out[elastic_mask])

        return out

    def _build_integral_image(self, vessel_map: torch.Tensor) -> torch.Tensor:
        integral = vessel_map.cumsum(dim=0).cumsum(dim=1)
        return F.pad(integral, (1, 0, 1, 0), value=0.0)

    def _score_boxes(self, integral: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        y0 = boxes[:, 0]
        x0 = boxes[:, 1]
        h = boxes[:, 2]
        w = boxes[:, 3]
        y1 = y0 + h
        x1 = x0 + w

        sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        return sums / (h * w).clamp_min(1)

    def _sample_candidate_boxes(self, parent_h, parent_w, scale, num_candidates, device):
        area = float(parent_h * parent_w)
        scales = torch.empty(num_candidates, device=device).uniform_(scale[0], scale[1])
        ratios = torch.empty(num_candidates, device=device).uniform_(self.crop_ratio[0], self.crop_ratio[1])

        h = torch.round(torch.sqrt(scales * area * ratios)).long().clamp_(1, parent_h)
        w = torch.round(torch.sqrt(scales * area / ratios)).long().clamp_(1, parent_w)

        max_i = (parent_h - h + 1).clamp_min(1)
        max_j = (parent_w - w + 1).clamp_min(1)
        i = torch.floor(torch.rand(num_candidates, device=device) * max_i.float()).long()
        j = torch.floor(torch.rand(num_candidates, device=device) * max_j.float()).long()
        return torch.stack([i, j, h, w], dim=1)

    def _extract_resized_crops(self, batch_img: torch.Tensor, crop_specs, output_size: int) -> torch.Tensor:
        if not crop_specs:
            return torch.empty((0, batch_img.size(1), output_size, output_size), device=batch_img.device, dtype=batch_img.dtype)

        rois = torch.tensor(
            [[b, x0, y0, x1, y1] for b, y0, x0, y1, x1 in crop_specs],
            device=batch_img.device,
            dtype=torch.float32,
        )
        return roi_align(
            batch_img,
            rois,
            output_size=(output_size, output_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )

    def _get_random_crop_params_inside(self, parent_h, parent_w, scale):
        target_area = parent_h * parent_w * random.uniform(scale[0], scale[1])
        aspect_ratio = random.uniform(self.crop_ratio[0], self.crop_ratio[1])
        
        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))
        h, w = min(h, parent_h), min(w, parent_w)
        
        i = random.randint(0, parent_h - h)
        j = random.randint(0, parent_w - w)
        return i, j, h, w

    def __call__(self, batch_img: torch.Tensor, vessel_batch: torch.Tensor):
        """vessel_batch: Frangi guidance maps (B, 1, H, W) in [0, 1]."""
        B, C, H, W = batch_img.shape
        device = batch_img.device
        vessel_batch = vessel_batch.to(device=device, dtype=torch.float32, non_blocking=True).squeeze(1)

        global_specs = [[] for _ in range(self.num_global_crops)]
        local_specs = [[] for _ in range(self.num_local_crops)]

        for b in range(B):
            vessel_map = vessel_batch[b]
            integral = self._build_integral_image(vessel_map)

            parents = []
            for g in range(self.num_global_crops):
                candidates = self._sample_candidate_boxes(H, W, self.global_scale, self.max_global_retries, device)
                scores = self._score_boxes(integral, candidates)
                best_idx = torch.argmax(scores)
                best_g_score = scores[best_idx].item()
                best_g_params = candidates[best_idx].tolist()

                if best_g_score < self.global_vessel_threshold:
                    best_g_params = (0, 0, H, W)
                
                g_i, g_j, g_h, g_w = best_g_params
                global_specs[g].append((b, g_i, g_j, g_i + g_h, g_j + g_w))
                parents.append((g_i, g_j, g_h, g_w))

            n_vessel = int(round(self.num_local_crops * self.local_vessel_fraction))
            modes = ['vessel'] * n_vessel + ['background'] * (self.num_local_crops - n_vessel)
            random.shuffle(modes)

            for l, mode in enumerate(modes):
                p_idx = random.randrange(len(parents))
                g_i, g_j, g_h, g_w = parents[p_idx]

                candidates = self._sample_candidate_boxes(g_h, g_w, self.local_scale, self.max_local_retries, device)
                candidates[:, 0] += g_i
                candidates[:, 1] += g_j
                scores = self._score_boxes(integral, candidates)

                if mode == 'vessel':
                    best_idx = torch.argmax(scores)
                else:
                    best_idx = torch.argmin(scores)

                abs_i, abs_j, l_h, l_w = candidates[best_idx].tolist()
                local_specs[l].append((b, abs_i, abs_j, abs_i + l_h, abs_j + l_w))

        global_views = [self._apply_stochastic_aug_batch(self._extract_resized_crops(batch_img, specs, self.img_size)) for specs in global_specs]
        local_views = [self._apply_stochastic_aug_batch(self._extract_resized_crops(batch_img, specs, self.img_size)) for specs in local_specs]
        # Extract Frangi crops with the same spatial specs as the image crops.
        # No stochastic aug applied: these are soft spatial weights (not pixel targets),
        # so approximate correspondence is sufficient even after image flips/elastic.
        vessel_4d = vessel_batch.unsqueeze(1)  # (B, 1, H, W) — required by roi_align
        global_frangi = [self._extract_resized_crops(vessel_4d, specs, self.img_size) for specs in global_specs]
        local_frangi  = [self._extract_resized_crops(vessel_4d, specs, self.img_size) for specs in local_specs]
        return {
            'global_crops':  global_views,
            'local_crops':   local_views,
            'global_frangi': global_frangi,
            'local_frangi':  local_frangi,
        }

# ==========================================
# 3. Model Architecture (Spatial DENSE LeJEPA)
# ==========================================
class LeJepaModel(nn.Module):
    def __init__(self, encoder_name='swinv2_tiny_window8_256', proj_dim=128, spatial_tokens=4, ema_momentum=0.996):
        super().__init__()
        self.context_backbone = timm.create_model(
            encoder_name,
            pretrained=False,
            in_chans=1,
            features_only=True,
        )
        self.target_backbone = copy.deepcopy(self.context_backbone)
        self.channels_list = self.context_backbone.feature_info.channels()
        self.spatial_tokens = spatial_tokens
        self.pool = nn.AdaptiveAvgPool2d(spatial_tokens)          
        
        self.context_proj = MLP(self.channels_list[-1], [512, proj_dim], norm_layer=nn.LayerNorm)
        self.target_proj = copy.deepcopy(self.context_proj)
        
        # New spatially-aware predictor
        self.predictor = JepaPredictor(proj_dim=proj_dim, spatial_tokens_total=spatial_tokens*spatial_tokens)
        self.ema_momentum = ema_momentum

        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False

    def _encode(self, x, backbone, proj):
        features = list(backbone(x))
        for i in range(len(features)):
            if features[i].dim() == 4 and features[i].shape[-1] == self.channels_list[i]:
                features[i] = features[i].permute(0, 3, 1, 2).contiguous()
        
        last_map = features[-1]
        sp = self.pool(last_map)
        B, C, S, _ = sp.shape
        tokens = sp.flatten(2).permute(0, 2, 1)
        proj_out = proj(tokens.reshape(B * S * S, C)).view(B, S * S, -1)
        return features, proj_out

    def forward_context(self, x):
        return self._encode(x, self.context_backbone, self.context_proj)

    @torch.no_grad()
    def forward_target(self, x):
        return self._encode(x, self.target_backbone, self.target_proj)

    @torch.no_grad()
    def update_target_encoder(self):
        m = self.ema_momentum
        for p_t, p_c in zip(self.target_backbone.parameters(), self.context_backbone.parameters()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1.0 - m)
        for p_t, p_c in zip(self.target_proj.parameters(), self.context_proj.parameters()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1.0 - m)

    def forward(self, x):
        return self.forward_context(x)

class SIGReg(nn.Module):
    def __init__(self, knots: int = 17):
        super().__init__()
        # Paper uses t ∈ [-3, 3]; we integrate on [0, 3] leveraging the symmetry of the CF.
        # t_max=3 is what the authors use — going wider just adds near-zero terms.
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.proj_dim = proj_dim
        self.num_random_features = num_random_features
        # Store a fallback buffer for when global_step is unavailable (e.g. validation).
        # During training, directions are RESAMPLED every forward pass keyed on global_step
        # so that the cumulative directions cover the full embedding space (paper §4.3,
        # Figure 7: resampling M=16 beats fixed M=4096).
        # Column-normalize A: for z ~ N(0,I) and unit-column a: z@a ~ N(0,1). The loss
        # then measures deviation of z from N(0,I), detecting both directional and scale collapse.
        projection = torch.randn(proj_dim, num_random_features, dtype=torch.float32)
        projection = projection / projection.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.register_buffer("projection", projection)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))               
        x_t = (proj @ A).unsqueeze(-1) * self.t      
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class LinearSegProbe(nn.Module):
    def __init__(self, in_channels_list, num_classes=1):
        super().__init__()
        self.probe = nn.Conv2d(in_channels_list[-1], num_classes, kernel_size=1, bias=True)

    def forward(self, features, original_size):
        last = features[-1]
        p = self.probe(last)
        return F.interpolate(p, size=original_size, mode='bilinear', align_corners=False)

# ==========================================
# 4. Training Loop (with Dynamic Masking)
# ==========================================
def train_epoch(model, probe, dataloader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, epoch, augment, config, writer):
    model.train()
    probe.train()
    running_lejepa_loss = 0
    running_probe_loss = 0
    epoch_lejepa_loss = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    for batch_idx, (img, mask, is_syntax, frangi_mask) in pbar:
        img, mask = img.cuda(), mask.cuda()
        is_syntax = is_syntax.cuda().bool()
        original_size = img.shape[2:]

        # Augmentation runs fully on GPU now
        aug_data = augment(img)
        global_crops = aug_data['global_crops']
        local_crops = aug_data['local_crops']
        local_parent_idx = aug_data['local_parent_idx']
        local_boxes = aug_data['local_boxes']

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            features_original, _ = model(img)
            features_probe = [f.detach() for f in features_original]

            pred_probe = probe(features_probe, original_size)

            if is_syntax.any():
                probe_loss = criterion_probe(pred_probe[is_syntax], mask[is_syntax])
            else:
                probe_loss = torch.tensor(0.0, device='cuda', requires_grad=True)

            # --- TARGET ENCODER (Run first to get ground truth) ---
            model_ref = model.module if hasattr(model, 'module') else model
            tgt_proj_local = []
            for l in local_crops:
                _, p_t = model_ref.forward_target(l)                
                tgt_proj_local.append(p_t)

            # --- CONTEXT ENCODER WITH DYNAMIC PATCH MASKING ---
            ctx_proj_global = []
            B = img.size(0)
            
            for g_idx, g in enumerate(global_crops):
                g_masked = g.clone()
                H, W = g_masked.shape[-2:]
                
                # Apply anti-leakage mask
                for l_idx in range(len(local_crops)):
                    parent_indices = local_parent_idx[l_idx] 
                    boxes = local_boxes[l_idx]               
                    
                    for b_idx in range(B):
                        if parent_indices[b_idx] == g_idx:
                            y_norm, x_norm, h_norm, w_norm = boxes[b_idx]
                            py, px = int(y_norm * H), int(x_norm * W)
                            ph, pw = int(h_norm * H), int(w_norm * W)
                            
                            # Mask the pixels (0.0 is mean for [-1, 1] normalized images)
                            g_masked[b_idx, :, py:py+ph, px:px+pw] = 0.0
                            
                # Context model sees the masked (holey) image
                _, p = model(g_masked)                                     
                ctx_proj_global.append(p)

            # --- PREDICTOR & LOSS CALCULATION ---
            ctx_stack = torch.stack(ctx_proj_global, dim=0)         
            batch_idx_ar = torch.arange(B, device=img.device)

            pred_losses = []
            pooled_views_for_sigreg = []
            for g in ctx_proj_global:
                pooled_views_for_sigreg.append(g.mean(dim=1))

            for li in range(len(local_crops)):
                parent = local_parent_idx[li]                       
                boxes = local_boxes[li]                             
                
                ctx_tokens = ctx_stack[parent, batch_idx_ar]        
                pred_local = model_ref.predictor(ctx_tokens, boxes) # Predictor deduces missing patch
                
                # Target is the averaged projection of the local crop
                target_local = tgt_proj_local[li].mean(dim=1).detach()
                pred_losses.append(F.mse_loss(pred_local, target_local))
                pooled_views_for_sigreg.append(target_local)

            pred_loss = torch.stack(pred_losses).mean()
            proj_for_sigreg = torch.stack(pooled_views_for_sigreg, dim=0)  
            sigreg_loss = sigreg(proj_for_sigreg)

            lejepa_loss = sigreg_loss * config['training']['labda'] + pred_loss * (1 - config['training']['labda'])
            total_loss  = lejepa_loss + probe_loss

            # --- Frangi-weighted spatial token pooling ---
            # Each view's Frangi crop (same spatial box as the image crop) is pooled to
            # the token grid (S×S). Token weight: w = 1 + alpha * frangi_score.
            # Vessel tokens (high Frangi response) get (1+alpha)× the gradient weight of
            # background tokens in the invariance loss — biasing the SSL objective toward
            # vessel-consistent representations with zero supervised signal.
            S = int(S_sq ** 0.5)
            all_frangi = global_frangi + local_frangi            # V × (B, 1, H, W)
            all_frangi_t = torch.cat(all_frangi, dim=0).float()  # (V*B, 1, img_size, img_size)
            frangi_grid   = F.adaptive_avg_pool2d(all_frangi_t, (S, S))           # (V*B, 1, S, S)
            frangi_tokens = frangi_grid.view(V, B, S_sq, 1).to(proj_views.dtype)  # (V, B, S*S, 1)
            vessel_alpha  = config['training']['vessel_alpha']
            token_weights = 1.0 + vessel_alpha * frangi_tokens                    # (V, B, S*S, 1)
            pooled_views  = (proj_views * token_weights).sum(dim=2) \
                            / token_weights.sum(dim=2)                            # (V, B, D)

            # Invariance loss: push ALL views toward the stop-gradient global anchor.
            # Stop-grad is critical: without it the trivial minimiser (all views = same
            # constant vector) is always accessible regardless of SIGReg strength.
            # Global views receive NO inv_loss gradient — only SIGReg drives them,
            # ensuring they maintain diverse, Gaussian-distributed representations.
            anchor = pooled_views[:num_global_views].mean(dim=0).detach()  # (B, D)
            # Only LOCAL views are pulled toward the per-image global anchor.
            # Global views have no inv_loss gradient — they are shaped exclusively by
            # SIGReg below, keeping them spread rather than collapsing toward each other.
            inv_loss = (pooled_views[num_global_views:] - anchor.unsqueeze(0)).square().mean()

            # Diagnostics only (not part of the loss)
            global_pooled = pooled_views[:num_global_views]
            local_pooled = pooled_views[num_global_views:]
            global_center = global_pooled.mean(dim=0)
            pooled_flat = pooled_views.reshape(V * B, D)
            var_std = pooled_flat.std(dim=0)  # monitored to detect collapse

            # --- SIGReg ON POOLED VIEWS (matches original paper exactly) ---
            # sigreg(proj) in the paper operates on (V*B, D) pooled embeddings.
            # This enforces cross-image Gaussian diversity in the POOLED space,
            # directly blocking the collapse we observed: Pooled_Std → 0.04 while
            # Token_Std stayed at 1.0 (tokens were fine, pooled reps were not).
            global_step = epoch * len(dataloader) + batch_idx
            sigreg_loss = sigreg(pooled_views.reshape(1, V * B, D), global_step=global_step)

            # Diagnostics
            token_flat = proj_views.reshape(V * B * S_sq, D)
            pooled_std = var_std.mean()
            token_std = token_flat.std(dim=0).mean()
            global_local_cosine = torch.tensor(0.0, device=img.device, dtype=proj_views.dtype)
            if local_pooled.numel() > 0:
                global_local_cosine = F.cosine_similarity(
                    local_pooled, global_center.unsqueeze(0), dim=-1
                ).mean()

            # Original LeJEPA single-lambda objective. No var_loss needed:
            # SIGReg with column-normalized A already penalizes scale collapse.
            lejepa_loss = (
                sigreg_loss * config['training']['labda']
                + inv_loss * (1 - config['training']['labda'])
            )
            total_loss  = lejepa_loss + probe_loss

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(probe.parameters()), max_norm=1.0
        )
        scaler.step(optimiser)
        scaler.update()
        
        model_ref = model.module if hasattr(model, 'module') else model
        model_ref.update_target_encoder()
        scheduler.step()

        running_lejepa_loss += lejepa_loss.item()
        running_probe_loss  += probe_loss.item()
        epoch_lejepa_loss   += lejepa_loss.item()

        pbar.set_postfix({
            'LeJEPA': running_lejepa_loss / (batch_idx + 1),
            'Probe':  running_probe_loss  / (batch_idx + 1),
        })

        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Train/LeJepa_Loss", lejepa_loss.item(), global_step)
        if is_syntax.any():
            writer.add_scalar("Train/Probe_Loss", probe_loss.item(), global_step)
        writer.add_scalar("Train/SIGReg",   sigreg_loss.item(), global_step)
        writer.add_scalar("Train/Pred_Loss", pred_loss.item(),    global_step)

    return epoch_lejepa_loss / len(dataloader)

def _log_saliency(model, imgs, masks, epoch, writer, num_vis=4):
    model.eval()
    m = model.module if hasattr(model, 'module') else model
    num_vis = min(num_vis, imgs.size(0))

    imgs_in = imgs[:num_vis].detach().float()

    with torch.enable_grad():
        feats, _ = m(imgs_in)
        last = feats[-1].float()         
        last.retain_grad()
        last.mean().backward()           

    with torch.no_grad():
        grad  = last.grad                                              
        alpha = grad.mean(dim=(2, 3), keepdim=True)                    
        cam   = F.relu((alpha * last).sum(dim=1, keepdim=True))        
        cam   = F.interpolate(cam, size=imgs.shape[2:],
                              mode='bilinear', align_corners=False)    
        cam   = cam / (cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)  

    img_vis  = (imgs[:num_vis] * 0.5 + 0.5).float().cpu().clamp(0, 1)  
    mask_vis = masks[:num_vis].float().cpu()                              
    cam_np   = cam.cpu().numpy()                                          

    grid_items = []
    for i in range(num_vis):
        inp_rgb = img_vis[i].repeat(3, 1, 1)           
        jet_np  = cm.jet(cam_np[i, 0])[:, :, :3]       
        jet_t   = torch.from_numpy(jet_np).float().permute(2, 0, 1)  
        blend   = 0.55 * inp_rgb + 0.45 * jet_t        
        msk_rgb = mask_vis[i].repeat(3, 1, 1)          
        grid_items += [inp_rgb, blend, msk_rgb]

    grid = torchvision.utils.make_grid(grid_items, nrow=3, padding=2, normalize=False)
    writer.add_image("Val/GradCAM", grid, epoch)

    return epoch_lejepa_loss / len(dataloader)

def _log_saliency(model, imgs, masks, epoch, writer, num_vis=4, probe=None, save_dir=None):
    model.eval()
    if probe is not None:
        probe.eval()
    m = model.module if hasattr(model, 'module') else model
    p = probe.module if (probe is not None and hasattr(probe, 'module')) else probe
    num_vis = min(num_vis, imgs.size(0))

    img_vis  = (imgs[:num_vis] * 0.5 + 0.5).float().cpu().clamp(0, 1)
    mask_vis = masks[:num_vis].float().cpu()

    # --- SmoothGrad probe-guided saliency (sharper than raw input gradients) ---
    smooth_samples = 6
    noise_std = 0.08
    sal_accum = torch.zeros_like(imgs[:num_vis].float())

    with torch.enable_grad():
        for _ in range(smooth_samples):
            noisy = imgs[:num_vis].float() + torch.randn_like(imgs[:num_vis].float()) * noise_std
            noisy = noisy.requires_grad_(True)

            m.zero_grad(set_to_none=True)
            if p is not None:
                p.zero_grad(set_to_none=True)

            feats, _ = m(noisy)
            if p is not None:
                prob = p(feats, noisy.shape[2:]).sigmoid()
                flat_prob = prob.flatten(1)
                thresh = torch.quantile(flat_prob, 0.90, dim=1, keepdim=True).view(-1, 1, 1, 1)
                focus_mask = (prob >= thresh).float()
                score = (prob * focus_mask).sum() / focus_mask.sum().clamp_min(1.0)
            else:
                fallback = feats[-1].float().mean(dim=1, keepdim=True)
                fallback = F.interpolate(fallback, size=noisy.shape[2:], mode='bilinear', align_corners=False)
                score = fallback.mean()

            score.backward()
            sal_accum = sal_accum + noisy.grad.abs()

    with torch.no_grad():
        sal = sal_accum / smooth_samples
        high = torch.quantile(sal.flatten(1), 0.995, dim=1, keepdim=True).view(-1, 1, 1, 1)
        sal = (sal / high.clamp_min(1e-8)).clamp(0, 1)
        sal = sal.pow(0.65)
        sal_np = sal.cpu().numpy()

    grid_items = []
    for i in range(num_vis):
        inp_rgb = img_vis[i].repeat(3, 1, 1)
        jet_np  = cm.jet(sal_np[i, 0])[:, :, :3]
        jet_t   = torch.from_numpy(jet_np).float().permute(2, 0, 1)
        blend   = 0.55 * inp_rgb + 0.45 * jet_t
        msk_rgb = mask_vis[i].repeat(3, 1, 1)
        grid_items += [inp_rgb, blend, msk_rgb]

    grid = torchvision.utils.make_grid(grid_items, nrow=3, padding=2, normalize=False)
    writer.add_image("Val/Saliency", grid, epoch)
    
    # Save saliency grid to file if save_dir is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        saliency_path = os.path.join(save_dir, f"epoch_{epoch+1:04d}_saliency.png")
        torchvision.utils.save_image(grid, saliency_path)

def validate_epoch(model, probe, dataloader, f1_metric, epoch, writer, perepoch_dir=None):
    model.eval()
    probe.eval()
    val_f1 = 0.0
    first_img, first_mask = None, None

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation {epoch+1}")
        for batch_idx, (img, mask) in pbar:
            img, mask = img.cuda(), mask.cuda()
            original_size = img.shape[2:]
            
            features_maps, _ = model(img)
            pred_probe = probe(features_maps, original_size)
            
            val_f1 += f1_metric(pred_probe.sigmoid(), mask.int()).item()
            pbar.set_postfix({'val_f1': val_f1 / (batch_idx + 1)})
            
            if batch_idx == 0:
                first_img  = img.clone()
                first_mask = mask.clone()

                img_vis = img * 0.5 + 0.5
                num_samples = min(4, img_vis.size(0))
                grid_images = []
                preds_vis = pred_probe.sigmoid()
                for i in range(num_samples):
                    grid_images.append(img_vis[i].cpu())
                    grid_images.append(preds_vis[i].float().cpu())
                    grid_images.append(mask[i].float().cpu())
                grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
                writer.add_image("Val/Predictions", grid, epoch)
        
        avg_f1 = val_f1 / len(dataloader)
        writer.add_scalar("Val/F1", avg_f1, epoch)
        print(f"Validation F1: {avg_f1:.4f}")

    if first_img is not None:
        _log_saliency(model, first_img, first_mask, epoch, writer)

    return avg_f1

def reload_checkpoint(checkpoint_path, model, probe, optimiser, scheduler, scaler, num_gpus):
    if os.path.isfile(checkpoint_path):
        print(f"=> Se încarcă checkpoint-ul '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_lejepa_loss = checkpoint.get('best_lejepa_loss', float('inf'))
        
        if num_gpus > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            probe.module.load_state_dict(checkpoint['probe_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            probe.load_state_dict(checkpoint['probe_state_dict'])
            
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"=> Reluare cu succes de la epoca {start_epoch} (Best F1: {best_f1:.4f}, Best LeJEPA: {best_lejepa_loss:.4f})")
        return start_epoch, best_f1, best_lejepa_loss
    else:
        print(f"=> Niciun checkpoint găsit la '{checkpoint_path}'. Antrenarea începe de la zero.")
        return 0, 0.0, float('inf')

    if first_img is not None:
        _log_saliency(model, first_img, first_mask, epoch, writer, probe=probe, save_dir=perepoch_dir)

    return avg_f1

def reload_checkpoint(checkpoint_path, model, probe, optimiser, scheduler, scaler, num_gpus):
    if os.path.isfile(checkpoint_path):
        print(f"=> Se încarcă checkpoint-ul '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_lejepa_loss = checkpoint.get('best_lejepa_loss', float('inf'))
        
        if num_gpus > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            probe.module.load_state_dict(checkpoint['probe_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            probe.load_state_dict(checkpoint['probe_state_dict'])
        # SIGReg projection is a non-learnable buffer — never load from checkpoint
        # so a corrected initialization is always used.
            
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"=> Reluare cu succes de la epoca {start_epoch} (Best F1: {best_f1:.4f}, Best LeJEPA: {best_lejepa_loss:.4f})")
        return start_epoch, best_f1, best_lejepa_loss
    else:
        print(f"=> Niciun checkpoint găsit la '{checkpoint_path}'. Antrenarea începe de la zero.")
        return 0, 0.0, float('inf')

def trainScript(model, probe, train_loader, val_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, augment, config, writer, scaler):
    checkpoint_dir = config['logging']['checkpoint_dir'].format(experiment_name=config['experiment_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    pacience = 300
    epochs_no_improve = 0

    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
    done_file_path = os.path.join(checkpoint_dir, "DONE")
    
    start_epoch, best_f1, best_lejepa_loss = reload_checkpoint(last_model_path, model, probe, optimiser, scheduler, scaler, num_gpus)

    for epoch in range(start_epoch, config['training']['epochs']):
        avg_lejepa_loss = train_epoch(model, probe, train_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, epoch, augment, config, writer)
        val_f1 = validate_epoch(model, probe, val_loader, f1_metric, epoch, writer)
        writer.add_scalar("Train/Epoch_LeJepa_Loss", avg_lejepa_loss, epoch)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
            'probe_state_dict': probe.module.state_dict() if num_gpus > 1 else probe.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1,
            'best_lejepa_loss': best_lejepa_loss,
        }
        torch.save(checkpoint, last_model_path)
        
        backbone_to_save = model.module.context_backbone if num_gpus > 1 else model.context_backbone

        if avg_lejepa_loss < best_lejepa_loss:
            best_lejepa_loss = avg_lejepa_loss
            epochs_no_improve = 0
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
            torch.save(backbone_to_save.state_dict(), os.path.join(checkpoint_dir, "best_backbone.pth"))
            print(f"--- Best backbone saved at epoch {epoch+1} with LeJEPA loss: {best_lejepa_loss:.4f} (val F1: {val_f1:.4f}) ---")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs (lejepa_loss={avg_lejepa_loss:.4f}, best={best_lejepa_loss:.4f}).")

        save_every = config['training'].get('save_every', 10)
        if (epoch + 1) % save_every == 0:
            snap_path = os.path.join(checkpoint_dir, f"backbone_ep{epoch+1}.pth")
            torch.save(backbone_to_save.state_dict(), snap_path)
            print(f"  [Snapshot] Backbone saved at epoch {epoch+1} → {snap_path}")
            
        if epochs_no_improve >= pacience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    with open(done_file_path, "w") as f:
        f.write("Training completed successfully.")
    print(f"\n✅ Antrenament complet pentru {config['model']['encoder_name']}! Fișierul DONE a fost creat.")

if __name__ == "__main__":
    encoders_to_train = [ 'swinv2_tiny_window8_256']

    for encoder in encoders_to_train:
        experiment_name = f"{encoder}_lejepa_SIGREG"
        checkpoint_dir = f"checkpoints/{experiment_name}"
        
        if os.path.exists(os.path.join(checkpoint_dir, "DONE")):
            print(f"\n{'='*60}\n⏭️  Modelul {encoder} a fost deja antrenat (Găsit fișier DONE). Trecem la următorul...\n{'='*60}")
            continue
            
        print(f"\n{'='*60}\n🚀 Începe antrenamentul pentru: {encoder}\n{'='*60}")

        config = {
            'experiment_name': experiment_name,
            'logging': {
                'log_dir': f'runs/{experiment_name}',
                'checkpoint_dir': checkpoint_dir
            },
            'training': {
                'img_size': 256,
                'batch_size': 20,
                'epochs': 100,
                'lr_probe': 1e-5,
                'lr_model': 1e-4,
                'weight_decay': 5e-2,
                'num_global_crops': 2,
                'num_local_crops': 4,
                'global_scale': (0.8, 1.0),
                'local_scale': (0.10, 0.35),
                'global_vessel_threshold': 0.015,
                'local_vessel_threshold': 0.02,
                'local_background_threshold': 0.01,
                'max_global_retries': 12,
                'max_local_retries': 20,
                'labda': 0.05,
                'warmup_epochs': 20,
                'save_every': 10,
            },
            'model': {
                'proj_dim': 128,
                'spatial_tokens': 4,
            }
        }
        
        model = LeJepaModel(
            encoder_name=config['model']['encoder_name'],
            proj_dim=config['model']['proj_dim'],
            spatial_tokens=config['model']['spatial_tokens'],
        ).cuda()
        sigreg = SIGReg().cuda()
        
        augment = augmentariLeJepa(
            img_size=config['training']['img_size'],
            num_global_crops=config['training']['num_global_crops'],
            num_local_crops=config['training']['num_local_crops'],
            global_scale=tuple(config['training']['global_scale']),
            local_scale=tuple(config['training']['local_scale']),
            global_vessel_threshold=config['training']['global_vessel_threshold'],
            local_vessel_threshold=config['training']['local_vessel_threshold'],
            local_background_threshold=config['training']['local_background_threshold'],
            max_global_retries=config['training']['max_global_retries'],
            max_local_retries=config['training']['max_local_retries'],
        ).cuda() # Mutăm direct pe CUDA

        dummy_input = torch.randn(1, 1, config['training']['img_size'], config['training']['img_size']).cuda()
        with torch.no_grad():
            feats, _ = model(dummy_input)
        encoder_channels = [f.shape[1] for f in feats]

        probe = LinearSegProbe(in_channels_list=encoder_channels, num_classes=1).cuda()
        
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = nn.DataParallel(model)
            probe = nn.DataParallel(probe)

        lr1 = {"params": probe.parameters(), "lr": config['training']['lr_probe'], "weight_decay": config['training']['weight_decay']}
        lr2 = {"params": model.parameters(), "lr": config['training']['lr_model'], "weight_decay": config['training']['weight_decay']}
        opt = torch.optim.AdamW([lr1, lr2])
        
        total_iters_per_epoch = len(train_loader)
        warmup_iters = config['training']['warmup_epochs'] * total_iters_per_epoch
        total_iters = config['training']['epochs'] * total_iters_per_epoch
        
        scheduler1 = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
        scheduler2 = CosineAnnealingLR(opt, T_max=total_iters - warmup_iters, eta_min=1e-6)
        scheduler = SequentialLR(opt, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
        
        _tversky_probe = TverskyLoss(mode='binary', beta=0.7, gamma=0.75, log_loss=False)
        _bce_probe = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).cuda())
        def criterion_probe(pred, target):
            return _tversky_probe(pred, target) + _bce_probe(pred, target)
        f1_metric = BinaryF1Score().cuda()
        
        trainScript(
            model=model,
            probe=probe,
            train_loader=train_loader,
            val_loader=val_loader,
            optimiser=opt,
            scheduler=scheduler,
            sigreg=sigreg,
            criterion_probe=criterion_probe,
            f1_metric=f1_metric,
            augment=augment,
            config=config,
            writer=writer
        )
        
        writer.close()
        del model, probe, opt, scheduler, train_loader, val_loader, sigreg
        torch.cuda.empty_cache()
        gc.collect()

    print("\n🎉 Toate modelele din listă au fost procesate!")