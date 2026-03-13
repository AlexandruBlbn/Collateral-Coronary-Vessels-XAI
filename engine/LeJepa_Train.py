import os
import sys
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
from torchvision.ops import MLP
from monai.losses import DiceCELoss
from segmentation_models_pytorch.losses import TverskyLoss, SoftBCEWithLogitsLoss
from torchmetrics.classification import BinaryF1Score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import gc
import matplotlib.cm as cm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper
from utils.helpers import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler()

def loader(img_size, batch_size, split='train', mode='train'):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    ds_mode = 'pretrain' if mode == 'lejepa' else 'syntax'
    base = ArcadeDataset(split=split, mode=ds_mode, transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
    ds = TransformsWrapper(base, input_size=img_size, mode=mode)
    g = torch.Generator()
    g.manual_seed(42)
    
    return DataLoader(
        ds,
        batch_size=batch_size, 
        shuffle=(split=='train'),
        num_workers=4, 
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

def configCreate(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)

class augmentariLeJepa(nn.Module):
    """
    Fully label-free multi-crop augmentation.

    Strategic change from original:
      OLD local scale: (0.05, 0.6)  — 5% minimum ≈ ~57px crop, vessels missed ~90% of the time
      NEW local scale: (0.4,  0.8)  — 40% minimum ≈ ~102px crop, always captures macroscopic
                                       cardiac structures and a large fraction of the vessel tree

    With 40-80% crops every local view substantially overlaps with the global view.
    The invariance objective then forces the model to learn WHAT is structurally consistent
    across those partially overlapping views, which in angiography is the vessel tree —
    not the background or catheter (which appear at different positions across crops).
    No labels, no pseudolabels required.
    """
    def __init__(self, img_size=224):
        super().__init__()
        self.img_size = img_size
        # BorderJitter: randomly crops 5-15% from each side before the main crop.
        # Angiography images have a circular FOV: the corners are always black.
        # Without this, BOTH global views often contain the same border transition
        # at the same spatial position → invariance loss uses it as a free shortcut.
        # By randomly varying which portion of the border is included in each view,
        # the border becomes INCONSISTENT across views and cannot be exploited.
        self.BorderJitter = transforms.Compose([
            transforms.RandomCrop(
                int(img_size * 0.88),   # removes ~12% worst-case border per side
                pad_if_needed=True,
                fill=0
            ),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        ])
        self.Global_Crops = transforms.RandomResizedCrop(
            img_size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.Local_Crops = transforms.RandomResizedCrop(
            img_size, scale=(0.4, 0.8), interpolation=transforms.InterpolationMode.BICUBIC
        )
        # ElasticTransform: locally warps each crop independently.
        # Catheters/guide-wires are straight, high-contrast lines — consistent across
        # un-augmented crops and therefore easy invariance shortcuts. After elastic
        # deformation they appear differently curved in different views → inconsistent
        # → unusable as an invariance target. Vessel structure (tortuous, distributed)
        # survives mild elastic deformation better than thin straight instruments.
        self.elastic = transforms.ElasticTransform(alpha=60.0, sigma=6.0)

    def _apply_stochastic_aug(self, crop: torch.Tensor) -> torch.Tensor:
        # Per-crop random flips: the catheter/instrument always enters from one edge
        # (typically top-center) in angiography. Without per-crop flips, all 5 views
        # of an image share the same catheter entry direction → stable spatial shortcut.
        # Flipping independently per-crop makes position inconsistent across views.
        if torch.rand(1).item() < 0.5:
            crop = TF.hflip(crop)
        if torch.rand(1).item() < 0.5:
            crop = TF.vflip(crop)
        # Inversion REMOVED: at p=0.5 most angiography frames (which are predominantly
        # dark) become nearly white after negation. The backbone then has to reconcile
        # a normal-contrast view with a photographic-negative view — the only invariant
        # features are structural positions, but the vessel/background contrast polarity
        # (the primary vessel cue) is completely destroyed. Both per-crop flips and
        # BorderJitter already make brightness-polarity shortcuts inconsistent across
        # views without inverting the diagnostic content.
        if torch.rand(1).item() < 0.3:
            crop = self.elastic(crop)
        return crop

    @staticmethod
    def _fill_fov_border(img: torch.Tensor) -> torch.Tensor:
        """
        Replace circular-FOV black border pixels AND bright edge bands with mean intensity.

        TWO shortcuts are handled:

        1. DARK corners (< -0.85): circular FOV mask outside the sensor circle.
           Both augmented views share black corners at the same position → free shortcut.

        2. BRIGHT top/bottom bands (> 0.7 in top/bottom 8% of rows): vendor-imprinted
           frame borders, imaging metadata strips, or reconstruction artefacts that appear
           as a brighter horizontal band at the image edge. GradCAM at epoch 17 shows
           activations locked on the top edge in images where the catheter entry and
           coronary origin are also at the top — the model conflates the bright band with
           the real structure. Filling it with the mean removes the per-image consistent
           spatial anchor so per-crop flips can fully decorrelate this region.
        """
        img = img.clone()
        fill_val = img.mean()

        # 1. Dark FOV corners
        border_mask = img < -0.85
        if border_mask.float().mean() >= 0.01:
            img[border_mask] = fill_val

        # 2. Bright top/bottom edge bands
        H = img.shape[-2]
        fringe = max(1, int(H * 0.08))          # top and bottom 8% of rows
        top_band    = img[..., :fringe, :]
        bottom_band = img[..., -fringe:, :]
        if (top_band > 0.7).float().mean() > 0.35:      # >35% pixels very bright → vendor strip
            img[..., :fringe, :] = fill_val
        if (bottom_band > 0.7).float().mean() > 0.35:
            img[..., -fringe:, :] = fill_val

        return img

    def __call__(self, img: torch.Tensor):
        # 1. Remove circular FOV border (shared across all views — eliminates that shortcut).
        img = self._fill_fov_border(img)
        crops = []
        # 2. Each crop gets its own independent stochastic augmentation AFTER cropping,
        #    so the same shortcut feature (catheter angle, brightness) appears differently
        #    across views and cannot be used to minimise the invariance loss.
        for _ in range(2):
            c = self.Global_Crops(self.BorderJitter(img))
            crops.append(self._apply_stochastic_aug(c))
        for _ in range(3):
            c = self.Local_Crops(self.BorderJitter(img))
            crops.append(self._apply_stochastic_aug(c))
        return crops

class LeJepaModel(nn.Module):
    """
    Dense spatial projection instead of global average pool.

    WHY: Global average pool collapses the (B, C, h, w) feature map to a single (B, C)
    vector. For segmentation, two crops may contain very different spatial arrangements
    of vessels; their global-average embeddings will differ even if local vessel features
    are consistent. This makes the invariance loss push the backbone toward global
    image-level descriptors ("this looks like a cardiac image") rather than local
    spatial descriptors ("here there is a vessel branch").

    FIX: Use AdaptiveAvgPool2d(spatial_tokens) to produce S×S spatial tokens per image
    (default S=4, giving 16 tokens). Each token represents a spatial region. Invariance
    is computed per-token across views, forcing the backbone to learn what is spatially
    consistent within corresponding regions — which is vessel structure.
    """
    def __init__(self, encoder_name='swinv2_tiny_window8_256', proj_dim=128, spatial_tokens=4):
        super().__init__()
        self.backbone = timm.create_model(
            encoder_name,
            pretrained=False,
            in_chans=1,
            features_only=True,
        )
        self.channels_list = self.backbone.feature_info.channels()
        self.spatial_tokens = spatial_tokens
        self.pool = nn.AdaptiveAvgPool2d(spatial_tokens)          # (B, C, S, S)
        self.proj = MLP(self.channels_list[-1], [512, proj_dim], norm_layer=nn.LayerNorm)

    def forward(self, x):
        features = list(self.backbone(x))
        for i in range(len(features)):
            # Fix channel order for Transformer models (SwinV2 outputs B,H,W,C)
            if features[i].dim() == 4 and features[i].shape[-1] == self.channels_list[i]:
                features[i] = features[i].permute(0, 3, 1, 2).contiguous()
        last_map = features[-1]                                   # (B, C, h, w)
        sp = self.pool(last_map)                                   # (B, C, S, S)
        B, C, S, _ = sp.shape
        tokens = sp.flatten(2).permute(0, 2, 1)                    # (B, S*S, C)
        proj_out = self.proj(
            tokens.reshape(B * S * S, C)
        ).view(B, S * S, -1)                                       # (B, S*S, proj_dim)
        return features, proj_out

class SIGReg(nn.Module):
    """
    Sliced Independence Gaussian Regularizer.

    Tests whether the distribution of projected embeddings matches a Gaussian
    characteristic function via random projections (slicing). Acts as a light
    distributional regulariser: if embeddings collapse to a single point or a
    degenerate manifold, SIGReg detects the non-Gaussian distribution and penalises.

    Used alongside the invariance MSE loss:
      lejepa_loss = labda * sigreg_loss + (1 − labda) * inv_loss

    labda=0.05 keeps SIGReg as a stabiliser — it prevents collapse without
    dominating the invariance signal that drives feature learning.
    """
    def __init__(self, knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: (V, B, proj_dim) — spatially-averaged projections across views
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))               # random unit projection directions
        x_t = (proj @ A).unsqueeze(-1) * self.t      # (V, B, 256, knots)
        # mean(-3) averages over the batch (dim B) to get empirical char. function
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class LinearSegProbe(nn.Module):
    """
    Last-scale linear probe: single 1×1 conv on the final backbone feature map only.

    WHY last-scale only:
    Multi-scale probes average predictions from all stages including shallow ones.
    Shallow features respond to local contrast edges regardless of backbone quality —
    vessel walls are simply high-contrast, so even a badly-trained backbone produces
    edge responses in layer1/layer2 that a 1×1 conv can exploit to reconstruct a
    rough vessel shape. The segmentation then looks deceptively vessel-like while
    GradCAM shows the backbone actually focuses on borders/catheters.

    By probing ONLY the last feature map, probe F1 measures the same representation
    that GradCAM targets. If GradCAM shows corner activations, probe F1 will be near 0.
    If GradCAM shifts to vessels, probe F1 will rise. Both signals stay consistent,
    making probe F1 in TensorBoard a reliable backbone quality diagnostic.
    """
    def __init__(self, in_channels_list, num_classes=1):
        super().__init__()
        # Only the last (semantically richest) feature map
        self.probe = nn.Conv2d(in_channels_list[-1], num_classes, kernel_size=1, bias=True)

    def forward(self, features, original_size):
        last = features[-1]
        p = self.probe(last)
        return F.interpolate(p, size=original_size, mode='bilinear', align_corners=False)

def train_epoch(model, probe, dataloader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, epoch, augment, config, writer):
    model.train()
    probe.train()
    running_lejepa_loss = 0
    running_probe_loss = 0
    epoch_lejepa_loss = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    for batch_idx, (img, mask, is_syntax) in pbar:
        img, mask = img.cuda(), mask.cuda()
        is_syntax = is_syntax.cuda().bool()
        original_size = img.shape[2:]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            features_original, _ = model(img)
            features_probe = [f.detach() for f in features_original]

            pred_probe = probe(features_probe, original_size)

            if is_syntax.any():
                probe_loss = criterion_probe(pred_probe[is_syntax], mask[is_syntax])
            else:
                probe_loss = torch.tensor(0.0, device='cuda', requires_grad=True)

            crops = augment(img)                                   # List[5] of (B, C, H, W)
            global_crops = torch.cat(crops[:2], dim=0)             # (2B, C, H, W)
            local_crops  = torch.cat(crops[2:], dim=0)             # (3B, C, H, W)

            _, p_proj_global = model(global_crops)                 # (2B, S*S, proj_dim)
            _, p_proj_local  = model(local_crops)                  # (3B, S*S, proj_dim)
            p_proj_all = torch.cat([p_proj_global, p_proj_local], dim=0)  # (5B, S*S, proj_dim)

            V          = len(crops)                                # 5
            current_bs = img.size(0)
            S_sq       = p_proj_all.shape[1]                       # S*S spatial tokens

            proj_views = p_proj_all.view(V, current_bs, S_sq, -1) # (V, B, S*S, proj_dim)
            # Stop-gradient on the mean: BYOL/SimSiam style asymmetric invariance loss.
            # Prevents the feedback loop that causes invariance loss spikes.
            proj_mean  = proj_views.mean(dim=0).detach()           # (B, S*S, proj_dim)
            inv_loss   = (proj_mean - proj_views).square().mean()

            # SIGReg operates on the spatially-averaged projection (avoid non-i.i.d.
            # statistics from spatial tokens by collapsing S*S first).
            proj_for_sigreg = proj_views.mean(dim=2)               # (V, B, proj_dim)
            sigreg_loss = sigreg(proj_for_sigreg)

            lejepa_loss = sigreg_loss * config['training']['labda'] + inv_loss * (1 - config['training']['labda'])
            total_loss  = lejepa_loss + probe_loss

        optimiser.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(probe.parameters()), max_norm=1.0
        )
        scaler.step(optimiser)
        scaler.update()
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
        writer.add_scalar("Train/Inv_Loss", inv_loss.item(),    global_step)

    return epoch_lejepa_loss / len(dataloader)


def _log_saliency(model, imgs, masks, epoch, writer, num_vis=4):
    """
    GradCAM on the last backbone feature map — label-free.

    WHY GradCAM instead of input-gradient saliency:
      Input gradients backprop to the pixel level and pick up high-frequency
      noise (especially bad with transformer backbones whose patch structure
      creates a visible grid artefact). GradCAM stays in feature-map space:
        1. Forward to get last feature map A  (B, C, h, w)
        2. Scalar target = mean(A)  → backward to get dTarget/dA
        3. Channel weights α_k = gap(grad_k)
        4. CAM = ReLU(Σ_k  α_k · A_k)  → upsample to input resolution
      Result: smooth spatial heatmap showing WHICH REGIONS of the image most
      activate the backbone, without pixel-level noise.

    Interpretation guide (Val/GradCAM in TensorBoard):
      GOOD — bright CAM regions overlap the vessel tree in the GT mask column.
      BAD  — diffuse /uniform CAM, or concentrated on catheter / image border.

    Grid (nrow=3): [input image | GradCAM heatmap | GT vessel mask]
    """
    model.eval()
    m = model.module if hasattr(model, 'module') else model
    num_vis = min(num_vis, imgs.size(0))

    imgs_in = imgs[:num_vis].detach().float()

    with torch.enable_grad():
        feats, _ = m(imgs_in)
        last = feats[-1].float()         # (B, C, h, w) — forward already permutes SwinV2
        last.retain_grad()
        last.mean().backward()           # uniform scalar target: gradient shows which
                                         # channels / locations drive overall activation

    with torch.no_grad():
        grad  = last.grad                                              # (B, C, h, w)
        alpha = grad.mean(dim=(2, 3), keepdim=True)                    # (B, C, 1, 1)
        cam   = F.relu((alpha * last).sum(dim=1, keepdim=True))        # (B, 1, h, w)
        cam   = F.interpolate(cam, size=imgs.shape[2:],
                              mode='bilinear', align_corners=False)    # (B, 1, H, W)
        cam   = cam / (cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)  # per-sample norm

    img_vis  = (imgs[:num_vis] * 0.5 + 0.5).float().cpu().clamp(0, 1)  # (N,1,H,W)
    mask_vis = masks[:num_vis].float().cpu()                              # (N,1,H,W)
    cam_np   = cam.cpu().numpy()                                          # (N,1,H,W)

    grid_items = []
    for i in range(num_vis):
        # --- input: replicate 1-ch grey to 3-ch RGB ---
        inp_rgb = img_vis[i].repeat(3, 1, 1)           # (3,H,W)

        # --- GradCAM: jet colormap blended on top of the image ---
        jet_np  = cm.jet(cam_np[i, 0])[:, :, :3]       # (H,W,3)  float32 in [0,1]
        jet_t   = torch.from_numpy(jet_np).float().permute(2, 0, 1)  # (3,H,W)
        blend   = 0.55 * inp_rgb + 0.45 * jet_t        # semi-transparent overlay

        # --- GT mask: white vessels on black, replicated to 3-ch ---
        msk_rgb = mask_vis[i].repeat(3, 1, 1)          # (3,H,W)

        grid_items += [inp_rgb, blend, msk_rgb]

    grid = torchvision.utils.make_grid(grid_items, nrow=3, padding=2, normalize=False)
    writer.add_image("Val/GradCAM", grid, epoch)


def validate_epoch(model, probe, dataloader, f1_metric, epoch, writer):
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

    # Saliency needs gradients — must be outside torch.no_grad()
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

def trainScript(model, probe, train_loader, val_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, augment, config, writer):
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
        
        backbone_to_save = model.module.backbone if num_gpus > 1 else model.backbone

        # Checkpoint selection uses LeJEPA loss, not probe F1.
        # Probe F1 is 0% throughout (probe never sees labeled data in pretrain mode
        # and has no nonlinearity to compensate), so it cannot guide backbone selection.
        # LeJEPA loss measures whether the backbone produces consistent spatial
        # representations across augmented views — the actual pretraining objective.
        if avg_lejepa_loss < best_lejepa_loss:
            best_lejepa_loss = avg_lejepa_loss
            epochs_no_improve = 0
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
            torch.save(backbone_to_save.state_dict(), os.path.join(checkpoint_dir, "best_backbone.pth"))
            print(f"--- Best backbone saved at epoch {epoch+1} with LeJEPA loss: {best_lejepa_loss:.4f} (val F1: {val_f1:.4f}) ---")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs (lejepa_loss={avg_lejepa_loss:.4f}, best={best_lejepa_loss:.4f}).")

        # Periodic backbone snapshot every 10 epochs, independently of probe F1.
        # Use these for fine-tuning evaluation — the linear probe F1 is a weak signal;
        # a snapshot at epoch 50 may have better backbone features than the "best" by F1.
        save_every = config['training'].get('save_every', 10)
        if (epoch + 1) % save_every == 0:
            snap_path = os.path.join(checkpoint_dir, f"backbone_ep{epoch+1}.pth")
            torch.save(backbone_to_save.state_dict(), snap_path)
            print(f"  [Snapshot] Backbone saved at epoch {epoch+1} → {snap_path}")
            
        if epochs_no_improve >= pacience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Marcăm antrenamentul ca fiind complet la ieșirea din buclă
    with open(done_file_path, "w") as f:
        f.write("Training completed successfully.")
    print(f"\n✅ Antrenament complet pentru {config['model']['encoder_name']}! Fișierul DONE a fost creat.")

if __name__ == "__main__":
    # Lista cu modelele pe care vrei să le antrenezi succesiv
    encoders_to_train = ['resnet50', 'swinv2_tiny_window8_256', 'convnextv2_tiny']

    for encoder in encoders_to_train:
        experiment_name = f"{encoder}_lejepa_SIGREG"
        checkpoint_dir = f"checkpoints/{experiment_name}"
        
        # 1. Verificăm dacă modelul a fost deja antrenat complet
        if os.path.exists(os.path.join(checkpoint_dir, "DONE")):
            print(f"\n{'='*60}\n⏭️  Modelul {encoder} a fost deja antrenat (Găsit fișier DONE). Trecem la următorul...\n{'='*60}")
            continue
            
        print(f"\n{'='*60}\n🚀 Începe antrenamentul pentru: {encoder}\n{'='*60}")

        # Configurația dinamică pentru modelul curent
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
                # labda=0.05: SIGReg is a light distributional regulariser.
                # Reduced from 0.2 → 0.05: SIGReg should stabilise, not dominate.
                # High labda caused the invariance loss spikes seen in earlier runs.
                'labda': 0.05,
                'warmup_epochs': 20,
                # Backbone snapshots every 10 epochs regardless of probe F1,
                # because the linear probe F1 is a weak checkpoint criterion.
                'save_every': 10,
            },
            'model': {
                'encoder_name': encoder,
                'proj_dim': 64,
                # 4×4 = 16 spatial tokens from the last feature map (8×8 → 4×4).
                # Forces invariance to be learned per spatial region, not globally.
                'spatial_tokens': 4,
            }
        }
        
        # 2. Inițializare Dataloaders și Writer
        writer = SummaryWriter(log_dir=config['logging']['log_dir'])
        configCreate(os.path.join(config['logging']['log_dir'], 'config.yaml'), config)
        
        train_loader = loader(config['training']['img_size'], config['training']['batch_size'], split='train', mode='lejepa')
        val_loader = loader(config['training']['img_size'], config['training']['batch_size'], split='validation', mode='validation')
        
        # 3. Crearea Modelelor
        model = LeJepaModel(
            encoder_name=config['model']['encoder_name'],
            proj_dim=config['model']['proj_dim'],
            spatial_tokens=config['model']['spatial_tokens'],
        ).cuda()
        sigreg = SIGReg().cuda()
        augment = augmentariLeJepa(img_size=config['training']['img_size'])

        dummy_input = torch.randn(1, 1, config['training']['img_size'], config['training']['img_size']).cuda()
        with torch.no_grad():
            feats, _ = model(dummy_input)
        encoder_channels = [f.shape[1] for f in feats]

        probe = LinearSegProbe(in_channels_list=encoder_channels, num_classes=1).cuda()
        
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = nn.DataParallel(model)
            probe = nn.DataParallel(probe)

        # 4. Optimizatori și Schedulere
        lr1 = {"params": probe.parameters(), "lr": config['training']['lr_probe'], "weight_decay": config['training']['weight_decay']}
        lr2 = {"params": model.parameters(), "lr": config['training']['lr_model'], "weight_decay": config['training']['weight_decay']}
        opt = torch.optim.AdamW([lr1, lr2])
        
        total_iters_per_epoch = len(train_loader)
        warmup_iters = config['training']['warmup_epochs'] * total_iters_per_epoch
        total_iters = config['training']['epochs'] * total_iters_per_epoch
        
        scheduler1 = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
        scheduler2 = CosineAnnealingLR(opt, T_max=total_iters - warmup_iters, eta_min=1e-6)
        scheduler = SequentialLR(opt, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
        
        # Vessel pixels are ~3-7% of the image. TverskyLoss(beta=0.7) penalises
        # false negatives more heavily; pos_weight=10 in BCE compensates for the
        # background-dominated pixel distribution in the CE gradient.
        _tversky_probe = TverskyLoss(mode='binary', beta=0.7, gamma=0.75, log_loss=False)
        _bce_probe = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).cuda())
        def criterion_probe(pred, target):
            return _tversky_probe(pred, target) + _bce_probe(pred, target)
        f1_metric = BinaryF1Score().cuda()
        
        # 5. Pornire Script
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
        
        # 6. Clean-up după fiecare model pentru a elibera memoria GPU pentru următorul
        writer.close()
        del model, probe, opt, scheduler, train_loader, val_loader, sigreg
        torch.cuda.empty_cache()
        gc.collect()

    print("\n🎉 Toate modelele din listă au fost procesate!")