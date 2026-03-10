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
import torchvision
import timm
from torchvision.ops import MLP
from monai.losses import DiceCELoss
from torchmetrics.classification import BinaryF1Score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import gc

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
        self.Global_Crops = transforms.RandomResizedCrop(
            img_size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.Local_Crops = transforms.RandomResizedCrop(
            img_size, scale=(0.4, 0.8), interpolation=transforms.InterpolationMode.BICUBIC
        )

    def __call__(self, img: torch.Tensor):
        crops = []
        for _ in range(2):
            crops.append(self.Global_Crops(img))
        for _ in range(3):
            crops.append(self.Local_Crops(img))
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
            pretrained=True,
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

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class LinearSegProbe(nn.Module):
    """
    True linear probe: one independent 1×1 conv per backbone scale, no normalization,
    no activation, no cross-scale fusion. Each scale independently predicts the mask;
    predictions are upsampled and averaged.

    WHY this matters for checkpoint selection:
    The previous StrictMultiScaleProbe had BN + ReLU + cross-scale fusion — expressive
    enough to compensate for a weak backbone. This caused best_backbone.pth to be saved
    whenever the PROBE performance improved, not the backbone quality.

    A purely linear probe cannot compensate for missing vessel features in the backbone:
    if a channel does not correlate with vessels, a 1×1 conv cannot make it do so.
    Validation F1 from this probe is therefore a direct (if noisy) measure of how much
    vessel-relevant linear information exists in the raw backbone feature maps.
    """
    def __init__(self, in_channels_list, num_classes=1):
        super().__init__()
        self.probes = nn.ModuleList([
            nn.Conv2d(c, num_classes, kernel_size=1, bias=True)
            for c in in_channels_list
        ])

    def forward(self, features, original_size):
        preds = []
        for feat, probe in zip(features, self.probes):
            p = probe(feat)
            p = F.interpolate(p, size=original_size, mode='bilinear', align_corners=False)
            preds.append(p)
        return torch.stack(preds).mean(0)

def train_epoch(model, probe, dataloader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, epoch, augment, config, writer):
    model.train()
    probe.train()
    running_lejepa_loss = 0
    running_probe_loss = 0
    running_sigreg = 0
    running_inv = 0
    
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
            _, p_proj_local  = model(local_crops)                   # (3B, S*S, proj_dim)
            p_proj_all = torch.cat([p_proj_global, p_proj_local], dim=0)  # (5B, S*S, proj_dim)

            V          = len(crops)                                 # 5
            current_bs = img.size(0)
            S_sq       = p_proj_all.shape[1]                        # S*S spatial tokens

            proj_views = p_proj_all.view(V, current_bs, S_sq, -1)  # (V, B, S*S, proj_dim)
            proj_mean  = proj_views.mean(dim=0)                     # (B, S*S, proj_dim)
            inv_loss   = (proj_mean - proj_views).square().mean()

            # SIGReg operates on the spatially-averaged projection to avoid
            # non-i.i.d. statistics across spatial tokens
            proj_for_sigreg = proj_views.mean(dim=2)                # (V, B, proj_dim)
            sigreg_loss = sigreg(proj_for_sigreg)
            
            lejepa_loss = sigreg_loss * config['training']['labda'] + inv_loss * (1 - config['training']['labda'])
            total_loss = lejepa_loss + probe_loss

        optimiser.zero_grad()
        scaler.scale(total_loss).backward()
        # Gradient clipping prevents the invariance loss spikes that occur when
        # crops are very dissimilar (large gradient norm from hard negative pairs)
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(probe.parameters()), max_norm=1.0
        )
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        
        running_lejepa_loss += lejepa_loss.item()
        running_probe_loss += probe_loss.item()
        running_sigreg += sigreg_loss.item()
        running_inv += inv_loss.item()
        
        pbar.set_postfix({
            'LeJEPA': running_lejepa_loss / (batch_idx + 1),
            'Probe': running_probe_loss / (batch_idx + 1)
        })
        
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Train/LeJepa_Loss", lejepa_loss.item(), global_step)
        if is_syntax.any():
            writer.add_scalar("Train/Probe_Loss", probe_loss.item(), global_step)
        writer.add_scalar("Train/SIGReg", sigreg_loss.item(), global_step)
        writer.add_scalar("Train/Inv_Loss", inv_loss.item(), global_step)

def validate_epoch(model, probe, dataloader, f1_metric, epoch, writer):
    model.eval()
    probe.eval()
    val_f1 = 0.0
    
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
        return avg_f1

def reload_checkpoint(checkpoint_path, model, probe, optimiser, scheduler, scaler, num_gpus):
    if os.path.isfile(checkpoint_path):
        print(f"=> Se încarcă checkpoint-ul '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        
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
            
        print(f"=> Reluare cu succes de la epoca {start_epoch} (Best F1: {best_f1:.4f})")
        return start_epoch, best_f1
    else:
        print(f"=> Niciun checkpoint găsit la '{checkpoint_path}'. Antrenarea începe de la zero.")
        return 0, 0.0

def trainScript(model, probe, train_loader, val_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, augment, config, writer):
    checkpoint_dir = config['logging']['checkpoint_dir'].format(experiment_name=config['experiment_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    pacience = 300
    epochs_no_improve = 0

    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
    done_file_path = os.path.join(checkpoint_dir, "DONE")
    
    start_epoch, best_f1 = reload_checkpoint(last_model_path, model, probe, optimiser, scheduler, scaler, num_gpus)

    for epoch in range(start_epoch, config['training']['epochs']):
        train_epoch(model, probe, train_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, epoch, augment, config, writer)
        val_f1 = validate_epoch(model, probe, val_loader, f1_metric, epoch, writer)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
            'probe_state_dict': probe.module.state_dict() if num_gpus > 1 else probe.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1,
        }
        torch.save(checkpoint, last_model_path)
        
        backbone_to_save = model.module.backbone if num_gpus > 1 else model.backbone

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
            torch.save(backbone_to_save.state_dict(), os.path.join(checkpoint_dir, "best_backbone.pth"))
            print(f"--- Best backbone saved at epoch {epoch+1} with F1: {best_f1:.4f} ---")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

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
    encoders_to_train = ['convnextv2_tiny', 'swinv2_tiny_window8_256', 'resnet50']

    for encoder in encoders_to_train:
        experiment_name = f"{encoder}_lejepa_strict_probe_imagenet"
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
                'log_dir': f'runs/{experiment_name}_imagenet',
                'checkpoint_dir': checkpoint_dir
            },
            'training': {
                'img_size': 256,
                'batch_size': 20,
                'epochs': 100,
                'lr_probe': 1e-5,
                'lr_model': 1e-4,
                'weight_decay': 5e-2,
                # lambda reduced 0.2→0.05: SIGReg is a distributional regulariser,
                # it should not dominate the invariance signal. High lambda was
                # causing the invariance loss spikes seen in TensorBoard.
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
        
        criterion_probe = DiceCELoss(include_background=True, sigmoid=True, lambda_ce=1, lambda_dice=1)
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
        del model, probe, opt, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    print("\n🎉 Toate modelele din listă au fost procesate!")