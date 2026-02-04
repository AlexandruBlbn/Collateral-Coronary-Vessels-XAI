"""
LeJepa Training Script for Coronary Angiography Images
Based on: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
Implements self-supervised learning with SIGReg objective and invariance loss.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- HF CACHE ---
hf_cache_dir = os.path.join(project_root, '.hf_cache')
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ.setdefault('HF_HOME', hf_cache_dir)
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(hf_cache_dir, 'hub'))

# --- 2. IMPORTS ---
from data.ARCADE.dataloader import ARCADEDataset
from data.ARCADE.LeJepa import ArcadeDatasetLeJepa, collate_lejepa
from utils.helpers import set_seed
from torch.utils.data import DataLoader
import timm
from torchvision.ops import MLP


# --- 3. SIGREG LOSS ---
class SIGReg(nn.Module):
    """
    SIGReg objective from LeJepa paper.
    Implements spectral independence via random features and characteristic functions.
    """
    def __init__(self, knots=17, t_max=3.0, num_random_features=256):
        super().__init__()
        # Quadrature points and weights (trapezoidal rule)
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Trapezoidal correction
        
        # Window function for better numerical stability
        window = torch.exp(-t.square() / 2.0)
        
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        self.num_random_features = num_random_features
        
    def forward(self, proj):
        """
        Args:
            proj: Projected features of shape (V, B, D) where:
                  V = number of views
                  B = batch size
                  D = projection dimension
                  
        Returns:
            statistic: SIGReg loss value (scalar)
        """
        # Generate random projection matrix A ~ N(0, I)
        D = proj.size(-1)
        A = torch.randn(D, self.num_random_features, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)  # Normalize columns
        
        # Project features: (V, B, D) @ (D, M) = (V, B, M)
        proj_random = proj @ A  # (V, B, M)
        
        # Compute characteristic function at different t values
        # x_t = proj_random * t, shape: (V, B, M, knots)
        x_t = proj_random.unsqueeze(-1) * self.t.view(1, 1, 1, -1)
        
        # Empirical characteristic function: E[cos(t*X)] + i*E[sin(t*X)]
        # SIGReg measures how Gaussian the SAMPLE distribution is
        # So we average over SAMPLES (dim=1, the batch dimension), NOT over views!
        cos_part = x_t.cos().mean(dim=1)  # (V, M, knots) - mean over batch
        sin_part = x_t.sin().mean(dim=1)  # (V, M, knots)
        
        # Error: ||E[cos] - phi||^2 + ||E[sin]||^2
        # phi is the characteristic function of standard normal
        err = (cos_part - self.phi).square() + sin_part.square()
        
        # Integrate over t using quadrature weights
        # (V, M, knots) @ (knots,) = (V, M)
        statistic = (err @ self.weights)  # (V, M)
        
        # Scale by number of SAMPLES (batch size)
        # Original: proj.size(-2) where proj is (V, N, D), so -2 = N (samples)
        # Our proj is (V, B, D), so size(1) = B (samples)
        B = proj.size(1)
        statistic = statistic * B
        return statistic.mean()


# --- 4. MODEL DEFINITION ---
class LeJepaEncoder(nn.Module):
    """
    LeJepa encoder with backbone and projection head.
    """
    def __init__(self, config):
        super().__init__()
        
        # Create backbone
        # ConvNeXt doesn't accept img_size, only transformers (ViT, Swin) do
        backbone_name = config['model']['backbone']
        backbone_kwargs = {
            'pretrained': False,
            'num_classes': config['model']['num_classes'],
            'drop_path_rate': config['model']['drop_path_rate'],
            'in_chans': 3  # Will convert grayscale to RGB in dataloader
        }
        
        # Only add img_size for transformers that need it
        if 'vit' in backbone_name or 'swin' in backbone_name or 'deit' in backbone_name:
            backbone_kwargs['img_size'] = config['data']['image_size']
        
        self.backbone = timm.create_model(backbone_name, **backbone_kwargs)
        
        # Load pretrained weights if specified
        if config['model']['pretrained_backbone']:
            pretrained_path = config['model']['pretrained_backbone_path']
            if pretrained_path and os.path.exists(pretrained_path):
                print(f"Loading pretrained backbone from: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.backbone.load_state_dict(state_dict, strict=False)
        
        # Projection head (3-layer MLP with BatchNorm)
        hidden_dim = 2048
        proj_dim = config['model']['proj_dim']
        num_classes = config['model']['num_classes']
        
        self.proj = MLP(
            in_channels=num_classes,
            hidden_channels=[hidden_dim, hidden_dim, proj_dim],
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.GELU,
            inplace=None
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B*V, C, H, W)
            
        Returns:
            emb: Backbone embeddings (B*V, num_classes)
            proj: Projected features (B*V, proj_dim)
        """
        emb = self.backbone(x)
        proj = self.proj(emb)
        return emb, proj


# --- 5. CONFIGURATION ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def flatten_config(config, parent_key='', sep='_'):
    """Flatten nested config for tensorboard hparams."""
    items = []
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                v = str(v)
            items.append((new_key, v))
    return dict(items)


# --- 6. DATALOADER ---
def get_lejepa_dataloader(json_path, config, is_training=True):
    """Create dataloader for LeJepa training."""
    base_dataset = ARCADEDataset(
        json_path=json_path,
        split='train' if is_training else 'valid',
        task='Unsupervised'  # Includes stenoza, segmentare, syntax, and extra data
    )
    
    lejepa_dataset = ArcadeDatasetLeJepa(
        arcade_dataset=base_dataset,
        image_size=config['data']['image_size'],
        num_views=config['data']['num_views'] if is_training else 1,
        is_training=is_training
    )
    
    num_workers = config['data'].get('num_workers', 0)
    batch_size = config['data']['batch_size']
    
    loader = DataLoader(
        lejepa_dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training,
        collate_fn=collate_lejepa,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return loader


# --- 7. LEARNING RATE SCHEDULE ---
def get_scheduler(optimizer, config, steps_per_epoch):
    """
    Create learning rate scheduler with linear warmup and cosine decay.
    """
    warmup_steps = config['optimizer']['warmup_epochs'] * steps_per_epoch
    total_steps = config['optimizer']['epochs'] * steps_per_epoch
    
    # Linear warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_steps
    )
    
    # Cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config['optimizer']['min_lr']
    )
    
    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler


# --- 8. LINEAR PROBE ---
class LinearProbe(nn.Module):
    """Linear probe for online evaluation during training."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(self.norm(x))


# --- 9. TRAINING LOOP ---
def train_epoch(model, probe, sigreg, loader, optimizer, scheduler, scaler, config, device, epoch, writer, global_step):
    """Train for one epoch."""
    model.train()
    if probe is not None:
        probe.train()
    
    total_lejepa_loss = 0
    total_sigreg_loss = 0
    total_inv_loss = 0
    total_probe_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['optimizer']['epochs']}")
    
    for batch_idx, views in enumerate(pbar):
        # views: (B, V, C, H, W)
        B, V = views.shape[:2]
        views = views.to(device, non_blocking=True)
        
        # Flatten batch and views: (B*V, C, H, W)
        views_flat = views.flatten(0, 1)
        
        with autocast(device.type, dtype=torch.bfloat16, enabled=config['system']['use_amp']):
            # Forward pass
            emb, proj = model(views_flat)
            
            # Reshape projections: (B*V, D) -> (V, B, D)
            proj = proj.view(B, V, -1).transpose(0, 1)
            
            # LeJepa losses
            sigreg_loss = sigreg(proj)
            
            # Invariance loss: each view should be close to the mean of all views
            # proj shape: (V, B, D)
            # mean over views (dim=0) -> (B, D) = mean representation per sample
            # Expand back to (V, B, D) and compute difference
            mean_proj = proj.mean(dim=0, keepdim=True)  # (1, B, D)
            inv_loss = (proj - mean_proj).square().mean()
            
            lejepa_loss = sigreg_loss * config['optimizer']['lamb'] + inv_loss * (1 - config['optimizer']['lamb'])
            
            # Probe loss (if enabled)
            probe_loss = torch.tensor(0.0, device=device)
            if probe is not None and config['probe']['enabled']:
                # For probe, we don't have labels in unsupervised setting
                # This is a placeholder - you'd need labels for real evaluation
                pass
            
            total_loss = lejepa_loss + probe_loss
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        
        # Gradient clipping
        if config['optimizer'].get('gradient_clip'):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['gradient_clip'])
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Logging
        total_lejepa_loss += lejepa_loss.item()
        total_sigreg_loss += sigreg_loss.item()
        total_inv_loss += inv_loss.item()
        if probe_loss.item() > 0:
            total_probe_loss += probe_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'lejepa': f'{lejepa_loss.item():.4f}',
            'sigreg': f'{sigreg_loss.item():.4f}',
            'inv': f'{inv_loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Tensorboard logging
        if batch_idx % config['system']['log_interval'] == 0:
            writer.add_scalar('train/lejepa_loss', lejepa_loss.item(), global_step)
            writer.add_scalar('train/sigreg_loss', sigreg_loss.item(), global_step)
            writer.add_scalar('train/inv_loss', inv_loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            if probe_loss.item() > 0:
                writer.add_scalar('train/probe_loss', probe_loss.item(), global_step)
        
        global_step += 1
    
    # Epoch averages
    avg_lejepa = total_lejepa_loss / num_batches
    avg_sigreg = total_sigreg_loss / num_batches
    avg_inv = total_inv_loss / num_batches
    
    return avg_lejepa, avg_sigreg, avg_inv, global_step


# --- 10. VISUALIZATION ---
def save_embeddings_visualization(model, loader, epoch, save_dir, device, num_samples=100):
    """Visualize learned embeddings using t-SNE or PCA."""
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        samples_collected = 0
        for views in loader:
            if samples_collected >= num_samples:
                break
            
            # Use first view only
            views = views[:, 0]  # (B, C, H, W)
            views = views.to(device, non_blocking=True)
            
            emb, _ = model(views)
            embeddings.append(emb.cpu())
            
            samples_collected += views.shape[0]
    
    if len(embeddings) == 0:
        return
    
    embeddings = torch.cat(embeddings, dim=0)[:num_samples]
    
    # Save embeddings
    save_path = os.path.join(save_dir, f'embeddings_epoch_{epoch+1}.pt')
    torch.save(embeddings, save_path)
    
    model.train()


# --- 11. CHECKPOINTING ---
def save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, config, save_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config
    }
    
    if probe is not None:
        checkpoint['probe_state_dict'] = probe.state_dict()
    
    # Save last checkpoint
    last_path = os.path.join(save_dir, f"{config['experiment_name']}_checkpoint_last.pth")
    torch.save(checkpoint, last_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, f"{config['experiment_name']}_checkpoint_best.pth")
        torch.save(checkpoint, best_path)
    
    # Save backbone separately for easy loading
    backbone_path = os.path.join(save_dir, f"{config['experiment_name']}_backbone_last.pth")
    torch.save(model.backbone.state_dict(), backbone_path)
    
    print(f"Checkpoint saved to {last_path}")


def load_checkpoint(model, probe, optimizer, scheduler, scaler, checkpoint_path):
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    if probe is not None and 'probe_state_dict' in checkpoint:
        probe.load_state_dict(checkpoint['probe_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")
    
    return start_epoch


# --- 12. MAIN TRAINING FUNCTION ---
def main():
    # Load config
    CONFIG_PATH = os.path.join(project_root, 'config', 'lejepa_config.yaml')
    config = load_config(CONFIG_PATH)
    
    # Setup
    set_seed(config['system']['seed'])
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print(f"LeJepa Training - {config['experiment_name']}")
    print("="*80)
    print(f"Device: {device}")
    print(f"Backbone: {config['model']['backbone']}")
    print(f"Image size: {config['data']['image_size']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Num views: {config['data']['num_views']}")
    print(f"Projection dim: {config['model']['proj_dim']}")
    print(f"Lambda (λ): {config['optimizer']['lamb']}")
    print("="*80)
    
    # Create save directory
    save_dir = config['system']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(project_root, 'runs', config['experiment_name'], timestamp)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Save config
    config_save_path = os.path.join(log_dir, 'config.json')
    import json
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Dataloader
    json_path = os.path.join(project_root, 'data/ARCADE/processed/dataset.json')
    train_loader = get_lejepa_dataloader(json_path, config, is_training=True)
    
    print(f"\nDataset size: {len(train_loader.dataset)}")
    print(f"Number of batches per epoch: {len(train_loader)}")
    
    # Model
    model = LeJepaEncoder(config).to(device)
    
    # SIGReg loss
    sigreg = SIGReg(
        knots=config['model']['sigreg']['knots'],
        t_max=config['model']['sigreg']['t_max'],
        num_random_features=config['model']['sigreg']['num_random_features']
    ).to(device)
    
    # Linear probe (optional)
    probe = None
    if config['probe']['enabled']:
        probe = LinearProbe(
            input_dim=config['model']['num_classes'],
            num_classes=config['probe']['num_classes']
        ).to(device)
    
    # Optimizer
    param_groups = [
        {'params': model.parameters(), 'lr': config['optimizer']['lr'], 'weight_decay': config['optimizer']['weight_decay']}
    ]
    if probe is not None:
        param_groups.append({
            'params': probe.parameters(),
            'lr': config['optimizer']['probe_lr'],
            'weight_decay': config['optimizer']['probe_weight_decay']
        })
    
    optimizer = optim.AdamW(param_groups)
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=config['system']['use_amp'])
    
    # Resume from checkpoint
    start_epoch = 0
    if config['system']['resume'] and config['system']['resume_from']:
        start_epoch = load_checkpoint(model, probe, optimizer, scheduler, scaler, config['system']['resume_from'])
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Training loop
    global_step = start_epoch * len(train_loader)
    best_loss = float('inf')
    
    print("\nStarting training...")
    print("="*80)
    
    for epoch in range(start_epoch, config['optimizer']['epochs']):
        avg_lejepa, avg_sigreg, avg_inv, global_step = train_epoch(
            model, probe, sigreg, train_loader, optimizer, scheduler, scaler,
            config, device, epoch, writer, global_step
        )
        
        # Log epoch metrics
        writer.add_scalar('epoch/lejepa_loss', avg_lejepa, epoch)
        writer.add_scalar('epoch/sigreg_loss', avg_sigreg, epoch)
        writer.add_scalar('epoch/inv_loss', avg_inv, epoch)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  LeJepa Loss: {avg_lejepa:.4f}")
        print(f"  SIGReg Loss: {avg_sigreg:.4f}")
        print(f"  Inv Loss: {avg_inv:.4f}")
        
        # Save checkpoint
        is_best = avg_lejepa < best_loss
        if is_best:
            best_loss = avg_lejepa
        
        if (epoch + 1) % config['system']['save_interval'] == 0 or is_best:
            save_checkpoint(model, probe, optimizer, scheduler, scaler, epoch, config, save_dir, is_best)
        
        # Visualizations
        if (epoch + 1) % config['system']['eval_interval'] == 0:
            save_embeddings_visualization(model, train_loader, epoch, log_dir, device)
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best LeJepa loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print("="*80)
    
    writer.close()


if __name__ == "__main__":
    main()
