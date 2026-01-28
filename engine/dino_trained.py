import sys
import os
import torch
import torch.optim as optim
import yaml
import json
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. IMPORTS ---
from data.ARCADE.dataloader import ARCADEDataset
from data.ARCADE.DINO import ArcadeDatasetDINO, collate_dino
from zoo.dino import DINOv3, DINOv3Loss, get_attention_maps
from utils.helpers import set_seed
from torch.utils.data import DataLoader


# --- 3. CONFIGURATION ---
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


CONFIG_PATH = os.path.join(project_root, 'config', 'dino_config.yaml')
config = load_config(CONFIG_PATH)


# --- 4. DATALOADER ---
def get_dino_dataloader(json_path, config):
    """Create dataloader for DINO training."""
    base_dataset = ARCADEDataset(
        json_path=json_path,
        split='train',
        task='Unsupervised'
    )
    
    dino_dataset = ArcadeDatasetDINO(
        arcade_dataset=base_dataset,
        global_crop_size=config['data']['global_crop_size'],
        local_crop_size=config['data']['local_crop_size'],
        global_crop_scale=tuple(config['data']['global_crop_scale']),
        local_crop_scale=tuple(config['data']['local_crop_scale']),
        num_local_crops=config['data']['num_local_crops']
    )
    
    num_workers = config['data'].get('num_workers', 0)
    loader = DataLoader(
        dino_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_dino,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return loader


def setup_system(config):
    """Setup random seeds and device."""
    set_seed(config['system']['seed'])
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    return device


# --- 5. LEARNING RATE SCHEDULE ---
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """
    Cosine learning rate schedule with warmup.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_teacher_momentum_schedule(base_momentum, final_momentum, epochs, niter_per_ep):
    """Teacher EMA momentum schedule."""
    return cosine_scheduler(base_momentum, final_momentum, epochs, niter_per_ep)


# --- 6. VISUALIZATION ---
def save_attention_maps(model, loader, epoch, save_dir, device, num_samples=8):
    """
    Visualize and save attention/feature maps from the model.
    """
    model.eval()
    plot_dir = os.path.join(save_dir, 'attention_maps', f'epoch_{epoch+1}')
    os.makedirs(plot_dir, exist_ok=True)
    
    samples_collected = 0
    tensorboard_imgs = []
    
    with torch.no_grad():
        for crops in loader:
            if samples_collected >= num_samples:
                break
            
            # Use only global crop 1 for visualization
            global_crop = crops[0].to(device)
            batch_size = global_crop.shape[0]
            
            for i in range(min(batch_size, num_samples - samples_collected)):
                img = global_crop[i:i+1]  # Keep batch dim
                
                # Get attention maps
                attn_maps = get_attention_maps(model, img, device)
                
                if attn_maps is not None:
                    # Handle different output shapes
                    if len(attn_maps.shape) == 4:
                        attn_avg = attn_maps[0, 0].cpu().numpy()  # (H, W)
                    else:
                        attn_avg = attn_maps[0].cpu().numpy()
                    
                    # Original image
                    orig_img = img[0, 0].cpu().numpy()
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    
                    # Original
                    axes[0].imshow(orig_img, cmap='gray')
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    # Attention heatmap - resize to match original
                    attn_tensor = torch.from_numpy(attn_avg).float()
                    if len(attn_tensor.shape) == 2:
                        attn_tensor = attn_tensor.unsqueeze(0).unsqueeze(0)
                    attn_resized = F.interpolate(
                        attn_tensor,
                        size=(256, 256),
                        mode='bilinear',
                        align_corners=False
                    )[0, 0].numpy()
                    
                    axes[1].imshow(attn_resized, cmap='hot')
                    axes[1].set_title('Feature Importance Map')
                    axes[1].axis('off')
                    
                    # Overlay
                    axes[2].imshow(orig_img, cmap='gray')
                    axes[2].imshow(attn_resized, cmap='hot', alpha=0.5)
                    axes[2].set_title('Overlay')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(plot_dir, f'attention_sample_{samples_collected}.png')
                    plt.savefig(save_path, dpi=150)
                    plt.close(fig)
                    
                    # For tensorboard
                    tensorboard_imgs.append(torch.from_numpy(orig_img).unsqueeze(0))
                    tensorboard_imgs.append(torch.from_numpy(attn_resized).unsqueeze(0))
                
                samples_collected += 1
    
    model.train()
    
    if len(tensorboard_imgs) > 0:
        grid = torchvision.utils.make_grid(
            tensorboard_imgs, nrow=2, normalize=True, padding=2
        )
        return grid
    return None


def measure_vram_usage(model, dino_loss_fn, config, device):
    """
    Estimates VRAM usage by performing a dry run with dummy data.
    """
    if device.type != 'cuda':
        return

    print(f"\n{'='*40}\n       VRAM USAGE ESTIMATOR (Dry Run)\n{'='*40}")
    
    # 1. Model Stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 2. Memory Measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    
    # Create dummy inputs
    B = config['data']['batch_size']
    C = int(config['data']['in_channels'])
    G_size = config['data']['global_crop_size']
    L_size = config['data']['local_crop_size']
    n_local = config['data']['num_local_crops']
    
    crops = []
    # Global
    for _ in range(2):
        crops.append(torch.randn(B, C, G_size, G_size, device=device))
    # Local
    for _ in range(n_local):
        crops.append(torch.randn(B, C, L_size, L_size, device=device))
        
    try:
        use_fp16 = config['system']['use_fp16']
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
        
        with torch.amp.autocast('cuda', enabled=use_fp16):
            s_out, t_out, g_loss = model(crops, update_gram=True)
            loss = dino_loss_fn(s_out, t_out, 0) + g_loss
            
        scaler.scale(loss).backward()
        
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"Estimated Peak VRAM:  {peak_mem / (1024**3):.2f} GB")
        print(f"Batch Size:           {B}")
        
    except Exception as e:
        print(f"! Could not estimate VRAM: {e}")
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("="*40 + "\n")


# --- 7. TRAINING ---
def train():
    device = setup_system(config)
    print(f"--> Training DINOv3 on Device: {device}")
    print(f"--> Backbone: {config['model']['backbone']}")
    print(f"--> Gram Anchoring: {config['model']['gram_anchoring']['enabled']}")
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Setup logging
    run_name = config['experiment_name']
    log_dir = os.path.join(project_root, 'runs', run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"--> Logs saving to: {log_dir}")
    
    # Save config
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Data
    json_path = os.path.join(project_root, 'data', 'ARCADE', 'processed', 'dataset.json')
    save_dir = config['system']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = get_dino_dataloader(json_path, config)
    print(f"--> Total Pretraining Images: {len(train_loader.dataset)}")
    
    # Model - embed_dim is auto-detected from backbone
    model = DINOv3(
        backbone_name=config['model']['backbone'],
        in_channels=int(config['data']['in_channels']),
        embed_dim=None,  # Auto-detect from backbone
        projection_dim=int(config['model']['projection_dim']),
        hidden_dim=int(config['model']['hidden_dim']),
        bottleneck_dim=int(config['model']['bottleneck_dim']),
        use_gram_anchoring=config['model']['gram_anchoring']['enabled'],
        gram_momentum=float(config['model']['gram_anchoring']['momentum']),
        use_checkpoint=config['system']['gradient_checkpointing']
    ).to(device)
    
    # Loss
    dino_loss = DINOv3Loss(
        out_dim=config['model']['projection_dim'],
        num_local_crops=config['data']['num_local_crops'],
        warmup_teacher_temp=config['teacher']['warmup_teacher_temp'],
        teacher_temp=config['teacher']['teacher_temp'],
        warmup_teacher_temp_epochs=config['teacher']['warmup_teacher_temp_epochs'],
        num_epochs=config['optimizer']['epochs'],
        student_temp=config['student']['temperature']
    ).to(device)
    
    # Optimizer (only student parameters)
    optimizer = optim.AdamW(
        model.student.parameters(),
        lr=float(config['optimizer']['lr']),
        weight_decay=float(config['optimizer']['weight_decay'])
    )
    
    # FP16 scaler
    use_fp16 = config['system']['use_fp16']
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    
    # Schedules
    epochs = int(config['optimizer']['epochs'])
    niter_per_ep = len(train_loader)
    
    lr_schedule = cosine_scheduler(
        float(config['optimizer']['lr']),
        float(config['optimizer']['min_lr']),
        epochs,
        niter_per_ep,
        warmup_epochs=int(config['optimizer']['warmup_epochs'])
    )
    
    wd_schedule = cosine_scheduler(
        float(config['optimizer']['weight_decay']),
        float(config['optimizer']['weight_decay_end']),
        epochs,
        niter_per_ep
    )
    
    momentum_schedule = get_teacher_momentum_schedule(
        float(config['teacher']['momentum_start']),
        float(config['teacher']['momentum_end']),
        epochs,
        niter_per_ep
    )
    
    # Estimate VRAM usage
    measure_vram_usage(model, dino_loss, config, device)
    
    # Training loop
    best_loss = float('inf')
    global_step = 0
    gram_update_freq = int(config['model']['gram_anchoring']['update_freq'])
    lambda_gram = float(config['model']['gram_anchoring']['lambda_gram'])
    accum_steps = int(config['optimizer'].get('gradient_accumulation_steps', 1))
    
    print(f"--> Epochs: {epochs}")
    print(f"--> Batch size: {config['data']['batch_size']}")
    print(f"--> Gradient accumulation steps: {accum_steps} (Effective batch: {config['data']['batch_size'] * accum_steps})")
    print(f"--> FP16: {use_fp16}")
    print(f"--> Gradient checkpointing: {config['system']['gradient_checkpointing']}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_dino_loss = 0.0
        running_gram_loss = 0.0
        
        optimizer.zero_grad()
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for batch_idx, crops in enumerate(loop):
            # Move crops to device
            crops = [c.to(device, non_blocking=True) for c in crops]
            
            # Determine if we should update Gram anchor
            update_gram = (global_step % gram_update_freq == 0) and config['model']['gram_anchoring']['enabled']
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=use_fp16):
                student_output, teacher_output, gram_loss = model(crops, update_gram=update_gram)
                
                # DINO loss
                loss_dino = dino_loss(student_output, teacher_output, epoch)
                
                # Total loss
                if config['model']['gram_anchoring']['enabled']:
                    loss = loss_dino + lambda_gram * gram_loss
                else:
                    loss = loss_dino
                
                # Scale loss for gradient accumulation
                loss = loss / accum_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1 == len(train_loader)):
                # Update LR and WD
                it = epoch * niter_per_ep + batch_idx
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[min(it, len(lr_schedule) - 1)]
                    param_group['weight_decay'] = wd_schedule[min(it, len(wd_schedule) - 1)]
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=3.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update teacher EMA
                with torch.no_grad():
                    m = momentum_schedule[min(it, len(momentum_schedule) - 1)]
                    model.update_teacher(m)
                
                global_step += 1
                
                # TensorBoard step logging
                if global_step % 10 == 0:
                    # Log unscaled loss
                    writer.add_scalar('Loss/train_step', loss.item() * accum_steps, global_step)
                    writer.add_scalar('Loss/dino_step', loss_dino.item(), global_step)
                    if config['model']['gram_anchoring']['enabled']:
                        writer.add_scalar('Loss/gram_step', gram_loss.item(), global_step)
            
            # Logging
            running_loss += loss.item() * accum_steps
            running_dino_loss += loss_dino.item()
            if config['model']['gram_anchoring']['enabled']:
                running_gram_loss += gram_loss.item()
            
            loop.set_postfix({
                'loss': f"{loss.item() * accum_steps:.4f}",
                'dino': f"{loss_dino.item():.4f}",
                'gram': f"{gram_loss.item():.8f}" if config['model']['gram_anchoring']['enabled'] else "N/A"
            })
        
        # Epoch statistics
        num_batches = len(train_loader)
        avg_loss = running_loss / num_batches
        avg_dino_loss = running_dino_loss / num_batches
        avg_gram_loss = running_gram_loss / num_batches if config['model']['gram_anchoring']['enabled'] else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # TensorBoard epoch logging
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/dino_epoch', avg_dino_loss, epoch)
        if config['model']['gram_anchoring']['enabled']:
            writer.add_scalar('Loss/gram_epoch', avg_gram_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Teacher_Momentum', momentum_schedule[min(epoch * niter_per_ep, len(momentum_schedule) - 1)], epoch)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.5f} | DINO: {avg_dino_loss:.5f} | Gram: {avg_gram_loss:.8f} | LR: {current_lr:.2e}")
        
        # Save best model (student backbone only)
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Extract student backbone state
            student_backbone_state = model.get_student_backbone().state_dict()
            
            # Save backbone
            backbone_path = os.path.join(save_dir, f"{config['experiment_name']}_backbone_best.pth")
            torch.save(student_backbone_state, backbone_path)
            
            # Save full student (backbone + head)
            student_full_path = os.path.join(save_dir, f"{config['experiment_name']}_student_best.pth")
            torch.save(model.student.state_dict(), student_full_path)
            
            # Save full teacher (backbone + head)
            teacher_full_path = os.path.join(save_dir, f"{config['experiment_name']}_teacher_best.pth")
            torch.save(model.teacher.state_dict(), teacher_full_path)
            
            # Save full best checkpoint
            best_ckpt_path = os.path.join(save_dir, f"{config['experiment_name']}_checkpoint_best.pth")
            torch.save({
                'epoch': epoch,
                'student_state_dict': model.student.state_dict(),
                'teacher_state_dict': model.teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'config': config
            }, best_ckpt_path)
            
            print(f"  --> Saved best model (loss: {best_loss:.5f})")
        
        # Save latest model (every epoch)
        latest_backbone_path = os.path.join(save_dir, f"{config['experiment_name']}_backbone_last.pth")
        torch.save(model.get_student_backbone().state_dict(), latest_backbone_path)
        
        latest_student_path = os.path.join(save_dir, f"{config['experiment_name']}_student_last.pth")
        torch.save(model.student.state_dict(), latest_student_path)
        
        latest_ckpt_path = os.path.join(save_dir, f"{config['experiment_name']}_checkpoint_last.pth")
        torch.save({
            'epoch': epoch,
            'student_state_dict': model.student.state_dict(),
            'teacher_state_dict': model.teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'config': config
        }, latest_ckpt_path)
        
        # Save checkpoint periodically
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir, f"{config['experiment_name']}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'student_state_dict': model.student.state_dict(),
                'teacher_state_dict': model.teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'config': config
            }, checkpoint_path)
        
        # Visualize attention maps
        if (epoch + 1) % 10 == 0 or epoch == 0:
            attn_grid = save_attention_maps(model, train_loader, epoch, log_dir, device, num_samples=8)
            if attn_grid is not None:
                writer.add_image('Attention_Maps', attn_grid, global_step=epoch)
    
    # Final logging
    flat_config = flatten_config(config)
    writer.add_hparams(flat_config, {'hparam/loss': best_loss})
    writer.close()
    
    print("-" * 50)
    print(f"Training complete! Best loss: {best_loss:.5f}")
    print(f"Backbone saved to: {os.path.join(save_dir, config['experiment_name'] + '_backbone_best.pth')}")


if __name__ == "__main__":
    import traceback
    from datetime import datetime
    
    try:
        train()
    except Exception as e:
        # Save traceback to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = os.path.join(project_root, f"error_log_{timestamp}.txt")
        
        with open(error_file, 'w') as f:
            f.write(f"Error occurred at: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Exception Type: " + str(type(e).__name__) + "\n")
            f.write("Exception Message: " + str(e) + "\n\n")
            f.write("Full Traceback:\n")
            f.write("=" * 80 + "\n")
            traceback.print_exc(file=f)
        
        print(f"\n{'='*80}")
        print(f"ERROR: {type(e).__name__}: {e}")
        print(f"Full traceback saved to: {error_file}")
        print(f"{'='*80}\n")
        
        # Re-raise the exception
        raise