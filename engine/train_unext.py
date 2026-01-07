import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import numpy as np
import torchvision
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import importlib.util

# --- 1. SETUP PATHS & IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Dataset (handling space in folder name)
seg_data_path = os.path.join(project_root, 'data', 'Segmentation Data')
if seg_data_path not in sys.path:
    sys.path.append(seg_data_path)

try:
    from dataloader import SegmentationDataset, get_training_augmentation, get_validation_augmentation
except ImportError:
    print("Error: Could not import dataloader. Check path:", seg_data_path)
    sys.exit(1)

# Import Model (handling hyphen in filename)
model_path = os.path.join(project_root, 'zoo', 'UNetX-S.py')
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("UNetX_S_Module", model_path)
unext_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unext_module)
UNeXt_S = unext_module.UNeXt_S

# --- 2. CONFIGURATION ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG_PATH = os.path.join(project_root, 'config', 'UNet_config.yaml')
config = load_config(CONFIG_PATH)

# --- 3. UTILS ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_predictions_to_tensorboard(model, loader, writer, epoch, device, num_samples=4):
    model.eval()
    images_list = []
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            preds = model(imgs)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            
            # Take first N samples from batch
            B = min(imgs.shape[0], num_samples)
            for i in range(B):
                img = imgs[i].cpu()
                # Denormalize image: [-1, 1] -> [0, 1]
                img = img * 0.5 + 0.5
                img = torch.clamp(img, 0, 1)
                
                mask = masks[i].cpu()
                pred = preds[i].cpu()
                
                # Stack: Image | Ground Truth | Prediction
                # Ensure all are 1-channel for visualization or convert to RGB
                combined = torch.cat([img, mask, pred], dim=2) # Concatenate width-wise
                images_list.append(combined)
            
            break # Only one batch needed
            
    if images_list:
        grid = torchvision.utils.make_grid(images_list, nrow=1, padding=2, normalize=False)
        writer.add_image('Val/Predictions', grid, epoch)

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice_score.mean()
        
        return 0.5 * bce_loss + dice_loss

# --- 4. TRAINING LOOP ---
def train():
    # System Setup
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(config['system']['seed'])
    
    # Directories
    run_name = config['experiment_name']
    log_dir = os.path.join(project_root, 'runs', run_name)
    save_dir = os.path.join(project_root, config['system']['save_dir'], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"--> Training {run_name} on {device}")
    print(f"--> Logs: {log_dir}")
    print(f"--> Checkpoints: {save_dir}")

    # Save config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    # Data
    with open(config['data']['dataset_json'], 'r') as f:
        data_map = json.load(f)
    all_keys = list(data_map.keys())
    
    # Split 85-15
    random.seed(config['system']['seed'])
    random.shuffle(all_keys)
    split_idx = int(len(all_keys) * 0.85)
    train_keys = all_keys[:split_idx]
    val_keys = all_keys[split_idx:]
    
    print(f"--> Dataset Split: {len(train_keys)} Train | {len(val_keys)} Validation")

    train_dataset = SegmentationDataset(
        json_path=config['data']['dataset_json'],
        keys=train_keys,
        transform=get_training_augmentation()
    )
    val_dataset = SegmentationDataset(
        json_path=config['data']['dataset_json'],
        keys=val_keys,
        transform=get_validation_augmentation()
    )
    
    batch_size = config['data'].get('batch_size')
    if not batch_size:
        batch_size = 8
        print(f"Warning: Batch size not set in config. Using default: {batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Model
    model = UNeXt_S(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        base_channels=config['model']['base_channels'],
        depths=config['model']['depths'],
        mlp_ratio=config['model']['mlp_ratio'],
        drop_rate=config['model']['drop_rate'],
        attention=config['model'].get('attention', False)
    ).to(device)
    
    # Save model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> Model Parameters: {num_params:,}")
    with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
        f.write(f"Experiment: {run_name}\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Total Trainable Parameters: {num_params:,}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Config:\n{json.dumps(config['model'], indent=4)}\n")

    # Optimization
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['optimizer']['lr']),
        weight_decay=float(config['optimizer']['weight_decay'])
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['optimizer']['epochs'], eta_min=1e-5)
    
    epochs = config['optimizer']['epochs']
    global_step = 0
    best_mean_dice = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for batch in loop:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
            writer.add_scalar('Train/Step_Loss', loss.item(), global_step)
            global_step += 1
            
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        
        # Validation
        model.eval()
        val_loss = 0.0
        dice_fg_accum = 0.0
        iou_fg_accum = 0.0
        dice_bg_accum = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Metrics
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Dice FG (Vessels)
                inter_fg = (preds * masks).sum(dim=(2,3))
                union_fg = preds.sum(dim=(2,3)) + masks.sum(dim=(2,3))
                dice_fg = (2. * inter_fg + 1e-6) / (union_fg + 1e-6)
                dice_fg_accum += dice_fg.mean().item()
                
                # IoU FG
                iou_fg = (inter_fg + 1e-6) / (union_fg - inter_fg + 1e-6)
                iou_fg_accum += iou_fg.mean().item()
                
                # Dice BG (Background)
                preds_bg = 1 - preds
                masks_bg = 1 - masks
                inter_bg = (preds_bg * masks_bg).sum(dim=(2,3))
                union_bg = preds_bg.sum(dim=(2,3)) + masks_bg.sum(dim=(2,3))
                dice_bg = (2. * inter_bg + 1e-6) / (union_bg + 1e-6)
                dice_bg_accum += dice_bg.mean().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dice_fg = dice_fg_accum / len(val_loader)
        avg_iou_fg = iou_fg_accum / len(val_loader)
        avg_f1_fg = avg_dice_fg # F1 score is equivalent to Dice coefficient for binary segmentation
        avg_dice_bg = dice_bg_accum / len(val_loader)
        mean_dice = (avg_dice_fg + avg_dice_bg) / 2.0
        
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Dice_Vessels', avg_dice_fg, epoch)
        writer.add_scalar('Val/IoU_Vessels', avg_iou_fg, epoch)
        writer.add_scalar('Val/F1_Vessels', avg_f1_fg, epoch)
        writer.add_scalar('Val/Dice_Background', avg_dice_bg, epoch)
        writer.add_scalar('Val/Mean_Dice', mean_dice, epoch)
        
        scheduler.step()
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"    Val Loss: {avg_val_loss:.4f} | Vessel Dice: {avg_dice_fg:.4f} | Mean Dice: {mean_dice:.4f} | IoU: {avg_iou_fg:.4f} | F1: {avg_f1_fg:.4f}")

        # Save Best Model by Mean Dice
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(model.state_dict(), os.path.join(save_dir, f"{run_name}_best_dice.pth"))
            print(f"    --> New Best Mean Dice: {best_mean_dice:.4f} (Saved)")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{run_name}_epoch_{epoch+1}.pth"))
            
        # Visualize
        save_predictions_to_tensorboard(model, val_loader, writer, epoch, device)
        
    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    train()