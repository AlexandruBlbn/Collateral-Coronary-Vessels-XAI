import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np

# Metrics
from torchmetrics.classification import (
    BinaryJaccardIndex, 
    BinaryF1Score, 
    BinarySpecificity, 
    BinaryRecall
)

# --- SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from zoo.models import SegmentatorCoronare, SegmentatorCoronarePlusPlus
from data.dataloader import ArcadeDataset

# --- DATASET WRAPPER WITH AUGMENTATIONS & TOP-HAT ---
class SegmentationWrapper(Dataset):
    def __init__(self, base_dataset, input_size=256, root_dir='.', augment=False):
        self.base_dataset = base_dataset
        self.root_dir = root_dir
        self.input_size = input_size
        self.augment = augment
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def apply_tophat(self, img_pil, kernel_size=15):
        img_np = np.array(img_pil)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat_img = cv2.morphologyEx(img_np, cv2.MORPH_TOPHAT, kernel)
        enhanced_img = cv2.add(img_np, tophat_img)
        return Image.fromarray(enhanced_img)

    def transform(self, image, mask):
        image = TF.resize(image, (self.input_size, self.input_size))
        mask = TF.resize(mask, (self.input_size, self.input_size), interpolation=transforms.InterpolationMode.NEAREST)

        if self.augment:
            # 1. Top-Hat Enhancement
            if random.random() > 0.3:
                image = self.apply_tophat(image, kernel_size=random.choice([9, 15, 21]))

            # 2. Geometric Augmentations
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 3. Intensity Augmentations
            if random.random() > 0.3:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

            # 4. Noise
            if random.random() > 0.5:
                img_tensor = TF.to_tensor(image)
                noise = torch.randn_like(img_tensor) * 0.02
                image = TF.to_pil_image(torch.clamp(img_tensor + noise, 0, 1))

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        image = self.normalize(image)
        mask = (mask > 0).float()
        
        return image, mask

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_pil, label_path = self.base_dataset[idx]
        if label_path is None:
            raise ValueError(f"Label path missing for index {idx}")
            
        full_label_path = os.path.join(self.root_dir, label_path)
        mask_pil = Image.open(full_label_path).convert('L')
        
        return self.transform(image_pil, mask_pil)

# --- DICE + FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1.0):
        super(TverskyFocalLoss, self).__init__()
        self.alpha = alpha # Penalizare pentru False Positive (Zgomot)
        self.beta = beta   # Penalizare pentru False Negative (Vase Ratate) -> MAI MARE
        self.gamma = gamma # Focal parameter
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # 1. Tversky Index
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        
        TP = (inputs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * inputs_flat).sum()
        FN = (targets_flat * (1 - inputs_flat)).sum()
        
        tversky_score = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1 - tversky_score
        
        # 2. Focal Loss (pentru exemple grele)
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        # Combinare: Tversky se ocupă de Recall, Focal se ocupă de Hard Mining
        return tversky_loss + 0.5 * focal_loss.mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, smooth=1.0, focal_alpha=0.8, focal_gamma=2.0):
        super(DiceFocalLoss, self).__init__()
        self.smooth = smooth
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        focal = self.focal_loss(inputs, targets)
        return dice_loss + focal

# --- TENSORBOARD LOGGING ---
def log_images_to_tensorboard(writer, images, true_masks, pred_logits, step, prefix="Train"):
    n_images = min(4, images.shape[0])
    for i in range(n_images):
        img_display = (images[i] * 0.5) + 0.5
        pred_prob = torch.sigmoid(pred_logits[i])
        grid = torch.cat([img_display, true_masks[i], pred_prob], dim=2)
        writer.add_image(f'{prefix}_Visuals/Sample_{i}', grid, step)

# --- TRAINING LOGIC WITH DIFFERENTIAL LEARNING RATE ---
def train_model_run(run_name, config, train_loader, val_loader, model_config, is_frozen=True):
    print(f"\n{'='*50}\nRun: {run_name} | Frozen: {is_frozen}\n{'='*50}")
    device = torch.device(config['training']['device'])
    epochs = config['training']['epochs']
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Instantiate Model
    model = SegmentatorCoronarePlusPlus(backbone=model_config['backbone_type'], in_channels=1, num_classes=1).to(device)
    
    # 1. Load weights (Resume logic)
    frozen_checkpoint = os.path.join(project_root, config['training']['save_dir'], f"{model_config['name']}_FROZEN_UnetP", "best.pth")
    
    # If Unfrozen, try to load Frozen checkpoint first
    if not is_frozen and os.path.exists(frozen_checkpoint):
        state_dict = torch.load(frozen_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        # Standard load from DINO weights or previous run
        pth_path = os.path.join(project_root, model_config['weights_path'])
        if os.path.exists(pth_path):
            state_dict = torch.load(pth_path, map_location='cpu')
            clean_state = {k.replace("backbone.", "").replace("encoder.", ""): v for k, v in state_dict.items()}
            model.backbone.load_state_dict(clean_state, strict=False)
    
    # 2. Configure Optimizer (Differential LR)
    base_lr = float(config['training']['lr'])
    
    if is_frozen:
        # CASE 1: FROZEN - Train ONLY Decoder
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=base_lr, weight_decay=0.05)
    else:
        # CASE 2: UNFROZEN - Differential Learning Rates
        for param in model.backbone.parameters():
            param.requires_grad = True
            
        # Separate Backbone params from the rest (Decoder params)
        backbone_ids = list(map(id, model.backbone.parameters()))
        decoder_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())
        
        # Backbone gets 1/100 of the base LR (e.g., 1e-5), Decoder gets standard LR (e.g., 1e-3)
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': base_lr * 0.01}, # Very slow fine-tuning
            {'params': decoder_params,              'lr': base_lr}         # Normal speed
        ], weight_decay=0.05)
        

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr*0.001)
    
    criterion = TverskyFocalLoss(alpha=0.3, beta=0.7)
    scaler = torch.amp.GradScaler('cuda')
    
    metrics = {
        "IoU": BinaryJaccardIndex().to(device),
        "F1": BinaryF1Score().to(device),
        "Spec": BinarySpecificity().to(device),
        "Sens": BinaryRecall().to(device)
    }

    best_iou = 0.0
    
    # Save directory
    save_folder_path = os.path.join(project_root, config['training']['save_dir'], run_name)
    os.makedirs(save_folder_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for m in metrics.values(): m.reset()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(imgs)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            probs = torch.sigmoid(logits)
            for m in metrics.values(): m.update(probs, masks.int())
            
            if batch_idx == 0: log_images_to_tensorboard(writer, imgs, masks, logits, epoch, "Train")

        # Logging Train
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch) # Logs backbone LR (group 0)
        writer.add_scalar("Loss/Train", train_loss/len(train_loader), epoch)
        for name, m in metrics.items(): writer.add_scalar(f"{name}/Train", m.compute(), epoch)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        for m in metrics.values(): m.reset()
        
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(val_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                for m in metrics.values(): m.update(probs, masks.int())
                if batch_idx == 0: log_images_to_tensorboard(writer, imgs, masks, logits, epoch, "Val")

        val_iou = metrics["IoU"].compute().item()
        writer.add_scalar("Loss/Val", val_loss/len(val_loader), epoch)
        for name, m in metrics.items(): writer.add_scalar(f"{name}/Val", m.compute(), epoch)
        
        print(f"Epoch {epoch+1} | IoU: {val_iou:.4f} | F1: {metrics['F1'].compute():.4f} | Sens: {metrics['Sens'].compute():.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(save_folder_path, "best.pth"))

        scheduler.step()

    writer.close()
    
    
def evaluate_model_run(run_name, config, test_loader, model_config):
    print(f"\n{'='*50}\nEvaluating Run: {run_name}\n{'='*50}")
    device = torch.device(config['training']['device'])
    
    # Instantiate Model
    model = SegmentatorCoronare(backbone=model_config['backbone_type'], in_channels=1, num_classes=1).to(device)
    
    # Load best checkpoint
    checkpoint_path = os.path.join(project_root, config['training']['save_dir'], run_name, "best.pth")
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    model.eval()
    
    metrics = {
        "IoU": BinaryJaccardIndex().to(device),
        "F1": BinaryF1Score().to(device),
        "Spec": BinarySpecificity().to(device),
        "Sens": BinaryRecall().to(device)
    }
    
    for m in metrics.values(): m.reset()
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(imgs)
            
            probs = torch.sigmoid(logits)
            for m in metrics.values(): m.update(probs, masks.int())
            
    print("\nTest Results:")
    for name, m in metrics.items():
        print(f"{name}: {m.compute().item():.4f}")


def main():
    with open("config/segmentation_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_train = ArcadeDataset(config['data']['json_path'], split='train', mode='syntax')
    base_val = ArcadeDataset(config['data']['json_path'], split='validation', mode='syntax')
    base_test = ArcadeDataset(config['data']['json_path'], split='test', mode='syntax')
    
    train_ds = SegmentationWrapper(base_train, root_dir=config['data']['root_dir'], augment=True)
    val_ds = SegmentationWrapper(base_val, root_dir=config['data']['root_dir'], augment=False)
    test_ds = SegmentationWrapper(base_test, root_dir=config['data']['root_dir'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['data']['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    for m_cfg in config['models_to_test']:
        # Step 1: Frozen (if not already done)
        # train_model_run(f"{m_cfg['name']}_FROZEN_UnetP", config, train_loader, val_loader, m_cfg, is_frozen=True)

        # Step 2: Unfrozen with Differential LR
        run_name = f"{m_cfg['name']}_FROZEN"
        # train_model_run(run_name, config, train_loader, val_loader, m_cfg, is_frozen=False)
        
        # Step 3: Evaluation
        evaluate_model_run(run_name, config, test_loader, m_cfg)

if __name__ == "__main__":
    main()
