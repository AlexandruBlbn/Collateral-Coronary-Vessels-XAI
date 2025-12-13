from utils.helpers import set_seed
from utils.zoo import UNet
import torchvision 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.ARCADE.dataloader import ArcadeCoronarySegmentation
from tqdm import tqdm
from utils.metrics import dice_coefficient, iou, DiceLoss
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import os

set_seed(42)

# Optimizări CUDA pentru Tensor Cores (RTX 5050 suportă TF32)
if torch.cuda.is_available():
    cudnn.benchmark = True  # Auto-optimize CUDA kernels
    cudnn.enabled = True
    # TF32 accelerează calcule pe Tensor Cores (cu tradeoff acuratețe)
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'


dataset_path = 'D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\dataset.json'

dataset = ArcadeCoronarySegmentation(dataset_path)
train_dataset = ArcadeCoronarySegmentation(dataset_path, split='train', augment=False)
val_dataset = ArcadeCoronarySegmentation(dataset_path, split='test', augment=False)  # Folosesc test pentru validare
test_dataset = ArcadeCoronarySegmentation(dataset_path, split='test', augment=False)

# Batch size optimizat pentru RTX 5050 cu mixed precision
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(in_channels=1, out_channels=1).to(device)

def train(epochs, num_samples=5):
    criterion = DiceLoss(smooth=1.0)  # Pure DiceLoss - mai agresiv pentru segmentație binară
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')  # Mixed Precision scaling (noua API)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    num_epochs = epochs
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.unsqueeze(1).float() / 255.0
            masks = masks.unsqueeze(1).float() / 255.0
            
            # Normalizare z-score per imagine
            images = (images - images.mean(dim=[2, 3], keepdim=True)) / (images.std(dim=[2, 3], keepdim=True) + 1e-8)
            
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision: autocast cu float16 pe Tensor Cores
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward cu scaling (GradScaler previne underflow-ul în float16)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
            # Eliberează memorie intermediară
            del images, masks, outputs, loss
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader.dataset)

        # Validation (float32 pentru acuratețe mai bună)
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_images = None
        val_masks = None
        val_preds = None
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.unsqueeze(1).float() / 255.0
                masks = masks.unsqueeze(1).float() / 255.0
                
                # Normalizare z-score per imagine
                images = (images - images.mean(dim=[2, 3], keepdim=True)) / (images.std(dim=[2, 3], keepdim=True) + 1e-8)
                
                images = images.to(device)
                masks = masks.to(device)
                
                # Validation rămâne în float32
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                
                preds = torch.sigmoid(outputs)
                val_dice += dice_coefficient(preds, masks).item() * images.size(0)
                val_iou += iou(preds, masks).item() * images.size(0)
                
                # Salvează imagini din primul batch pentru plotare
                if val_images is None:
                    val_images = images.cpu()
                    val_masks = masks.cpu()
                    val_preds = preds.cpu()
        
        # Evită diviziunea cu zero dacă val_loader e gol
        if len(val_loader.dataset) > 0:
            val_loss /= len(val_loader.dataset)
            val_dice /= len(val_loader.dataset)
            val_iou /= len(val_loader.dataset)
            scheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Validation Dice: {val_dice:.4f} - Validation IoU: {val_iou:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
            
            # Plotează doar dacă avem date din validation
            if num_samples > 0 and val_images is not None:
                # Creează directorul dacă nu există
                os.makedirs('logs/validation', exist_ok=True)
                
                indices = np.random.choice(val_images.size(0), min(num_samples, val_images.size(0)), replace=False)
                
                fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5*len(indices)))
                if len(indices) == 1:
                    axes = axes.reshape(1, -1)
                
                for i, idx in enumerate(indices):
                    axes[i, 0].imshow(val_images[idx, 0].numpy(), cmap='gray')
                    axes[i, 0].set_title('Data')
                    axes[i, 0].axis('off')
                    axes[i, 1].imshow(val_masks[idx, 0].numpy(), cmap='gray')
                    axes[i, 1].set_title('Label')
                    axes[i, 1].axis('off')
                    axes[i, 2].imshow(val_preds[idx, 0].numpy(), cmap='gray')
                    axes[i, 2].set_title('Prediction')
                    axes[i, 2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'logs/validation/predictions_epoch_{epoch+1:03d}.png')
                plt.close()
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f} - (Validation dataset e gol)")

        
if __name__ == "__main__":
    train(epochs=20, num_samples=3)
    

        