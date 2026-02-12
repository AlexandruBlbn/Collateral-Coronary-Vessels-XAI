import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Setăm backend-ul pentru server (fără GUI)
plt.switch_backend('Agg')

# Adăugăm path-ul root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from zoo.backbones import get_backbone
from data.dataloader import ArcadeDataset
from utils.metrics import dice_coefficient

class LinearProbeLeJEPA(nn.Module):
    def __init__(self, backbone, in_channels, num_classes=1):
        super().__init__()
        self.backbone = backbone
        
        # 1. ÎNGHEȚARE BACKBONE
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Determinare Dimensiuni
        with torch.no_grad():
            dummy = torch.randn(1, 1, 256, 256)
            feats = self.backbone(dummy)
            self.embed_dim = feats.shape[1]
            print(f"Detected backbone feature dim: {self.embed_dim}")

        # 3. Capul Liniar
        self.head = nn.Conv2d(self.embed_dim, num_classes, kernel_size=1)
        
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        
        logits = self.head(features)
        
        output = torch.nn.functional.interpolate(
            logits, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False
        )
        return output

def visualize_results(model, loader, device, epoch, save_dir, n_samples=4):
    """Salvează un grid cu predicții pentru inspectare vizuală."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Luăm un singur batch
    imgs, masks = next(iter(loader))
    imgs = imgs.to(device)
    
    with torch.no_grad():
        preds_logits = model(imgs)
        preds_probs = torch.sigmoid(preds_logits)
    
    # Mutăm pe CPU
    imgs = imgs.cpu()
    masks = masks.cpu()
    preds_probs = preds_probs.cpu()
    
    # Plotare
    fig, axes = plt.subplots(n_samples, 4, figsize=(15, 4 * n_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i in range(min(n_samples, len(imgs))):
        # 1. Imagine Originală (Denormalizare: x * 0.5 + 0.5)
        img_show = imgs[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        img_show = np.clip(img_show, 0, 1)
        
        # 2. Masca Reală
        mask_show = masks[i].permute(1, 2, 0).numpy()
        
        # 3. Predicție (Heatmap)
        pred_show = preds_probs[i].permute(1, 2, 0).numpy()
        
        # 4. Overlay (Predicție binară peste imagine)
        pred_binary = (pred_show > 0.5).astype(np.float32)
        overlay = img_show.copy()
        # Punem roșu unde e vas
        if img_show.shape[2] == 1: # Dacă e grayscale, facem RGB
            overlay = np.repeat(overlay, 3, axis=2)
            
        # Colorăm cu roșu pixelii prezisi (canalul R + 0.5)
        overlay[:, :, 0] = np.where(pred_binary[:, :, 0] == 1, 1.0, overlay[:, :, 0])

        # Afișare pe rând
        ax_row = axes[i] if n_samples > 1 else axes
        
        ax_row[0].imshow(img_show, cmap='gray' if img_show.shape[2]==1 else None)
        ax_row[0].set_title("Input")
        ax_row[0].axis('off')
        
        ax_row[1].imshow(mask_show, cmap='gray')
        ax_row[1].set_title("Ground Truth")
        ax_row[1].axis('off')
        
        ax_row[2].imshow(pred_show, cmap='jet', vmin=0, vmax=1)
        ax_row[2].set_title("Prediction (Prob)")
        ax_row[2].axis('off')
        
        ax_row[3].imshow(overlay)
        ax_row[3].set_title("Overlay (Pred > 0.5)")
        ax_row[3].axis('off')
        
    save_path = os.path.join(save_dir, f"epoch_{epoch}_vis.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    model.backbone.eval() 
    total_loss, total_dice = 0, 0
    
    for imgs, targets in tqdm(loader, desc="Training"):
        imgs = imgs.to(device)
        targets = targets.to(device).float()
        if targets.dim() == 3: targets = targets.unsqueeze(1)
            
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        with torch.no_grad():
            total_dice += dice_coefficient(torch.sigmoid(preds), targets).item()
            
    return total_loss / len(loader), total_dice / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Validating"):
            imgs = imgs.to(device)
            targets = targets.to(device).float()
            if targets.dim() == 3: targets = targets.unsqueeze(1)
                
            preds = model(imgs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            total_dice += dice_coefficient(torch.sigmoid(preds), targets).item()
    return total_loss / len(loader), total_dice / len(loader)

def collate_fn(batch):
    imgs, masks = [], []
    transform_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    for img, label_path in batch:
        imgs.append(img)
        mask = Image.open(label_path).convert('L')
        masks.append(transform_mask(mask))
    return torch.stack(imgs), torch.stack(masks)

def main():
    # --- Config ---
    config_path = "config/lejepa_config.yaml"
    if not os.path.exists(config_path): config_path = "../../config/lejepa_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    backbone_name = config["model"]["backbone"]
    experiment_name = f"lejepa_pretrain_{backbone_name}"
    
    checkpoint_path = f"./checkpoints/lejepa/{experiment_name}/best_backbone.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"../../checkpoints/lejepa/{experiment_name}/best_backbone.pth"
    
    # Folder pentru vizualizări
    vis_dir = f"./vis_probing/{experiment_name}"

    print(f"Running Linear Probe for: {backbone_name}")
    print(f"Outputs will be saved to: {vis_dir}")

    # --- Data ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    json_path = "/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json"
    root_dir = "/workspace/Collateral-Coronary-Vessels-XAI"
    
    train_ds = ArcadeDataset(json_path, split='train', transform=transform, mode='syntax', root_dir=root_dir)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    val_split = 'validation' if 'validation' in train_ds.data else 'test'
    val_ds = ArcadeDataset(json_path, split=val_split, transform=transform, mode='syntax', root_dir=root_dir)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- Model ---
    backbone = get_backbone(model_name=backbone_name, in_channels=1, pretrained=False)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        clean_state_dict = {k.replace("backbone.", "").replace("encoder.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(clean_state_dict, strict=False)
        print("Backbone loaded successfully.")
    else:
        print("WARNING: Using random weights!")

    probe_model = LinearProbeLeJEPA(backbone, in_channels=1).to(device)

    optimizer = optim.AdamW(probe_model.head.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 50
    best_dice = 0.0
    
    for epoch in range(epochs):
        train_loss, train_dice = train_one_epoch(probe_model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(probe_model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} | Val Dice: {val_dice:.4f}")
        
        # Vizualizare la fiecare 5 epoci + prima + ultima
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            visualize_results(probe_model, val_loader, device, epoch + 1, vis_dir)
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(probe_model.state_dict(), f"{vis_dir}/best_probe.pth")

if __name__ == "__main__":
    main()