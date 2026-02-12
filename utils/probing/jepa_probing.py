import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from PIL import Image

# Project imports
from zoo.backbones import get_backbone
from data.dataloader import ArcadeDataset


class LinearProbeLeJEPA(nn.Module):
    def __init__(self, backbone, img_size=256, patch_size=16, embed_dim=768, num_classes=1):
        super().__init__()
        
        # 1. Backbone-ul LeJEPA (Encoderul)
        self.backbone = backbone
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Calculăm dimensiunea grid-ului de patch-uri (ex: 224/16 = 14x14)
        self.grid_size = img_size // patch_size
        
        # 2. ÎNGHEȚARE (Esențial pentru Linear Probing)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Capul Liniar (Sonda)
        # Transformă vectorul latent (ex: 768) în logits (1 clasă)
        # Folosim Conv2d(1x1) care este echivalent matematic cu un Linear layer aplicat spațial
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
        # Inițializare standard pentru cap
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        
        # 1. Extragem Feature-urile din Backbone
        with torch.no_grad():
            # LeJEPA (ViT) returnează de obicei (Batch, N_patches, Dim)
            # Verifică dacă backbone-ul tău are o funcție forward_encoder sau doar forward
            if hasattr(self.backbone, 'forward_encoder'):
                features = self.backbone.forward_encoder(x)
            else:
                features = self.backbone(x)
                
            # Dacă output-ul este un dicționar sau tuplu, extragem doar tensorul
            if isinstance(features, (tuple, list)):
                features = features[0]
            
            # Eliminăm CLS token dacă există (ViT standard are CLS pe poz 0)
            # Dacă LeJEPA tău nu are CLS token, comentează linia asta.
            if features.shape[1] == (self.grid_size ** 2) + 1:
                features = features[:, 1:, :]
                
            # 2. Reshape din secvență în imagine 2D
            # (B, N, D) -> (B, D, H_grid, W_grid)
            B, N, D = features.shape
            H_grid = W_grid = int(N ** 0.5) # ar trebui să fie 14
            
            features = features.transpose(1, 2) # (B, D, N)
            features = features.view(B, D, H_grid, W_grid)

        # 3. Aplicăm Sonda Liniară
        # (B, 768, 14, 14) -> (B, 1, 14, 14)
        logits = self.head(features)
        
        # 4. Upsampling la rezoluția originală
        # De la 14x14 înapoi la 224x224
        masks = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        return masks


def main():
    # --- CONFIGURATION ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    epochs = 10  # Linear probing converges quickly
    lr = 1e-3
    
    # Path to the trained LeJEPA backbone checkpoint
    checkpoint_path = "checkpoints/lejepa/lejepa_pretrain_vit_small_patch16_224/best_backbone.pth"
    
    # Data and model paths
    json_path = "data/ARCADE/processed/dataset.json"
    root_dir = "."
    
    print(f"🚀 Starting Linear Probing for LeJEPA on {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # 1. Dataset - Simple transforms (resize + normalize)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # ImageNet normalization
    ])
    
    # Train set (syntax mode: returns image and label path)
    train_dataset = ArcadeDataset(
        json_path=json_path,
        split='train',
        transform=transform,
        mode='syntax',  # Lowercase: returns (image, label_path)
        root_dir=root_dir
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_with_label_loading
    )

    # Validation set (if available, else use test)
    try:
        val_dataset = ArcadeDataset(
            json_path=json_path,
            split='validation',
            transform=transform,
            mode='syntax',  # Lowercase
            root_dir=root_dir
        )
        print(f"✓ Validation split found: {len(val_dataset)} samples")
    except:
        print("⚠ Validation split not found, using test split...")
        try:
            val_dataset = ArcadeDataset(
                json_path=json_path,
                split='test',
                transform=transform,
                mode='syntax',
                root_dir=root_dir
            )
            print(f"✓ Using test split: {len(val_dataset)} samples")
        except:
            print("✗ Neither validation nor test split found. Using 10% of train as val.")
            val_size = len(train_dataset) // 10
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [len(train_dataset) - val_size, val_size]
            )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        collate_fn=collate_with_label_loading
    )

    # 2. Load Backbone LeJEPA
    print("Loading Backbone...")
    
    # Initialize backbone architecture - ViT-Small (adjust if using different model)
    backbone = get_backbone(
        img_size=256, patch_size=16, embed_dim=384, depth=12, num_heads=6  # ViT-Small
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Clean state dict keys (handle 'encoder.', 'backbone.', 'module.' prefixes)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    clean_dict = {}
    for k, v in state_dict.items():
        # Keep only backbone/encoder weights, skip predictor
        if 'predictor' not in k:
            name = k.replace('encoder.', '').replace('backbone.', '').replace('module.', '')
            clean_dict[name] = v
             
    msg = backbone.load_state_dict(clean_dict, strict=False)
    print(f"✓ Backbone loaded: {msg}")

    # 3. Create Linear Probe Model
    # embed_dim must match ViT dimension (384 for Small, 768 for Base)
    model = LinearProbeLeJEPA(backbone, img_size=256, patch_size=16, embed_dim=384).to(device)
    print(f"✓ Probe model created on {device}")

    # 4. Training Setup
    # Optimize ONLY the probe head (backbone is frozen in __init__)
    optimizer = optim.AdamW(model.head.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"{'='*60}\n")

    best_val_dice = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device).float()
            
            # Ensure mask shape (B, 1, H, W)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            elif masks.dim() == 4 and masks.shape[1] > 1:
                # If multi-channel, take first channel
                masks = masks[:, 0:1, :, :]
            
            optimizer.zero_grad()
            
            preds = model(imgs)
            loss = criterion(preds, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Compute Dice for monitoring
            with torch.no_grad():
                pred_binary = (torch.sigmoid(preds) > 0.5).float()
                dice = compute_dice(pred_binary, masks)
                train_dice += dice.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': dice.item()
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # --- VALIDATION ---
        val_loss, val_dice = validate(model, val_loader, device, criterion)
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss
            }
            os.makedirs('runs/probing', exist_ok=True)
            torch.save(checkpoint, 'runs/probing/best_probe.pth')
            print(f"  ✓ Best model saved at epoch {epoch+1}")
        
        # Print epoch results
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

def compute_dice(pred, target):
    """Compute Dice score for a batch."""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice


def collate_with_label_loading(batch):
    """Custom collate function to load label images from paths."""
    images = []
    masks = []
    
    for img, label_path in batch:
        images.append(img)
        
        # Load label image from path
        try:
            label_img = Image.open(label_path).convert('L')  # Grayscale
            label_tensor = transforms.ToTensor()(label_img)
            masks.append(label_tensor)
        except Exception as e:
            print(f"Warning: Could not load label {label_path}: {e}")
            # Create dummy mask if loading fails
            masks.append(torch.zeros(1, 224, 224))
    
    images_batch = torch.stack(images)
    masks_batch = torch.stack(masks)
    
    return images_batch, masks_batch


def validate(model, loader, device, criterion):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device).float()
            
            # Ensure mask shape
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            elif masks.dim() == 4 and masks.shape[1] > 1:
                masks = masks[:, 0:1, :, :]
            
            preds = model(imgs)
            loss = criterion(preds, masks)
            total_loss += loss.item()
            
            # Binary prediction
            pred_binary = (torch.sigmoid(preds) > 0.5).float()
            dice = compute_dice(pred_binary, masks)
            total_dice += dice.item()
    
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    
    return avg_loss, avg_dice

if __name__ == "__main__":
    main()