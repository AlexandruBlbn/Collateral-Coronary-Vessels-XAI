import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from PIL import Image

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset
from zoo.backbones import get_backbone

# --- CONFIGURARE DEFAULT ---
DEFAULT_CONFIG = "config/lejepa_config.yaml"
# Ajusteaza daca e cazul
DEFAULT_JSON_PATH = "/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json"
DEFAULT_ROOT_DIR = "/workspace/Collateral-Coronary-Vessels-XAI"

class LinearProbeWrapper(Dataset):
    def __init__(self, base_dataset, input_size, root_dir):
        self.base_dataset = base_dataset
        self.root_dir = root_dir
        self.input_size = input_size
        
        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label_path = self.base_dataset[idx]
        
        # Procesare Imagine
        img_tensor = self.img_transform(img)
        
        # Procesare Masca
        full_label_path = os.path.join(self.root_dir, label_path)
        mask = Image.open(full_label_path).convert('L')
        mask = mask.resize((self.input_size, self.input_size), resample=Image.NEAREST)
        
        mask_tensor = self.to_tensor(mask)
        mask_tensor = (mask_tensor > 0).float()
        
        return img_tensor, mask_tensor

class LinearProbeSegmenter(nn.Module):
    def __init__(self, backbone, in_channels=1, num_classes=1, input_size=256):
        super().__init__()
        self.backbone = backbone
        
        # Determine output channels dynamically
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, input_size, input_size)
            feats = self.backbone(dummy)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            out_channels = feats.shape[1]

        # Linear Head (1x1 Conv)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if isinstance(features, (list, tuple)):
                features = features[-1]
        
        logits_small = self.head(features)
        logits = F.interpolate(logits_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

def train_probe(model, train_loader, optimizer, criterion, device, epoch):
    model.head.train()
    model.backbone.eval() # Ensure backbone stats are frozen
    
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(train_loader)

def validate_probe(model, val_loader, device, f1_metric, iou_metric):
    model.eval()
    f1_metric.reset()
    iou_metric.reset()
    
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            f1_metric.update(probs, masks.int())
            iou_metric.update(probs, masks.int())
            
    return f1_metric.compute().item(), iou_metric.compute().item()

def main():
    parser = argparse.ArgumentParser(description="Linear Probe Evaluation for Pretrained Backbone")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the backbone .pth file")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help="Path to config yaml")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train the linear head")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the head")
    args = parser.parse_args()

    # 1. Load Config
    if not os.path.exists(args.config):
        # Try finding it relative to project root
        args.config = os.path.join(project_root, args.config)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data Setup
    json_path = config["data"].get("json_path", DEFAULT_JSON_PATH)
    root_dir = config["data"].get("root_dir", DEFAULT_ROOT_DIR)
    input_size = config["model"]["input_size"]
    batch_size = config["data"]["batch_size"]

    print(f"Loading Data from: {root_dir}")
    train_base = ArcadeDataset(json_path, split='train', transform=None, root_dir=root_dir)
    val_base = ArcadeDataset(json_path, split='validation', transform=None, root_dir=root_dir)
    
    train_ds = LinearProbeWrapper(train_base, input_size, root_dir)
    val_ds = LinearProbeWrapper(val_base, input_size, root_dir)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Model Setup
    print(f"Loading Backbone: {config['model']['backbone']}")
    backbone = get_backbone(config["model"]["backbone"], in_channels=1, pretrained=False)
    
    # LOAD WEIGHTS
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"Loading weights from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    
    # Handle potentially messy keys (remove 'backbone.' prefix if exists)
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("backbone.", "")
        clean_state_dict[new_k] = v
        
    missing, unexpected = backbone.load_state_dict(clean_state_dict, strict=False)
    print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # Freeze Backbone
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Initialize Full Model
    model = LinearProbeSegmenter(backbone, in_channels=1, input_size=input_size).to(device)
    
    # 4. Optimization (Only Head)
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    f1_metric = BinaryF1Score().to(device)
    iou_metric = BinaryJaccardIndex().to(device)
    
    # 5. Training Loop
    print(f"\nStarting Linear Probe Training for {args.epochs} epochs...")
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        loss = train_probe(model, train_loader, optimizer, criterion, device, epoch)
        f1, iou = validate_probe(model, val_loader, device, f1_metric, iou_metric)
        
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Val F1: {f1:.4f} | Val IoU: {iou:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            
    print(f"\nFinal Result for {args.checkpoint}:")
    print(f"Best Validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()