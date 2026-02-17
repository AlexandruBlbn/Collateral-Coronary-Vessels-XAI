import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryF1Score
from PIL import Image  # --- NECESAR PENTRU INCARCAREA MASTILOR

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataloader import ArcadeDataset
from zoo.backbones import get_backbone
from zoo.lejepa_components.lejepa_loss import SIGReg
from utils.logger import TensorboardLogger

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# --- DEFAULT PATHS (Fallback) ---
DEFAULT_JSON_PATH = "/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json"
DEFAULT_ROOT_DIR = "/workspace/Collateral-Coronary-Vessels-XAI"

# --- WRAPPER PENTRU PRE-ANTRENARE (IGNORA LABELURILE) ---
class PretrainWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        views, _ = self.base_dataset[idx]
        return views

# --- WRAPPER NOU PENTRU LINEAR PROBE (INCARCA SI PROCESEAZA MASTILE) ---
class LinearProbeWrapper(Dataset):
    def __init__(self, base_dataset, input_size, root_dir):
        self.base_dataset = base_dataset
        self.root_dir = root_dir
        self.input_size = input_size
        
        # Transformari pentru imagine (Bicubic pentru calitate)
        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Transformari pentru masca (Nearest Neighbor pentru a pastra clasele distincte, apoi Binarizare)
        # Nota: facem resize manual in getitem pentru masca
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # ArcadeDataset returneaza (img_pil, cale_relativa_string)
        img, label_path = self.base_dataset[idx]
        
        # 1. Procesare Imagine
        img_tensor = self.img_transform(img)
        
        # 2. Procesare Masca
        full_label_path = os.path.join(self.root_dir, label_path)
        mask = Image.open(full_label_path).convert('L')
        
        # Resize Nearest pentru masca
        mask = mask.resize((self.input_size, self.input_size), resample=Image.NEAREST)
        
        mask_tensor = self.to_tensor(mask)
        mask_tensor = (mask_tensor > 0).float() # Binarizare (0 sau 1)
        
        return img_tensor, mask_tensor

# --- LEJEPA COMPONENTS ---
class LeJEPAProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)

class LeJEPAEncoder(nn.Module):
    def __init__(self, backbone_name='swinv2_tiny_window16_256', in_channels=1,
                 proj_dim=128, proj_hidden_dim=2048, input_size=256):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone_name, in_channels=in_channels, pretrained=False)

        with torch.no_grad():
            dummy = torch.randn(1, in_channels, input_size, input_size)
            feats = self.backbone(dummy)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            self.feat_dim = feats.shape[1]

        self.projector = LeJEPAProjector(self.feat_dim, proj_hidden_dim, proj_dim)

    def _pool(self, feat):
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.dim() == 4:
            return feat.mean(dim=[-2, -1])
        if feat.dim() == 3:
            return feat.mean(dim=1)
        return feat

    def forward(self, x):
        B, V = x.shape[:2]
        flat = x.flatten(0, 1)
        feats = self.backbone(flat)
        emb = self._pool(feats)
        proj = self.projector(emb)
        proj = proj.reshape(B, V, -1).permute(1, 0, 2)
        return emb, proj

class MultiViewAugmentation:
    def __init__(self, input_size=256, n_views=4, scale=(0.3, 1.0)):
        self.n_views = n_views
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.0, 0.0)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(7, (0.1, 2.0))], p=0.5),
            transforms.RandomApply([transforms.RandomSolarize(128)], p=0.2),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __call__(self, image):
        return torch.stack([self.transform(image) for _ in range(self.n_views)])

class LinearProbeSegmenter(nn.Module):
    def __init__(self, backbone, in_channels=1, num_classes=1, input_size=256):
        super().__init__()
        self.backbone = backbone
        
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, input_size, input_size)
            feats = self.backbone(dummy)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            out_channels = feats.shape[1]

        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if isinstance(features, (list, tuple)):
                features = features[-1]
        
        logits_small = self.head(features)
        logits = F.interpolate(logits_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

def run_linear_evaluation(backbone_state_dict, config, device):
    # Paths
    json_path = config["data"].get("json_path", DEFAULT_JSON_PATH)
    root_dir = config["data"].get("root_dir", DEFAULT_ROOT_DIR)
    input_size = config["model"]["input_size"]

    # 1. Instantiem Dataset-urile de baza (fara transform, ca il face wrapper-ul)
    train_base = ArcadeDataset(json_path, split='train', transform=None, root_dir=root_dir)
    val_base = ArcadeDataset(json_path, split='validation', transform=None, root_dir=root_dir)
    
    # 2. Folosim Wrapper-ul care incarca mastile corect
    train_ds = LinearProbeWrapper(train_base, input_size, root_dir)
    val_ds = LinearProbeWrapper(val_base, input_size, root_dir)
    
    # 3. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=2)
    
    # 4. Model Setup
    backbone = get_backbone(config["model"]["backbone"], in_channels=config["data"]["in_channels"], pretrained=False)
    backbone.load_state_dict(backbone_state_dict, strict=False)
    
    probe_model = LinearProbeSegmenter(backbone, in_channels=config["data"]["in_channels"], input_size=input_size).to(device)
    probe_model.backbone.eval()
    
    optimizer = optim.AdamW(probe_model.head.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    metric = BinaryF1Score().to(device)
    
    # 5. Training Loop (5 epochs)
    for _ in range(5):
        probe_model.head.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            # masks e deja tensor float din wrapper
            
            optimizer.zero_grad()
            logits = probe_model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
    # 6. Validation Loop
    probe_model.eval()
    val_f1 = 0.0
    steps = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            logits = probe_model(imgs)
            probs = torch.sigmoid(logits)
            val_f1 += metric(probs, masks).item()
            steps += 1
            
    return val_f1 / steps if steps > 0 else 0.0

def train_one_epoch(model, sigreg, dataloader, optimizer, scheduler, device, epoch, logger, scaler, lamb):
    model.train()
    running_lejepa = 0.0
    running_sigreg = 0.0
    running_inv = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for step, views in enumerate(progress_bar):
        views = views.to(device, non_blocking=True)  

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            emb, proj = model(views)
            inv_loss = (proj.mean(0, keepdim=True) - proj).square().mean()
            sigreg_loss = sigreg(proj)
            lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)

        scaler.scale(lejepa_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_lejepa += lejepa_loss.item()
        running_sigreg += sigreg_loss.item()
        running_inv += inv_loss.item()
        
        progress_bar.set_postfix({
            "loss": f"{lejepa_loss.item():.4f}",
            "inv": f"{inv_loss.item():.4f}",
            "reg": f"{sigreg_loss.item():.4f}"
        })

        global_step = epoch * len(dataloader) + step
        logger.log_scalar("Train/Step_LeJEPA", lejepa_loss.item(), global_step)
        logger.log_scalar("Train/Step_SIGReg", sigreg_loss.item(), global_step)
        logger.log_scalar("Train/Step_Invariance", inv_loss.item(), global_step)
        logger.log_scalar("Train/LR", optimizer.param_groups[0]["lr"], global_step)

    n = len(dataloader)
    return running_lejepa / n, running_sigreg / n, running_inv / n

def main():
    config_path = "config/lejepa_config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "lejepa_config.yaml")

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    backbone_name = base_config["model"]["backbone"]
    device = torch.device(base_config["system"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(base_config["system"]["seed"])

    print(f"Pretraining Backbone: {backbone_name}")

    config = base_config.copy()
    config["experiment_name"] = f"lejepa_pretrain_{backbone_name}"

    logger = TensorboardLogger(log_dir="runs", experiment_name=config["experiment_name"])

    transform = MultiViewAugmentation(
        input_size=config["model"]["input_size"],
        n_views=config["data"]["n_views"],
        scale=(config["data"]["global_scale_min"], config["data"]["global_scale_max"]),
    )

    # --- FALLBACK PATHS ---
    json_path = config["data"].get("json_path", DEFAULT_JSON_PATH)
    root_dir = config["data"].get("root_dir", DEFAULT_ROOT_DIR)

    base_dataset = ArcadeDataset(
        json_path=json_path, split='train', transform=transform,
        mode='pretrain', root_dir=root_dir
    )
    
    wrapped_dataset = PretrainWrapper(base_dataset)

    train_loader = DataLoader(
        wrapped_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    model = LeJEPAEncoder(
        backbone_name=backbone_name,
        in_channels=config["data"]["in_channels"],
        proj_dim=config["model"]["proj_dim"],
        proj_hidden_dim=config["model"]["proj_hidden_dim"],
        input_size=config["model"]["input_size"]
    ).to(device)

    sigreg = SIGReg(knots=config["lejepa"]["sigreg_knots"]).to(device)
    lamb = config["lejepa"]["lamb"]

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler('cuda')

    epochs = config["optimizer"]["epochs"]
    warmup_epochs = config["optimizer"].get("warmup_epochs", 10)
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=float(config["optimizer"]["lr"]) / 1000)
    scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    save_dir = os.path.join(config["system"]["save_dir"], config["experiment_name"])
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        lejepa_loss, sigreg_loss, inv_loss = train_one_epoch(
            model, sigreg, train_loader, optimizer, scheduler,
            device, epoch, logger, scaler, lamb
        )
        
        logger.log_scalar("Train/Epoch_LeJEPA", lejepa_loss, epoch)
        logger.log_scalar("Train/Epoch_SIGReg", sigreg_loss, epoch)
        logger.log_scalar("Train/Epoch_Invariance", inv_loss, epoch)

        print(f"Ep {epoch+1}/{epochs} | Loss: {lejepa_loss:.4f} (Inv: {inv_loss:.4f}, Reg: {sigreg_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(model.backbone.state_dict(), os.path.join(save_dir, "last_backbone.pth"))

        if lejepa_loss < best_loss:
            best_loss = lejepa_loss
            torch.save(model.backbone.state_dict(), os.path.join(save_dir, "best_backbone.pth"))

        if (epoch + 1) % 50 == 0:
            print(f"Running Linear Probe at epoch {epoch+1}...")
            f1_score = run_linear_evaluation(model.backbone.state_dict(), config, device)
            logger.log_scalar("Probe/Validation_F1", f1_score, epoch)
            print(f"Linear Probe F1: {f1_score:.4f}")
            torch.save(model.backbone.state_dict(), os.path.join(save_dir, f"checkpoint_ep{epoch+1}_f1_{f1_score:.2f}.pth"))

    logger.close()
    print("Training finished.")

if __name__ == "__main__":
    main()