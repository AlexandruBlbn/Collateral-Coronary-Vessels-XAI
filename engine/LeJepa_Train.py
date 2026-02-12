import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import torchvision.transforms as transforms
from data.dataloader import ArcadeDataset
from zoo.backbones import get_backbone
from zoo.lejepa_components.lejepa_loss import SIGReg
from utils.logger import TensorboardLogger


class LeJEPAProjector(nn.Module):
    """MLP projector: maps pooled backbone features to a low-dim embedding."""
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
    def __init__(self, backbone_name='vit_small_patch16_224', in_channels=1,
                 proj_dim=128, proj_hidden_dim=2048):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone_name, in_channels=in_channels, pretrained=False)

        # Determine feature dim by a dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 256, 256)
            feats = self.backbone(dummy)
            self.feat_dim = feats.shape[1]

        self.projector = LeJEPAProjector(self.feat_dim, proj_hidden_dim, proj_dim)

    def _pool(self, feat):
        """Global average pool spatial features (B, C, H, W) -> (B, C)."""
        if feat.dim() == 4:
            return feat.mean(dim=[-2, -1])
        if feat.dim() == 3:
            return feat.mean(dim=1)
        return feat

    def forward(self, x):
        """
        Args:
            x: (B, V, C, H, W) — batch of V views stacked along dim 1.
        Returns:
            emb:  (B*V, D)       — pooled backbone embeddings (for optional probing)
            proj: (V, B, proj_dim) — projected embeddings arranged per-view
        """
        B, V = x.shape[:2]
        flat = x.flatten(0, 1)                        # (B*V, C, H, W)
        feats = self.backbone(flat)                    # (B*V, D, h, w)
        emb = self._pool(feats)                        # (B*V, D)
        proj = self.projector(emb)                     # (B*V, proj_dim)
        proj = proj.reshape(B, V, -1).permute(1, 0, 2) # (V, B, proj_dim)
        return emb, proj


class MultiViewAugmentation:
    """Generates V augmented views from a single image, following LeJEPA's strategy."""
    def __init__(self, input_size=256, n_views=4, scale=(0.3, 1.0)):
        self.n_views = n_views
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                input_size, scale=scale,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))],
                p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomSolarize(threshold=128)],
                p=0.2
            ),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __call__(self, image):
        return torch.stack([self.transform(image) for _ in range(self.n_views)])


def train_one_epoch(model, sigreg, dataloader, optimizer, scheduler,
                    device, epoch, logger, scaler, lamb):
    model.train()
    running_lejepa = 0.0
    running_sigreg = 0.0
    running_inv = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, views in enumerate(progress_bar):
        views = views.to(device, non_blocking=True)  # (B, V, C, H, W)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            emb, proj = model(views)  # proj: (V, B, proj_dim)

            # Invariance loss: MSE between each view's projection and the mean
            inv_loss = (proj.mean(0, keepdim=True) - proj).square().mean()

            # SIGReg regularisation
            sigreg_loss = sigreg(proj)

            # Combined LeJEPA objective
            lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)

        scaler.scale(lejepa_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_lejepa += lejepa_loss.item()
        running_sigreg += sigreg_loss.item()
        running_inv += inv_loss.item()
        progress_bar.set_postfix({
            "lejepa": f"{lejepa_loss.item():.4f}",
            "sigreg": f"{sigreg_loss.item():.4f}",
            "inv": f"{inv_loss.item():.4f}",
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
        print(f"Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    backbones = [
        "swinv2_tiny_window16_256"
    ]

    device = torch.device(base_config["system"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(base_config["system"]["seed"])
    print(f"Using device: {device}")

    for backbone_name in backbones:
        print(f"\n{'='*50}")
        print(f"Backbone: {backbone_name}")
        print(f"{'='*50}\n")

        config = base_config.copy()
        config["model"]["backbone"] = backbone_name
        config["experiment_name"] = f"lejepa_pretrain_{backbone_name}"

        logger = TensorboardLogger(log_dir="runs", experiment_name=config["experiment_name"])

        # ---------- data ----------
        n_views = config["data"]["n_views"]
        transform = MultiViewAugmentation(
            input_size=config["model"]["input_size"],
            n_views=n_views,
            scale=(config["data"]["global_scale_min"], config["data"]["global_scale_max"]),
        )

        json_path = "/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json"
        root_dir = "/workspace/Collateral-Coronary-Vessels-XAI"

        dataset = ArcadeDataset(
            json_path=json_path, split='train', transform=transform,
            mode='pretrain', root_dir=root_dir
        )

        train_loader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        # ---------- model ----------
        model = LeJEPAEncoder(
            backbone_name=backbone_name,
            in_channels=config["data"]["in_channels"],
            proj_dim=config["model"]["proj_dim"],
            proj_hidden_dim=config["model"]["proj_hidden_dim"],
        ).to(device)

        sigreg = SIGReg(knots=config["lejepa"]["sigreg_knots"]).to(device)
        lamb = config["lejepa"]["lamb"]

        # ---------- optimizer & scheduler ----------
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config["optimizer"]["lr"]),
            weight_decay=config["optimizer"]["weight_decay"],
        )
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

        epochs = config["optimizer"]["epochs"]
        warmup_epochs = config["optimizer"].get("warmup_epochs", 10)
        warmup_steps = warmup_epochs * len(train_loader)
        total_steps = epochs * len(train_loader)

        s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        s2 = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=float(config["optimizer"]["lr"]) / 1000,
        )
        scheduler = SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

        # ---------- training ----------
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

            print(f"Backbone: {backbone_name} | Epoch {epoch+1}/{epochs} | LeJEPA: {lejepa_loss:.4f} | "
                  f"SIGReg: {sigreg_loss:.4f} | Inv: {inv_loss:.4f}")

            # Save backbone weights (excluding projector)
            torch.save(model.backbone.state_dict(),
                       os.path.join(save_dir, "last_backbone.pth"))

            if lejepa_loss < best_loss:
                best_loss = lejepa_loss
                torch.save(model.backbone.state_dict(),
                           os.path.join(save_dir, "best_backbone.pth"))
                print(f"  -> New best backbone saved (loss {best_loss:.4f})")

        logger.close()
        print(f"Finished training for {backbone_name}")


if __name__ == "__main__":
    main()
