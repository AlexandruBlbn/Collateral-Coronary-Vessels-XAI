import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryF1Score
import torchvision.transforms.functional as tf
from PIL import Image 
import timm as timm
import torchvision
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataloader import ArcadeDataset
from utils.logger import TensorboardLogger
import segmentation_models_pytorch as smp
import cv2
from monai.networks.nets.basic_unet import BasicUNet
from torchinfo import summary
from torchvision.ops import MLP
import monai
from torchmetrics.classification import (
    BinaryJaccardIndex, 
    BinaryF1Score
)
from zoo.models import SegmentatorCoronare, SegmentatorCoronarePlusPlus
from torch.utils.tensorboard import SummaryWriter
from utils.helpers import set_seed
import numpy as np
from deeplab import TransformsWrapper

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

img_size = 224
batch_size = 20
labda = 0.04

train_base = ArcadeDataset(split='train',mode='pretrain', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
train_ds = TransformsWrapper(train_base, input_size=img_size, mode='lejepa')
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)

test_base = ArcadeDataset(split='train', mode='syntax', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
test_ds = TransformsWrapper(test_base, input_size=img_size, mode='train')
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

val_base = ArcadeDataset(split='validation', mode='syntax', transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
val_ds = TransformsWrapper(val_base, input_size=img_size, mode='validation')
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class LeJepaModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            'swinv2_tiny_window8_256', 
            pretrained=False, 
            in_chans=1, 
            features_only=True,
        )
        feature_info = self.backbone.feature_info.channels()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = MLP(feature_info[-1], [512, proj_dim], norm_layer=nn.LayerNorm)
    
    def forward(self, x):
        features = self.backbone(x)
        last_map = features[-1]
        emb_vec = self.pool(last_map).flatten(1)
        p_loss = self.proj(emb_vec)
        return features, p_loss

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

sigreg = SIGReg().to('cuda')
dice_loss = smp.losses.DiceLoss(mode='binary')
f1_metric = BinaryF1Score().to('cuda')
iou = BinaryJaccardIndex().to('cuda')
scaler = torch.amp.GradScaler()

class ProbeHeadUNet(nn.Module):
    def __init__(self, encoder_channels, num_classes=1):
        super().__init__()
        self.center = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=encoder_channels[3], out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=encoder_channels[2], out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=encoder_channels[1], out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=encoder_channels[0], out_channels=32)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, features):
        x = self.center(features[-1]) 
        x = self.dec4(x, features[3])
        x = self.dec3(x, features[2]) 
        x = self.dec2(x, features[1]) 
        x = self.dec1(x, features[0]) 
        x = self.final_up(x) 
        return x

model = LeJepaModel().to('cuda')

dummy_input = torch.randn(1, 1, img_size, img_size).to('cuda')
with torch.no_grad():
    feats, _ = model(dummy_input)
encoder_channels = [f.shape[1] for f in feats]

probe = ProbeHeadUNet(encoder_channels=encoder_channels, num_classes=1).to('cuda')

lr1 = {"params": probe.parameters(), "lr": 1e-4, "weight_decay": 5e-2}
lr2 = {"params": model.parameters(), "lr": 1e-5, "weight_decay": 5e-2}
opt = torch.optim.AdamW([lr1, lr2])
scheduler1 = LinearLR(opt, start_factor=0.1, end_factor=0.1, total_iters=20)
scheduler2 = CosineAnnealingLR(opt, T_max=100 - 5, eta_min=1e-6)
scheduler3 = SequentialLR(opt, schedulers=[scheduler1, scheduler2], milestones=[20])
v=4

writer = SummaryWriter(log_dir=f"runs/LeJepa_coatnet_1_rw_224_detach")
num_gpus = torch.cuda.device_count()

if num_gpus > 1:
    model = nn.DataParallel(model)
    probe = nn.DataParallel(probe)

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
])

class augmentariLeJepa(nn.Module):
    def __init__(self, img_size=224, local_size=112):
        super().__init__()
        self.img_size = img_size
        self.local_size = local_size
        self.Global_Crops = transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC)
        self.Local_Crops = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.6), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)
        ])

    def __call__(self, img):
            crops = []
            for _ in range(2):
                crops.append(self.Global_Crops(img))
            for _ in range(3):
                crops.append(self.Local_Crops(img))
            return crops

augment = augmentariLeJepa()    
        
V=5
best_f1 = 0
save_dir = "checkpoints/LeJepa_coatnet_detach"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(300):
    model.train(), probe.train()
    pbar = tqdm(enumerate(test_loader),total=len(test_loader), desc=f"Epoch {epoch+1}/300")
    
    for batch_idx, (img, mask) in pbar:
        img_device = img.to('cuda')
        mask = mask.to('cuda')
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            features_original, _ = model(img_device)
            
            features_probe = [f.detach() for f in features_original]
            
            pred_probe = probe(features_probe)
            probe_loss = dice_loss(pred_probe, mask)
            
            crops = augment(img_device) 
            global_crops = torch.cat(crops[:2], dim=0) 
            local_crops = torch.cat(crops[2:], dim=0)
            
            _, p_loss_global = model(global_crops)
            _, p_loss_local = model(local_crops)
            p_loss_all = torch.cat([p_loss_global, p_loss_local], dim=0)
            
            current_bs = img.size(0)
            proj_views = p_loss_all.view(V, current_bs, -1)
            proj_mean = proj_views.mean(dim=0)
            inv_loss = (proj_mean - proj_views).square().mean()
            sigreg_loss = sigreg(proj_views)
            lejepa_loss = sigreg_loss * labda + inv_loss * (1-labda)
            
            loss = lejepa_loss + probe_loss
            
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler3.step()
        pbar.set_postfix({
            "LeJepa Loss": lejepa_loss.item(),
            "Probe Loss": probe_loss.item(),
            "SIGReg Loss": sigreg_loss.item(),
            "Inv Loss": inv_loss.item(),
        })
        writer.add_scalar("Train/LeJepa Loss", lejepa_loss.item(), epoch * len(test_loader) + batch_idx)
        writer.add_scalar("Train/F1 Score", f1_metric(pred_probe, mask).item(), epoch * len(test_loader) + batch_idx)
        writer.add_scalar("Train/SIGReg Loss", sigreg_loss.item(), epoch * len(test_loader) + batch_idx)
        writer.add_scalar("Train/Inv Loss", inv_loss.item(), epoch * len(test_loader) + batch_idx)
        
    model.eval(), probe.eval()
    f1=0
    pbar = tqdm(enumerate(val_loader),total = len(val_loader), desc=f"Validation Epoch {epoch+1}/300")
    with torch.no_grad():
            with torch.inference_mode():
                for batch_idx, (img, mask) in pbar:
                    img = img.to('cuda')
                    mask = mask.to('cuda')
                    
                    features_maps, p_loss = model(img)
                    pred_probe = probe(features_maps)
                    
                    f1 += f1_metric(pred_probe, mask).item()
                    if batch_idx == 0:
                        img = img * 0.5 + 0.5
                        num_samples = min(4, img.size(0))
                        grid_images = []
                        for i in range(num_samples):
                            grid_images.append(img[i].cpu())
                            grid_images.append(pred_probe[i].float().cpu())
                            grid_images.append(mask[i].float().cpu())
                        grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
                        writer.add_image("Val Predictions", grid, epoch)
                f1 = f1 /len(val_loader)
                writer.add_scalar("Validation/F1 Score", f1, epoch)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                    'probe_state_dict': probe.module.state_dict() if num_gpus > 1 else probe.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler3.state_dict(),
                    'best_f1': best_f1,
                }
                torch.save(checkpoint, os.path.join(save_dir, "last_model.pth"))
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
                    print(f"--- Model salvat la epoca {epoch+1} cu F1: {best_f1:.4f} ---")
                    
                    backbone_to_save = model.module.backbone if num_gpus > 1 else model.backbone
                    backbone_path = os.path.join(save_dir, f"best_backbone.pth")
                    torch.save(backbone_to_save.state_dict(), backbone_path)
                    
                    decoder_path = os.path.join(save_dir, f"best_probe.pth")
                    torch.save(probe.module.state_dict() if num_gpus > 1 else probe.state_dict(), decoder_path)