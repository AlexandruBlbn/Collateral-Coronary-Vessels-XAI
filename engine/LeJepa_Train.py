import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
import timm
from torchvision.ops import MLP
from monai.losses import DiceCELoss
from torchmetrics.classification import BinaryF1Score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset
from data.transformWrapper import TransformsWrapper
from utils.helpers import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler()

def loader(img_size, batch_size, split='train', mode='train'):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    ds_mode = 'pretrain' if mode == 'lejepa' else 'syntax'
    base = ArcadeDataset(split=split, mode=ds_mode, transform=None, root_dir='.', json_path='data/ARCADE/processed/dataset.json')
    ds = TransformsWrapper(base, input_size=img_size, mode=mode)
    g = torch.Generator()
    g.manual_seed(42)
    
    return DataLoader(
        ds,
        batch_size=batch_size, 
        shuffle=(split=='train'),
        num_workers=4, 
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

def configCreate(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)

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

class LeJepaModel(nn.Module):
    def __init__(self, encoder_name='swinv2_tiny_window8_256', proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            encoder_name, 
            pretrained=False, 
            in_chans=1, 
            features_only=True,
        )
        self.channels_list = self.backbone.feature_info.channels()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = MLP(self.channels_list[-1], [512, proj_dim], norm_layer=nn.LayerNorm)
    
    def forward(self, x):
        features = list(self.backbone(x))
        for i in range(len(features)):
            if features[i].dim() == 4 and features[i].shape[-1] == self.channels_list[i]:
                features[i] = features[i].permute(0, 3, 1, 2).contiguous()
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
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class MultiScaleLinearProbe(nn.Module):
    def __init__(self, in_channels_list, probe_dim=256, num_classes=1):
        super().__init__()
        self.probes = nn.ModuleList([
            nn.Conv2d(c, probe_dim, kernel_size=1) for c in in_channels_list
        ])
        self.fuse = nn.Conv2d(probe_dim * len(in_channels_list), num_classes, kernel_size=1)

    def forward(self, features, original_size):
        upscaled = []
        target_size = features[0].shape[2:]
        
        for feat, probe_conv in zip(features, self.probes):
            p = probe_conv(feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            upscaled.append(p)
            
        out = torch.cat(upscaled, dim=1)
        out = self.fuse(out)
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        return out

def train_epoch(model, probe, dataloader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, epoch, augment, config, writer):
    model.train()
    probe.train()
    running_lejepa_loss = 0
    running_probe_loss = 0
    running_sigreg = 0
    running_inv = 0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
    for batch_idx, (img, mask, is_syntax) in pbar:
        img, mask = img.cuda(), mask.cuda()
        is_syntax = is_syntax.cuda().bool()
        original_size = img.shape[2:]
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            features_original, _ = model(img)
            features_probe = [f.detach() for f in features_original]
            
            pred_probe = probe(features_probe, original_size)
            
            if is_syntax.any():
                probe_loss = criterion_probe(pred_probe[is_syntax], mask[is_syntax])
            else:
                probe_loss = torch.tensor(0.0, device='cuda', requires_grad=True)
            
            crops = augment(img) 
            global_crops = torch.cat(crops[:2], dim=0) 
            local_crops = torch.cat(crops[2:], dim=0)
            
            _, p_loss_global = model(global_crops)
            _, p_loss_local = model(local_crops)
            p_loss_all = torch.cat([p_loss_global, p_loss_local], dim=0)
            
            V = len(crops)
            current_bs = img.size(0)
            proj_views = p_loss_all.view(V, current_bs, -1)
            proj_mean = proj_views.mean(dim=0)
            inv_loss = (proj_mean - proj_views).square().mean()
            sigreg_loss = sigreg(proj_views)
            
            lejepa_loss = sigreg_loss * config['training']['labda'] + inv_loss * (1 - config['training']['labda'])
            total_loss = lejepa_loss + probe_loss

        optimiser.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        
        running_lejepa_loss += lejepa_loss.item()
        running_probe_loss += probe_loss.item()
        running_sigreg += sigreg_loss.item()
        running_inv += inv_loss.item()
        
        pbar.set_postfix({
            'LeJEPA': running_lejepa_loss / (batch_idx + 1),
            'Probe': running_probe_loss / (batch_idx + 1)
        })
        
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Train/LeJepa_Loss", lejepa_loss.item(), global_step)
        if is_syntax.any():
            writer.add_scalar("Train/Probe_Loss", probe_loss.item(), global_step)
        writer.add_scalar("Train/SIGReg", sigreg_loss.item(), global_step)
        writer.add_scalar("Train/Inv_Loss", inv_loss.item(), global_step)

def validate_epoch(model, probe, dataloader, f1_metric, epoch, writer):
    model.eval()
    probe.eval()
    val_f1 = 0.0
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation {epoch+1}")
        for batch_idx, (img, mask, _) in pbar:
            img, mask = img.cuda(), mask.cuda()
            original_size = img.shape[2:]
            
            features_maps, _ = model(img)
            pred_probe = probe(features_maps, original_size)
            
            val_f1 += f1_metric(pred_probe.sigmoid(), mask.int()).item()
            pbar.set_postfix({'val_f1': val_f1 / (batch_idx + 1)})
            
            if batch_idx == 0:
                img_vis = img * 0.5 + 0.5
                num_samples = min(4, img_vis.size(0))
                grid_images = []
                preds_vis = pred_probe.sigmoid()
                for i in range(num_samples):
                    grid_images.append(img_vis[i].cpu())
                    grid_images.append(preds_vis[i].float().cpu())
                    grid_images.append(mask[i].float().cpu())
                grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
                writer.add_image("Val/Predictions", grid, epoch)
        avg_f1 = val_f1 / len(dataloader)
        writer.add_scalar("Val/F1", avg_f1, epoch)
        print(f"Validation F1: {avg_f1:.4f}")
        return avg_f1

def reload_checkpoint(checkpoint_path, model, probe, optimiser, scheduler, scaler, num_gpus):
    if os.path.isfile(checkpoint_path):
        print(f"=> Se încarcă checkpoint-ul '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        
        if num_gpus > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            probe.module.load_state_dict(checkpoint['probe_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            probe.load_state_dict(checkpoint['probe_state_dict'])
            
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"=> Reluare cu succes de la epoca {start_epoch} (Best F1: {best_f1:.4f})")
        return start_epoch, best_f1
    else:
        print(f"=> Niciun checkpoint găsit la '{checkpoint_path}'. Antrenarea începe de la zero.")
        return 0, 0.0


def trainScript(model, probe, train_loader, val_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, augment, config, writer):
    checkpoint_dir = config['logging']['checkpoint_dir'].format(experiment_name=config['experiment_name'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    pacience = 300
    epochs_no_improve = 0

    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
    
    # 1. Încărcăm checkpoint-ul dacă există
    start_epoch, best_f1 = reload_checkpoint(last_model_path, model, probe, optimiser, scheduler, scaler, num_gpus)

    # 2. Începem bucla de la start_epoch
    for epoch in range(start_epoch, config['training']['epochs']):
        train_epoch(model, probe, train_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, epoch, augment, config, writer)
        val_f1 = validate_epoch(model, probe, val_loader, f1_metric, epoch, writer)
        
        # Salvăm starea completă (inclusiv scaler)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
            'probe_state_dict': probe.module.state_dict() if num_gpus > 1 else probe.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_f1,
        }
        torch.save(checkpoint, last_model_path)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
            
            backbone_to_save = model.module.backbone if num_gpus > 1 else model.backbone
            torch.save(backbone_to_save.state_dict(), os.path.join(checkpoint_dir, "best_backbone.pth"))
            print(f"--- Backbone & Model salvat la epoca {epoch+1} cu F1: {best_f1:.4f} ---")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            
        if epochs_no_improve >= pacience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# def trainScript(model, probe, train_loader, val_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, augment, config, writer):
#     checkpoint_dir = config['logging']['checkpoint_dir'].format(experiment_name=config['experiment_name'])
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     best_f1 = 0.0
#     num_gpus = torch.cuda.device_count()
#     pacience = 50
#     epochs_no_improve = 0

#     for epoch in range(config['training']['epochs']):
#         train_epoch(model, probe, train_loader, optimiser, scheduler, sigreg, criterion_probe, f1_metric, epoch, augment, config, writer)
#         val_f1 = validate_epoch(model, probe, val_loader, f1_metric, epoch, writer)
        
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
#             'probe_state_dict': probe.module.state_dict() if num_gpus > 1 else probe.state_dict(),
#             'optimizer_state_dict': optimiser.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'best_f1': best_f1,
#         }
#         torch.save(checkpoint, os.path.join(checkpoint_dir, "last_model.pth"))
        
#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             epochs_no_improve = 0
#             torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
            
#             backbone_to_save = model.module.backbone if num_gpus > 1 else model.backbone
#             torch.save(backbone_to_save.state_dict(), os.path.join(checkpoint_dir, "best_backbone.pth"))
#             print(f"--- Backbone & Model salvat la epoca {epoch+1} cu F1: {best_f1:.4f} ---")
#         else:
#             epochs_no_improve += 1
#             print(f"No improvement for {epochs_no_improve} epochs.")
            
#         if epochs_no_improve >= pacience:
#             print(f"Early stopping triggered after {epoch+1} epochs.")
#             break

if __name__ == "__main__":
    config = {
        'experiment_name': 'ConvNexTV2_Tiny_lejepa_linear_probe',
        'logging': {
            'log_dir': 'runs/{experiment_name}',
            'checkpoint_dir': 'checkpoints/{experiment_name}'
        },
        'training': {
            'img_size': 256,
            'batch_size': 10,
            'epochs': 500,
            'lr_probe': 1e-4,
            'lr_model': 1e-5,
            'weight_decay': 5e-2,
            'labda': 0.04,
            'warmup_epochs': 20,
        },
        'model': {
            'encoder_name': 'convnextv2_tiny',
            'proj_dim': 256
        }
    }
    
    writer = SummaryWriter(log_dir=config['logging']['log_dir'].format(experiment_name=config['experiment_name']))
    configCreate(os.path.join(config['logging']['log_dir'].format(experiment_name=config['experiment_name']), 'config.yaml'), config)
    
    train_loader = loader(config['training']['img_size'], config['training']['batch_size'], split='train', mode='lejepa')
    val_loader = loader(config['training']['img_size'], config['training']['batch_size'], split='validation', mode='validation')
    
    model = LeJepaModel(encoder_name=config['model']['encoder_name'], proj_dim=config['model']['proj_dim']).cuda()
    sigreg = SIGReg().cuda()
    augment = augmentariLeJepa(img_size=config['training']['img_size'], local_size=config['training']['img_size']//2)
    
    dummy_input = torch.randn(1, 1, config['training']['img_size'], config['training']['img_size']).cuda()
    with torch.no_grad():
        feats, _ = model(dummy_input)
    encoder_channels = [f.shape[1] for f in feats]
    
    probe = MultiScaleLinearProbe(in_channels_list=encoder_channels, probe_dim=256, num_classes=1).cuda()
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
        probe = nn.DataParallel(probe)

    lr1 = {"params": probe.parameters(), "lr": config['training']['lr_probe'], "weight_decay": config['training']['weight_decay']}
    lr2 = {"params": model.parameters(), "lr": config['training']['lr_model'], "weight_decay": config['training']['weight_decay']}
    opt = torch.optim.AdamW([lr1, lr2])
    
    total_iters_per_epoch = len(train_loader)
    warmup_iters = config['training']['warmup_epochs'] * total_iters_per_epoch
    total_iters = config['training']['epochs'] * total_iters_per_epoch
    
    scheduler1 = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
    scheduler2 = CosineAnnealingLR(opt, T_max=total_iters - warmup_iters, eta_min=1e-6)
    scheduler = SequentialLR(opt, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
    
    criterion_probe = DiceCELoss(include_background=True, sigmoid=True, lambda_ce=1, lambda_dice=1)
    f1_metric = BinaryF1Score().cuda()
    
    trainScript(
        model=model,
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        optimiser=opt,
        scheduler=scheduler,
        sigreg=sigreg,
        criterion_probe=criterion_probe,
        f1_metric=f1_metric,
        augment=augment,
        config=config,
        writer=writer
    )