import sys
import os
import torch
import torch.optim as optim
import yaml
import json
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. IMPORTS ---
from data.ARCADE.VasoMIM import VasoMIMMaskGenerator, ArcadeDatasetVasoMIM
from data.ARCADE.dataloader import ARCADEDataset
from zoo.mim import SimMIM
from utils.helpers import set_seed
from torch.utils.data import DataLoader

# --- 3. CONFIGURATION ---
def configLoader(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def flatten_config(config, parent_key='', sep='_'):
    items = []
    for k, v in config.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                v = str(v)
            items.append((new_key, v))
    return dict(items)

CONFIG_PATH = os.path.join(project_root, 'config', 'mim_config.yaml')
config = configLoader(CONFIG_PATH)

# --- 4. DATALOADER ---
def get_vasomim_dataloader(json_path, batch_size=32):
    base_dataset = ARCADEDataset(
        json_path=json_path, 
        split='train', 
        task='Unsupervised' 
    )

    # VasoMIM Mask Generator (folosește logica vasculară)
    mask_gen = VasoMIMMaskGenerator(
        input_size=config['model']['input_size'], 
        mask_patch_size=config['data']['mask_patch_size'], 
        mask_ratio=config['data']['mask_ratio']
    )

    # Dataset Wrapper specific pentru VasoMIM (pasează imaginea la generator)
    mim_dataset = ArcadeDatasetVasoMIM(
        arcade_dataset=base_dataset, 
        mask_generator=mask_gen
    )

    loader = DataLoader(
        mim_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config['data'].get('num_workers', 0), 
        pin_memory=True,
        drop_last=True
    )
    
    return loader

def setupSystem():
    set_seed(config['system']['seed'])
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    return device

# --- 5. VISUALIZATION ---
def save_plotting_samples(model, loader, epoch, save_dir, device, num_samples=40):
    model.eval()
    plot_dir = os.path.join(save_dir, 'plottings_vasomim', f'epoch_{epoch+1}')
    os.makedirs(plot_dir, exist_ok=True)
    
    samples_collected = 0
    tensorboard_imgs = [] 
    
    with torch.no_grad():
        for imgs, masks, anatomy_masks in loader:
            if samples_collected >= num_samples:
                break
                
            imgs = imgs.to(device)
            masks = masks.to(device)
            # anatomy_masks nu e folosit la inferență, doar la loss, dar îl putem vizualiza

            _, rec_imgs = model(imgs, masks)
            
            batch_size = imgs.shape[0]
            for i in range(batch_size):
                if samples_collected >= num_samples:
                    break
                
                orig_img = imgs[i, 0].cpu().numpy()
                rec_raw = rec_imgs[i, 0].cpu().float().numpy()
                mask_np = masks[i].cpu().numpy()
                anatomy_np = anatomy_masks[i].cpu().numpy()
                
                if rec_raw.max() < 0.1 and rec_raw.max() > 0:
                     rec_viz = (rec_raw - rec_raw.min()) / (rec_raw.max() - rec_raw.min() + 1e-6)
                else:
                     rec_viz = rec_raw

                part_original = orig_img * (1 - mask_np)
                part_reconstructed = rec_viz * mask_np
                combined_img = part_original + part_reconstructed

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                axes[0].imshow(orig_img, cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(mask_np, cmap='gray')
                axes[1].set_title('VasoMIM Mask')
                axes[1].axis('off')

                axes[2].imshow(anatomy_np, cmap='gray')
                axes[2].set_title('Anatomy (Vessels)')
                axes[2].axis('off')

                axes[3].imshow(combined_img, cmap='gray')
                axes[3].set_title('Reconstruit')
                axes[3].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(plot_dir, f'sample_{samples_collected}.png')
                plt.savefig(save_path)
                plt.close(fig) 
                
                tensorboard_imgs.extend([imgs[i].cpu(), torch.tensor(combined_img).unsqueeze(0)])
                samples_collected += 1
                
    if len(tensorboard_imgs) > 0:
        grid = torchvision.utils.make_grid(tensorboard_imgs, nrow=2, normalize=True, padding=2)
        return grid
    return None


def train():
    device = setupSystem()
    print(f"--> Training VasoMIM on Device: {device}")
    
    run_name = f"{config['experiment_name']}_vasomim"
    log_dir = os.path.join(project_root, 'runs', run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"--> Logs saving to: {log_dir}")

    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    json_path = os.path.join(project_root, 'data', 'ARCADE', 'processed', 'dataset.json')
    save_dir = config['system']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = get_vasomim_dataloader(json_path, batch_size=config['data']['batch_size'])
    print(f"--> Initializing SimMIM Backbone: {config['model']['backbone']}")
    
    encoder_stride = 32
    if 'vit' in config['model']['backbone']:
        encoder_stride = 16
        
    model = SimMIM(
        backbone_name=config['model']['backbone'],
        in_channels=config['data']['in_channels'],
        encoder_stride=encoder_stride
    ).to(device)

    start_lr = 1e-4
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=start_lr,
        weight_decay=float(config['optimizer']['weight_decay'])
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['optimizer']['epochs'], 
        eta_min=1e-6
    )
    
    best_loss = float('inf')
    epochs = config['optimizer']['epochs']
    global_step = 0
    ANATOMY_WEIGHT = 10.0  # Ponderea pentru reconstrucția vaselor

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for imgs, masks, anatomy_masks in loop:
            imgs = imgs.to(device)
            masks = masks.to(device)
            anatomy_masks = anatomy_masks.to(device).float()
            
            optimizer.zero_grad()

            loss, rec_imgs = model(imgs, masks)
            # SimMIM returnează loss standard și reconstrucția.
            # Ignorăm loss-ul standard și calculăm loss-ul VasoMIM ponderat.
            _, rec_imgs = model(imgs, masks)
            
            # Asigurăm dimensiuni corecte pentru broadcasting (B, 1, H, W)
            if masks.dim() == 3: masks = masks.unsqueeze(1)
            if anatomy_masks.dim() == 3: anatomy_masks = anatomy_masks.unsqueeze(1)
            
            # 1. Calculăm L1 Loss per pixel
            loss_l1 = torch.abs(rec_imgs - imgs)
            
            # 2. Ne interesează doar pixelii mascați (MIM logic)
            loss_masked = loss_l1 * masks
            
            # 3. Separăm în Anatomie (Vase) și Fundal
            # Masca pentru vasele ascunse
            mask_anatomy = masks * anatomy_masks
            # Masca pentru fundalul ascuns
            mask_bg = masks * (1 - anatomy_masks)
            
            # Calculăm mediile (evitând împărțirea la zero)
            count_anatomy = mask_anatomy.sum() + 1e-8
            count_bg = mask_bg.sum() + 1e-8
            
            loss_anatomy = (loss_masked * anatomy_masks).sum() / count_anatomy
            loss_bg = (loss_masked * (1 - anatomy_masks)).sum() / count_bg
            
            # 4. Loss Final Ponderat
            loss = ANATOMY_WEIGHT * loss_anatomy + loss_bg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Loss/anatomy_step', loss_anatomy.item(), global_step)
            writer.add_scalar('Loss/bg_step', loss_bg.item(), global_step)
            global_step += 1
            
        avg_loss = running_loss / len(train_loader)

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.5f} | LR: {current_lr:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            full_state = model.state_dict()
            backbone_state = {k.replace('encoder.', ''): v for k, v in full_state.items() if k.startswith('encoder.')}
            
            best_backbone_path = os.path.join(save_dir, f"{config['experiment_name']}_vasomim_backbone_best.pth")
            torch.save(backbone_state, best_backbone_path)
            
            best_full_path = os.path.join(save_dir, f"{config['experiment_name']}_vasomim_full_best.pth")
            torch.save(full_state, best_full_path)

        tb_grid = save_plotting_samples(model, train_loader, epoch, log_dir, device, num_samples=40)
        if tb_grid is not None:
            writer.add_image('Reconstruction_Samples_Grid', tb_grid, global_step=epoch)

    flat_config = flatten_config(config)
    writer.add_hparams(flat_config, {'hparam/loss': best_loss})
    writer.close()
    print("Train complete")

if __name__ == "__main__":
    train()
