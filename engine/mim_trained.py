import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
from data.ARCADE.MIM import MaskGenerator, ArcadeDatasetMIM
from data.ARCADE.dataloader import ARCADEDataset
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch.optim as optim
from utils.helpers import set_seed
import torch
import tqdm
from zoo.mim import SimMIM



def configLoader(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = configLoader(r'D:\Collateral Coronary Vessels XAI\config\mim_config.yaml')

def get_mim_dataloader(json_path, batch_size=32):
    base_dataset = ARCADEDataset(
        json_path=json_path, 
        split='train', 
        task='Unsupervised' 
    )

    mask_gen = MaskGenerator(
        input_size=config['model']['input_size'], 
        mask_patch_size=config['data']['mask_patch_size'], 
        mask_ratio=config['data']['mask_ratio']
    )

    mim_dataset = ArcadeDatasetMIM(
        arcade_dataset=base_dataset, 
        mask_generator=mask_gen
    )

    loader = DataLoader(
        mim_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return loader

def optimizer(lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay']):
    optimizer = optim.AdamW(
        params=[],  
        lr=lr,
        weight_decay=weight_decay
    )
    return optimizer

def setupSystem():
    set_seed(config['system']['seed'])
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    return device

    

data_loader = get_mim_dataloader(
    json_path=r'D:\Collateral Coronary Vessels XAI\data\ARCADE\processed\dataset.json',
    batch_size=config['data']['batch_size']
)

#epochs = config['optimizer']['epochs']
def train():
    device = setupSystem()
    print("device: "    , device)
    
    json_path = r'D:\Collateral Coronary Vessels XAI\data\ARCADE\processed\dataset.json'
    train_loader = get_mim_dataloader(
        json_path=json_path,
        batch_size=config['data']['batch_size']
    )
    model = SimMIM(
        backbone_name=config['model']['backbone_name'],
        in_channels=config['model']['in_channels']
    ).to(device)
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=float(config['optimizer']['lr']),
        weight_decay=float(config['optimizer']['weight_decay'])
    )
    epochs = config['optimizer']['epochs']
    save_dir = config['system']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        loop = tqdm.tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for imgs, masks in loop:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            loss, img_recreate = model(imgs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
=