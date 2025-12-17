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
from zoo.mim import 



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


def train(epochs = config['optimizer']['epochs']):
    running_loss = 0.0
    device = setupSystem()
    optimizier = optimizer(
        lr=config['optimizer']['lr'], 
        weight_decay=config['optimizer']['weight_decay']
    )
    
    for epoch in range(epochs):
        
    







































# if __name__ == "__main__":
#     loader = get_mim_dataloader(r"D:\Collateral Coronary Vessels XAI\data\ARCADE\processed\dataset.json")
#     imgs, masks = next(iter(loader))

#     print(f"img: {imgs.shape}") # [32, 1, 256, 256]
#     print(f"masca:  {masks.shape}") # [32, 256, 256]
    
#     fig, axes = plt.subplots(2, 4, figsize=(12, 6))
#     for i in range(8):
#         row = i // 4
#         col = i % 4
#         ax = axes[row, col]
        
#         img = imgs[i].cpu().numpy().squeeze()  # [256, 256]
#         mask = masks[i].cpu().numpy()  # [256, 256]
        
#         masked_img = img * mask
        
#         ax.imshow(masked_img, cmap='gray')
#         ax.set_title(f"Sample {i+1}")
#         ax.axis('off')
    
#     plt.tight_layout()
#     plt.show()