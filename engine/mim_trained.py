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

def get_mim_dataloader(json_path, batch_size=32):
    base_dataset = ARCADEDataset(
        json_path=json_path, 
        split='train', 
        task='Unsupervised' 
    )

    mask_gen = MaskGenerator(
        input_size=256, 
        mask_patch_size=32, 
        mask_ratio=0.5
    )

    mim_dataset = ArcadeDatasetMIM(
        arcade_dataset=base_dataset, 
        mask_generator=mask_gen
    )

    loader = DataLoader(
        mim_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    return loader

if __name__ == "__main__":
    loader = get_mim_dataloader(r"D:\Collateral Coronary Vessels XAI\data\ARCADE\processed\dataset.json")
    imgs, masks = next(iter(loader))

    print(f"img: {imgs.shape}") # [32, 1, 256, 256]
    print(f"masca:  {masks.shape}") # [32, 256, 256]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        img = imgs[i].cpu().numpy().squeeze()  # [256, 256]
        mask = masks[i].cpu().numpy()  # [256, 256]
        
        masked_img = img * mask
        
        ax.imshow(masked_img, cmap='gray')
        ax.set_title(f"Sample {i+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()