#mask generator

import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset
try:
    from data.dataloader import ArcadeDataset
except ImportError:
    from dataloader import ArcadeDataset
from torch.utils.data import DataLoader
import json
from PIL import Image
import matplotlib.pyplot as plt
import os

class MaskGenerator():
        def __init__(self, input_size=256, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
            self.input_size = input_size
            self.mask_patch_size = mask_patch_size
            self.model_patch_size = model_patch_size
            self.mask_ratio = mask_ratio      
            self.rand_size = self.input_size // self.mask_patch_size
            self.scale_mask = self.mask_patch_size // self.model_patch_size
            self.token_count = self.rand_size ** 2
            self.mask_count = int(np.ceil(self.token_count * mask_ratio))
        
        def __call__(self):
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
            mask = np.zeros(self.token_count, dtype=int)
            mask[mask_idx] = 1 
            mask = mask.reshape((self.rand_size, self.rand_size))
            mask = mask.repeat(self.mask_patch_size, axis=0).repeat(self.mask_patch_size, axis=1)
            return torch.from_numpy(mask)
        

class ArcadeDatasetMIM():
    def __init__(self, arcade_dataset, mask_generator):
        self.arcade_dataset = arcade_dataset
        self.mask_generator = mask_generator
        
    def __len__(self):
        return len(self.arcade_dataset)
    
    def __getitem__(self, idx):
        image = self.arcade_dataset[idx]
        mask = self.mask_generator()
        return image, mask
    

    
    
def test():
    json_path = '/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json'
    root_dir = '/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE'
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])
    
    dataset = ArcadeDataset(json_path=json_path, split='train', transform=transform, mode='pretrain', root_dir=root_dir)
    mask_gen = MaskGenerator(input_size=256, mask_patch_size=4, mask_ratio=0.65)
    mim_dataset = ArcadeDatasetMIM(dataset, mask_gen)
    img, mask = mim_dataset[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.savefig('mim_test_plot.png')

if __name__ == "__main__":
    test()
