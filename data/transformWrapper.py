# transformWrapper.py

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import cv2
from PIL import Image
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

class TransformsWrapper():
    def __init__(self, dataset, input_size=224, mode='train'):
        self.dataset = dataset
        self.input_size = input_size
        self.mode = mode
        self.root_dir = '.'
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if len(data) == 3:
            img, mask, is_syntax = data
        else:
            img, mask = data
            is_syntax = True 

        img = tf.resize(img, [self.input_size, self.input_size])
        mask = tf.resize(mask, [self.input_size, self.input_size])

        if self.mode == 'train':
            if torch.rand(1).item() > 0.5:
                img = tf.hflip(img)
                mask = tf.hflip(mask)
            if torch.rand(1).item() > 0.5:
                img = tf.vflip(img)
                mask = tf.vflip(mask)
            
            angle = torch.randint(-15, 15, (1,)).item()
            img = tf.rotate(img, angle)
            mask = tf.rotate(mask, angle)
            
        if self.mode == "lejepa":
            lejepa_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.2),
                transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=0.2)
            ])
            img = lejepa_transforms(img)
                
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        img_tensor = tf.to_tensor(img_np)
        img_tensor = tf.normalize(img_tensor, [0.5], [0.5])
        
        mask_tensor = tf.to_tensor(mask)

        return img_tensor, mask_tensor, is_syntax