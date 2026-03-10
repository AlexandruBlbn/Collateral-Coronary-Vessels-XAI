import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import cv2
from skimage.filters import frangi
from PIL import Image
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# from engine.frangiPreproces import FrangiFilter
from torch.utils.data import Dataset


class TransformsWrapper(Dataset):
    def __init__(self, dataset, input_size=224, mode='train'):
        self.dataset = dataset
        self.input_size = input_size
        self.mode = mode
        self.root_dir = '.'
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # ---> Citim dinamic output-ul dataset-ului de bază <---
        dataset_output = self.dataset[idx]
        
        if len(dataset_output) == 3:
            img, label, is_syntax = dataset_output
        else:
            img, label = dataset_output
            is_syntax = None
        
        img = img.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        mask = label.resize((self.input_size, self.input_size), resample=Image.NEAREST)

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
            if torch.rand(1).item() > 0.5:
                img = tf.hflip(img)
                mask = tf.hflip(mask)
            if torch.rand(1).item() > 0.5:
                img = tf.vflip(img)
                mask = tf.vflip(mask)
                
            if torch.rand(1).item() > 0.8:
                sigma = torch.empty(1).uniform_(0.1, 2.0).item()
                img = tf.gaussian_blur(img, kernel_size=[7, 7], sigma=[sigma, sigma])
            if torch.rand(1).item() > 0.8:
                img = tf.solarize(img, threshold=128) 
                
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        img_tensor = tf.to_tensor(img_np)
        img_tensor = tf.normalize(img_tensor, [0.5], [0.5])
        
        img = img_tensor
        mask = tf.to_tensor(mask)

        # ---> Returnăm dinamic <---
        if is_syntax is not None:
            return img, mask, torch.tensor(is_syntax, dtype=torch.bool)
            
        return img, mask