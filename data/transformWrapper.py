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
from engine.frangiPreproces import FrangiFilter


class TransformsWrapper():
    def __init__(self, dataset, input_size=224, mode='train'):
        self.dataset = dataset
        self.input_size = input_size
        self.mode = mode
        self.root_dir = '.'
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
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
            img = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(transforms.RandomVerticalFlip()(img), p=0.5),
            transforms.RandomApply(transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))(img), p=0.2),
            transforms.RandomApply(transforms.RandomSolarize(threshold=0.5)(img), p=0.2)])
                
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        # img_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_np)
        
        frangi = FrangiFilter(img_np, 256)
        frangi_tensor = torch.from_numpy(frangi).unsqueeze(0).float() 
        
        img_tensor = tf.to_tensor(img_np)
        img_tensor = tf.normalize(img_tensor, [0.5], [0.5])
        
        # combined_img = img_tensor
        # combined_img = torch.cat([img_tensor, frangi_tensor], dim=0)
        img = img_tensor
        mask = tf.to_tensor(mask)

        return img, mask