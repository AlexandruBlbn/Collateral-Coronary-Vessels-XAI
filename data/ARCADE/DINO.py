"""
Multi-crop augmentation dataset for DINOv3 training.
"""

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random


class GaussianBlur:
    """Gaussian blur augmentation."""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
        
    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        # img is a tensor (C, H, W)
        img_np = img.numpy()
        if len(img_np.shape) == 3:
            img_np = img_np[0]  # Grayscale
        
        # Apply Gaussian blur
        ksize = int(sigma * 6) | 1  # Make odd
        ksize = max(3, min(ksize, 23))  # Clamp kernel size
        blurred = cv2.GaussianBlur(img_np, (ksize, ksize), sigma)
        
        return torch.from_numpy(blurred).unsqueeze(0)


class Solarization:
    """Solarization augmentation."""
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def __call__(self, img):
        # img is tensor (C, H, W) in [0, 1]
        mask = img > self.threshold
        img = torch.where(mask, 1.0 - img, img)
        return img


class MultiCropAugmentation:
    """
    Multi-crop augmentation for DINO.
    Creates 2 global crops and N local crops from each image.
    """
    def __init__(
        self,
        global_crop_size: int = 256,
        local_crop_size: int = 96,
        global_crop_scale: tuple = (0.4, 1.0),
        local_crop_scale: tuple = (0.05, 0.4),
        num_local_crops: int = 4
    ):
        self.num_local_crops = num_local_crops
        
        # Global crop transforms (stronger augmentation)
        self.global_transform_1 = T.Compose([
            T.RandomResizedCrop(
                global_crop_size,
                scale=global_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4)
            ], p=0.8),
            T.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=1.0),
        ])
        
        self.global_transform_2 = T.Compose([
            T.RandomResizedCrop(
                global_crop_size,
                scale=global_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4)
            ], p=0.8),
            T.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.1),
            T.RandomApply([Solarization(threshold=0.5)], p=0.2),
        ])
        
        # Local crop transform (smaller patches)
        self.local_transform = T.Compose([
            T.RandomResizedCrop(
                local_crop_size,
                scale=local_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4)
            ], p=0.8),
            T.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
        ])
        
    def __call__(self, image):
        """
        Apply multi-crop augmentation.
        
        Args:
            image: PIL Image or tensor (C, H, W)
            
        Returns:
            crops: list of tensors [global1, global2, local1, ..., localN]
        """
        crops = []
        
        # Convert to tensor if needed
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if len(image.shape) == 2:
                image = image.unsqueeze(0)
        
        # Ensure image is (C, H, W) and float
        if image.dtype != torch.float32:
            image = image.float()
        
        # Global crops
        crops.append(self.global_transform_1(image))
        crops.append(self.global_transform_2(image))
        
        # Local crops
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))
        
        return crops


class ArcadeDatasetDINO(Dataset):
    """
    ARCADE dataset wrapper for DINO training with multi-crop augmentation.
    """
    def __init__(
        self,
        arcade_dataset,
        global_crop_size: int = 256,
        local_crop_size: int = 96,
        global_crop_scale: tuple = (0.4, 1.0),
        local_crop_scale: tuple = (0.05, 0.4),
        num_local_crops: int = 4
    ):
        self.arcade_dataset = arcade_dataset
        self.multi_crop = MultiCropAugmentation(
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            global_crop_scale=global_crop_scale,
            local_crop_scale=local_crop_scale,
            num_local_crops=num_local_crops
        )
        
    def __len__(self):
        return len(self.arcade_dataset)
    
    def __getitem__(self, idx):
        # Get image from base dataset (returns tensor)
        image = self.arcade_dataset[idx]
        
        # If it's a tuple (image, label), take just image
        if isinstance(image, tuple):
            image = image[0]
        
        # Apply multi-crop augmentation
        crops = self.multi_crop(image)
        
        return crops


def collate_dino(batch):
    """
    Custom collate function for DINO multi-crop batches.
    
    Args:
        batch: list of crop lists, each containing [global1, global2, local1, ..., localN]
        
    Returns:
        crops: list of batched tensors
    """
    num_crops = len(batch[0])
    
    # Reorganize: for each crop position, stack all samples
    crops = []
    for i in range(num_crops):
        crop_batch = torch.stack([sample[i] for sample in batch], dim=0)
        crops.append(crop_batch)
    
    return crops