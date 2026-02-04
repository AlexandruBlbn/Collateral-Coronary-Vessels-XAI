"""
LeJepa dataset wrapper for coronary angiography images.
Implements multi-view augmentation strategy for self-supervised learning.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
from PIL import Image
import random


class LeJepaAugmentation:
    """
    Multi-view augmentation for LeJepa on coronary angiography images.
    Creates V different augmented views of each image.
    """
    def __init__(self, image_size: int = 256, num_views: int = 2):
        """
        Args:
            image_size: Target image size (will be resized to image_size x image_size)
            num_views: Number of augmented views to generate per image (V parameter)
        """
        self.num_views = num_views
        self.image_size = image_size
        
        # Augmentation pipeline for training views
        self.aug = v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),  # Additional flip for medical images
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Test/validation transform (single view, no heavy augmentation)
        self.test = v2.Compose([
            v2.Resize(image_size),
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, image, is_training=True):
        """
        Apply multi-view augmentation.
        
        Args:
            image: PIL Image or tensor
            is_training: If True, apply augmentation; if False, use test transform
            
        Returns:
            views: tensor of shape (V, C, H, W) or (1, C, H, W) for test
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = np.stack([image, image, image], axis=-1)
            image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL
            if image.dim() == 2:
                image = image.unsqueeze(0).repeat(3, 1, 1)
            elif image.dim() == 3 and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            image = TF.to_pil_image(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate views
        if is_training and self.num_views > 1:
            views = torch.stack([self.aug(image) for _ in range(self.num_views)])
        else:
            # For validation/test, return single view
            views = self.test(image).unsqueeze(0)
        
        return views


class ArcadeDatasetLeJepa(Dataset):
    """
    ARCADE dataset wrapper for LeJepa training with multi-view augmentation.
    Supports unsupervised pretraining including extra data.
    """
    def __init__(
        self,
        arcade_dataset,
        image_size: int = 256,
        num_views: int = 2,
        is_training: bool = True
    ):
        """
        Args:
            arcade_dataset: Base ARCADE dataset with task='Unsupervised'
            image_size: Target image size (256x256 for coronary angiography)
            num_views: Number of augmented views per image (V parameter)
            is_training: Whether in training mode (applies augmentation)
        """
        self.arcade_dataset = arcade_dataset
        self.is_training = is_training
        self.augmentation = LeJepaAugmentation(
            image_size=image_size,
            num_views=num_views
        )
        
    def __len__(self):
        return len(self.arcade_dataset)
    
    def __getitem__(self, idx):
        """
        Get item with multi-view augmentation.
        
        Returns:
            views: tensor of shape (V, C, H, W) containing V augmented views
        """
        # Get image from base dataset
        item = self.arcade_dataset[idx]
        
        # Handle different return types from base dataset
        if isinstance(item, tuple):
            image = item[0]  # (image, label) or (image, None)
        else:
            image = item
        
        # Apply multi-view augmentation
        views = self.augmentation(image, is_training=self.is_training)
        
        return views


def collate_lejepa(batch):
    """
    Custom collate function for LeJepa multi-view batches.
    
    Args:
        batch: list of view tensors, each of shape (V, C, H, W)
        
    Returns:
        views: tensor of shape (B, V, C, H, W) where B is batch size
    """
    # Stack all samples: (B, V, C, H, W)
    views = torch.stack(batch, dim=0)
    return views
    