#mask generator

import torch
import torch.nn as nn
from dataloader import ARCADEDataset
import torchvision
import numpy as np

class MaskGenerator():
        def __init__(self, input_size=256, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
            self.input_size = input_size
            self.mask_patch_size = mask_patch_size
            self.model_patch_size = model_patch_size
            self.mask_ratio = mask_ratio
            
            self.rand_size = self.input_size // self.mask_patch_size