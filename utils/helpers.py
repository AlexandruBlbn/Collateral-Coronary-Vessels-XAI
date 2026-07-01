# utils/helpers.py
import os
import sys
import json
import re
import csv
import yaml
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import timm
import cv2
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

# Definim curat interfața publică a acestui modul (fără duplicate)
__all__ = [
    "set_seed", "torch", "nn", "torch_f", "Dataset", "DataLoader", "transforms",
    "np", "timm", "cv2", "nib", "Image", "tqdm", "plt",
    "os", "sys", "json", "re", "csv", "yaml", "random", "Path",
    "Dict", "List", "Optional", "Tuple"
]