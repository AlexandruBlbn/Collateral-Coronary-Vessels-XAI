from PIL import Image
import numpy as np
import cv2
import torch
import os
import sys
from skimage import io, color
from skimage.filters import frangi
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataloader import ArcadeDataset
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects
from scipy.ndimage import binary_fill_holes
import cv2
import numpy as np


dataset = ArcadeDataset(json_path="data/ARCADE/processed/dataset.json", split='train', mode='pretrain', root_dir='.')

def FrangiFilter(img, img_size=256): 
    img_np = np.array(img)
    img_np = cv2.GaussianBlur(img_np, (7,7), sigmaX=3)
    pad=16
    img_np = img_np[pad:-pad, pad:-pad]
    img_np = cv2.resize(img_np, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img_np = img_np.astype('float32')
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    vesselness = frangi(img_np,sigmas=range(1, 16, 2), alpha=0.5, beta=1, gamma=10, mode='reflect', black_ridges=True)
    p_low, p_high = np.percentile(vesselness, (1, 99.9))
    vesselness_clipped = np.clip(vesselness, p_low, p_high)
    v_norm = (vesselness_clipped - p_low) / (p_high - p_low + 1e-8)
    threshold_value = 0.1
    vessel_mask_bool = v_norm > threshold_value
    vessel_mask_clean = remove_small_objects(vessel_mask_bool, max_size=70)
    gamma_value = 0.5
    vesselness_enhanced = np.power(v_norm, gamma_value)
    threshold_value = 0.05
    vessel_mask = vessel_mask_clean.astype(float)
    vessel_mask = (v_norm > threshold_value).astype(float)
    vesselness_final = vesselness_enhanced * vessel_mask
    vesselness_final = cv2.GaussianBlur(vesselness_final, (3, 3), 0)
    return vesselness_final



item = dataset[6]
img, label = item

img_np = np.array(img)
V = FrangiFilter(img_np)

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_np, cmap='gray'); plt.title('1. Input Original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(V, cmap='gray'); plt.title('2. Frangi (Trecerea 1)')
plt.axis('off')

plt.savefig('frangi_output.png', bbox_inches='tight')
