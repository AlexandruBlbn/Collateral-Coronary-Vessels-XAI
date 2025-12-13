import torch
import numpy as np
from data.ARCADE.dataloader import ArcadeCoronarySegmentation
from torch.utils.data import DataLoader

dataset_path = 'D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\dataset.json'
val_dataset = ArcadeCoronarySegmentation(dataset_path, split='test', augment=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print("Debugging labels și predictions...")
print("=" * 60)

for i, (images, masks) in enumerate(val_loader):
    if i >= 3:
        break
    
    print(f"\nBatch {i+1}:")
    print(f"  Images: shape={images.shape}, min={images.min()}, max={images.max()}, mean={images.mean()}")
    print(f"  Masks (raw): shape={masks.shape}, min={masks.min()}, max={masks.max()}, mean={masks.mean()}")
    print(f"  Masks unique values: {torch.unique(masks)}")
    
    # Normalizeaza ca in training
    images_norm = images.unsqueeze(1).float() / 255.0
    masks_norm = masks.unsqueeze(1).float() / 255.0
    masks_bin = (masks_norm > 0).float()
    
    print(f"  Images (normalized): min={images_norm.min()}, max={images_norm.max()}")
    print(f"  Masks (normalized): min={masks_norm.min()}, max={masks_norm.max()}")
    print(f"  Masks (binarized): min={masks_bin.min()}, max={masks_bin.max()}, sum={masks_bin.sum().item()}")
    print(f"  Ratio positive pixels: {(masks_bin.sum() / masks_bin.numel()).item():.2%}")

print("\n" + "=" * 60)
