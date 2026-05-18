import sys, os
sys.path.insert(0, '.')
from train_jepa import JEPAModel, generate_block_mask, apply_mask
import torch

model = JEPAModel(backbone_name='tf_efficientnetv2_s', in_channels=1, pretrained=False)

B, C, H, W = 2, 1, 256, 256
images = torch.randn(B, C, H, W)
conf = torch.rand(B, 1, 32, 32) * 0.5
masks = torch.stack([generate_block_mask(H, W) for _ in range(B)], dim=0)
masked = apply_mask(images, masks)
out = model(masked, images, masks, conf, alpha=1.0)
print(f'Loss: {out["loss"]:.6f}')
print(f'Per layer: {out["loss_per_layer"]}')

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total:,} ({total/1e6:.2f}M)')
print(f'Trainable: {trainable:,} ({trainable/1e6:.2f}M)')
