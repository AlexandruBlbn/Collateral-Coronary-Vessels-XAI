import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from zoo.mim import SimMIM
from data.ARCADE.MIM import MaskGenerator

def visualize_mim_comparison(image_path, models_config, save_path='mim_comparison.png', device='cuda'):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img = cv2.resize(img, (256, 256))
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 256, 256)

    num_models = len(models_config)
    fig, axes = plt.subplots(num_models, 3, figsize=(15, 5 * num_models))
    
    # Handle single model case (axes is 1D array)
    if num_models == 1:
        axes = np.expand_dims(axes, axis=0)

    print(f"--> Processing image: {image_path}")

    for idx, config in enumerate(models_config):
        print(f"  --> Running model: {config['name']}")
        
        # 2. Setup Model
        stride = 16 if 'vit' in config['backbone'] else 32
        model = SimMIM(
            backbone_name=config['backbone'], 
            in_channels=1, 
            encoder_stride=stride
        ).to(device)
        
        # 3. Load Weights
        try:
            checkpoint = torch.load(config['ckpt_path'], map_location=device)
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint for {config['name']}: {e}")
            continue
            
        model.eval()

        # 4. Generate Mask (Specific to model config)
        patch_size = config.get('mask_patch_size', 8)
        mask_gen = MaskGenerator(input_size=256, mask_patch_size=patch_size, mask_ratio=0.5)
        mask = mask_gen().to(device).unsqueeze(0) # (1, 256, 256)

        # 5. Inference
        with torch.no_grad():
            # SimMIM forward returns (loss, reconstruction)
            _, rec = model(img_tensor, mask)

        # 6. Prepare for Plotting
        orig_np = img_tensor.cpu().squeeze().numpy()
        mask_np = mask.cpu().squeeze().numpy()
        rec_np = rec.cpu().squeeze().numpy()
        
        # Create masked view (0 is visible, 1 is masked)
        masked_view = orig_np * (1 - mask_np)
        
        # Combine: Show original where visible, reconstruction where masked
        combined_view = orig_np * (1 - mask_np) + rec_np * mask_np
        combined_view = np.clip(combined_view, 0, 1)
        
        # Row for this model
        ax_orig = axes[idx, 0]
        ax_mask = axes[idx, 1]
        ax_rec = axes[idx, 2]
        
        # Original
        ax_orig.imshow(orig_np, cmap='gray')
        ax_orig.set_title("Original", fontsize=14)
        ax_orig.axis('off')
        
        # Masked Input
        ax_mask.imshow(masked_view, cmap='gray')
        ax_mask.set_title(f"Masked Input\n(Patch: {patch_size})", fontsize=14)
        ax_mask.axis('off')
        
        # Reconstruction (Merged)
        ax_rec.imshow(combined_view, cmap='gray')
        ax_rec.set_title(f"Reconstruction (Merged)\n{config['name']}", fontsize=14, fontweight='bold')
        ax_rec.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"--> Saved comparison to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Example Usage
    models_to_compare = [
        {
            'name': 'ConvNeXt Tiny',
            'backbone': 'convnext_tiny',
            'ckpt_path': 'checkpoints/mim/simmim_pretrain_convnext_v1_full_best.pth',
            'mask_patch_size': 8 # Was trained with 8
        },
        {
            'name': 'SwinV2 Tiny',
            'backbone': 'swinv2_tiny_window16_256',
            'ckpt_path': 'checkpoints/mim/simmim_pretrain_swinv2_v1_full_best.pth',
            'mask_patch_size': 8 # Was trained with 8
        },
         {
            'name': 'ViT Small',
            'backbone': 'vit_small_patch16_224',
            'ckpt_path': 'checkpoints/mim/simmim_pretrain_vit_small_patch16_224_v1_full_best.pth',
            'mask_patch_size': 8 
        }
    ]
    
    # Replace with a real image path from your dataset
    test_image = "data/ARCADE/processed/stenoza/2.png" 
    
    # Check if image exists, otherwise pick one from directory
    if not os.path.exists(test_image):
        # Fallback to find first png
        for root, dirs, files in os.walk("data/ARCADE/processed"):
            for file in files:
                if file.endswith(".png"):
                    test_image = os.path.join(root, file)
                    break
            if os.path.exists(test_image): break
            
    if os.path.exists(test_image):
        visualize_mim_comparison(test_image, models_to_compare)
    else:
        print("No image found to test.")