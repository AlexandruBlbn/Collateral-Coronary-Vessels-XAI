import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
from sklearn.decomposition import PCA

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from zoo.mim import SimMIM
from data.ARCADE.MIM import MaskGenerator


# =============================================================================
# DINO Visualization Functions (Similar to DINOv3 paper figures)
# =============================================================================

def get_patch_features(model, image, device='cuda'):
    """
    Extract patch-level features from a DINO model.
    
    Args:
        model: DINOv3 model or backbone
        image: (1, C, H, W) tensor
        device: torch device
        
    Returns:
        patch_features: (num_patches, embed_dim) tensor
        grid_size: (H, W) tuple of patch grid dimensions
    """
    model.eval()
    
    # Get the backbone
    if hasattr(model, 'student'):
        backbone = model.student.backbone
    elif hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        backbone = model
    
    # Get inner model if wrapped
    if hasattr(backbone, 'model'):
        inner_model = backbone.model
    else:
        inner_model = backbone
    
    with torch.no_grad():
        image = image.to(device)
        
        # Resize to 256x256 if needed
        if image.shape[-1] != 256 or image.shape[-2] != 256:
            image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Get features from backbone
        features = backbone(image)
        
        # Handle different output formats
        if len(features.shape) == 4:  # (B, C, H, W) - ConvNeXt/Swin
            B, C, H, W = features.shape
            # Reshape to (num_patches, embed_dim)
            patch_features = features[0].permute(1, 2, 0).reshape(-1, C)
            grid_size = (H, W)
        elif len(features.shape) == 3:  # (B, N, D) - ViT tokens
            B, N, D = features.shape
            # Remove CLS token if present
            if hasattr(inner_model, 'cls_token'):
                patch_features = features[0, 1:]  # Skip CLS
                N = N - 1
            else:
                patch_features = features[0]
            H = W = int(N ** 0.5)
            grid_size = (H, W)
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
            
    return patch_features, grid_size


def visualize_cosine_similarity(model, image, save_path=None, device='cuda', query_point=None):
    """
    Visualize cosine similarity between a query patch and all other patches.
    Similar to Figure 6 in DINOv3 paper.
    
    Args:
        model: DINOv3 model or backbone
        image: (1, C, H, W) tensor or numpy array
        save_path: path to save figure
        device: torch device
        query_point: (row, col) in patch grid, or None for center
        
    Returns:
        similarity_map: (H, W) numpy array
    """
    # Convert numpy to tensor if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        image = image / 255.0 if image.max() > 1 else image
    
    image = image.to(device)
    
    # Get patch features
    patch_features, grid_size = get_patch_features(model, image, device)
    H, W = grid_size
    
    # Normalize features for cosine similarity
    patch_features = F.normalize(patch_features, dim=-1)
    
    # Default query point is center
    if query_point is None:
        query_point = (H // 2, W // 2)
    
    query_idx = query_point[0] * W + query_point[1]
    query_feature = patch_features[query_idx:query_idx+1]
    
    # Compute cosine similarity
    similarity = torch.mm(query_feature, patch_features.T)  # (1, num_patches)
    similarity_map = similarity.reshape(H, W).cpu().numpy()
    
    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        orig = image[0, 0].cpu().numpy() if image.shape[1] == 1 else image[0].permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(orig, cmap='gray' if len(orig.shape) == 2 else None)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Mark query point on original
        patch_h = 256 // H
        patch_w = 256 // W
        rect_y = query_point[0] * patch_h
        rect_x = query_point[1] * patch_w
        from matplotlib.patches import Rectangle
        rect = Rectangle((rect_x, rect_y), patch_w, patch_h, 
                         linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        
        # Similarity heatmap
        sim_resized = cv2.resize(similarity_map, (256, 256), interpolation=cv2.INTER_NEAREST)
        axes[1].imshow(sim_resized, cmap='inferno', vmin=0, vmax=1)
        axes[1].set_title('Cosine Similarity', fontsize=14)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(orig, cmap='gray' if len(orig.shape) == 2 else None)
        axes[2].imshow(sim_resized, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
        axes[2].set_title('Overlay', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"--> Saved cosine similarity visualization to {save_path}")
    
    return similarity_map


def visualize_pca_features(model, image, save_path=None, device='cuda', n_components=3):
    """
    Visualize PCA of patch features mapped to RGB.
    Similar to Figure 4 in DINOv3 paper.
    
    Args:
        model: DINOv3 model or backbone
        image: (1, C, H, W) tensor or numpy array
        save_path: path to save figure
        device: torch device
        n_components: number of PCA components (3 for RGB)
        
    Returns:
        pca_rgb: (H, W, 3) numpy array
    """
    # Convert numpy to tensor if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        image = image / 255.0 if image.max() > 1 else image
    
    image = image.to(device)
    
    # Get patch features
    patch_features, grid_size = get_patch_features(model, image, device)
    H, W = grid_size
    
    # Apply PCA
    features_np = patch_features.cpu().numpy()
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_np)
    
    # Normalize each component to [0, 1] for RGB
    pca_normalized = np.zeros_like(pca_features)
    for i in range(n_components):
        col = pca_features[:, i]
        pca_normalized[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)
    
    # Reshape to spatial grid
    pca_rgb = pca_normalized.reshape(H, W, n_components)
    
    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        orig = image[0, 0].cpu().numpy() if image.shape[1] == 1 else image[0].permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(orig, cmap='gray' if len(orig.shape) == 2 else None)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # PCA RGB (upsampled)
        pca_resized = cv2.resize(pca_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
        axes[1].imshow(pca_resized)
        axes[1].set_title('PCA Features (RGB)', fontsize=14)
        axes[1].axis('off')
        
        # PCA with bilinear interpolation (smoother)
        pca_smooth = cv2.resize(pca_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        axes[2].imshow(pca_smooth)
        axes[2].set_title('PCA Features (Smooth)', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"--> Saved PCA visualization to {save_path}")
    
    return pca_rgb


def visualize_attention_grid(model, image, save_path=None, device='cuda', num_queries=9):
    """
    Visualize cosine similarity for multiple query points in a grid.
    Shows where the model "looks" from different positions.
    
    Args:
        model: DINOv3 model or backbone
        image: (1, C, H, W) tensor or numpy array
        save_path: path to save figure
        device: torch device
        num_queries: number of query points (should be a perfect square)
    """
    # Convert numpy to tensor if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        image = image / 255.0 if image.max() > 1 else image
    
    image = image.to(device)
    
    # Get patch features
    patch_features, grid_size = get_patch_features(model, image, device)
    H, W = grid_size
    
    # Normalize features
    patch_features = F.normalize(patch_features, dim=-1)
    
    # Create query points grid
    n = int(np.sqrt(num_queries))
    query_rows = np.linspace(1, H-2, n, dtype=int)
    query_cols = np.linspace(1, W-2, n, dtype=int)
    
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    
    orig = image[0, 0].cpu().numpy() if image.shape[1] == 1 else image[0].permute(1, 2, 0).cpu().numpy()
    
    for i, row in enumerate(query_rows):
        for j, col in enumerate(query_cols):
            query_idx = row * W + col
            query_feature = patch_features[query_idx:query_idx+1]
            
            similarity = torch.mm(query_feature, patch_features.T)
            similarity_map = similarity.reshape(H, W).cpu().numpy()
            sim_resized = cv2.resize(similarity_map, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            ax = axes[i, j] if n > 1 else axes
            ax.imshow(orig, cmap='gray' if len(orig.shape) == 2 else None)
            ax.imshow(sim_resized, cmap='inferno', alpha=0.65, vmin=0, vmax=1)
            
            # Mark query point
            patch_h = 256 // H
            patch_w = 256 // W
            ax.scatter(col * patch_w + patch_w//2, row * patch_h + patch_h//2, 
                      c='cyan', s=50, marker='x', linewidths=2)
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"--> Saved attention grid to {save_path}")
    else:
        plt.show()


def compare_backbones_visualization(image_path, models_config, save_dir='visualizations', device='cuda'):
    """
    Compare visualizations across different backbones.
    
    Args:
        image_path: path to input image
        models_config: list of dicts with 'name', 'model' (loaded model)
        save_dir: directory to save outputs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (256, 256))
    
    num_models = len(models_config)
    
    # Create comparison figure for PCA
    fig_pca, axes_pca = plt.subplots(1, num_models + 1, figsize=(4 * (num_models + 1), 4))
    axes_pca[0].imshow(img, cmap='gray')
    axes_pca[0].set_title('Input', fontsize=12)
    axes_pca[0].axis('off')
    
    # Create comparison figure for cosine similarity
    fig_cos, axes_cos = plt.subplots(1, num_models + 1, figsize=(4 * (num_models + 1), 4))
    axes_cos[0].imshow(img, cmap='gray')
    axes_cos[0].set_title('Input', fontsize=12)
    axes_cos[0].axis('off')
    
    for idx, config in enumerate(models_config):
        model = config['model']
        name = config['name']
        
        # PCA visualization
        pca_rgb = visualize_pca_features(model, img, device=device)
        H, W = pca_rgb.shape[:2]
        pca_resized = cv2.resize(pca_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        axes_pca[idx + 1].imshow(pca_resized)
        axes_pca[idx + 1].set_title(name, fontsize=12)
        axes_pca[idx + 1].axis('off')
        
        # Cosine similarity (center query)
        sim_map = visualize_cosine_similarity(model, img, device=device)
        sim_resized = cv2.resize(sim_map, (256, 256), interpolation=cv2.INTER_LINEAR)
        axes_cos[idx + 1].imshow(img, cmap='gray')
        axes_cos[idx + 1].imshow(sim_resized, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
        axes_cos[idx + 1].set_title(name, fontsize=12)
        axes_cos[idx + 1].axis('off')
    
    fig_pca.tight_layout()
    fig_pca.savefig(os.path.join(save_dir, 'pca_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_pca)
    
    fig_cos.tight_layout()
    fig_cos.savefig(os.path.join(save_dir, 'cosine_similarity_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_cos)
    
    print(f"--> Saved comparison visualizations to {save_dir}/")


# =============================================================================
# MIM Visualization (existing)
# =============================================================================

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


# =============================================================================
# DINO Visualization Example
# =============================================================================
def demo_dino_visualization():
    """
    Demo function showing how to use DINO visualizations.
    """
    from zoo.dino import DINOv3
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load a trained DINO model
    # Option 1: Load full checkpoint
    checkpoint_path = "checkpoints/dino/dinov3_convnext_checkpoint_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Create model
    model = DINOv3(
        backbone_name='convnext_tiny',
        in_channels=1,
        embed_dim=None,
        projection_dim=8192,
        hidden_dim=2048,
        bottleneck_dim=512,
        use_gram_anchoring=False
    ).to(device)
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.student.load_state_dict(ckpt['student_state_dict'])
    model.eval()
    
    # Find test image
    test_image = "data/ARCADE/processed/stenoza/2.png"
    if not os.path.exists(test_image):
        for root, dirs, files in os.walk("data/ARCADE/processed"):
            for file in files:
                if file.endswith(".png"):
                    test_image = os.path.join(root, file)
                    break
            if os.path.exists(test_image):
                break
    
    if not os.path.exists(test_image):
        print("No test image found")
        return
    
    # Load image
    img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    
    # Create output directory
    os.makedirs("visualizations/dino", exist_ok=True)
    
    # Generate visualizations
    print("--> Generating DINO visualizations...")
    
    # 1. PCA Features (like Figure 4)
    visualize_pca_features(model, img, 
                          save_path="visualizations/dino/pca_features.png",
                          device=device)
    
    # 2. Cosine Similarity from center (like Figure 6)
    visualize_cosine_similarity(model, img,
                               save_path="visualizations/dino/cosine_similarity_center.png",
                               device=device)
    
    # 3. Attention grid (multiple query points)
    visualize_attention_grid(model, img,
                            save_path="visualizations/dino/attention_grid.png",
                            device=device,
                            num_queries=9)
    
    print("--> Done! Check visualizations/dino/ folder")


if __name__ == "__main__":
    # Run MIM comparison
    # visualize_mim_comparison(test_image, models_to_compare)
    
    # Or run DINO visualization demo
    demo_dino_visualization()