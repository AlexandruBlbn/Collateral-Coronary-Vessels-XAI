"""
Quick test script for LeJepa setup - verifies configuration and data loading
"""
import sys
import os

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import yaml
from data.ARCADE.dataloader import ARCADEDataset
from data.ARCADE.LeJepa import ArcadeDatasetLeJepa, collate_lejepa
from torch.utils.data import DataLoader

print("="*80)
print("LeJepa Configuration Test")
print("="*80)

# Load config
config_path = os.path.join(project_root, 'config', 'lejepa_config.yaml')
print(f"\n1. Loading config from: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"   ✓ Config loaded successfully")
print(f"   - Experiment: {config['experiment_name']}")
print(f"   - Backbone: {config['model']['backbone']}")
print(f"   - Image size: {config['data']['image_size']}")
print(f"   - Batch size: {config['data']['batch_size']}")
print(f"   - Num views: {config['data']['num_views']}")
print(f"   - Lambda (λ): {config['optimizer']['lamb']}")

# Test dataset loading
print("\n2. Testing dataset loading...")
json_path = os.path.join(project_root, 'data/ARCADE/processed/dataset.json')

try:
    base_dataset = ARCADEDataset(
        json_path=json_path,
        split='train',
        task='Unsupervised'
    )
    print(f"   ✓ Base dataset loaded: {len(base_dataset)} samples")
    
    # Wrap with LeJepa augmentation
    lejepa_dataset = ArcadeDatasetLeJepa(
        arcade_dataset=base_dataset,
        image_size=config['data']['image_size'],
        num_views=config['data']['num_views'],
        is_training=True
    )
    print(f"   ✓ LeJepa dataset wrapper created")
    
    # Test dataloader
    loader = DataLoader(
        lejepa_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        collate_fn=collate_lejepa,
        drop_last=True
    )
    print(f"   ✓ DataLoader created: {len(loader)} batches per epoch")
    
    # Test one batch
    print("\n3. Testing batch loading...")
    views = next(iter(loader))
    print(f"   ✓ Batch loaded successfully")
    print(f"   - Shape: {views.shape}")  # (B, V, C, H, W)
    print(f"   - Expected: ({config['data']['batch_size']}, {config['data']['num_views']}, 3, {config['data']['image_size']}, {config['data']['image_size']})")
    print(f"   - Dtype: {views.dtype}")
    print(f"   - Range: [{views.min():.3f}, {views.max():.3f}]")
    
    # Verify shape
    B, V, C, H, W = views.shape
    assert B == config['data']['batch_size'], f"Batch size mismatch: {B} != {config['data']['batch_size']}"
    assert V == config['data']['num_views'], f"Num views mismatch: {V} != {config['data']['num_views']}"
    assert C == 3, f"Channels mismatch: {C} != 3"
    assert H == config['data']['image_size'], f"Height mismatch: {H} != {config['data']['image_size']}"
    assert W == config['data']['image_size'], f"Width mismatch: {W} != {config['data']['image_size']}"
    
    print("\n4. Testing model creation...")
    import timm
    
    backbone = timm.create_model(
        config['model']['backbone'],
        pretrained=False,
        num_classes=config['model']['num_classes'],
        drop_path_rate=config['model']['drop_path_rate'],
        img_size=config['data']['image_size'],
        in_chans=3
    )
    print(f"   ✓ Backbone created: {config['model']['backbone']}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   ✓ Device: {device}")
    
    backbone = backbone.to(device)
    test_input = views[0, 0].unsqueeze(0).to(device)  # (1, C, H, W)
    
    with torch.no_grad():
        output = backbone(test_input)
    
    print(f"   ✓ Forward pass successful")
    print(f"   - Input shape: {test_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Expected output dim: {config['model']['num_classes']}")
    
    # Test SIGReg
    print("\n5. Testing SIGReg loss...")
    from engine.LeJepa_trained import SIGReg
    
    sigreg = SIGReg(
        knots=config['model']['sigreg']['knots'],
        t_max=config['model']['sigreg']['t_max'],
        num_random_features=config['model']['sigreg']['num_random_features']
    ).to(device)
    
    # Create dummy projections (V, B, D)
    dummy_proj = torch.randn(
        config['data']['num_views'],
        config['data']['batch_size'],
        config['model']['proj_dim'],
        device=device
    )
    
    with torch.no_grad():
        loss = sigreg(dummy_proj)
    
    print(f"   ✓ SIGReg loss computed: {loss.item():.4f}")
    
    print("\n" + "="*80)
    print("✓ All tests passed! LeJepa is ready for training.")
    print("="*80)
    print("\nTo start training, run:")
    print("  python engine/LeJepa_trained.py")
    print("\nTo monitor training:")
    print("  tensorboard --logdir runs/lejepa_vit_small/")
    print("="*80)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
