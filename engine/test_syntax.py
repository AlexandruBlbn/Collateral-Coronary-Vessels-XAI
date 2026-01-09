import sys
import os
import torch
import yaml
import json
import numpy as np
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import importlib.util

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- CONFIGURATION ---
CONFIG_PATH = os.path.join(project_root, 'config', 'UNet_config.yaml')

DATASET_JSON_PATH = r"D:\Collateral Coronary Vessels XAI\data\ARCADE\processed\dataset.json"

# --- 2. IMPORT MODEL DYNAMICALLY ---
model_path = os.path.join(project_root, 'zoo', 'UNetX-S.py')

spec = importlib.util.spec_from_file_location("UNetX_S_Module", model_path)
unext_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unext_module)
UNeXt_S = unext_module.UNeXt_S

# --- 3. UTILS ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_test_augmentation(input_size):
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2()
    ])

def calculate_dice(pred, target):
    """
    Calculates Dice Score for a single sample.
    pred: (H, W) tensor, binary (0 or 1)
    target: (H, W) tensor, binary (0 or 1)
    """
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

# --- 4. DATASET ---
class TestDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.json_path = json_path
        self.transform = transform
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'test' in data and 'syntax' in data['test']:
            print("--> Selecting data['test']['syntax'] subset.")
            data = data['test']['syntax']
        
        self.samples = []
        self._find_samples(data)
        print(f"--> Found {len(self.samples)} samples in JSON.")
        
        if len(self.samples) == 0:
            print("DEBUG: JSON structure snippet (top level keys):", list(data.keys()))
            if isinstance(data, dict) and len(data) > 0:
                first_key = list(data.keys())[0]
                print(f"DEBUG: Inside '{first_key}':", type(data[first_key]))

    def _find_samples(self, node):
        if isinstance(node, dict):
            keys = set(node.keys())
            
            # Define possible keys
            img_candidates = ['image', 'img_path', 'img', 'image_path', 'images', 'data']
            mask_candidates = ['mask', 'mask_path', 'label', 'seg', 'masks', 'segmentation']
            
            img_key = next((k for k in img_candidates if k in keys), None)
            mask_key = next((k for k in mask_candidates if k in keys), None)
            
            if img_key and mask_key:
                img_val = node[img_key]
                mask_val = node[mask_key]
                
                # Case 1: Values are lists of paths (e.g. {'images': [p1, p2], 'masks': [m1, m2]})
                if isinstance(img_val, list) and isinstance(mask_val, list):
                    if len(img_val) == len(mask_val):
                        for i, m in zip(img_val, mask_val):
                            self.samples.append({'image': i, 'mask': m})
                    return
                # Case 2: Values are strings (e.g. {'image': p1, 'mask': m1})
                elif isinstance(img_val, str) and isinstance(mask_val, str):
                    self.samples.append({'image': img_val, 'mask': mask_val})
                    return

            for v in node.values():
                self._find_samples(v)
        elif isinstance(node, list):
            for item in node:
                self._find_samples(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img_path = sample['image']
        mask_path = sample['mask']
        
        # Attempt to resolve relative paths if file not found
        if not os.path.exists(img_path):
            # Try relative to JSON directory
            alt_path = os.path.join(os.path.dirname(self.json_path), img_path)
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                # Try relative to grandparent (e.g. data/ARCADE)
                alt_path = os.path.join(os.path.dirname(os.path.dirname(self.json_path)), img_path)
                if os.path.exists(alt_path):
                    img_path = alt_path

        if not os.path.exists(mask_path):
            alt_path = os.path.join(os.path.dirname(self.json_path), mask_path)
            if os.path.exists(alt_path):
                mask_path = alt_path
            else:
                alt_path = os.path.join(os.path.dirname(os.path.dirname(self.json_path)), mask_path)
                if os.path.exists(alt_path):
                    mask_path = alt_path

        # Load Image (Grayscale)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        # Load Mask (Grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Binarize mask (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask / 255.0 # Normalize to [0, 1]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is tensor and has channel dim if needed (Albumentations usually returns mask without channel dim if input was 2D)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask, os.path.basename(img_path)

# --- 5. MAIN EVALUATION ---
def evaluate():
    # Load Config
    config = load_config(CONFIG_PATH)
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"--> Evaluating on {device}")

    # Initialize Model
    model = UNeXt_S(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        base_channels=config['model']['base_channels'],
        depths=config['model']['depths'],
        mlp_ratio=config['model']['mlp_ratio'],
        drop_rate=0.0, # No dropout for inference
        attention=config['model'].get('attention', False)
    ).to(device)

    # Load Checkpoint
    checkpoint_path = os.path.join(project_root, config['system']['save_dir'], config['experiment_name'], f"{config['experiment_name']}_best_dice.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"--> Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Create results directory
    results_dir = os.path.join(project_root, 'results', config['experiment_name'], 'syntax_test')
    os.makedirs(results_dir, exist_ok=True)
    print(f"--> Saving visualizations to: {results_dir}")

    # Dataset & Loader
    print(f"--> Loading data from: {DATASET_JSON_PATH}")
    try:
        dataset = TestDataset(
            json_path=DATASET_JSON_PATH,
            transform=get_test_augmentation(config['model']['input_size'])
        )
    except FileNotFoundError as e:
        print(e)
        print(f"Please ensure your dataset JSON exists at {DATASET_JSON_PATH}")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    dice_scores = []
    
    print("--> Starting Inference...")
    with torch.no_grad():
        for img, mask, fname in tqdm(loader):
            img = img.to(device)
            mask = mask.to(device)
            
            # Forward pass
            output = model(img)
            
            # Post-processing
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).float()
            
            # Calculate Dice
            dice = calculate_dice(pred, mask)
            dice_scores.append(dice)
            
            # --- VISUALIZATION ---
            # Denormalize Image: (x * 0.5 + 0.5) * 255
            img_np = img[0, 0].cpu().numpy()
            img_np = (img_np * 0.5 + 0.5) * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            
            # Prepare Masks
            mask_np = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
            pred_np = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
            
            mask_color = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
            pred_color = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
            
            # Create Overlay (Red prediction on original image)
            overlay = img_color.copy()
            overlay[pred_np > 0] = [0, 0, 255] # Red BGR
            cv2.addWeighted(overlay, 0.4, img_color, 0.6, 0, img_color)
            
            # Combine: Image (with Overlay) | Ground Truth | Prediction
            combined = np.hstack([img_color, mask_color, pred_color])
            
            # Save
            save_name = f"{os.path.splitext(fname[0])[0]}_dice_{dice:.4f}.png"
            cv2.imwrite(os.path.join(results_dir, save_name), combined)

    avg_dice = np.mean(dice_scores)
    print("\n" + "="*40)
    print(f"RESULTS for {config['experiment_name']}")
    print(f"Data Source: {DATASET_JSON_PATH}")
    print(f"Total Images: {len(dice_scores)}")
    print(f"Average Mean Dice: {avg_dice:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate()