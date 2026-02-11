import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def apply_clahe(img_rgb):
    """
    Applies CLAHE to the L channel of an RGB image converted to LAB.
    """
    # Convert to LAB
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back to RGB
    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img_clahe

def main():
    parser = argparse.ArgumentParser(description="Preprocess images with CLAHE and overwrite.")
    parser.add_argument("--json_path", type=str, 
                        default="/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json",
                        help="Path to dataset.json")
    parser.add_argument("--root_dir", type=str, 
                        default="/workspace/Collateral-Coronary-Vessels-XAI",
                        help="Root directory for image paths")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        return

    print(f"Loading dataset from {args.json_path}")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
        
    tasks = []
    # Traverse the json structure: split -> source -> id -> info
    for split, split_data in data.items():
        for source, source_data in split_data.items():
            for sample_id, sample_info in source_data.items():
                if 'data' in sample_info:
                    rel_path = sample_info['data']
                    tasks.append(rel_path)
    
    print(f"Found {len(tasks)} images to process.")
    
    for rel_path in tqdm(tasks, desc="Processing"):
        full_path = os.path.join(args.root_dir, rel_path)
        
        if not os.path.exists(full_path):
            if os.path.exists(rel_path):
                full_path = rel_path
            else:
                continue
                
        try:
            # Load image
            img = Image.open(full_path).convert('RGB')
            img_np = np.array(img)
            
            # Apply CLAHE
            img_clahe = apply_clahe(img_np)
            
            # Save (Overwrite)
            Image.fromarray(img_clahe).save(full_path)
            
        except Exception as e:
            print(f"Error processing {full_path}: {e}")

    print(f"Processing complete. Overwritten images in {args.root_dir}")

if __name__ == "__main__":
    main()