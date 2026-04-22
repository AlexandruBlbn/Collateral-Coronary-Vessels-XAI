import os
import random
import glob
from pathlib import Path
import numpy as np
import cv2
from skimage import io, filters
from PIL import Image
from tqdm import tqdm
import sys

# Ensure XA-SSL-REPO is in the path so we can import if needed, or we just redefine the filter here.
# Actually, redefining it is simpler and safer to avoid import issues.

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.array(img)

def save_image(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def sato_filter(image: np.ndarray, sigmas: list = [1, 2, 3, 4], border: int = 3) -> np.ndarray:
    """
    Apply Sato Hessian-based vesselness filter and zero out borders.
    """
    sato = filters.sato(image.astype(np.uint8), sigmas=sigmas, black_ridges=True, mode="reflect", cval=0)
    
    # Normalize to 0-255 for better visual inspection
    if sato.max() > 0:
        result = (sato / sato.max() * 255).astype(np.uint8)
    else:
        result = sato.astype(np.uint8)
        
    # mask borders
    h, w = result.shape
    result[:border, :] = 0
    result[-border:, :] = 0
    result[:, :border] = 0
    result[:, -border:] = 0
    return result

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE and Gaussian Blurring to smooth background noise."""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    
    # Apply Gaussian Blur to smooth noise
    blurred = cv2.GaussianBlur(cl_img, (5, 5), 0)
    return blurred

def main():
    dataset_dir = "/workspace/Collateral-Coronary-Vessels-XAI/XA-170K/dataset"
    output_dir = "/workspace/Collateral-Coronary-Vessels-XAI/XA-170K/frangi_samples"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Finding all images in the dataset...")
    # Gather all image files
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    all_images = []
    
    for ext in extensions:
        all_images.extend(Path(dataset_dir).rglob(ext))
        all_images.extend(Path(dataset_dir).rglob(ext.upper()))
        
    print(f"Found {len(all_images)} images.")
    
    if len(all_images) == 0:
        print("No images found! Please check the dataset path.")
        return
        
    # Sample 3000 images
    num_samples = min(3000, len(all_images))
    sampled_images = random.sample(all_images, num_samples)
    
    print(f"Processing {num_samples} random images...")
    
    for i, img_path in enumerate(tqdm(sampled_images)):
        try:
            img = load_image(str(img_path))
            
            # Reduce background noise with CLAHE & Gaussian
            img = preprocess_image(img)
            
            # Apply sato/frangi filter
            vesselness = sato_filter(img, sigmas=[1, 2, 3, 4], border=20)
            
            # We don't threshold yet since user said:
            # "I will inspect these images and then convert them to binary mask"
            # We'll save the raw normalized filter output as PNG
            
            out_name = f"{img_path.parent.name}_{img_path.name}"
            out_path = os.path.join(output_dir, out_name)
            
            save_image(vesselness, out_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
