import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def zscore_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

def minmax_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data

def zscore_normalize_image(img_array):
    mean = np.mean(img_array)
    std = np.std(img_array)
    if std == 0:
        return img_array - mean
    return (img_array - mean) / std

pathJson = "data/ARCADE/processed/dataset.json"

def dataPlot():
    with open(pathJson, 'r') as f:
        data = json.load(f)
        
    meanStenosis = []
    meanData = []
    meanDifference = []
    
    # Per-image normalized
    perImageMeanStenosis = []
    perImageMeanData = []
    perImageMeanDifference = []
    
    # Normalized variants
    normalizedStenosis = []
    normalizedData = []
    normalizedDifference = []
    
    minmaxStenosis = []
    minmaxData = []
    minmaxDifference = []
    
    for patient_id, pacienti in data['train']['stenoza'].items():
        image = Image.open(os.path.join("data/ARCADE/", pacienti['data'])).convert('L')
        label = Image.open(os.path.join("data/ARCADE/", pacienti['label'])).convert('L')
        
        image_array = np.array(image)
        label_array = np.array(label)
        img_normalized = zscore_normalize_image(image_array)
        
        
        mask = label_array > 0
        
        if np.any(mask):
            mean_image = np.mean(img_normalized[mask])
            mean_label = np.mean(label_array[mask])
            mean_difference = mean_image - mean_label
            
            # Z-score normalizat
            normalized_image = zscore_normalize_image(image_array)[mask]
            normalized_label = zscore_normalize_image(label_array)[mask]
            
            # Minmax normalizat
            minmax_image = minmax_normalize(image_array)[mask]
            minmax_label = minmax_normalize(label_array)[mask]
            
            meanStenosis.append(mean_label)
            meanData.append(mean_image)
            meanDifference.append(mean_difference)
            
            normalizedStenosis.append(np.mean(normalized_label))
            normalizedData.append(np.mean(normalized_image))
            normalizedDifference.append(np.mean(normalized_image) - np.mean(normalized_label))
            
            minmaxStenosis.append(np.mean(minmax_label))
            minmaxData.append(np.mean(minmax_image))
            minmaxDifference.append(np.mean(minmax_image) - np.mean(minmax_label))
            
            perImageMeanStenosis.append(mean_label)
            perImageMeanData.append(mean_image)
            perImageMeanDifference.append(mean_difference)

    for patient_id, pacienti in data['train']['stenoza'].items():
        image = Image.open(os.path.join("data/ARCADE/", pacienti['data'])).convert('L')
        label = Image.open(os.path.join("data/ARCADE/", pacienti['label'])).convert('L')
        
        image_array = np.array(image)
        label_array = np.array(label)
        img_normalized = zscore_normalize_image(image_array)
        
        
        mask = label_array > 0
        
        if np.any(mask):
            mean_image = np.mean(img_normalized[mask])
            mean_label = np.mean(label_array[mask])
            mean_difference = mean_image - mean_label
            
            meanStenosis.append(mean_label)
            meanData.append(mean_image)
            meanDifference.append(mean_difference)
            
            perImageMeanStenosis.append(mean_label)
            perImageMeanData.append(mean_image)
            perImageMeanDifference.append(mean_difference)

    for patient_id, pacienti in data['test']['stenoza'].items():
        image = Image.open(os.path.join("data/ARCADE/", pacienti['data'])).convert('L')
        label = Image.open(os.path.join("data/ARCADE/", pacienti['label'])).convert('L')
        
        image_array = np.array(image)
        label_array = np.array(label)
        img_normalized = zscore_normalize_image(image_array)
        
        
        mask = label_array > 0
        
        if np.any(mask):
            mean_image = np.mean(img_normalized[mask])
            mean_label = np.mean(label_array[mask])
            mean_difference = mean_image - mean_label
            
            meanStenosis.append(mean_label)
            meanData.append(mean_image)
            meanDifference.append(mean_difference)
            
            perImageMeanStenosis.append(mean_label)
            perImageMeanData.append(mean_image)
            perImageMeanDifference.append(mean_difference)
            
    fig, axe = plt.subplots(4, 3,  figsize=(35, 35))
    axe[0, 0].set_title('Intensitate regiuni cu stenoza')
    axe[0, 0].scatter(range(len(meanStenosis)), meanStenosis)
    axe[0, 0].set_xlabel("nr pacienti")
    axe[0, 0].set_ylabel("Intensitate medie")
    axe[0, 1].set_title('Intensitate medie imagine')
    axe[0, 1].scatter(range(len(meanData)), meanData)
    axe[0, 1].set_xlabel("nr pacienti")
    axe[0, 1].set_ylabel("Intensitate medie")
    axe[0, 2].set_title('Diferenta intensitate medie stenoza - intensitate medie imagine')
    axe[0, 2].scatter(range(len(meanDifference)), meanDifference)
    axe[0, 2].set_xlabel("nr pacienti")
    axe[0, 2].set_ylabel("Diferenta")
    axe[1, 0].set_title('Intensitate regiuni cu stenoza normalizata')
    axe[1, 0].scatter(range(len(normalizedStenosis)), normalizedStenosis)
    axe[1, 0].set_xlabel("nr pacienti")
    axe[1, 0].set_ylabel("Intensitate medie")
    
    axe[1, 1].set_title('Intensitate medie imagine normalizata')
    axe[1, 1].scatter(range(len(normalizedData)), normalizedData)
    axe[1, 1].set_xlabel("nr pacienti")
    axe[1, 1].set_ylabel("Intensitate medie")
   # axe[1, 2].set_title('Diferenta intensitate medie stenoza - intensitate medie imagine normalizata')
    axe[1, 2].scatter(range(len(normalizedDifference)), normalizedDifference)
    axe[1, 2].set_xlabel("nr pacienti")
    axe[1, 2].set_ylabel("Diferenta")
    
    axe[2, 0].set_title('Intensitate regiuni cu stenoza minmax normalizata')
    axe[2, 0].scatter(range(len(minmaxStenosis)), minmaxStenosis)


    axe[2, 0].set_xlabel("nr pacienti")
    axe[2, 0].set_ylabel("Intensitate medie")
    axe[2, 1].set_title('Intensitate medie imagine minmax normalizata')
    axe[2, 1].scatter(range(len(minmaxData)), minmaxData)
    axe[2, 1].set_xlabel("nr pacienti")
    axe[2, 1].set_ylabel("Intensitate medie")
    axe[2, 2].set_title('intensitate medie imagine minmax normalizata')
    axe[2, 2].scatter(range(len(minmaxDifference)), minmaxDifference)
    axe[2, 2].set_xlabel("nr pacienti")
    axe[2, 2].set_ylabel("Diferenta")
    
    axe[3, 0].set_title('Intensitate regiuni cu stenoza z-score per imagine')
    axe[3, 0].scatter(range(len(perImageMeanStenosis)), perImageMeanStenosis, alpha=0.6, color='purple')
    axe[3, 0].set_xlabel("nr pacienti")
    axe[3, 0].set_ylabel("Intensitate medie")
    
    axe[3, 1].set_title('Intensitate medie imagine z-score per imagine')
    axe[3, 1].scatter(range(len(perImageMeanData)), perImageMeanData, alpha=0.6, color='orange')
    axe[3, 1].set_xlabel("nr pacienti")
    axe[3, 1].set_ylabel("Intensitate medie")
    
    axe[3, 2].set_title('intensitate medie imagine z-score per imagine')
    axe[3, 2].scatter(range(len(perImageMeanDifference)), perImageMeanDifference, alpha=0.6, color='green')
    axe[3, 2].set_xlabel("nr pacienti")
    axe[3, 2].set_ylabel("Diferenta")
    
    
    plt.tight_layout()
    plt.show()
    plt.savefig("data/ARCADE/processed/data_vizualiation.png")
   
   
# dataPlot()


class ARCADEDataset(Dataset):
    def __init__(self, json_path, split='train', task='segmentare', transform=None):
        '''
        task-uri:
        - 'SegStenoza' - segmentare stenoza
        - 'SegCoronare' - segmentare coronare
        - 'Clasificare' - clasificare stenoza/non-stenoza
        - 'Unsupervised' - return date fara label
        '''
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            self.split = split
            self.task = task
            self.transform = transform
            
            # Find the project root by looking for 'Collateral-Coronary-Vessels-XAI' folder
            json_path_abs = os.path.abspath(json_path)
            path_parts = json_path_abs.split(os.sep)
            
            # Find the project folder (handle both hyphenated and underscored versions)
            self.project_root = None
            for i, part in enumerate(path_parts):
                if 'collateral' in part.lower() and 'coronary' in part.lower():
                    self.project_root = os.sep.join(path_parts[:i+1])
                    break
            
            if not self.project_root:
                # Fallback: use directory containing 'data' folder
                self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(json_path_abs)))
            
            self.samples = []
            potential_samples = []
            
            if task == 'SegStenoza':
                # 1. Original Stenoza (Positives with masks)
                if 'stenoza' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['stenoza'].items():
                        potential_samples.append((pacienti['data'], pacienti['label']))
                
                # 2. DataValidation (Negatives only - generate empty masks)
                if 'DataValidation' in self.data:
                    dataval = self.data['DataValidation']
                    negative_samples = []
                    for key, item in dataval.items():
                        if key.startswith('negative_'):
                            negative_samples.append(item['data'])
                    
                    # Split logic (consistent with Clasificare task)
                    num_neg = len(negative_samples)
                    train_neg_count = int(num_neg * 0.60)
                    val_neg_count = int(num_neg * 0.20)
                    extra_neg = 6 
                    
                    if split == 'train':
                        for img_path in negative_samples[:train_neg_count]:
                            potential_samples.append((img_path, None)) # None = Empty Mask
                    elif split == 'validation':
                        val_end_neg = train_neg_count + val_neg_count + extra_neg
                        for img_path in negative_samples[train_neg_count:val_end_neg]:
                            potential_samples.append((img_path, None))
                    elif split == 'test':
                        val_end_neg = train_neg_count + val_neg_count + extra_neg
                        for img_path in negative_samples[val_end_neg:]:
                            potential_samples.append((img_path, None))

            elif task == 'SegCoronare':
                if 'syntax' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['syntax'].items():
                        potential_samples.append((pacienti['data'], pacienti['label']))
            elif task == 'Clasificare':
                # Toggle this to False if you want to train ONLY on the new DataValidation data
                use_original_data = True
                # DISABLE Extra data to fix domain shift (High Train Acc, 0 Val Specificity)
                use_extra_data = False

                # Stenoza = 1 (positive class)
                if use_original_data and 'stenoza' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['stenoza'].items():
                        potential_samples.append((pacienti['data'], 1))
                
                # Extra/Normal images = 0 (negative class) - USE ALL for training
                if use_original_data and use_extra_data and split == 'train':
                    if 'extra' in self.data and 'pretrain' in self.data['extra']:
                        for _, item in self.data['extra']['pretrain'].items():
                            potential_samples.append((item['data'], 0))
                
                # DataValidation data distribution: 60% train, 20% val (+20 from test), 20% test (-20)
                if 'DataValidation' in self.data:
                    dataval = self.data['DataValidation']
                    
                    # Separate positive and negative samples by checking key prefixes
                    positive_samples = []
                    negative_samples = []
                    
                    for key, item in dataval.items():
                        if key.startswith('positive_'):
                            positive_samples.append(item['data'])
                        elif key.startswith('negative_'):
                            negative_samples.append(item['data'])
                    
                    # Calculate split sizes maintaining the positive/negative ratio
                    num_pos = len(positive_samples)
                    num_neg = len(negative_samples)
                    
                    # 60/20/20 split
                    train_pos_count = int(num_pos * 0.60)
                    val_pos_count = int(num_pos * 0.20)
                    test_pos_count = num_pos - train_pos_count - val_pos_count  # remaining ~20%
                    
                    train_neg_count = int(num_neg * 0.60)
                    val_neg_count = int(num_neg * 0.20)
                    test_neg_count = num_neg - train_neg_count - val_neg_count  # remaining ~20%
                    
                    # Add 20 extra samples from test to val (14 pos + 6 neg to maintain ratio)
                    extra_pos = 14
                    extra_neg = 6
                    
                    if split == 'train':
                        # First 60% of positives and negatives
                        for img_path in positive_samples[:train_pos_count]:
                            potential_samples.append((img_path, 1))
                        for img_path in negative_samples[:train_neg_count]:
                            potential_samples.append((img_path, 0))
                    
                    elif split == 'validation':
                        # 20% + 20 extra from test (14 pos + 6 neg)
                        val_end_pos = train_pos_count + val_pos_count + extra_pos
                        val_end_neg = train_neg_count + val_neg_count + extra_neg
                        
                        for img_path in positive_samples[train_pos_count:val_end_pos]:
                            potential_samples.append((img_path, 1))
                        for img_path in negative_samples[train_neg_count:val_end_neg]:
                            potential_samples.append((img_path, 0))
                    
                    elif split == 'test':
                        # Remaining 20% minus the 20 moved to validation
                        test_start_pos = train_pos_count + val_pos_count + extra_pos
                        test_start_neg = train_neg_count + val_neg_count + extra_neg
                        
                        for img_path in positive_samples[test_start_pos:]:
                            potential_samples.append((img_path, 1))
                        for img_path in negative_samples[test_start_neg:]:
                            potential_samples.append((img_path, 0))      
            elif task == 'Unsupervised':
                if 'stenoza' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['stenoza'].items():
                        potential_samples.append((pacienti['data'], None))
                if 'segmentare' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['segmentare'].items():
                        potential_samples.append((pacienti['data'], None))
                if 'syntax' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['syntax'].items():
                        potential_samples.append((pacienti['data'], None))
                
                if 'extra' in self.data and 'pretrain' in self.data['extra']:
                    for _, item in self.data['extra']['pretrain'].items():
                        potential_samples.append((item['data'], None))

            # Filter missing files
            missing_count = 0
            missing_dirs = set()
            for sample in potential_samples:
                img_path = self._resolve_path(sample[0])
                if os.path.exists(img_path):
                    self.samples.append(sample)
                else:
                    if missing_count == 0:
                        print(f"DEBUG: First missing file: {sample[0]}")
                        print(f"DEBUG: Resolved to: {img_path}")
                    missing_count += 1
                    missing_dirs.add(os.path.dirname(img_path))
            
            if missing_count > 0:
                print(f"Warning: {missing_count} samples were missing and removed from the dataset.")
                if len(missing_dirs) < 5:
                    print(f"Missing directories: {list(missing_dirs)}")

            # Print Class Distribution for Classification
            if self.task == 'Clasificare':
                pos_count = sum(1 for _, label in self.samples if label == 1)
                neg_count = sum(1 for _, label in self.samples if label == 0)
                print(f"Dataset Split: {split.upper()} | Total: {len(self.samples)} | Pos (Stenoza): {pos_count} | Neg (Normal): {neg_count}")

    def __len__(self):
            return len(self.samples)
        
    def _resolve_path(self, path_str):
            """Resolve path intelligently from project root and fallback locations."""
            # Normalize backslashes to forward slashes (Windows to Unix)
            path_str = path_str.replace('\\', '/')
            
            # Check if path contains the project folder name (case insensitive)
            project_folder_variants = [
                'Collateral Coronary Vessels XAI',
                'Collateral-Coronary-Vessels-XAI',
                'collateral coronary vessels xai',
                'collateral-coronary-vessels-xai'
            ]
            
            relative_part = None
            for variant in project_folder_variants:
                if variant.lower() in path_str.lower():
                    # Find the position after the project folder name
                    idx = path_str.lower().find(variant.lower())
                    # Move past the folder name
                    start_pos = idx + len(variant)
                    # Skip any separator
                    while start_pos < len(path_str) and path_str[start_pos] in ['/', '\\']:
                        start_pos += 1
                    relative_part = path_str[start_pos:]
                    break
            
            # If no project folder found, assume it's already relative
            if relative_part is None:
                relative_part = path_str
            
            # Ensure relative_part doesn't start with / or \ to allow os.path.join to work
            relative_part = relative_part.lstrip('/\\')
            
            # Generate variants for case sensitivity issues (specifically TrainB -> trainB)
            variants = [relative_part]
            if 'TrainB' in relative_part:
                variants.append(relative_part.replace('TrainB', 'trainB'))
            
            # Try multiple locations
            candidates = []
            for variant in variants:
                candidates.extend([
                    os.path.join(self.project_root, variant),  # Primary location
                    os.path.join(self.project_root, 'data', 'ARCADE', variant), # Try inside data/ARCADE
                    os.path.join('/workspace/Collateral-Coronary-Vessels-XAI', variant),  # Workspace fallback
                    os.path.join('/root/Collateral-Coronary-Vessels-XAI', variant),  # Root fallback
                ])
                
                # Special handling: if path starts with 'data/', try stripping it to look inside data/ARCADE
                # e.g. 'data/Extra/...' -> 'data/ARCADE/Extra/...'
                if variant.startswith('data/'):
                    stripped = variant[5:]
                    candidates.append(os.path.join(self.project_root, 'data', 'ARCADE', stripped))
            
            # Return first candidate that exists
            for candidate in candidates:
                normalized = os.path.normpath(candidate)
                if os.path.exists(normalized):
                    return normalized
            
            # If none exist, return the primary candidate (will raise FileNotFoundError with proper path)
            return os.path.normpath(candidates[0])
    
    def __getitem__(self, idx):
            samples = self.samples[idx]
            image_path = self._resolve_path(samples[0])
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None: raise FileNotFoundError(image_path)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            image = image.astype(np.float32) / 255.0
            
            if self.task == 'SegStenoza':
                if samples[1] is None:
                    # Negative sample -> Empty mask (Black)
                    label = np.zeros((256, 256), dtype=np.float32)
                else:
                    label_path = self._resolve_path(samples[1])
                    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
                    label = np.array(label, dtype=np.float32)
                    label = label / 255.0
                    label = (label > 0.5).astype(np.float32)
                
                if self.transform:
                    augmented = self.transform(image=image, mask=label)
                    image = augmented['image']
                    label = augmented['mask']
                
                image = zscore_normalize_image(image)
                image = torch.tensor(image).unsqueeze(0) 
                label = torch.tensor(label).unsqueeze(0) 
                
                return image, label
            
            elif self.task == 'SegCoronare':
                label_path = self._resolve_path(samples[1])
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
                label = np.array(label, dtype=np.float32)
                label = label / 255.0
                label = (label > 0.5).astype(np.float32)
                
                if self.transform:
                    augmented = self.transform(image=image, mask=label)
                    image = augmented['image']
                    label = augmented['mask']
                
                image = zscore_normalize_image(image)
                image = torch.tensor(image).unsqueeze(0) 
                label = torch.tensor(label).unsqueeze(0) 
                
                return image, label
            
            elif self.task == 'Clasificare':
                label = samples[1]
                
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                
                image = zscore_normalize_image(image)
                image = torch.tensor(image).unsqueeze(0) 
                label = torch.tensor(label, dtype=torch.long)
                
                return image, label
            
            elif self.task == 'Unsupervised':
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                
                image = zscore_normalize_image(image)
                image = torch.tensor(image).unsqueeze(0) 
                
                return image
            
