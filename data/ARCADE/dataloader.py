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
            self.samples = []
            
            if task == 'SegStenoza':
                for patient_id, pacienti in self.data[split]['stenoza'].items():
                    self.samples.append((pacienti['data'], pacienti['label']))
            elif task == 'SegCoronare':
                for patient_id, pacienti in self.data[split]['segmentare'].items():
                    self.samples.append((pacienti['data'], pacienti['label']))
            elif task == 'Clasificare':
                if 'stenoza' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['stenoza'].items():
                        self.samples.append((pacienti['data'], 1))
                if 'segmentare' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['segmentare'].items():
                        self.samples.append((pacienti['data'], 0))      
            elif task == 'Unsupervised':
                if 'stenoza' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['stenoza'].items():
                        self.samples.append((pacienti['data'], None))
                if 'segmentare' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['segmentare'].items():
                        self.samples.append((pacienti['data'], None))
                if 'syntax' in self.data[split]:
                    for patient_id, pacienti in self.data[split]['syntax'].items():
                        self.samples.append((pacienti['data'], None))
                
                if 'extra' in self.data and 'pretrain' in self.data['extra']:
                    for _, item in self.data['extra']['pretrain'].items():
                        self.samples.append((item['data'], None))

    def __len__(self):
            return len(self.samples)
        
    def __getitem__(self, idx):
            samples = self.samples[idx]
            image_path = os.path.join("data/ARCADE/", samples[0])
            # OpenCV este mult mai rapid decat PIL pentru citire si resize
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None: raise FileNotFoundError(image_path)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32) / 255.0
            
            if self.task == 'SegStenoza':
                label_path = os.path.join("data/ARCADE/", samples[1])
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
                label = np.array(label, dtype=np.float32)
                label = label / 255.0
                label = (label > 0.5).astype(np.float32)
                
                if self.transform:
                    augmented = self.transform(image=image, mask=label)
                    image = augmented['image']
                    label = augmented['mask']
                
                image = torch.tensor(image).unsqueeze(0) 
                label = torch.tensor(label).unsqueeze(0) 
                
                return image, label
            
            elif self.task == 'SegCoronare':
                label_path = os.path.join("data/ARCADE/", samples[1])
                label = Image.open(label_path).convert('L')
                label = label.resize((256, 256), Image.NEAREST)
                label = np.array(label, dtype=np.float32)
                label = label / 255.0
                label = (label > 0.5).astype(np.float32)
                
                if self.transform:
                    augmented = self.transform(image=image, mask=label)
                    image = augmented['image']
                    label = augmented['mask']
                
                image = torch.tensor(image).unsqueeze(0) 
                label = torch.tensor(label).unsqueeze(0) 
                
                return image, label
            
            elif self.task == 'Clasificare':
                label = samples[1]
                
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                
                image = torch.tensor(image).unsqueeze(0) 
                label = torch.tensor(label, dtype=torch.long)
                
                return image, label
            
            elif self.task == 'Unsupervised':
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                
                image = torch.tensor(image).unsqueeze(0) 
                
                return image
            
