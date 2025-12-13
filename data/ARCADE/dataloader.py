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


def plot_Data():
    with open('D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\dataset.json', 'r') as f:
        data = json.load(f)
    root_path = 'D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\'
    
    meanStenosis = []
    meanData = []
    meanDifference = []
    
    # Per-image normalized
    perImageMeanStenosis = []
    perImageMeanData = []
    perImageMeanDifference = []
    
    for pacient in data['train'].values():
        img_path = pacient['data'].replace('src\\data\\processed\\ARCADE\\', root_path)
        img = Image.open(img_path).convert('L')
        
        if pacient['label'] == 'None':
            label = np.zeros_like(np.array(img))
        else: 
            label_path = pacient['label'].replace('src\\data\\processed\\ARCADE\\', root_path)
            label = Image.open(label_path).convert('L')
        
        img_array = np.array(img)
        label_array = np.array(label)
        img_normalized = zscore_normalize_image(img_array)
        
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(img_array[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(img_array) / 255)
            meanDifference.append(mean_value - np.mean(img_array) / 255)
            
            perImageMeanStenosis.append(np.mean(img_normalized[mask]))
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanDifference.append(np.mean(img_normalized[mask]) - np.mean(img_normalized))
        else:
            meanData.append(np.mean(img_array) / 255)
            meanStenosis.append(0)
            meanDifference.append(0)
            
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanStenosis.append(0)
            perImageMeanDifference.append(0)

    for pacient in data['validation'].values():
        img_path = pacient['data'].replace('src\\data\\processed\\ARCADE\\', root_path)
        img = Image.open(img_path).convert('L')
        
        if pacient['label'] == 'None':
            label = np.zeros_like(np.array(img))
        else: 
            label_path = pacient['label'].replace('src\\data\\processed\\ARCADE\\', root_path)
            label = Image.open(label_path).convert('L')
        
        img_array = np.array(img)
        label_array = np.array(label)
        img_normalized = zscore_normalize_image(img_array)
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(img_array[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(img_array) / 255)
            meanDifference.append(mean_value - np.mean(img_array) / 255)
            
            perImageMeanStenosis.append(np.mean(img_normalized[mask]))
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanDifference.append(np.mean(img_normalized[mask]) - np.mean(img_normalized))
        else:
            meanData.append(np.mean(img_array) / 255)
            meanStenosis.append(0)
            meanDifference.append(0)
            
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanStenosis.append(0)
            perImageMeanDifference.append(0)

    for pacient in data['test'].values():
        img_path = pacient['data'].replace('src\\data\\processed\\ARCADE\\', root_path)
        img = Image.open(img_path).convert('L')
        
        if pacient['label'] == 'None':
            label = np.zeros_like(np.array(img))
        else: 
            label_path = pacient['label'].replace('src\\data\\processed\\ARCADE\\', root_path)
            label = Image.open(label_path).convert('L')
        
        img_array = np.array(img)
        label_array = np.array(label)
        img_normalized = zscore_normalize_image(img_array)
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(img_array[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(img_array) / 255)
            meanDifference.append(mean_value - np.mean(img_array) / 255)
            
            perImageMeanStenosis.append(np.mean(img_normalized[mask]))
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanDifference.append(np.mean(img_normalized[mask]) - np.mean(img_normalized))
        else:
            meanData.append(np.mean(img_array) / 255)
            meanStenosis.append(0)
            meanDifference.append(0)
            
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanStenosis.append(0)
            perImageMeanDifference.append(0)
    

    normalizedStenosis = zscore_normalize(np.array(meanStenosis))
    normalizedData = zscore_normalize(np.array(meanData))
    normalizedDifference = zscore_normalize(np.array(meanDifference))
    
    minmaxStenosis = minmax_normalize(np.array(meanStenosis))
    minmaxData = minmax_normalize(np.array(meanData))
    minmaxDifference = minmax_normalize(np.array(meanDifference))

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
    plotPNG = 'D:\\Collateral Coronary Vessels XAI\\data_analysis.png'
    fig.savefig(plotPNG)
    #z-score per image cel mai bun
    #todo: augmentari pentru z-score per image ca sa facem diferenta intre stenoza si restul imaginii - equalizare histogramica
    #redistribuirea intensitatilor in functie de histograma imaginii
    #constrast limited adaptive histrogram equalization (CLAHE)
    
        # clahe_img = cv2.GaussianBlur(img_array, (3, 3),2)
        # clahe_img = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(clahe_img)
        # clahe_img = zscore_normalize_image(clahe_img)

# plot_Data()




def plot_Data_Augmented():
    with open('D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\dataset.json', 'r') as f:
        data = json.load(f)
    root_path = 'D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\'
    
    meanStenosis = []
    meanData = []
    meanDifference = []
    
    perImageMeanStenosis = []
    perImageMeanData = []
    perImageMeanDifference = []
    
    for pacient in data['train'].values():
        img_path = pacient['data'].replace('src\\data\\processed\\ARCADE\\', root_path)
        img = Image.open(img_path).convert('L')
        
        if pacient['label'] == 'None':
            label = np.zeros_like(np.array(img))
        else: 
            label_path = pacient['label'].replace('src\\data\\processed\\ARCADE\\', root_path)
            label = Image.open(label_path).convert('L')
        
        img_array = np.array(img)
        label_array = np.array(label)
        
        clahe_img = cv2.GaussianBlur(img_array, (3, 3), 2)
        clahe_img = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(clahe_img)
        img_normalized = zscore_normalize_image(clahe_img)
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(clahe_img[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(clahe_img) / 255)
            meanDifference.append(mean_value - np.mean(clahe_img) / 255)
            
            perImageMeanStenosis.append(np.mean(img_normalized[mask]))
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanDifference.append(np.mean(img_normalized[mask]) - np.mean(img_normalized))
        else:
            meanData.append(np.mean(clahe_img) / 255)
            meanStenosis.append(0)
            meanDifference.append(0)
            
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanStenosis.append(0)
            perImageMeanDifference.append(0)

    for pacient in data['validation'].values():
        img_path = pacient['data'].replace('src\\data\\processed\\ARCADE\\', root_path)
        img = Image.open(img_path).convert('L')
        
        if pacient['label'] == 'None':
            label = np.zeros_like(np.array(img))
        else: 
            label_path = pacient['label'].replace('src\\data\\processed\\ARCADE\\', root_path)
            label = Image.open(label_path).convert('L')
        
        img_array = np.array(img)
        label_array = np.array(label)
        
        clahe_img = cv2.GaussianBlur(img_array, (3, 3), 2)
        clahe_img = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(clahe_img)
        img_normalized = zscore_normalize_image(clahe_img)
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(clahe_img[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(clahe_img) / 255)
            meanDifference.append(mean_value - np.mean(clahe_img) / 255)
            
            perImageMeanStenosis.append(np.mean(img_normalized[mask]))
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanDifference.append(np.mean(img_normalized[mask]) - np.mean(img_normalized))
        else:
            meanData.append(np.mean(clahe_img) / 255)
            meanStenosis.append(0)
            meanDifference.append(0)
            
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanStenosis.append(0)
            perImageMeanDifference.append(0)

    for pacient in data['test'].values():
        img_path = pacient['data'].replace('src\\data\\processed\\ARCADE\\', root_path)
        img = Image.open(img_path).convert('L')
        
        if pacient['label'] == 'None':
            label = np.zeros_like(np.array(img))
        else: 
            label_path = pacient['label'].replace('src\\data\\processed\\ARCADE\\', root_path)
            label = Image.open(label_path).convert('L')
        
        img_array = np.array(img)
        label_array = np.array(label)
        
        clahe_img = cv2.GaussianBlur(img_array, (3, 3), 2)
        clahe_img = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(clahe_img)
        img_normalized = zscore_normalize_image(clahe_img)
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(clahe_img[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(clahe_img) / 255)
            meanDifference.append(mean_value - np.mean(clahe_img) / 255)
            
            perImageMeanStenosis.append(np.mean(img_normalized[mask]))
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanDifference.append(np.mean(img_normalized[mask]) - np.mean(img_normalized))
        else:
            meanData.append(np.mean(clahe_img) / 255)
            meanStenosis.append(0)
            meanDifference.append(0)
            
            perImageMeanData.append(np.mean(img_normalized))
            perImageMeanStenosis.append(0)
            perImageMeanDifference.append(0)
    
    normalizedStenosis = zscore_normalize(np.array(meanStenosis))
    normalizedData = zscore_normalize(np.array(meanData))
    normalizedDifference = zscore_normalize(np.array(meanDifference))
    
    minmaxStenosis = minmax_normalize(np.array(meanStenosis))
    minmaxData = minmax_normalize(np.array(meanData))
    minmaxDifference = minmax_normalize(np.array(meanDifference))

    fig, axe = plt.subplots(4, 3, figsize=(35, 35))
    axe[0, 0].set_title('Intensitate regiuni cu stenoza (Augmented)')
    axe[0, 0].scatter(range(len(meanStenosis)), meanStenosis)
    axe[0, 0].set_xlabel("nr pacienti")
    axe[0, 0].set_ylabel("Intensitate medie")
    axe[0, 1].set_title('Intensitate medie imagine (Augmented)')
    axe[0, 1].scatter(range(len(meanData)), meanData)
    axe[0, 1].set_xlabel("nr pacienti")
    axe[0, 1].set_ylabel("Intensitate medie")
    axe[0, 2].set_title('Diferenta intensitate (Augmented)')
    axe[0, 2].scatter(range(len(meanDifference)), meanDifference)
    axe[0, 2].set_xlabel("nr pacienti")
    axe[0, 2].set_ylabel("Diferenta")
    
    axe[1, 0].set_title('Intensitate regiuni cu stenoza normalizata (Augmented)')
    axe[1, 0].scatter(range(len(normalizedStenosis)), normalizedStenosis)
    axe[1, 0].set_xlabel("nr pacienti")
    axe[1, 0].set_ylabel("Intensitate medie")
    axe[1, 1].set_title('Intensitate medie imagine normalizata (Augmented)')
    axe[1, 1].scatter(range(len(normalizedData)), normalizedData)
    axe[1, 1].set_xlabel("nr pacienti")
    axe[1, 1].set_ylabel("Intensitate medie")
    axe[1, 2].scatter(range(len(normalizedDifference)), normalizedDifference)
    axe[1, 2].set_xlabel("nr pacienti")
    axe[1, 2].set_ylabel("Diferenta")
    
    axe[2, 0].set_title('Intensitate regiuni cu stenoza minmax (Augmented)')
    axe[2, 0].scatter(range(len(minmaxStenosis)), minmaxStenosis)
    axe[2, 0].set_xlabel("nr pacienti")
    axe[2, 0].set_ylabel("Intensitate medie")
    axe[2, 1].set_title('Intensitate medie imagine minmax (Augmented)')
    axe[2, 1].scatter(range(len(minmaxData)), minmaxData)
    axe[2, 1].set_xlabel("nr pacienti")
    axe[2, 1].set_ylabel("Intensitate medie")
    axe[2, 2].set_title('Diferenta minmax (Augmented)')
    axe[2, 2].scatter(range(len(minmaxDifference)), minmaxDifference)
    axe[2, 2].set_xlabel("nr pacienti")
    axe[2, 2].set_ylabel("Diferenta")
    
    axe[3, 0].set_title('Stenoza z-score per imagine (Augmented)')
    axe[3, 0].scatter(range(len(perImageMeanStenosis)), perImageMeanStenosis, alpha=0.6, color='purple')
    axe[3, 0].set_xlabel("nr pacienti")
    axe[3, 0].set_ylabel("Intensitate medie")
    axe[3, 1].set_title('Imagine z-score per imagine (Augmented)')
    axe[3, 1].scatter(range(len(perImageMeanData)), perImageMeanData, alpha=0.6, color='orange')
    axe[3, 1].set_xlabel("nr pacienti")
    axe[3, 1].set_ylabel("Intensitate medie")
    axe[3, 2].set_title('Diferenta z-score per imagine (Augmented)')
    axe[3, 2].scatter(range(len(perImageMeanDifference)), perImageMeanDifference, alpha=0.6, color='green')
    axe[3, 2].set_xlabel("nr pacienti")
    axe[3, 2].set_ylabel("Diferenta")
    
    plt.tight_layout()
    plt.show()
    plotPNG = 'D:\\Collateral Coronary Vessels XAI\\data_analysis_augmented.png'
    fig.savefig(plotPNG)
    
# plot_Data_Augmented()















    
def CLAHETEST(img_array):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    return clahe.apply(img_array)

from scipy.ndimage import gaussian_filter

def unsharp_mask(img_array, sigma=1.0, strength=1.5):
    blurred = gaussian_filter(img_array.astype(float), sigma=sigma)
    sharpened = img_array + strength * (img_array - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def plot_CLAHE():
    with open('D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\dataset.json', 'r') as f:
        data = json.load(f)
    root_path = 'D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\'
    for pacient in data['train'].values():
        img_path = pacient['data'].replace('src\\data\\processed\\ARCADE\\', root_path)
        img = Image.open(img_path).convert('L')

        img_array = np.array(img)
        clahe_img = cv2.GaussianBlur(img_array, (3, 3),2)
        clahe_img = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(clahe_img)
        clahe_img = zscore_normalize_image(clahe_img)
        fig, axe = plt.subplots(1, 2,  figsize=(10, 5))
        axe[0].set_title('Original Image')
        axe[0].imshow(img_array, cmap='gray')
        axe[1].set_title('CLAHE Image')
        axe[1].imshow(clahe_img, cmap='gray')
        plt.show()
        break 
    
# plot_CLAHE()

def segmentationExtract():
    # Modificat pentru a extrage segmentarile vaselor din datasetul SYNTAX (vase coronariene)
    base_raw_path = r'D:\Collateral Coronary Vessels XAI\data\ARCADE\RAW\arcade\syntax'
    dataset_json_path = r'D:\Collateral Coronary Vessels XAI\data\ARCADE\dataset.json'
    base_output_path = r'D:\Collateral Coronary Vessels XAI\data\ARCADE'
    
    splits_map = {
        'train': ('train', 'train'),
        'validation': ('val', 'Validation'),
        'test': ('test', 'Test')
    }

    with open(dataset_json_path, 'r') as f:
        main_dataset = json.load(f)

    print("Inceperea extragerii segmentarilor coronariene (din SYNTAX)...")

    for split_key, (raw_split_name, folder_name) in splits_map.items():
        print(f"\nProcesare split: {split_key} (Raw: {raw_split_name}, Folder: {folder_name})...")
        
        raw_json_path = os.path.join(base_raw_path, raw_split_name, 'annotations', f'{raw_split_name}.json')
        
        if not os.path.exists(raw_json_path):
            print(f"  [EROARE] Nu am gasit fisierul: {raw_json_path}")
            continue

        with open(raw_json_path, 'r') as f:
            coco_data = json.load(f)

        save_dir = os.path.join(base_output_path, folder_name, 'labels')
        os.makedirs(save_dir, exist_ok=True)

        img_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        imgs_info = {img['id']: img for img in coco_data['images']}

        print(f"  Total imagini de procesat: {len(img_to_anns)}")
        
        processed_count = 0
        for img_id, anns in img_to_anns.items():
            img_info = imgs_info.get(img_id)
            if not img_info:
                # print(f"  [WARN] Image ID {img_id} nu are info, skip")
                continue
                
            h, w = img_info['height'], img_info['width']
            file_name = img_info['file_name']
            patient_id_str = os.path.splitext(file_name)[0]
            
            mask = np.zeros((h, w), dtype=np.uint8)
            vessel_count = 0

            for ann in anns:
                if 'segmentation' in ann and ann['segmentation']:
                    for poly in ann['segmentation']:
                        if len(poly) >= 6:
                            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                            vessel_count += 1

            if vessel_count > 0:
                save_filename = f"label_coronary_{patient_id_str}.png"
                save_path = os.path.join(save_dir, save_filename)
                cv2.imwrite(save_path, mask)
                
                patient_key = f"patient_{patient_id_str}"
                if split_key in main_dataset and patient_key in main_dataset[split_key]:
                    main_dataset[split_key][patient_key]['coronary segmentation'] = save_path
                    processed_count += 1
                # else:
                #     print(f"  [WARN] {patient_key} nu exista in dataset.json la split {split_key}")
            # else:
            #     print(f"  [WARN] Pacient {patient_id_str}: nu am gasit vase de desenat")

        print(f"  Procesate cu succes: {processed_count}/{len(img_to_anns)} imagini")

    with open(dataset_json_path, 'w') as f:
        json.dump(main_dataset, f, indent=4)
    
    print("\n[OK] Extragere finalizata si dataset.json actualizat!")
        
# segmentationExtract()


















class ArcadeSegmentare(Dataset):
    def __init__(self, path, split='train', augment=False):
        with open(path, 'r') as f:
            self.data = json.load(f)
        self.root_path = os.path.dirname(path) + '\\'
        self.split = split
        self.augment = augment
        self.samples = list(self.data[split].values())
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pacient = self.samples[idx]
        img_path = pacient['data'].replace('src\\data\\processed\\ARCADE\\', self.root_path)
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        if pacient['label'] == 'None':
            label_array = np.zeros_like(img_array)
        else:
            label_path = pacient['label'].replace('src\\data\\processed\\ARCADE\\', self.root_path)
            label = Image.open(label_path).convert('L')
            label_array = np.array(label)
            
        

        
        #blur -> clahe -> zscore
            
            # if self.augment == True:
            # #augmentari data
            # img_array = zscore_normalize_image(img_array)
            # img_array = 0
            # if np.random.rand() > 0.5:
            # else:
            #     print("test")

                
            
        # if self.augment == True:
        #     data = dataTransform
        #     label = labelTransform
        # else:
            # data = T.Resize((256, 256))(img)
            # data = zscore_normalize_image(np.array(data))
            # label = T.Resize((256, 256))(Image.fromarray(label_array))
            # label = np.array(label)
        
        # data = torch.from_numpy(img_array).float().unsqueeze(0)
        # label = torch.from_numpy(label_array).float().unsqueeze(0)
        # label = (label > 0).float()
        
        # return data, label
        

def test():
    dataset = ArcadeSegmentare('D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\dataset.json', split='train', augment=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for data, label in dataloader:
        print("Data shape:", data.shape)
        print("Label shape:", label.shape)
        
        image = data.squeeze().numpy()
        mask = label.squeeze().numpy()
        fig, axe = plt.subplots(1, 2,  figsize=(10, 5))
        axe[0].set_title('data')
        axe[0].imshow(image, cmap='gray')
        axe[1].set_title('label')
        axe[1].imshow(mask, cmap='gray')
        plt.show()


        break
    
# test()


#todo: de facut diferenta dintre segmentarile coronarienelor si a stenozelor.


class ArcadeCoronarySegmentation(Dataset):
    """
    Dataset class pentru segmentările vaselor coronariene (syntax dataset).
    Încarcă pacienții din categoria 'segmentare' din dataset.json procesat.
    """
    def __init__(self, path, split='train', augment=False):
        with open(path, 'r') as f:
            self.data = json.load(f)
        # Base path should be the parent of "processed" folder
        self.base_path = Path(path).parent.parent  # Go up from dataset.json -> processed -> ARCADE
        self.split = split
        self.augment = augment
        
        # Map split names
        split_map = {'train': 'train', 'val': 'validation', 'validation': 'validation', 'test': 'test'}
        dataset_split = split_map.get(split, 'train')
        
        # Get segmentare samples from new dataset structure
        self.samples = list(self.data[dataset_split]['segmentare'].items())
        
        # print(f"[CoronarySegmentation] Loaded {len(self.samples)} samples from {dataset_split} split")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        patient_id, patient_data = self.samples[idx]
        
        # Construct full paths
        img_path = self.base_path / patient_data['data']
        label_path = self.base_path / patient_data['label']
        
        # Load image
        img = Image.open(img_path).convert('L')
        img = img.resize((256, 256), Image.BILINEAR)
        img_array = np.array(img)
        
        # Load label (vessel segmentation)
        label = Image.open(label_path).convert('L')
        label = label.resize((256, 256), Image.NEAREST)
        label_array = np.array(label)
        
        # Preprocessing: Gaussian blur -> CLAHE -> Z-score normalization
        # if self.augment:
        #     # img_array = cv2.GaussianBlur(img_array, (3, 3), 2)
        #     img_array = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(img_array)
        
        # img_array = zscore_normalize_image(img_array)
        
        # Convert to tensors
        data = torch.from_numpy(img_array).float().unsqueeze(0)
        label = torch.from_numpy(label_array).float().unsqueeze(0)
        label = (label > 0).float()
        
        return data, label


def test_coronary_segmentation():
    """Test function for ArcadeCoronarySegmentation dataset"""
    dataset_path = r'D:\Collateral Coronary Vessels XAI\data\ARCADE\processed\dataset.json'
    
    print("Testing ArcadeCoronarySegmentation...")
    
    # Test train split
    train_dataset = ArcadeCoronarySegmentation(dataset_path, split='train', augment=True)
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Test dataloader
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    for i, (data, label) in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Label shape: {label.shape}")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Label unique values: {torch.unique(label).numpy()}")
        
        # Visualize first sample
        if i == 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            image = data[0, 0].numpy()
            mask = label[0, 0].numpy()
            
            # Create overlay
            overlay = np.stack([image, image, image], axis=-1)
            overlay = ((overlay - overlay.min()) / (overlay.max() - overlay.min()) * 255).astype(np.uint8)
            overlay[mask > 0] = [0, 255, 0]
            
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Coronary Image (Preprocessed)')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Coronary Vessels Label')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay (vessels in green)')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(r'D:\Collateral Coronary Vessels XAI\data\ARCADE\processed\coronary_test.png', dpi=150)
            print("\nVisualization saved to: coronary_test.png")
            plt.show()
        
        if i >= 2:  # Test only 3 batches
            break
    
    print("\n[OK] Test completed successfully!")

# Uncomment to test:
# test_coronary_segmentation()

