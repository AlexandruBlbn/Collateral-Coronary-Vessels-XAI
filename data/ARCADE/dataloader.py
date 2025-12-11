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



def plot_Data():
    with open('D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\dataset.json', 'r') as f:
        data = json.load(f)
    root_path = 'D:\\Collateral Coronary Vessels XAI\\data\\ARCADE\\'
    
    meanStenosis = []
    meanData = []
    
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
        
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(img_array[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(img_array) / 255)
        else:
            meanData.append(np.mean(img_array) / 255)
            meanStenosis.append(0)
            

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
        
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(img_array[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(img_array) / 255)
        else:
            meanData.append(np.mean(img_array) / 255)
            meanStenosis.append(0)
            
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
        
        
        mask = label_array > 0
    
        if np.any(mask):
            mean_value = np.mean(img_array[mask]) / 255
            meanStenosis.append(mean_value)
            meanData.append(np.mean(img_array) / 255)
        else:
            meanData.append(np.mean(img_array) / 255)
            meanStenosis.append(0)   
            

    fig, axe = plt.subplots(1, 2, figsize=(12, 6))
    axe[0].set_title('Intensitate regiuni cu stenoza')
    axe[0].scatter(range(len(meanStenosis)), meanStenosis, color='blue')
    axe[0].set_xlabel("nr pacienti")
    axe[0].set_ylabel("Intensitate medie")
    axe[1].set_title('Intensitate medie imagine')
    axe[1].scatter(range(len(meanData)), meanData, color='green')
    axe[1].set_xlabel("nr pacienti")
    axe[1].set_ylabel("Intensitate medie")
    plt.show()
    
    #todo: calcul diferenta medie intre zone stenoza si nostenozate.
    
    
    
    
    
plot_Data()
    
