import torch
import torchvision
from torch.utils.data import Dataset
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np



class ArcadeDataset(Dataset):
    def __init__(self, json_path, split='train', transform=None, mode='syntax', root_dir=None):
        """
        Args:
            json_path: Path to the dataset JSON file
            split: 'train', 'validation', or 'test'
            transform: Optional image transforms
            mode: 'syntax', 'stenosis', or 'pretrain'
                - 'syntax': Returns data and label from syntax source
                - 'stenosis': Returns data and label from stenoza source
                - 'pretrain': Returns data from syntax, stenoza, cadica, and extra sources
            root_dir: Root directory to prepend to image paths
        """
        self.json_path = json_path
        self.split = split
        self.transform = transform
        self.mode = mode.lower() if mode else 'syntax'
        self.root_dir = root_dir
        self.frangi_dir = frangi_dir
        self.frangi_preview_dir = frangi_preview_dir
        self._frangi_failed_paths = set()

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in dataset.")

        self.samples = self._prepare_data()

    def _prepare_data(self):
        samples = []
        split_data = self.data[self.split]
        
        if self.mode == 'pretrain':
            # Load all sources: syntax, stenoza, cadica, extra
            sources_to_load = ['syntax', 'stenoza', 'cadica', 'extra']
            for source_name in sources_to_load:
                if source_name not in split_data:
                    continue
                source_data = split_data[source_name]
                for sample_id, sample_info in source_data.items():
                    samples.append({
                        'image_path': sample_info.get('data'),
                        'label': sample_info.get('label'),
                        'source': source_name,
                        'id': sample_id
                    })
        else:
            # For 'syntax' or 'stenosis' modes, use the corresponding source
            source_name = 'stenoza' if self.mode == 'stenosis' else 'syntax'
            if source_name not in split_data:
                raise ValueError(f"Source '{source_name}' not found in split '{self.split}'")
            
            source_data = split_data[source_name]
            for sample_id, sample_info in source_data.items():
                samples.append({
                    'image_path': sample_info.get('data'),
                    'label': sample_info.get('label'),
                    'source': source_name,
                    'id': sample_id
                })
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item['image_path']
        label_path = item.get('label', None)  # Previne KeyError și setează None dacă lipsește
        
        # Extragem informația, dar o returnăm doar dacă e nevoie
        is_syntax = (item.get('source', '') == 'syntax')

        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
            
        image = Image.open(img_path).convert('L')
        
        # --- VERIFICARE ȘI TRATARE LABEL_PATH ---
        # Dacă label_path este valid (string), îl combinăm și încărcăm
        if isinstance(label_path, str) and label_path != "":
            if self.root_dir:
                label_path = os.path.join(self.root_dir, label_path)
            label = Image.open(label_path).convert('L')
        else:
            # Dacă nu are label (ex: date extra), creăm o mască neagră falsă 
            # de aceeași dimensiune cu imaginea
            label = Image.new('L', image.size, 0)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # ---> VERIFICAREA AICI <---
        if self.mode == 'pretrain':
            return image, label, is_syntax
        
        return image, label
