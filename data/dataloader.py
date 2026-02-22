# dataloader.py

import torch
import torchvision
from torch.utils.data import Dataset
import json
import os
from PIL import Image

class ArcadeDataset(Dataset):
    def __init__(self, json_path, split='train', transform=None, mode='syntax', root_dir=None):
        self.json_path = json_path
        self.split = split
        self.transform = transform
        self.mode = mode.lower() if mode else 'syntax'
        self.root_dir = root_dir

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in dataset.")

        self.samples = self._prepare_data()

    def _prepare_data(self):
        samples = []
        split_data = self.data[self.split]
        
        if self.mode == 'pretrain':
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
        label_path = item['label']
        source = item['source']

        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
            
        image = Image.open(img_path).convert('L')

        is_syntax = (source == 'syntax' and label_path is not None and isinstance(label_path, str))

        if is_syntax:
            if self.root_dir:
                label_path = os.path.join(self.root_dir, label_path)
            label = Image.open(label_path).convert('L')
        else:
            label = Image.new('L', image.size, 0)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label, is_syntax