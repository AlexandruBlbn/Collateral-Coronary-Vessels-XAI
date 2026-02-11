import torch
import torchvision
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


class ArcadeDataset(Dataset):
    def __init__(self, json_path, split='train', transform=None, mode='finetune', root_dir=None):
        self.json_path = json_path
        self.split = split
        self.transform = transform
        self.mode = mode
        self.root_dir = root_dir

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in dataset.")

        self.samples = self._prepare_data()

    def _prepare_data(self):
        samples = []
        split_data = self.data[self.split]
        for source_name, source_data in split_data.items():
            for sample_id, sample_info in source_data.items():
                samples.append({
                    'image_path': sample_info['data'],
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

        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path).convert('RGB')

        # Apply CLAHE
        img_np = np.array(image)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

        if self.transform:
            image = self.transform(image)

        if self.mode == 'finetune':
            label = item['label']
            return image, label

        return image
