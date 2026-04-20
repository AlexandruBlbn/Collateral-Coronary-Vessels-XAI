import os
import json
import random
import torch
import cv2
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class LeJepaDenseDataset(Dataset):
    def __init__(self, base_dataset_json, crops_json_path, root_dir='.', 
                 num_global=2, num_local=4, global_size=224, local_size=96, max_jitter=4,
                 num_vessel_classes=26):
        """
        Dataloader engineered for Dense LeJEPA using pre-computed exact coordinate pools.
        """
        self.root_dir = root_dir
        self.num_global = num_global
        self.num_local = num_local
        self.global_size = global_size
        self.local_size = local_size
        self.max_jitter = max_jitter
        self.num_vessel_classes = num_vessel_classes

        # We load base to know the splits, but we only really care about train pretraining
        with open(base_dataset_json, 'r') as f:
            base_data = json.load(f)
            
        with open(crops_json_path, 'r') as f:
            self.crops_meta = json.load(f)
            
        self.samples = []
        train_split = base_data.get('train', {})
        for source, items in train_split.items():
            for s_id, s_info in items.items():
                img_path = s_info.get('data')
                if img_path and img_path in self.crops_meta:
                    meta = self.crops_meta[img_path]
                    # Only include if it has enough valid bounding boxes
                    if len(meta.get('global_crops', [])) > 0 and len(meta.get('local_crops', [])) > 0:
                        self.samples.append({
                            'path': img_path,
                            'source': source,
                            'meta': meta
                        })
                        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print(f"Loaded {len(self.samples)} valid samples for Dense SSL.")

    def __len__(self):
        return len(self.samples)

    def _apply_jitter(self, coord, h_img, w_img):
        y, x, h, w = coord['y'], coord['x'], coord['h'], coord['w']
        
        # apply jitter
        jy = random.randint(-self.max_jitter, self.max_jitter)
        jx = random.randint(-self.max_jitter, self.max_jitter)
        
        y = max(0, min(y + jy, h_img - h))
        x = max(0, min(x + jx, w_img - w))
        
        return int(y), int(x), int(h), int(w)
        
    def _stochastic_aug(self, crop_t):
        # We NO LONGER augment the individual patches independently because Dense SSL 
        # requires strict topological alignment. If we flip the Target patch independently, 
        # the Context Predictor still predicts the unflipped vector sequence, causing spatial collapse.
        # We will instead apply transformations at the FULL IMAGE level below.
        return crop_t

    @staticmethod
    def _center_in_abs_box(cy, cx, box):
        y, x, h, w = box
        return (cy >= y) and (cy <= y + h) and (cx >= x) and (cx <= x + w)

    def __getitem__(self, idx):
        item = self.samples[idx]
        abs_path = os.path.join(self.root_dir, item['path'])
        meta = item['meta']
        
        # Load and CLAHE
        img = np.array(Image.open(abs_path).convert('L'))
        img_h, img_w = img.shape
        img = self.clahe.apply(img)
        
        # Normalize to [-1, 1] tensor for backbone
        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        img_t = img_t * 2.0 - 1.0 # [-1, 1] normalization
        
        g_candidates = [dict(c) for c in meta['global_crops']]
        l_candidates = [dict(c) for c in meta['local_crops']]
        
        # Whole-Canvas Topolocial Augmentations: Apply flips to the source image 
        # AND correctly mathematically invert the bounding box tracking coordinates identically.
        if random.random() < 0.5:
            img_t = TF.hflip(img_t)
            for c in g_candidates + l_candidates:
                c['x'] = float(img_w - (c['x'] + c['w']))
                
        if random.random() < 0.5:
            img_t = TF.vflip(img_t)
            for c in g_candidates + l_candidates:
                c['y'] = float(img_h - (c['y'] + c['h']))
                
        # Simulating X-Ray Noise / Variance 
        if random.random() < 0.8:
            noise_sigma = random.uniform(0.01, 0.05)
            img_t = img_t + torch.randn_like(img_t) * noise_sigma
            img_t = torch.clamp(img_t, -1.0, 1.0)
        
        global_crops_t = []
        global_boxes = []
        local_crops_t = []
        local_boxes = []
        
        # Random pick global coords; prefer unique contexts for better local coverage.
        if len(g_candidates) >= self.num_global:
            g_coords = random.sample(g_candidates, k=self.num_global)
        else:
            g_coords = random.choices(g_candidates, k=self.num_global)

        selected_global_abs = []
        
        for c in g_coords:
            y, x, h, w = self._apply_jitter(c, img_h, img_w)
            crop = TF.crop(img_t, y, x, h, w)
            global_crops_t.append(crop)
            global_boxes.append([y / img_h, x / img_w, h / img_h, w / img_w])
            selected_global_abs.append((y, x, h, w))
            
        # Prefer local crops whose centers are inside at least one selected global crop.
        local_pool = []
        for c in l_candidates:
            cy = float(c['y']) + 0.5 * float(c['h'])
            cx = float(c['x']) + 0.5 * float(c['w'])
            if any(self._center_in_abs_box(cy, cx, g) for g in selected_global_abs):
                local_pool.append(c)

        if len(local_pool) == 0:
            local_pool = l_candidates

        if len(local_pool) >= self.num_local:
            l_coords = random.sample(local_pool, k=self.num_local)
        else:
            l_coords = random.choices(local_pool, k=self.num_local)
        
        for c in l_coords:
            y, x, h, w = self._apply_jitter(c, img_h, img_w)
            # Keep local-center alignment with selected globals after jitter when possible.
            valid = False
            for _ in range(3):
                cy = float(y) + 0.5 * float(h)
                cx = float(x) + 0.5 * float(w)
                if any(self._center_in_abs_box(cy, cx, g) for g in selected_global_abs):
                    valid = True
                    break
                y, x, h, w = self._apply_jitter(c, img_h, img_w)

            if not valid:
                y = int(max(0, min(float(c['y']), img_h - float(c['h']))))
                x = int(max(0, min(float(c['x']), img_w - float(c['w']))))
                h = int(float(c['h']))
                w = int(float(c['w']))

            crop = TF.crop(img_t, y, x, h, w)
            local_crops_t.append(crop)
            local_boxes.append([y / img_h, x / img_w, h / img_h, w / img_w])
            
        # Classification probe metadata
        syntax_classes = meta.get('syntax_classes', [])
        is_syntax = (item['source'] == 'syntax')
        
        cls_target = torch.zeros(self.num_vessel_classes, dtype=torch.float32)
        if is_syntax:
            for cls_id in syntax_classes:
                if 1 <= cls_id <= self.num_vessel_classes:
                    cls_target[cls_id - 1] = 1.0
                    
        return {
            'global_crops': torch.stack(global_crops_t),
            'global_boxes': torch.tensor(global_boxes, dtype=torch.float32),
            'local_crops': torch.stack(local_crops_t),
            'local_boxes': torch.tensor(local_boxes, dtype=torch.float32),
            'is_syntax': torch.tensor(is_syntax),
            'cls_target': cls_target
        }


# Original ArcadeDataset is kept here unmodified for backward compatibility in segmentation tasks.
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
            sources_to_load = ['syntax', 'stenoza', 'cadica', 'extra', 'coronarydominance']
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
        label_path = item.get('label', None) 
        
        is_syntax = (item.get('source', '') == 'syntax')

        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
            
        image = Image.open(img_path).convert('L')
        
        if isinstance(label_path, str) and label_path != "":
            if self.root_dir:
                label_path = os.path.join(self.root_dir, label_path)
            label = Image.open(label_path).convert('L')
        else:
            label = Image.new('L', image.size, 0)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        if self.mode == 'pretrain':
            return image, label, is_syntax
        
        return image, label
