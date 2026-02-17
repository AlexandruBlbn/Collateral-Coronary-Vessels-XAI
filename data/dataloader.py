import torch
import torchvision
from torch.utils.data import Dataset
import json
import os
from PIL import Image


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
        label_path = item['label']

        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
            label_path = os.path.join(self.root_dir, label_path)
        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # All modes return (image, label)
        return image, label


def test_dataloader():
    """Test function to verify dataloader works correctly for all modes."""
    import random
    
    json_path = os.path.join('data', 'ARCADE', 'processed', 'dataset.json')
    root_dir = '.'
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Skipping test.")
        return
    
    modes = ['syntax', 'stenosis', 'pretrain']
    
    for mode in modes:
        print(f"\n--- Testing mode: {mode} ---")
        try:
            ds = ArcadeDataset(json_path, split='test', transform=None, mode=mode, root_dir=root_dir)
            print(f"✓ Dataset created | size: {len(ds)}")
            
            if len(ds) == 0:
                print("  Dataset is empty, skipping.")
                continue
            
            # Get a random sample
            idx = random.randint(0, len(ds) - 1)
            image, label = ds[idx]
            
            print(f"  Sample {idx}: image type={type(image).__name__}")
            print(f"  Label path: {label}")
            print(f"  Image size: {image.size if hasattr(image, 'size') else 'unknown'}")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ All dataloader tests completed!")


if __name__ == '__main__':
    test_dataloader()
