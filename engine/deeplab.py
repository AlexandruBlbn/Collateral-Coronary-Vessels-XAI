import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
from torchvision.models.segmentation import deeplabv3_resnet50

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset
import cv2
import numpy as np

class SegmentationWrapper(Dataset):
    def __init__(self, base_dataset, input_size=256, root_dir='.', augment=False):
        self.base_dataset = base_dataset
        self.root_dir = root_dir
        self.input_size = input_size
        self.augment = augment
        # Base normalization
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def apply_tophat(self, img_pil, kernel_size=15):
        """
        Applies Morphological Top-Hat Transform to highlight small vessels.
        TopHat(I) = I - Open(I)
        """
        # Convert PIL to Numpy (Grayscale)
        img_np = np.array(img_pil)
        
        # Define structural element (kernel)
        # Size 15 is typical for coronary vessels (adjust based on vessel thickness)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Apply Top-Hat
        tophat_img = cv2.morphologyEx(img_np, cv2.MORPH_TOPHAT, kernel)
        
        # Add Top-Hat result back to original image to enhance details
        # Or return just tophat_img if you want to remove background entirely.
        # Strategy: Enhance features
        enhanced_img = cv2.add(img_np, tophat_img)
        
        return Image.fromarray(enhanced_img)

    def transform(self, image, mask):
        # Resize mandatory
        image = TF.resize(image, (self.input_size, self.input_size))
        mask = TF.resize(mask, (self.input_size, self.input_size), interpolation=transforms.InterpolationMode.NEAREST)

        if self.augment:
            # --- 1. Top-Hat Enhancement (Probabilistic) ---
            # We apply this BEFORE geometric transforms to enhance vessel contrast
            if random.random() > 0.3:
                image = self.apply_tophat(image, kernel_size=random.choice([9, 15, 21]))

            # --- 2. Geometric Augmentations ---
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # --- 3. Intensity Augmentations ---
            if random.random() > 0.3:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

            # --- 4. Noise ---
            if random.random() > 0.5:
                img_tensor = TF.to_tensor(image)
                noise = torch.randn_like(img_tensor) * 0.02
                image = TF.to_pil_image(torch.clamp(img_tensor + noise, 0, 1))

        # Final Conversion
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        image = self.normalize(image)
        mask = (mask > 0).float()
        
        return image, mask

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_pil, label_path = self.base_dataset[idx]
        if label_path is None:
            raise ValueError(f"Label path lipsă pentru indexul {idx}")
            
        full_label_path = os.path.join(self.root_dir, label_path)
        mask_pil = Image.open(full_label_path).convert('L')
        
        return self.transform(image_pil, mask_pil)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        return bce + (1 - dice)

def train_deeplab_pytorch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = "DeepLabV3_ResNet50_PyTorch"
    writer = SummaryWriter(f"runs/{run_name}")
    batch_size = 16
    epochs = 100
    lr = 1e-4

    json_path = '/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json'
    root_dir = '/workspace/Collateral-Coronary-Vessels-XAI'
    
    train_loader = DataLoader(
        SegmentationWrapper(ArcadeDataset(json_path, split='train', mode='syntax'), root_dir=root_dir, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        SegmentationWrapper(ArcadeDataset(json_path, split='validation', mode='syntax'), root_dir=root_dir, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"Starting Training: {run_name}")
    
    # Initialize standard DeepLabV3 from torchvision
    model = deeplabv3_resnet50(num_classes=1)
    
    # Modify the first convolutional layer to accept 1 channel instead of 3
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = DiceBCELoss()
    
    metric_iou = BinaryJaccardIndex().to(device)
    metric_f1 = BinaryF1Score().to(device)

    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Torchvision models return an OrderedDict: {'out': tensor, 'aux': tensor}
            outputs = model(imgs)
            logits = outputs['out']
            
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.eval()
        val_iou = 0.0
        val_f1 = 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                logits = outputs['out']
                probs = torch.sigmoid(logits)
                
                val_iou += metric_iou(probs, masks.int()).item()
                val_f1 += metric_f1(probs, masks.int()).item()
        
        avg_iou = val_iou / len(val_loader)
        avg_f1 = val_f1 / len(val_loader)
        avg_train_loss = train_loss / len(train_loader)
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("IoU/Val", avg_iou, epoch)
        writer.add_scalar("F1/Val", avg_f1, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1} | Val IoU: {avg_iou:.4f} | Val F1: {avg_f1:.4f} | Train Loss: {avg_train_loss:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            save_path = f"runs/{run_name}/best_deeplab.pth"
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved: {save_path}")

        scheduler.step()

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    train_deeplab_pytorch()