import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ArcadeDataset

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class GenericUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_filters=32):
        super(GenericUNet, self).__init__()
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters*2, base_filters*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters*4, base_filters*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters*8, base_filters*16))

        self.up1 = nn.ConvTranspose2d(base_filters*16, base_filters*8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_filters*16, base_filters*8)

        self.up2 = nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_filters*8, base_filters*4)

        self.up3 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_filters*4, base_filters*2)

        self.up4 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_filters*2, base_filters)

        self.outc = nn.Conv2d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        if x.shape != x4.shape: 
            x = torch.nn.functional.interpolate(x, size=x4.shape[2:], mode='bilinear')
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        if x.shape != x3.shape:
            x = torch.nn.functional.interpolate(x, size=x3.shape[2:], mode='bilinear')
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        if x.shape != x2.shape:
            x = torch.nn.functional.interpolate(x, size=x2.shape[2:], mode='bilinear')
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        if x.shape != x1.shape:
            x = torch.nn.functional.interpolate(x, size=x1.shape[2:], mode='bilinear')
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        return self.outc(x)

class SegmentationWrapper(Dataset):
    def __init__(self, base_dataset, input_size=256, root_dir='.', augment=False):
        self.base_dataset = base_dataset
        self.root_dir = root_dir
        self.input_size = input_size
        self.augment = augment

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_pil, label_path = self.base_dataset[idx]
        full_label_path = os.path.join(self.root_dir, label_path)
        mask_pil = Image.open(full_label_path).convert('L')

        image = TF.resize(image_pil, (self.input_size, self.input_size))
        mask = TF.resize(mask_pil, (self.input_size, self.input_size), interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5], std=[0.5])
        
        mask = TF.to_tensor(mask)
        mask = (mask > 0).float()
        
        return image, mask

# --- DICE + FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceFocalLoss(nn.Module):
    def __init__(self, smooth=1.0, focal_alpha=0.8, focal_gamma=2.0):
        super(DiceFocalLoss, self).__init__()
        self.smooth = smooth
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, inputs, targets):
        # Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        # Focal Loss
        focal = self.focal_loss(inputs, targets)
        
        return dice_loss + focal

def log_images_to_tensorboard(writer, images, true_masks, pred_logits, step, prefix="Train"):
    n_images = min(4, images.shape[0])
    for i in range(n_images):
        img_display = (images[i] * 0.5) + 0.5
        img_display = torch.clamp(img_display, 0, 1)
        pred_prob = torch.sigmoid(pred_logits[i])
        grid = torch.cat([img_display, true_masks[i], pred_prob], dim=2)
        writer.add_image(f'{prefix}_Visuals/Sample_{i}', grid, step)

def train_original_unet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = "nnUnet"
    writer = SummaryWriter(f"runs/{run_name}")
    batch_size = 8
    epochs = 50
    lr = 0.001

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
    
    model = GenericUNet(n_channels=1, n_classes=1, base_filters=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Updated Loss Function
    loss_fn = DiceFocalLoss(focal_alpha=0.8, focal_gamma=2.0)
    
    metric_iou = BinaryJaccardIndex().to(device)
    metric_f1 = BinaryF1Score().to(device)

    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if batch_idx == 0:
                log_images_to_tensorboard(writer, imgs, masks, logits, epoch, "Train")

        model.eval()
        val_iou = 0.0
        val_f1 = 0.0
        
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(val_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits)
                
                val_iou += metric_iou(probs, masks.int()).item()
                val_f1 += metric_f1(probs, masks.int()).item()
                
                if batch_idx == 0:
                    log_images_to_tensorboard(writer, imgs, masks, logits, epoch, "Val")
        
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
            save_path = f"runs/{run_name}/best_original_unet.pth"
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved: {save_path}")

        scheduler.step()

    writer.close()
    print("Training finished.")
    
def test_original_unet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = "nnUnet"
    checkpoint_path = f"runs/{run_name}/best_original_unet.pth"
    
    json_path = '/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json'
    root_dir = '/workspace/Collateral-Coronary-Vessels-XAI'
    
    test_loader = DataLoader(
        SegmentationWrapper(ArcadeDataset(json_path, split='test', mode='syntax'), root_dir=root_dir, augment=False),
        batch_size=1, shuffle=False, num_workers=4
    )

    print(f"Evaluating on Test Split...")
    model = GenericUNet(n_channels=1, n_classes=1, base_filters=32).to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.eval()
    metric_iou = BinaryJaccardIndex().to(device)
    metric_f1 = BinaryF1Score().to(device)
    
    total_iou = 0.0
    total_f1 = 0.0
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            total_iou += metric_iou(probs, masks.int()).item()
            total_f1 += metric_f1(probs, masks.int()).item()
            
    print(f"\nTest Results:\nIoU: {total_iou/len(test_loader):.4f}\nF1: {total_f1/len(test_loader):.4f}")

if __name__ == "__main__":
    # train_original_unet()
    test_original_unet()