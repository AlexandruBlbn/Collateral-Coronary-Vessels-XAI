import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import yaml
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import data.dataloader as dataloader
from data.dataloader import ArcadeDataset
from data.MIM import ArcadeDatasetMIM, MaskGenerator
from utils.logger import TensorboardLogger
from zoo.mim import SimMIM

def train_one_epoch(model, dataloader, optimizer, device, epoch, logger, scaler):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(images, masks)
            reconstruction = None
            if isinstance(outputs, tuple):
                loss = outputs[0]
                reconstruction = outputs[1]
            else:
                loss = outputs
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        global_step = epoch * len(dataloader) + step
        logger.log_scalar("Train/Step_Loss", loss.item(), global_step)
        
        if step == 0:
            with torch.no_grad():
                n_vis = 8
                vis_images = images[:n_vis]
                vis_masks = masks[:n_vis].unsqueeze(1).float()
                
                if vis_masks.shape[-2:] != vis_images.shape[-2:]:
                    vis_masks = F.interpolate(vis_masks, size=vis_images.shape[-2:], mode='nearest')
                
                vis_masked_images = vis_images * (1 - vis_masks)
                
                if reconstruction is not None:
                    vis_raw_rec = reconstruction[:n_vis]
                    vis_final_rec = (vis_images * (1 - vis_masks)) + (vis_raw_rec * vis_masks)
                    
                    combined = torch.stack([vis_images, vis_masked_images, vis_final_rec], dim=1).flatten(0, 1)
                    grid = torchvision.utils.make_grid(combined, nrow=3, normalize=True, padding=2)
                    logger.log_image("Train/Visualizations", grid, global_step)

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def main():
    config_path = "config/mim_config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    backbones = [
        "vit_small_patch16_224"
    ]

    device = torch.device(base_config["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for backbone_name in backbones:
        print(f"\n{'='*50}")
        print(f"Backbone: {backbone_name}")
        print(f"{'='*50}\n")
        
        config = base_config.copy()
        config["model"]["backbone"] = backbone_name
        config["experiment_name"] = f"simmim_pretrain_{backbone_name}"
        
        logger = TensorboardLogger(log_dir="runs", experiment_name=config["experiment_name"])
        
        transform_list = [transforms.Resize((config["model"]["input_size"], config["model"]["input_size"]))]
        transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        transform = transforms.Compose(transform_list)
        
        json_path = "/workspace/Collateral-Coronary-Vessels-XAI/data/ARCADE/processed/dataset.json"
        root_dir = "/workspace/Collateral-Coronary-Vessels-XAI"
        
        dataset = ArcadeDataset(json_path=json_path, split='train', transform=transform, mode='pretrain', root_dir=root_dir)
        
        mask_generator = MaskGenerator(
            input_size=config["model"]["input_size"],
            mask_patch_size=config["data"]["mask_patch_size"],
            mask_ratio=config["data"]["mask_ratio"]
        )
        
        mim_dataset = ArcadeDatasetMIM(dataset, mask_generator)
        
        dataloader = DataLoader(
            mim_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True
        )
        
        model = SimMIM(backbone_name=backbone_name, in_channels=config["data"]["in_channels"]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=float(config["optimizer"]["lr"]), weight_decay=config["optimizer"]["weight_decay"])
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        
        epochs = config["optimizer"]["epochs"]
        save_dir = os.path.join(config["system"]["save_dir"], config["experiment_name"])
        os.makedirs(save_dir, exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, dataloader, optimizer, device, epoch, logger, scaler)
            logger.log_scalar("Train/Epoch_Loss", train_loss, epoch)
            print(f"Backbone: {backbone_name} | Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f}")
            
            torch.save(model.encoder.state_dict(), os.path.join(save_dir, "last_backbone.pth"))
            
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.encoder.state_dict(), os.path.join(save_dir, "best_backbone.pth"))
                print(f"New best backbone saved with loss: {best_loss:.4f}")

        logger.close()
        print(f"Finished training for {backbone_name}")

if __name__ == "__main__":
    main()