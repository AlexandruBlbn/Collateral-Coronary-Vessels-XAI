import torch
import sys
import math
from tqdm import tqdm

def train_one_epoch_dino(model, dataloader, optimizer, device, epoch, max_epochs, logger, scaler=None):
    model.train()
    running_loss = 0.0
    base_m = 0.996
    final_m = 1.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
    
    for step, images in enumerate(progress_bar):
        images = [img.to(device, non_blocking=True) for img in images]
        global_step = epoch * len(dataloader) + step
        
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            student_out, teacher_out = model(images)
            dino_loss_val = model.dino_loss(student_out, teacher_out, epoch)
            koleo_loss_val = model.koleo_loss(student_out)
            loss = dino_loss_val + 0.1 * koleo_loss_val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.student_backbone.parameters(), 3.0)
        torch.nn.utils.clip_grad_norm_(model.student_head.parameters(), 3.0)
        
        optimizer.step()
        progress = global_step / (max_epochs * len(dataloader))
        curr_m = base_m + (final_m - base_m) * progress
        
        model.update_teacher(m=curr_m)
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "ema_m": f"{curr_m:.4f}"})
        
        if logger is not None and step % 10 == 0:
            logger.log_scalar("Train/Total_Loss", loss.item(), global_step)
            logger.log_scalar("Train/DINO_Loss", dino_loss_val.item(), global_step)

    return running_loss / len(dataloader)