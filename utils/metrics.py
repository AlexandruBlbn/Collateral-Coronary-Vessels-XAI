import numpy as np
import torch
import torch.nn as nn

def accuracy(preds, targets):
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def precision(preds, targets):
    true_positive = ((preds == 1) & (targets == 1)).sum().item()
    predicted_positive = (preds == 1).sum().item()
    if predicted_positive == 0:
        return 0.0
    return true_positive / predicted_positive

def recall(preds, targets):
    true_positive = ((preds == 1) & (targets == 1)).sum().item()
    actual_positive = (targets == 1).sum().item()
    if actual_positive == 0:
        return 0.0
    return true_positive / actual_positive

def f1_score(preds, targets):
    p = precision(preds, targets)
    r = recall(preds, targets)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def iou(preds, targets, threshold=0.5):
    """IoU pentru segmentație. Preds pot fi continuous."""
    if preds.dtype != torch.bool:
        preds = (preds > threshold).bool()
    if targets.dtype != torch.bool:
        targets = (targets > 0).bool()
    
    intersection = (preds & targets).sum().float()
    union = (preds | targets).sum().float()
    
    if union == 0:
        return torch.tensor(0.0, device=preds.device)
    return intersection / union

def dice_coefficient(preds, targets, threshold=0.5):
    """Dice/F1 score pentru segmentație."""
    if preds.dtype != torch.bool:
        preds = (preds > threshold).bool()
    if targets.dtype != torch.bool:
        targets = (targets > 0).bool()
    
    intersection = (preds & targets).sum().float()
    dice = 2 * intersection / (preds.sum().float() + targets.sum().float() + 1e-8)
    
    return dice


class DiceLoss(nn.Module):
    """Dice Loss pentru segmentație binară"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        predictions: logits din model (inainte de sigmoid)
        targets: binar (0 sau 1)
        """
        # Aplică sigmoid pentru a obține probabilități
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculeaza intersectie si dice
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1.0 - dice  # Loss = 1 - Dice


class CombinedLoss(nn.Module):
    """Combinație de BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss