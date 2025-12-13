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
    if preds.dtype != torch.bool:
        preds = (preds > threshold).bool()
    if targets.dtype != torch.bool:
        targets = (targets > 0).bool()
    
    intersection = (preds & targets).sum().float()
    dice = 2 * intersection / (preds.sum().float() + targets.sum().float() + 1e-8)
    
    return dice
