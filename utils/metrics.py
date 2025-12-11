import numpy as np

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

