import torch
import torch.nn as nn
import torch.nn.functional as F

class KoLeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        # Pairwise dot products
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        # Fill diagonal with -1 to ignore self
        dots.view(-1)[:: (n + 1)].fill_(-1)
        # Get max inner prod -> min distance
        _, indices = torch.max(dots, dim=1)
        return indices

    def forward(self, student_output, eps=1e-8):
        # Normalize vectors
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        # Find nearest neighbors
        indices = self.pairwise_NNs_inner(student_output)
        # Calculate distance to nearest neighbor
        distances = self.pdist(student_output, student_output[indices])
        # Log distance loss (maximize distance => minimize -log)
        loss = -torch.log(distances + eps).mean()
        return loss