import torch
import torch.nn as nn
import torch.nn.functional as F

class GramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Folosim MSE Loss, la fel ca în paper
        self.mse_loss = nn.MSELoss()

    def forward(self, student_patch_tokens, teacher_patch_tokens):
        # Intrări: (Batch, N_patches, Dim)
        s_norm = F.normalize(student_patch_tokens.float(), dim=-1, p=2)
        t_norm = F.normalize(teacher_patch_tokens.float(), dim=-1, p=2)

        gram_s = torch.bmm(s_norm, s_norm.transpose(1, 2))
        gram_t = torch.bmm(t_norm, t_norm.transpose(1, 2))

        loss = self.mse_loss(gram_s, gram_t)
        
        return loss