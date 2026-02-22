import torch
import torch.nn as nn
import torch.nn.functional as F

class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))

    @torch.no_grad()
    def step_center(self, teacher_patch_tokens):
        # (Batch, N, Dim) -> (1, 1, Dim)
        batch_center = torch.mean(teacher_patch_tokens, dim=(0, 1), keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks, epoch):

        teacher_temp = 0.04 + (0.07 - 0.04) * (epoch / 100)
 
        self.step_center(teacher_patch_tokens)

        mask_bool = student_masks.bool()
        s_masked = student_patch_tokens[mask_bool] # (N_masked_total, D)
        t_masked = teacher_patch_tokens[mask_bool] # (N_masked_total, D)
        
        if s_masked.numel() == 0:
            return torch.tensor(0.0, device=student_patch_tokens.device, requires_grad=True)
        
        t_centered = t_masked - self.center.squeeze()
        t_probs = F.softmax(t_centered / teacher_temp, dim=-1).detach()
        s_logits = F.log_softmax(s_masked / self.student_temp, dim=-1)
        loss = torch.sum(-t_probs * s_logits, dim=-1).mean()
        
        return loss