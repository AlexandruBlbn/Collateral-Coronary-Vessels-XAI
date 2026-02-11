import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        
        # Buffer pentru centru (Media mobilă a profesorului)
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        
    @torch.no_grad()
    def step_center(self, teacher_patch_tokens):
        # teacher_patch_tokens: (B, N, D)
        # Calculăm media pe batch curent
        batch_center = torch.mean(teacher_patch_tokens, dim=(0, 1), keepdim=True).unsqueeze(0) # (1, 1, D)
        
        # EMA Update
        # Center = m * Center + (1-m) * Batch_Mean
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks, epoch):
        """
        student_patch_tokens: (B, N, D) - Toate token-urile
        teacher_patch_tokens: (B, N, D) - Toate token-urile
        student_masks: (B, N) - Boolean (True = Mascat/Invizibil pentru student)
        """
        # Teacher Temp Schedule (Linear warmup)
        # De la 0.04 la 0.07 (valori standard DINO)
        teacher_temp = 0.04 + (0.07 - 0.04) * (epoch / 100)
        
        # 1. Update Center (folosind toți tokenii profesorului, nu doar cei mascați)
        self.step_center(teacher_patch_tokens)

        # 2. Selectăm doar tokenii MASCATI
        # Studentul trebuie să prezică tokenii pe care NU îi vede
        # Flatten pentru a selecta ușor
        B, N, D = student_patch_tokens.shape
        
        # Ne asigurăm că masca e boolean
        mask_bool = student_masks.bool()
        
        s_masked = student_patch_tokens[mask_bool] # (N_total_masked, D)
        t_masked = teacher_patch_tokens[mask_bool] # (N_total_masked, D)
        
        if s_masked.numel() == 0:
            return torch.tensor(0.0, device=student_patch_tokens.device, requires_grad=True)

        # 3. Teacher Targets (Sharpening + Centering)
        # Scădem centrul pentru a evita colapsul (toți tokenii să fie identici)
        t_centered = t_masked - self.center.squeeze()
        t_probs = F.softmax(t_centered / teacher_temp, dim=-1).detach()

        # 4. Student Predictions (Log Softmax)
        s_logits = F.log_softmax(s_masked / self.student_temp, dim=-1)

        # 5. Cross Entropy Loss
        loss = torch.sum(-t_probs * s_logits, dim=-1).mean()
        
        return loss