import torch
import torch.nn as nn
import torch.nn.functional as F

class GramLoss(nn.Module):
    def __init__(
        self,
        apply_norm=True,
        img_level=True,
        remove_neg=False,
        remove_only_teacher_neg=False,
    ):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.apply_norm = apply_norm
        self.img_level = img_level
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats):
        # Float casting for stability
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # 1. Normalize features (Cosine Similarity prep)
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)
            target_feats = F.normalize(target_feats, dim=-1)

        # 2. Flatten if needed (Global Gram vs Image-level Gram)
        # img_level=True means we compute Gram per image (B matrices of NxN)
        # img_level=False means we compute one giant Gram (1 matrix of BNxBN) - usually avoided due to memory
        if not self.img_level:
            output_feats = output_feats.flatten(0, 1)
            target_feats = target_feats.flatten(0, 1)

        # 3. Compute Gram Matrices (Similarity)
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # 4. Optional: Remove Negatives (ReLU on Gram)
        if self.remove_neg:
            target_sim = F.relu(target_sim)
            student_sim = F.relu(student_sim)
        elif self.remove_only_teacher_neg:
            # Zero out student where teacher is negative, and zero out teacher negatives
            mask = target_sim < 0
            target_sim[mask] = 0.0
            student_sim[mask] = 0.0
            # Also clamp student negatives? Official code logic:
            student_sim[student_sim < 0] = 0.0 

        # 5. MSE Loss
        return self.mse_loss(student_sim, target_sim)