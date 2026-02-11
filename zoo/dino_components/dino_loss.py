import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        
    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update(teacher_output)
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    def forward(self, student_output, teacher_output, epoch):

        teacher_temp = 0.04 + (0.07 - 0.04) * (epoch / 100)
        
        teacher_probs = self.softmax_center_teacher(teacher_output, teacher_temp)
        student_logits = F.log_softmax(student_output / self.student_temp, dim=-1)
        
        # Cross Entropy: -Sum(P_teacher * log(P_student))
        loss = torch.sum(-teacher_probs * student_logits, dim=-1).mean()
        return loss

    @torch.no_grad()
    def apply_center_update(self, teacher_output):
        # Update center with EMA
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        batch_center = batch_center / len(teacher_output)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)