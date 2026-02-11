import copy
import torch
import torch.nn as nn
from zoo.dino_components.dino_head import DINOHead
from zoo.dino_components.dino_clstoken_loss import DINOLoss
from zoo.dino_components.koleo_loss import KoLeoLoss
from zoo.dino_components.ibot_patch_loss import iBOTPatchLoss

class DINOv3Wrapper(nn.Module):
    def __init__(self, backbone, feature_dim, head_dim=65536):
        super().__init__()
        
        self.student_backbone = backbone

        self.student_head = DINOHead(
            in_dim=feature_dim,
            out_dim=head_dim,
            nlayers=3,
            hidden_dim=2048,
            bottleneck_dim=256
        )


        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = copy.deepcopy(self.student_head)

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        self.dino_loss = DINOLoss(out_dim=head_dim)
        self.koleo_loss = KoLeoLoss()
        self.ibot_loss = iBOTPatchLoss()

    def forward(self, inputs):
        student_cls_tokens = []
        for img in inputs:
            feat = self.student_backbone(img) # (B, feature_dim)
            student_cls_tokens.append(feat)
            
        student_cls = torch.cat(student_cls_tokens) 
        student_out = self.student_head(student_cls)

        with torch.no_grad():
            teacher_cls_tokens = []
            for img in inputs[:2]: # Doar primele 2
                feat = self.teacher_backbone(img)
                teacher_cls_tokens.append(feat)
            
            teacher_cls = torch.cat(teacher_cls_tokens)
            teacher_out = self.teacher_head(teacher_cls)

        return student_out, teacher_out
    
    @torch.no_grad()
    def update_teacher(self, m=0.996):
        # Update Backbone
        for param_q, param_k in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.data)
        
        # Update Head
        for param_q, param_k in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.data)