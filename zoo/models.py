import torch.nn as nn
import torch
import numpy as np
from zoo.backbones import get_backbone

#headuri etc

class upBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class SegmentatorCoronare(nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone, in_channels=in_channels, pretrained=pretrained)
        
        #todo: verificare pentru outputurile backbone-urilor ca sa seteze corect head-ul
    