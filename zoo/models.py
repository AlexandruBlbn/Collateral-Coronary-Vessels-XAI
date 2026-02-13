import torch.nn as nn
import torch
import numpy as np
from backbones import get_backbone

class UNetrUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upscale_factor=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=upscale_factor)
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block(x)
        return x

class SegmentatorCoronare(nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1, num_classes=1):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone, in_channels=in_channels, pretrained=pretrained, print_summary=False)
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, 256, 256)
            feats = self.backbone(dummy)
            enc_channels = feats.shape[1]
        
        self.decoder_dim = 128
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels, self.decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_dim),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = UNetrUpBlock(self.decoder_dim, 64)
        self.up2 = UNetrUpBlock(64, 32)
        self.up3 = UNetrUpBlock(32, 16)
        self.up4 = UNetrUpBlock(16, 16)
        self.up5 = UNetrUpBlock(16, 16)
        
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.out_conv(x)
        return x
    

def test():
    model = SegmentatorCoronare(backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1, num_classes=1)
    du = torch.randn(1, 1, 256, 256)
    predict = model(du)
    print(predict.shape)
    
