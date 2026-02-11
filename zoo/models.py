import torch.nn as nn
import torch
import numpy as np
from .backbones import get_backbone

#headuri etc

class upBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class SegmentatorCoronare(nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone, in_channels=in_channels, pretrained=pretrained, print_summary=False)
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, 256, 256)
            feats = self.backbone(dummy)
            enc_channels = feats.shape[1]
            enc_size = feats.shape[2]
            print(f'channels: {enc_channels}, size: {enc_size}')
        
        self.bridge = nn.Sequential(
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.SiLU(inplace=True),
        )
        
        curr_size = enc_size
        curr_channels = enc_channels
        layers = []
        target_size = 256
        while curr_size < target_size:
            out_ch = max(64, curr_channels // 2)
            layers.append(upBlock(curr_channels, out_ch))
            curr_channels = out_ch
            curr_size = curr_size * 2
        
        self.decoder = nn.Sequential(*layers)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(curr_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(curr_channels, curr_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(curr_channels // 2),
            nn.SiLU(inplace=True),
        )
        
        self.head = nn.Conv2d(curr_channels // 2, 1, kernel_size=1)
        
    def forward(self, x):
        feats = self.backbone(x)
        feats = self.bridge(feats)
        x = self.decoder(feats)
        x = self.final_conv(x)
        x = self.head(x)
        return x
    
    
    
class headClasificare(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        # Multi-scale pooling for richer feature representation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global context
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Salient features (stenoza)
        
        # Dual pooling: 2x features
        flatt_features = in_features * 2
        
        # Stronger architecture with higher dropout to prevent overfitting
        self.fc1 = nn.Linear(flatt_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(0.4)  # Increased for regularization
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(0.3)  # Increased for regularization
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.SiLU()
        self.drop3 = nn.Dropout(0.2)  # Additional layer for capacity
        
        self.fc_out = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Handle both CNN [B, C, H, W] and ViT [B, N, D] features
        if x.dim() == 3:  # ViT features [B, N, D]
            # For ViT, average over all tokens
            x = x.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, D, 1, 1]
        
        B, C = x.shape[0], x.shape[1]
        
        # Dual pooling: avg + max
        x_avg = self.avg_pool(x).view(B, C)  # [B, C]
        x_max = self.max_pool(x).view(B, C)  # [B, C]
        
        # Concatenate pooled features
        x = torch.cat([x_avg, x_max], dim=1)  # [B, 2C]
        
        # Classification layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)
        
        x = self.fc_out(x)
        
        return x
    
    
class ClasificatorStenoza(nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1, num_classes=2):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone, in_channels=in_channels, pretrained=pretrained, print_summary=False)
        with torch.no_grad():
            dummy = torch.randn(1, 1, 256, 256)
            feats = self.backbone(dummy)
            enc_channels = feats.shape[1] #384 sau 768
            print(f'channels: {enc_channels}')
        self.head = headClasificare(in_features=enc_channels, num_classes=num_classes)
    
    def forward(self, x):
        feats = self.backbone(x)
        x = self.head(feats)
        return x
            
# seg = SegmentatorCoronare(backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1)

    

#     print("SWIN")
#     model = SegmentatorCoronare(backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1)
#     dummy = torch.randn(2, 1, 256, 256)
#     out = model(dummy)
#     print(out.shape)
    
    
#     print('ConvNext')
#     model = SegmentatorCoronare(backbone='convnext_tiny', pretrained=False, in_channels=1)
#     dummy = torch.randn(2, 1, 256, 256)
#     out = model(dummy)
#     print(out.shape)
    
#     print('ViT')
#     model = SegmentatorCoronare(backbone='vit_small_patch16_224', pretrained=False, in_channels=1)
#     dummy = torch.randn(2, 1, 256, 256)
#     out = model(dummy)
#     print(out.shape)
