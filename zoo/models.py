import torch.nn as nn
import torch
import numpy as np
from backbones import get_backbone

#headuri etc

class upBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class SegmentatorCoronare(nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone, in_channels=in_channels, pretrained=pretrained, summary=False)
        
        #todo: verificare pentru outputurile backbone-urilor ca sa seteze corect head-ul
        with torch.no_grad():
            dummy = torch.randn(1, 1, 256, 256)
            feats = self.backbone(dummy)
            enc_channels = feats.shape[1] #384 sau 768
            enc_size = feats.shape[2] #16 sau 8
            print(f'channels: {enc_channels}, size: {enc_size}')
        
        curr_size = enc_size
        curr_channels = enc_channels
        layers = []
        target_size = 256
        while curr_size < target_size:
            out_ch = max(64,curr_channels // 2)
            layers.append(upBlock(curr_channels, out_ch))
            curr_channels = out_ch
            curr_size = curr_size * 2
            
        self.decoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(curr_channels, 1, kernel_size=1)
        
    def forward(self, x):
        feats = self.backbone(x)
        x = self.decoder(feats)
        x = self.head(x)
        return x
    
    
    
class headClasificare(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #mean global
        self.max_pool = nn.AdaptiveMaxPool2d(1) #extract stenoza
        flatt_features = in_features * 2    
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatt_features,256    ),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x_average = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat([x_average, x_max], dim=1)
        return self.classifier(x)
    
    
class ClasificatorStenoza(nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1, num_classes=2):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone, in_channels=in_channels, pretrained=pretrained, summary=False)
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
