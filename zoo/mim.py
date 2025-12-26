import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from zoo.backbones import get_backbone
except ImportError:
    from backbones import get_backbone
    
class SimMIM(nn.Module):
    def __init__(self, backbone_name='swinv2_tiny_window16_256', in_channels=1, encoder_stride=32):
        super().__init__()
        self.encoder = get_backbone(model_name=backbone_name, in_channels=in_channels, pretrained=False)
        self.encoder_stride = encoder_stride 
        with torch.no_grad():
            dummy = torch.randn(1, 1, 256, 256)
            feats = self.encoder(dummy)
            encoder_dim = feats.shape[1] 
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder_dim, 
                out_channels=(encoder_stride ** 2) * in_channels, 
                kernel_size=1
            ),
            nn.PixelShuffle(encoder_stride),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, mask):
        x_masked = x * (1 - mask.unsqueeze(1))
        z = self.encoder(x_masked)
        x_rec = self.decoder(z)
        loss = F.l1_loss(x, x_rec, reduction='none')
        mask_expanded = mask.unsqueeze(1)
        loss = (loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-5)
        
        return loss, x_rec