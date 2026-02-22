import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- SETUP CĂI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from zoo.backbones import get_backbone

# --- BLOCURI STANDARD CNN ---
class ConvBlock(nn.Module):
    """Bloc clasic de U-Net cu 2 convoluții, BatchNorm și ReLU"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- ATTENTION GATE ---
class AttentionGate(nn.Module):
    """
    Filtrează trăsăturile din Skip Connection (x) folosind semnalul de la nivelul inferior (g).
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # F_g: Canalele semnalului de gating (de jos, upsampled)
        # F_l: Canalele skip connection (local features)
        # F_int: Canalele intermediare (de obicei jumătate)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int, affine=True)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int, affine=True)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1, affine=True),
            nn.Sigmoid() # Harta de atenție (0 la 1)
        )
        
        self.silu = nn.SiLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Aliniere dimensională (Safety check)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.silu(g1 + x1)
        psi = self.psi(psi)
        
        # Înmulțim element cu element skip connection-ul cu harta de atenție
        return x * psi

# --- BLOC DECODER CU ATTENTION ---
class AttentionUNetUpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        # 1. Upsample
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        
        # 2. Attention Gate
        # Gating signal (g) are in_c // 2 canale după upsample
        # Skip connection (x) are skip_c canale
        self.att = AttentionGate(F_g=in_c // 2, F_l=skip_c, F_int=out_c // 2)
        
        # 3. Procesare finală cu ConvBlock standard
        self.conv = ConvBlock((in_c // 2) + skip_c, out_c)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Safety resize pentru 'x' înainte de attention
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        # Aplicăm Attention Gate: filtrăm 'skip' folosind 'x' ca ghid
        skip_filtered = self.att(g=x, x=skip)
        
        # Concatenare
        x = torch.cat([x, skip_filtered], dim=1)
        
        # Convoluții standard
        return self.conv(x)

# --- MODELUL PRINCIPAL ---
class SegmentatorCoronare(nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1, num_classes=1, input_size=256):
        super().__init__()
        self.backbone = get_backbone(model_name=backbone, in_channels=in_channels, pretrained=pretrained, print_summary=False)
        
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, input_size, input_size)
            feats = self.backbone(dummy)
            enc_channels = feats.shape[1]
            spatial_dim = feats.shape[2] 
            

        self.skip1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.skip2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.skip3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.skip4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        # 2. BOTTLENECK
        self.bottleneck = ConvBlock(enc_channels, 512)
        
        # 3. ADAPTARE DINAMICĂ
        self.needs_initial_up = (spatial_dim < 16)
        if self.needs_initial_up:
            self.up_initial = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
            
        # 4. DECODER CU ATTENTION
        # Folosim AttentionUNetUpBlock în loc de UNetUpBlock simplu
        self.up4 = AttentionUNetUpBlock(in_c=512, skip_c=256, out_c=256) 
        self.up3 = AttentionUNetUpBlock(in_c=256, skip_c=128, out_c=128) 
        self.up2 = AttentionUNetUpBlock(in_c=128, skip_c=64, out_c=64)   
        self.up1 = AttentionUNetUpBlock(in_c=64, skip_c=32, out_c=32)    
        
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        orig_size = x.shape[2:]
        
        # --- Encoder ---
        s1 = self.skip1(x)
        p1 = self.pool1(s1)
        
        s2 = self.skip2(p1)
        p2 = self.pool2(s2)
        
        s3 = self.skip3(p2)
        p3 = self.pool3(s3)
        
        s4 = self.skip4(p3)
        
        # --- Backbone ---
        feats = self.backbone(x)
        feats = self.bottleneck(feats)
        
        if self.needs_initial_up:
            feats = self.up_initial(feats)
            
        # --- Decoder + Attention ---
        d4 = self.up4(feats, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        
        out = self.out_conv(d1)
        
        # Safety resize
        if out.shape[2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
            
        return out

# --- U-Net++ NESTED BLOCK ---
class NestedConvBlock(nn.Module):
    """
    Standard Convolution Block used in the nested pathways.
    (Conv -> InstanceNorm -> SiLU) * 2
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- U-Net++ (NESTED U-NET) DECODER ---
class SegmentatorCoronarePlusPlus(nn.Module):
    def __init__(self, backbone='swinv2_tiny_window16_256', pretrained=False, in_channels=1, num_classes=1, input_size=256):
        super().__init__()
        # 1. Backbone (Custom)
        self.backbone = get_backbone(model_name=backbone, in_channels=in_channels, pretrained=pretrained, print_summary=False)
        
        # Determine bottleneck channels
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, input_size, input_size)
            feats = self.backbone(dummy)
            enc_channels = feats.shape[1]      # e.g., 768 for Swin-Tiny
            spatial_dim = feats.shape[2]       # e.g., 8x8
            
        # Standard filter sizes for decoder levels
        # We assume 4 levels: 32 -> 64 -> 128 -> 256 -> (Backbone 512)
        filters = [32, 64, 128, 256, 512] 

        # --- ENCODER SIDE (SKIP GENERATION) ---
        # We manually create the encoder pyramid because Swin/ViT usually only gives the final output.
        # This acts as a "CNN Encoder" parallel to the Transformer backbone.
        self.conv0_0 = NestedConvBlock(in_channels, filters[0])
        self.pool0 = nn.MaxPool2d(2)
        
        self.conv1_0 = NestedConvBlock(filters[0], filters[1])
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2_0 = NestedConvBlock(filters[1], filters[2])
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3_0 = NestedConvBlock(filters[2], filters[3])
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck Adapter (from Backbone to Decoder)
        self.bottleneck = NestedConvBlock(enc_channels, filters[4])
        
        # Dynamic Upsample if backbone output is too small (<16x16)
        self.needs_initial_up = (spatial_dim < 16)
        if self.needs_initial_up:
            self.up_initial = nn.ConvTranspose2d(filters[4], filters[4], kernel_size=2, stride=2)

        # --- NESTED DECODER PATHWAYS (L=Layer, N=Node) ---
        # Level 0 (32 filters)
        self.conv0_1 = NestedConvBlock(filters[0] + filters[1], filters[0])
        self.conv0_2 = NestedConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv0_3 = NestedConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv0_4 = NestedConvBlock(filters[0]*4 + filters[1], filters[0])

        # Level 1 (64 filters)
        self.conv1_1 = NestedConvBlock(filters[1] + filters[2], filters[1])
        self.conv1_2 = NestedConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv1_3 = NestedConvBlock(filters[1]*3 + filters[2], filters[1])

        # Level 2 (128 filters)
        self.conv2_1 = NestedConvBlock(filters[2] + filters[3], filters[2])
        self.conv2_2 = NestedConvBlock(filters[2]*2 + filters[3], filters[2])

        # Level 3 (256 filters)
        self.conv3_1 = NestedConvBlock(filters[3] + filters[4], filters[3])

        # Upsampling Layers
        self.up_to_3_1 = nn.ConvTranspose2d(filters[4], filters[4], kernel_size=2, stride=2)
        
        self.up_to_2_1 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2)
        self.up_to_2_2 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2)
        
        self.up_to_1_1 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        self.up_to_1_2 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        self.up_to_1_3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        
        self.up_to_0_1 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        self.up_to_0_2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        self.up_to_0_3 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        self.up_to_0_4 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)

        # Final Convolution (Deep Supervision is optional, here we just output the final node)
        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        orig_size = x.shape[2:]

        # --- 1. Backbone & Encoder Pyramid ---
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        
        # Backbone Feature Extraction
        feats = self.backbone(x)
        x4_0 = self.bottleneck(feats)
        if self.needs_initial_up:
            x4_0 = self.up_initial(x4_0)

        # --- 2. Nested Decoding ---
        
        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0.shape)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0.shape)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0.shape)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0.shape)], 1))

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0.shape)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0.shape)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0.shape)], 1))

        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0.shape)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0.shape)], 1))

        # Column 4 (Output Column)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0.shape)], 1))

        # Final Output
        out = self.final(x0_4)

        if out.shape[2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)

        return out

    def _up(self, x, size):
        """Helper for simple bilinear upsampling to target size"""
        return F.interpolate(x, size=size[2:], mode='bilinear', align_corners=False)
