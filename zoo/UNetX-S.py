import torch
import torch.nn as nn
from torchinfo import summary


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class TokenizedMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.drop = nn.Dropout(drop)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(hidden_channels)
    
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + identity


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 2, 2)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.up(x)
        x = self.norm(x)
        return x


class StageBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=2, mlp_ratio=4, drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            TokenizedMLP(
                in_channels=in_channels,
                hidden_channels=int(in_channels * mlp_ratio),
                drop=drop
            ) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNeXt(nn.Module):
    def __init__(
        self, 
        in_channels=3,
        num_classes=1,
        base_channels=32,
        depths=[1, 1, 1, 1],
        mlp_ratio=4,
        drop_rate=0.1,
        attention=False
    ):
        super().__init__()
        self.attention = attention
        
        channels = [base_channels * (2 ** i) for i in range(len(depths))]
        self.stem = ConvBNReLU(in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        # Stage 1
        self.encoder_stages.append(
            StageBlock(channels[0], num_blocks=depths[0], mlp_ratio=mlp_ratio, drop=drop_rate)
        )
        for i in range(1, len(depths)):
            self.downsample_layers.append(
                DownsampleBlock(channels[i-1], channels[i])
            )
            self.encoder_stages.append(
                StageBlock(channels[i], num_blocks=depths[i], mlp_ratio=mlp_ratio, drop=drop_rate)
            )
        self.bottleneck = StageBlock(
            channels[-1], 
            num_blocks=depths[-1], 
            mlp_ratio=mlp_ratio, 
            drop=drop_rate
        )
        self.upsample_layers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        for i in range(len(depths) - 1, 0, -1):
            self.upsample_layers.append(
                UpsampleBlock(channels[i], channels[i-1])
            )
            if self.attention:
                self.attention_gates.append(
                    AttentionBlock(F_g=channels[i-1], F_l=channels[i-1], F_int=channels[i-1] // 2)
                )
            self.decoder_stages.append(
                nn.Sequential(
                    ConvBNReLU(channels[i-1] * 2, channels[i-1], kernel_size=1, padding=0),
                    StageBlock(channels[i-1], num_blocks=depths[i-1], mlp_ratio=mlp_ratio, drop=drop_rate)
                )
            )
        self.head = nn.Sequential(
            ConvBNReLU(channels[0], channels[0], kernel_size=3, padding=1),
            nn.Conv2d(channels[0], num_classes, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        skip_connections = []
        x = self.encoder_stages[0](x)
        skip_connections.append(x)
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.encoder_stages[i+1](x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[:-1]
        
        for i in range(len(self.upsample_layers)):
            x = self.upsample_layers[i](x)
            skip = skip_connections[-(i+1)]
            
            if self.attention:
                skip = self.attention_gates[i](g=x, x=skip)
            
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_stages[i](x)
        
        x = self.head(x)
        
        return x


def UNeXt_S(in_channels=1, num_classes=1, base_channels=32, depths=[1, 1, 1, 1], mlp_ratio=4, drop_rate=0.5, attention=False):
    return UNeXt(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        depths=depths,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attention=attention
    )
    
    
if __name__ == "__main__":
    model = UNeXt_S(in_channels=1, num_classes=1)
    summary(model, input_size=(1, 1, 256, 256))
