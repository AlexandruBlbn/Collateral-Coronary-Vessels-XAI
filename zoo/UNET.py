import torch
import torch.nn as nn
import torchvision
import torchsummary as summary

def DoubleConv(in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        norm_layer(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        norm_layer(out_channels),
        nn.SiLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 512)  # Max 512 în bottleneck
        
        # Decoder: upconv reduce canale, apoi concatenez cu skip, dann dec
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 512, 256)  # 256 (upconv) + 512 (skip enc4) = 768
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 256, 128)  # 128 (upconv) + 256 (skip enc3) = 384
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 128, 64)    # 64 (upconv) + 128 (skip enc2) = 192
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + 64, 32)     # 32 (upconv) + 64 (skip enc1) = 96
        
        self.conv_last = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, x):

        enc1 = self.enc1(x)
        x = self.max_pool2d(enc1)
        
        enc2 = self.enc2(x)
        x = self.max_pool2d(enc2)
        
        enc3 = self.enc3(x)
        x = self.max_pool2d(enc3)
        
        enc4 = self.enc4(x)
        x = self.max_pool2d(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(x)
        
        # Decoder cu skip connections
        x = self.upconv4(bottleneck)
        x = torch.cat([x, enc4], dim=1)  
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)  
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)  
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1) 
        x = self.dec1(x)
        
        # Output layer
        x = self.conv_last(x)
        
        return x

from torchsummary import summary

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_channels=1, out_channels=1).to(device)
    summary(model, (1, 256, 256))  # Input shape: (channels, height, width)
    x = torch.randn((1, 1, 256, 256)).to(device)
    preds = model(x)
    print(preds.shape)