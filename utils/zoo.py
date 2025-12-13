import torch
import torch.nn as nn
import torchvision
import torchsummary as summary

def DoubleConv(in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
    """DoubleConv cu InstanceNorm (mai bun pentru batch size mic)"""
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
        self.bottleneck = DoubleConv(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
            #encoder
            enc1 = self.enc1(x)
            x = self.max_pool2d(enc1)
            enc2 = self.enc2(x)
            x = self.max_pool2d(enc2)
            enc3 = self.enc3(x)
            x = self.max_pool2d(enc3)
            bottleneck = self.bottleneck(x)
            #decoder
            x = self.upconv3(bottleneck)
            x = torch.cat((x, enc3), dim=1)
            x = self.dec3(x)
            x = self.upconv2(x)
            x = torch.cat((x, enc2), dim=1)
            x = self.dec2(x)
            x = self.upconv1(x)
            x = torch.cat((x, enc1), dim=1)
            x = self.dec1(x)
            return self.conv_last(x)
        

from torchsummary import summary

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_channels=1, out_channels=1).to(device)
    summary(model, (1, 256, 256))  # Input shape: (channels, height, width)
    x = torch.randn((1, 1, 256, 256)).to(device)
    preds = model(x)
    print(preds.shape)