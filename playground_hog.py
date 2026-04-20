import torch
import torch.nn.functional as F
import math

class HOGLayer(torch.nn.Module):
    def __init__(self, nbins=9, pool=8):
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        weight_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        weight_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('weight_x', weight_x.view(1, 1, 3, 3))
        self.register_buffer('weight_y', weight_y.view(1, 1, 3, 3))

    @torch.no_grad()
    def forward(self, x):
        # x: (B, 1, H, W)
        g_x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), self.weight_x)
        g_y = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), self.weight_y)
        
        magnitude = torch.sqrt(g_x**2 + g_y**2 + 1e-6)
        angle = torch.atan2(g_y, g_x) * 180 / math.pi
        angle = torch.where(angle < 0, angle + 180, angle)
        
        bins = torch.linspace(0, 180, self.nbins + 1, device=x.device)
        hog_feats = []
        for i in range(self.nbins):
            mask = (angle >= bins[i]) & (angle < bins[i+1])
            hog_feats.append(magnitude * mask.float())
            
        hog = torch.cat(hog_feats, dim=1) # (B, nbins, H, W)
        hog = F.avg_pool2d(hog, kernel_size=self.pool, stride=self.pool)
        # Normalize
        hog = hog / (torch.norm(hog, p=2, dim=1, keepdim=True) + 1e-6)
        return hog

hog = HOGLayer()
x = torch.randn(2, 1, 256, 256)
out = hog(x)
print(out.shape)
