import sys
sys.path.append(".")
from utils.helpers import *


class Denoiser(nn.Module):
    def __init__(self, input_proj=384):
        super(Denoiser, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_proj + 1, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, input_proj)     # output: same dim as clean z
            )
    
    def forward(self, z_noisy, sigma):
        x = torch.cat([z_noisy, sigma], dim=-1)
        return self.net(x)
    
class VesselHead(nn.Module):
        def __init__(self, in_dim=384):
            super(VesselHead, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.net(x)
        

class LDSBranch(nn.Module):
    def __init__(self, noise_levels=(0.1, 0.3, 0.5, 1.0), lambda_start=1.0, lambda_end=0.1):
        super(LDSBranch, self).__init__()
        self.denoiser    = Denoiser()
        self.vessel_head = VesselHead()
        self.noise_levels = noise_levels
        self.lambda_start = lambda_start
        self.lambda_end   = lambda_end
    
    def get_lambda(self, epoch, total_epochs):
        t = epoch / max(total_epochs - 1, 1)    # goes from 0.0 to 1.0
        return self.lambda_start + t * (self.lambda_end - self.lambda_start)
    
    def forward(self, z, prior, epoch, total_epochs):
        lambda_ = self.get_lambda(epoch, total_epochs)
        B, N, dim = z.shape
        z_flat = z.reshape(B*N, dim)
        idx = torch.randint(0, len(self.noise_levels), (1,)).item()
        sigma = self.noise_levels[idx]
        noise = torch.randn_like(z_flat)
        
        z_noisy = z_flat + sigma * noise
        
        sigma_input = torch.full((B*N, 1), sigma, device=z.device, dtype=z.dtype)
        
        z_denoised = self.denoiser(z_noisy, sigma_input)
       
        loss_denoised = F.mse_loss(z_denoised, z_flat)
        
        vessel_logits = self.vessel_head(z_denoised)
        prior_flat = prior.reshape(B*N, 1)
        loss_vessel = F.binary_cross_entropy_with_logits(vessel_logits, prior_flat)

        vessel_score = torch.sigmoid(vessel_logits).reshape(B, N)  # [B, N], values in [0, 1]
        return loss_denoised + lambda_ * loss_vessel, vessel_score


        