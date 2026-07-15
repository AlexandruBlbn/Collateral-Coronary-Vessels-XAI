import sys
sys.path.append(".")
from utils.helpers import *


def pairwise_square_distances(A, Z):
    #a2 -2ab +b2
    with torch.autocast(device_type=A.device.type, enabled=False):
        A = A.float()
        Z = Z.float()
        A_sqr = (A*A).sum(dim=-1, keepdim=True)
        Z_sqr = (Z*Z).sum(dim=-1, keepdim=True).t()

        dot = A @ Z.t()

        dist = A_sqr - 2*dot + Z_sqr

        return dist.clamp_min(min=0.0)


def logarithm_theta(d2, r, d=7):
    #d2 = result of the pariwise square_distance
    #N number of points in z
    N = d2.shape[-1]
    return (torch.logsumexp(( -d2 / (2*r**2)), dim=-1) - math.log(N) - d * math.log(r))
   
def cglt_loss(d2, r, d):
    return (logarithm_theta(d2=d2, r=2*r)-logarithm_theta(d2=d2, r=r))**2






class CGLTRegularizer(nn.Module):
    def __init__(self, in_dim=384, proj_dim=32, n=7, k=5):
        super(CGLTRegularizer, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, proj_dim)
        )
        self.n = n
        self.k = k
        

    def build_scale_ladder(self, Z):
        with torch.no_grad():
            centroid = Z.mean(dim=0)
            dists = (Z - centroid).norm(dim=1)   # distance from each point to center
            r_max = dists.median().clamp(min=1e-6)
        r_min = r_max * 0.05
    
        log_scales = torch.linspace(
            torch.log(r_max).item(),
            torch.log(r_min).item(),
            self.k + 1,
            device=Z.device
        )
        return torch.exp(log_scales)   # (K+1,) decreasing
    
    def forward(self, f, vessel_score=None):
        B, N, D = f.shape
        f_flat = f.reshape(B * N, -1)
        z = self.proj(f_flat)                        # [B*N, 32]

        d2 = pairwise_square_distances(z, z)         # [B*N, B*N]
        scales = self.build_scale_ladder(z)          # [K+1]

        loss = torch.tensor(0.0, device=f.device)
        for k in range(self.k):
            lt_coarse = logarithm_theta(d2, scales[k],   self.n)  # [B*N]
            lt_fine   = logarithm_theta(d2, scales[k+1], self.n)  # [B*N]
            delta = lt_coarse - lt_fine                            # [B*N]

            if vessel_score is not None:
                w = (1.0 - 0.8 * vessel_score.reshape(B * N))     # [B*N]
                loss = loss + (delta ** 2 * w).mean()
            else:
                loss = loss + (delta ** 2).mean()

        return loss




















# set_seed(42)
# Z_random = torch.randn(200, 384)
# Z_colapsed = torch.ones(200, 384) * 0.5
# basis = torch.randn(7, 384)
# basis = basis / basis.norm(dim=1, keepdim=True) 
# coeffs = torch.randn(2000, 7) *2 -1                   
# Z_manifold = coeffs @ basis  


# for name, Z in [("random", Z_random), ("collapsed", Z_colapsed), ("7D manifold", Z_manifold)]:
#     d2 = pairwise_square_distances(Z, Z)
#     k = 5
#     r = d2.sqrt().topk(k+1, dim=-1, largest=False).values[:, -1].mean().item()
#     if r < 1e-6:
#         r = 1.0
#     loss = cglt_loss(d2, r, d=7).mean().item()
#     print(f"{name:>12s}  |  r={r:.2f}  |  CGLT loss={loss:.4f}")