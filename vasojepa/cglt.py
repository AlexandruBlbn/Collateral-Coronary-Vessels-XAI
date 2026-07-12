import sys
sys.path.append(".")
from utils.helpers import *


def pairwise_square_distances(A, Z):
    #a2 -2ab +b2
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


set_seed(42)
Z_random = torch.randn(200, 384)
Z_colapsed = torch.ones(200, 384) * 0.5
basis = torch.randn(7, 384)
basis = basis / basis.norm(dim=1, keepdim=True) 
coeffs = torch.randn(2000, 7) *2 -1                   
Z_manifold = coeffs @ basis  

for name, Z in [("random", Z_random), ("collapsed", Z_colapsed), ("7D manifold", Z_manifold)]:
    d2 = pairwise_square_distances(Z, Z)
    k = 5
    r = d2.sqrt().topk(k+1, dim=-1, largest=False).values[:, -1].mean().item()
    if r < 1e-6:
        r = 1.0
    loss = cglt_loss(d2, r, d=7).mean().item()
    print(f"{name:>12s}  |  r={r:.2f}  |  CGLT loss={loss:.4f}")