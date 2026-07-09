import sys
sys.path.append(".")
from utils.helpers import *
import math


# --- Utilities ----------------------------------------------------------------

def pairwise_sq_distances(A, Z):
    """
    Squared Euclidean distances between every anchor in A and every point in Z.
    A : (Na, D)
    Z : (N,  D)
    Returns: (Na, N)
    """
    # ||a - z||^2 = ||a||^2 + ||z||^2 - 2 * <a, z>
    a2 = (A * A).sum(dim=-1, keepdim=True)       # (Na, 1)
    z2 = (Z * Z).sum(dim=-1, keepdim=True).T      # (1,  N)
    return (a2 + z2 - 2.0 * A @ Z.T).clamp_min(0.0)


def build_scale_ladder(Z, n_scales=5, r_min_factor=0.05, eps=1e-8):
    """
    Build a log-spaced scale ladder from the data distribution.
    r_max is estimated as the median pairwise distance of a random subsample.
    r_min = r_max * r_min_factor.
    Returns: (n_scales+1,) tensor, decreasing.
    """
    N = Z.shape[0]
    sub = Z[:min(N, 512)].detach()
    with torch.no_grad():
        d2 = pairwise_sq_distances(sub, sub)
        mask = ~torch.eye(sub.shape[0], dtype=torch.bool, device=Z.device)
        r_max = d2[mask].sqrt().median().clamp_min(eps)
    r_min = r_max * r_min_factor
    ladder = torch.exp(
        torch.linspace(
            math.log(r_max.item()), math.log(r_min.item()),
            n_scales + 1, device=Z.device
        )
    )
    return ladder   # (n_scales+1,) decreasing


# --- Projector ----------------------------------------------------------------

class CGLTProjector(nn.Module):
    """
    MLP that projects encoder features to a low-dimensional space where
    CGLT is computed.

    Why a separate projector?
    Pairwise distances in the raw 384-dim space are noisy and high-variance.
    At 32D the geometric structure of the manifold is exposed and CGLT
    gradients become meaningful.

    The output is L2-normalized so all embeddings lie on the unit sphere,
    giving stable scale-ladder estimates across batches.

    Input:  (B*N, in_dim)   e.g. 384 from Stage 2
    Output: (B*N, proj_dim) e.g. 32, L2-normalized
    """
    def __init__(self, in_dim=384, hidden_dim=128, proj_dim=32):
        super(CGLTProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return torch_f.normalize(z, dim=-1)   # L2 normalize onto unit sphere


# --- CGLT Carleson Loss -------------------------------------------------------

class CGLTLoss(nn.Module):
    """
    Gaussian-smoothed Carleson loss from the UR-JEPA paper.

    Core idea: define the Gaussian-smoothed density at anchor x and scale r:

        theta_r(x) = (1 / N*r^n) * sum_j exp(-||z_j - x||^2 / (2*r^2))

    On a uniform n-rectifiable manifold, theta_r(x) is constant across
    scales (the r^n denominator exactly cancels). The loss measures the
    log-increment of this density between adjacent dyadic scales:

        delta_k(x) = log theta_{r_{k+1}}(x) - log theta_{r_k}(x)
        L_CGLT = log_step * sum_k mean_x |delta_k(x)|^2

    Anti-collapse: a collapsed embedding (all z_j equal) has wildly
    varying density across scales, making this loss very large.
    Zero: only when embeddings lie on a uniform n-dimensional manifold.

    Args:
        n         : target intrinsic dimension (7 for vessel trees)
        n_scales  : number of dyadic scale levels
        n_anchors : number of anchor points to evaluate at (subset of N)
        eps       : numerical safety floor
    """
    def __init__(self, n=7, n_scales=5, n_anchors=64, eps=1e-8):
        super(CGLTLoss, self).__init__()
        self.n         = n
        self.n_scales  = n_scales
        self.n_anchors = n_anchors
        self.eps       = eps

    def _log_theta(self, d2, r, N):
        """
        log theta_r(x) for all anchors at one scale r.
        d2 : (Na, N)  squared distances
        r  : scalar
        Returns: (Na,)
        """
        log_w_sum = torch.logsumexp((-0.5 / (r * r)) * d2, dim=-1)
        return log_w_sum - self.n * torch.log(r.clamp_min(self.eps)) - math.log(max(N, 1))

    def forward(self, Z):
        """
        Z : (N, proj_dim)  L2-normalized projected embeddings
        Returns: scalar
        """
        N = Z.shape[0]

        # Random anchors
        n_a = min(self.n_anchors, N)
        A = Z[torch.randperm(N, device=Z.device)[:n_a]]   # (Na, D)

        r_ladder = build_scale_ladder(Z, n_scales=self.n_scales, eps=self.eps)
        d2 = pairwise_sq_distances(A, Z)   # (Na, N)

        # log theta at each scale: (Na, K+1)
        log_thetas = torch.stack(
            [self._log_theta(d2, r, N) for r in r_ladder], dim=-1
        )

        # Log increments between adjacent scales: (Na, K)
        delta = log_thetas[:, 1:] - log_thetas[:, :-1]

        # Carleson loss: sum over scales of mean squared increment
        loss = delta.pow(2).mean(0).sum()

        # Riemann weight: log step in scale space
        log_step = torch.log(r_ladder[0] / r_ladder[-1].clamp_min(self.eps)) / max(self.n_scales, 1)

        return loss * log_step


# --- AD Regularity Anchor -----------------------------------------------------

class ADRegularity(nn.Module):
    """
    Ahlfors-David regularity anchor that pairs with CGLTLoss.

    CGLTLoss checks that density is flat ACROSS SCALES at each anchor.
    ADRegularity checks that density is flat ACROSS ANCHORS at each scale.

    Together: density is uniform everywhere at all zoom levels.
    This enforces the AD-regularity assumption the CGLT theorem requires.

    Returns: variance of log theta_r(x) across anchors, averaged over scales.
    """
    def __init__(self, n=7, n_scales=5, n_anchors=64, eps=1e-8):
        super(ADRegularity, self).__init__()
        self.n         = n
        self.n_scales  = n_scales
        self.n_anchors = n_anchors
        self.eps       = eps

    def forward(self, Z):
        """
        Z : (N, proj_dim)
        Returns: scalar
        """
        N = Z.shape[0]

        n_a = min(self.n_anchors, N)
        A = Z[torch.randperm(N, device=Z.device)[:n_a]]

        r_ladder = build_scale_ladder(Z, n_scales=self.n_scales, eps=self.eps)
        d2 = pairwise_sq_distances(A, Z)

        log_thetas = []
        for r in r_ladder:
            log_w_sum = torch.logsumexp((-0.5 / (r * r)) * d2, dim=-1)
            lt = log_w_sum - self.n * torch.log(r.clamp_min(self.eps)) - math.log(max(N, 1))
            log_thetas.append(lt)

        lt = torch.stack(log_thetas, dim=-1)   # (Na, K+1)

        # Variance across anchors at each scale, averaged over scales
        return lt.var(unbiased=False, dim=0).mean()


# --- Combined Regularizer -----------------------------------------------------

class CGLTRegularizer(nn.Module):
    """
    Full CGLT regularizer for VasoJEPA.

    Wraps: CGLTProjector + CGLTLoss + ADRegularity into a single module.
    Call it on flattened Stage 2 encoder features every training step.

    Usage in train.py:
        regularizer = CGLTRegularizer().to(device)

        # f2 shape: [B, N, 384] from encoder
        z = f2.reshape(-1, 384)         # [B*N, 384]
        loss_reg = regularizer(z)       # scalar

    Args:
        in_dim      : encoder feature dim (384 for TinyViT Stage 2)
        proj_dim    : CGLT projection dim (32 recommended)
        n           : target intrinsic dimension (7 for vessel features)
        n_scales    : dyadic scale levels
        n_anchors   : evaluation anchors per batch
        lambda_cglt : weight for Carleson loss
        lambda_ad   : weight for AD regularity anchor (0.1 is enough)
    """
    def __init__(self, in_dim=384, proj_dim=32, n=7, n_scales=5, n_anchors=64, lambda_cglt=1.0, lambda_ad=0.1):
        super(CGLTRegularizer, self).__init__()
        self.lambda_cglt = lambda_cglt
        self.lambda_ad   = lambda_ad
        self.projector     = CGLTProjector(in_dim=in_dim, proj_dim=proj_dim)
        self.cglt_loss     = CGLTLoss(n=n, n_scales=n_scales, n_anchors=n_anchors)
        self.ad_regularity = ADRegularity(n=n, n_scales=n_scales, n_anchors=n_anchors)

    def forward(self, x):
        z = self.projector(x)                         # (B*N, 32) L2-normalized
        loss_cglt = self.cglt_loss(z)
        loss_ad   = self.ad_regularity(z)

        return self.lambda_cglt * loss_cglt + self.lambda_ad * loss_ad


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, D = 4, 196, 384
    f2 = torch.randn(B * N, D).to(device)
    reg = CGLTRegularizer().to(device)
    loss = reg(f2)
    print(f2.shape)
    print(loss)