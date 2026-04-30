"""
SIGReg: Sketched Isotropic Gaussian Regularization

Implements the Epps-Pulley characteristic function test from
Balestriero & LeCun (2025), "LeJEPA: Provable SSL Without Heuristics".

The loss enforces that token embeddings match an isotropic N(0,1)
distribution in randomly projected 1D directions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """
    SIGReg: Epps-Pulley CF-based isotropy regularizer.

    For each random projection direction, compute the empirical CF
    of the projected embeddings and compare it to the theoretical
    N(0,1) CF via L2 distance.
    """

    def __init__(
        self,
        knots: int = 64,
        t_max: float = 4.0,
        num_projections: int = 16,
    ):
        super().__init__()
        self.knots = knots
        self.t_max = t_max
        self.num_projections = num_projections

        # Precompute evaluation grid t ∈ [0, t_max]
        self.register_buffer("t_grid", torch.linspace(0.0, t_max, knots))

        # Theoretical N(0, 1) CF: φ(t) = exp(-t²/2)
        self.register_buffer(
            "cf_target", torch.exp(-0.5 * self.t_grid ** 2)
        )

    def _sample_projections(self, d: int, device: torch.device) -> torch.Tensor:
        """Sample num_projections random unit-norm directions."""
        v = torch.randn(d, self.num_projections, device=device)
        return F.normalize(v, dim=0)

    def _empirical_cf(
        self, z: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute empirical CF for each projection direction.
        """
        proj = z @ V  # [N, P]

        t = self.t_grid.unsqueeze(0).unsqueeze(2)  # [1, knots, 1]
        p = proj.unsqueeze(1)                       # [N, 1, P]

        cos_vals = torch.cos(t * p)
        cf_emp = cos_vals.mean(dim=0).T  # [P, knots]

        return cf_emp

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            z: [B, d] or [N, d] — token embeddings

        Returns:
            loss: scalar tensor
        """
        if z.dim() > 2:
            z = z.reshape(-1, z.shape[-1])

        z = z - z.mean(dim=0, keepdim=True)

        d = z.shape[-1]
        V = self._sample_projections(d, z.device)

        cf_emp = self._empirical_cf(z, V)
        cf_target = self.cf_target.unsqueeze(0)

        loss = F.mse_loss(cf_emp, cf_target.expand_as(cf_emp))

        return loss
