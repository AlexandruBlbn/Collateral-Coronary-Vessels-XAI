import sys
sys.path.append(".")
from utils.helpers import *
from vasojepa.encoder import Encoder
from vasojepa.predictor import Predictor
import copy

#Model structure — VasoJEPA: JEPA with optional EMA teacher, vessel-aware masking,
#and vesselness anchor head. Three independent toggles:
#  use_ema         — EMA teacher (I-JEPA style) vs single encoder (LeJEPA style)
#  vessel_masking  — bias target patch selection toward vessel patches (prior guides WHAT to predict)
#  vessel_anchor   — auxiliary head: encoder features must decode to vesselness prior (anti-collapse + vessel encoding)

class Model(nn.Module):
    def __init__(self, use_ema=True, vessel_masking=False, vessel_anchor=False,
                 ema_start=0.996, ema_end=1.0, lambda_anchor=0.1, lambda_consistency=0.05):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()

        self.use_ema = use_ema
        self.vessel_masking = vessel_masking
        self.vessel_anchor = vessel_anchor
        self.lambda_anchor = lambda_anchor
        self.lambda_consistency = lambda_consistency

        if use_ema:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            self.ema_start = ema_start
            self.ema_end = ema_end

        if vessel_anchor:
            self.vessel_head = nn.Linear(384, 1)  #f2: 384-dim, 14x14 patches

    @torch.no_grad()
    def update_target_encoder(self, progress):
        """progress: float in [0, 1], fraction of total optimizer steps completed."""
        if not self.use_ema:
            return
        m = self.ema_start + (self.ema_end - self.ema_start) * progress
        for p_t, p_o in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def forward(self, x, prior, epoch, total_epochs):
        f0, f1, f2, f3 = self.encoder(x)

        #targets: EMA encoder (if use_ema) or detached online encoder (EMA-free / LeJEPA)
        if self.use_ema:
            with torch.no_grad():
                _, _, tf2, tf3 = self.target_encoder(x)
        else:
            tf2 = f2.detach()
            tf3 = f3.detach()

        B, N, dim = f2.shape
        device = f2.device
        N_ctx = int(N * 0.65)
        N_tgt = N - N_ctx

        #patch selection: random (default) or vessel-biased (prior guides which patches to predict)
        if self.vessel_masking:
            prior_flat = prior.view(B, N)
            weights = prior_flat + 0.15  #base weight so non-vessel patches can still be sampled
            target_idx_list = []
            context_idx_list = []
            for b in range(B):
                tgt = torch.multinomial(weights[b], N_tgt, replacement=False)
                target_idx_list.append(tgt)
                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[tgt] = False
                ctx = mask.nonzero(as_tuple=True)[0]
                ctx = ctx[torch.randperm(len(ctx))]
                context_idx_list.append(ctx)
            target_idx = torch.stack(target_idx_list)
            context_idx = torch.stack(context_idx_list)
        else:
            context_idx = []
            target_idx = []
            for _ in range(B):
                perm = torch.randperm(N, device=device)
                context_idx.append(perm[:N_ctx])
                target_idx.append(perm[N_ctx:])
            context_idx = torch.stack(context_idx)
            target_idx = torch.stack(target_idx)

        #gather context tokens from online encoder (with gradient)
        K_context = context_idx.shape[1]
        idx_context = context_idx.unsqueeze(-1).expand(B, K_context, dim)
        context_tokens = torch.gather(f2, 1, idx_context)

        #predictor: context -> target predictions
        pred_f2, pred_f3 = self.predictor(context_tokens, N_tgt)

        #gather target features (always no-grad)
        idx_target = target_idx.unsqueeze(-1).expand(B, N_tgt, dim)
        target_f2 = torch.gather(tf2, 1, idx_target)

        row = target_idx // 14
        col = target_idx % 14
        f3_idx = (row // 2) * 7 + (col // 2)
        f3_idx = f3_idx.unsqueeze(-1).expand(B, N_tgt, tf3.shape[-1])
        target_f3 = torch.gather(tf3, 1, f3_idx)

        #dense prediction loss (primary JEPA objective)
        loss_dense = (pred_f2 - target_f2).pow(2).mean() + (pred_f3 - target_f3).pow(2).mean()
        loss = loss_dense

        #vesselness anchor: encoder features must decode to prior V
        #anti-collapse (exogenous signal breaks encoder-predictor collusion) + vessel structure encoding
        if self.vessel_anchor:
            prior_flat = prior.view(B, N)
            #anchor on online encoder (gradient to encoder + vessel_head) — dense, all patches
            vessel_pred_enc = self.vessel_head(f2).squeeze(-1)
            anchor_loss = (vessel_pred_enc - prior_flat).pow(2).mean()
            #consistency on predictor output (gradient to predictor + vessel_head) — target patches only
            prior_tgt = torch.gather(prior_flat, 1, target_idx)
            vessel_pred_pred = self.vessel_head(pred_f2).squeeze(-1)
            consistency_loss = (vessel_pred_pred - prior_tgt).pow(2).mean()
            loss = loss + self.lambda_anchor * anchor_loss + self.lambda_consistency * consistency_loss

        #collapse monitors — online encoder std vs target encoder std, per-channel, averaged
        f2_std = f2.detach().float().std(dim=(0, 1)).mean().item()
        tf2_std = tf2.detach().float().std(dim=(0, 1)).mean().item()

        loss_dict = {
            "dense":   loss_dense.item(),
            "f2_std":  f2_std,
            "tf2_std": tf2_std,
        }
        if self.vessel_anchor:
            loss_dict["anchor"] = anchor_loss.item()
            loss_dict["consistency"] = consistency_loss.item()

        return loss, loss_dict
