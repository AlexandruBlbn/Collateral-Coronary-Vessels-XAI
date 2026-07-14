import sys
sys.path.append(".")
from utils.helpers import *
from vasojepa.encoder import Encoder
from vasojepa.predictor import Predictor
from vasojepa.lds import LDSBranch
from vasojepa.cglt import CGLTRegularizer

#Model structure.

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.lds = LDSBranch()
        self.cglt = CGLTRegularizer()
        


    def forward(self, x, prior, epoch, total_epochs):
        f0, f1, f2, f3 = self.encoder(x)
        loss_lds, vessel_score = self.lds(f2.detach(), prior, epoch, total_epochs)
        vessel_score = vessel_score.detach()
        loss_cglt = self.cglt(f2, vessel_score.detach())
        
        #preparation of the context / target for predicotr
        B, N, dim = f2.shape
        device = f2.device
        N_ctx = int(N * 0.65)
        N_tgt = N - N_ctx

        context_idx = []
        target_idx  = []
        for _ in range(B):
            perm = torch.randperm(N, device=device)
            context_idx.append(perm[:N_ctx])
            target_idx.append(perm[N_ctx:])
            
        context_idx = torch.stack(context_idx)  # [B, N_ctx]
        target_idx  = torch.stack(target_idx)   # [B, N_tgt]

        K_context = context_idx.shape[1]
        K_target  = target_idx.shape[1]
        
        idx_context = context_idx.unsqueeze(-1).expand(B, K_context, dim)
        idx_target  = target_idx.unsqueeze(-1).expand(B, K_target, dim)
        
        context_tokens = torch.gather(f2, 1, idx_context)
        target_tokens  = torch.gather(f2, 1, idx_target)
        
        pred_f2, pred_f3 = self.predictor(context_tokens, N_tgt)
        
        target_f2 = target_tokens.detach()
        row = target_idx // 14
        col = target_idx % 14
        f3_idx = (row // 2) * 7 + (col // 2)              # [B, N_tgt]
        f3_idx = f3_idx.unsqueeze(-1).expand(B, N_tgt, f3.shape[-1])
        target_f3 = torch.gather(f3, 1, f3_idx).detach()  # [B, N_tgt, 576]
        vs_tgt = torch.gather(vessel_score, 1, target_idx)
        w = (1.0 + 2.0 * vs_tgt).unsqueeze(-1)                      # [B, N_tgt, 1]

        loss_dense = ((pred_f2 - target_f2).pow(2) * w).mean() + ((pred_f3 - target_f3).pow(2) * w).mean()
        loss = 1.0 * loss_dense + 20.0 * loss_cglt + 0.2 * loss_lds
        
        return loss, {"dense": loss_dense.item(), "cglt": loss_cglt.item(), "lds": loss_lds.item()}