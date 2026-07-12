import sys
sys.path.append(".")
from utils.helpers import *

class Predictor(nn.Module):
    def __init__(self, stage2_dim=384, stage3_dim=576,  pred_dim=192, n_heads=8, n_layers=2):
        super(Predictor, self).__init__()
        self.input_projection = nn.Linear(stage2_dim, pred_dim)
        self.head_f3 = nn.Linear(pred_dim, stage3_dim)
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, pred_dim))
        self.transformer = nn.Transformer(
            d_model=pred_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            dropout=0,
            batch_first=True,
            activation='gelu'
            )
        
        self.output_projection = nn.Linear(pred_dim, stage2_dim)

        
    def forward(self, context_tokens, N_tgt):
        search = self.input_projection(context_tokens)
        B, N_ctx, _ = context_tokens.shape
        target = self.mask_tokens.expand(B, N_tgt, -1)
        output = self.transformer(src=search, tgt=target)
        output_f2 = self.output_projection(output)
        output_f3 = self.head_f3(output)
        return output_f2, output_f3
    
    
    
    