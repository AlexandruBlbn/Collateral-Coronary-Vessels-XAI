import sys
sys.path.append(".")
from utils.helpers import *
import timm
from torch import nn

class Predictor(nn.Module):
    def __init__(self, stage2_dim=384, stage3_dim=576, pred_dim=192, depth=2, num_heads=6):
        super(Predictor,self).__init__()
        self.input_projection = nn.Linear(stage2_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim)) #masking token pos, learnable
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pred_dim,
            nhead=num_heads,
            dim_feedforward=pred_dim * 4,
            batch_first=True,
            dropout=0.0
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head_stage2 = nn.Linear(pred_dim, stage2_dim)  # predicts f2 targets
        self.head_stage3 = nn.Linear(pred_dim, stage3_dim)  # predicts f3 targets
        
        
    def forward(self, context_tokens, context_masks, target_masks):
        '''
        context_tokens [B, N_context, stage2_dim] - encoder output per visible patches/pos
        context_masks [B, N_context] which position is visible
        target_masks [B, N_target] which is to be predicted
        
        '''
        
        B, N_ctx, _ = context_tokens.shape
        N_target = target_masks.shape[1]
        context = self.input_projection(context_tokens) # B N_context pred_dim
        target = self.mask_token.expand(B, N_target, -1) # B N_target pred_dim
        
        x = torch.cat([context, target], dim=1) # context + target at channel 1 => N target+context pred_dim
        x = self.transformer(x)
        
        prediction = x[:, N_ctx:, :] # B, N_target, pred_Dim
        
        out_stage2= self.head_stage2(prediction) # B, N_target, stage2_dim
        out_stage3 = self.head_stage3(prediction) # B, N_target, stage3_dim
        
        return out_stage2, out_stage3