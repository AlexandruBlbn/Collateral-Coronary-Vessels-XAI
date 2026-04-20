import torch
import torch.nn as nn
import timm
from torchinfo import summary
import numpy as np
from typing import Optional, Sequence

class ViT_16windows256(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.model = vit_model
        
    def forward(self, x):
        x = self.model.forward_features(x)
        if self.model.global_pool == '':
            x = x[:, 1:, :]
        B, N, D = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(1,2).reshape(B, D, H, W)
        return x
            
class SWIN_permute(nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.model = swin_model
        
    def forward(self, x):
        x = self.model.forward_features(x)
        return x.permute(0,3,1,2)


class FeatureListToNCHW(nn.Module):
    """
    Wrapper ensuring intermediate features are returned in NCHW format.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def _to_nchw(feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4:
            return feat
        # Robust NHWC -> NCHW conversion used by some timm backbones.
        if feat.shape[-1] > feat.shape[1] and feat.shape[-1] > feat.shape[2]:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x):
        feats = self.model(x)
        if isinstance(feats, (list, tuple)):
            return [self._to_nchw(f) for f in feats]
        return self._to_nchw(feats)


def get_backbone(
    model_name='tf_efficientnetv2_s',
    in_channels=1,
    pretrained=False,
    print_summary=False,
    return_intermediates: bool = False,
    out_indices: Optional[Sequence[int]] = None,
):
    '''
    Returns a backbone feature extractor returning [B, C, H, W] dense spatial grid
    '''
    kwargs = {}
    if 'vit' in model_name or 'swin' in model_name:
        if return_intermediates:
            kwargs['features_only'] = True
            kwargs['out_indices'] = tuple(out_indices) if out_indices is not None else (2, 3)
        else:
            kwargs['global_pool'] = ''
            kwargs['num_classes'] = 0
        if 'swin' in model_name:
            kwargs['dynamic_img_size'] = True
    else:
        # For efficientnet / convnext, we want spatial features.
        kwargs['features_only'] = True
        # Extract at stride 16 (usually before the final stage) for better spatial resolution on small crops
        # In timm, out_indices=[-2] skips the 32x downsampling stage.
        kwargs['out_indices'] = list(out_indices) if out_indices is not None else [-2]
        
    model = timm.create_model(
        model_name, 
        pretrained=pretrained,
        in_chans=in_channels,
        **kwargs
    )

    if return_intermediates:
        model = FeatureListToNCHW(model)
    elif 'vit' in model_name:
        model = ViT_16windows256(model)
    elif 'swin' in model_name:
        model = SWIN_permute(model)
                
    if print_summary:
        print(summary(model, (1, in_channels, 256, 256)))
    
    return model

if __name__ == "__main__":
    m = get_backbone('tf_efficientnetv2_s')
    x = torch.randn(2, 1, 224, 224)
    # The output is a list of features because features_only=True
    # We only asked for out_indices=[-2], so there should be 1 tensor in the list.
    out = m(x)
    print(out[0].shape)