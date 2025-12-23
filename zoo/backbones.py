import torch
import torch.nn as nn
import timm
from torchinfo import summary
import numpy as np



class ViT_16windows256(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.model = vit_model
        
    def forward(self, x):
        x = self.model.forward_features(x) #b, n_tokene, dim
        if self.model.global_pool == '':
          x=  x[:, 1:, :]  # rem cls token
        B, N, D = x.shape
        H = W = int(np.sqrt(N)) #16
        x = x.transpose(1,2).reshape(B, D, H, W)  # b, dim, h, w
        return x
            
class SWIN_permute(nn.Module):
    def __init__(self, swin_model):
        super().__init__()
        self.model = swin_model
    def forward(self, x):
        x = self.model.forward_features(x) #b, n_tokene, dim
        return x.permute(0,3,1,2)






def get_backbone(model_name='convnext_tiny', in_channels=1, pretrained=False, print_summary=False):
    '''
    Docstring for get_backbone
    
    :param model_name: convnext_tiny, swinv2_tiny_window16_256, vit_small_patch16_224
    '''
    kwargs = {}
    if 'vit' in model_name or 'swin' in model_name:
        kwargs['img_size'] = 256
    
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=0, 
        global_pool='' ,
        **kwargs
    )
    
    if in_channels == 1:
        if 'convnext' in model_name:
            old_layer = model.stem[0]
            # print(old_layer)
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None
            )
            model.stem[0] = new_layer
        
        elif 'swin' in model_name:
            old_layer = model.patch_embed.proj
            # print(old_layer)
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None
            )
            model.patch_embed.proj = new_layer
            
            nn.init.xavier_uniform_(new_layer.weight)
            if new_layer.bias is not None:
                nn.init.constant_(new_layer.bias, 0)
                
        elif 'vit' in model_name:
            old_layer = model.patch_embed.proj
            # print(old_layer)
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None
            )
            model.patch_embed.proj = new_layer
            nn.init.xavier_uniform_(new_layer.weight)
            if new_layer.bias is not None:
                nn.init.constant_(new_layer.bias, 0)
        
        if 'vit' in model_name:
            model = ViT_16windows256(model)
        if 'swin' in model_name:
            model = SWIN_permute(model)
                
    if print_summary == True:
        print(summary(model, (1, in_channels, 256, 256)))
    
    
    return model


def sanityCheck():
    vit_model = get_backbone(model_name='vit_small_patch16_224', in_channels=1, pretrained=False, print_summary=True)
    # swin_model = get_backbone(model_name='swinv2_tiny_window16_256', in_channels=1, pretrained=False, summary=False)
    # convnext_model = get_backbone(model_name='convnext_tiny', in_channels=1, pretrained=False, summary=False)
    
    x = torch.randn((1, 1, 256, 256))
    
    # vit_out = vit_model(x)
    # swin_out = swin_model(x)
    # convnext_out = convnext_model(x)
    
    # print(f'ViT output shape: {vit_out.shape}') #1 384 16 16
    # print(f'Swin output shape: {swin_out.shape}') #1 768 8 8
    # print(f'ConvNeXT output shape: {convnext_out.shape}') #1 768 8 8
    
# sanityCheck()