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


if __name__ == "__main__":
    # --- Memory Estimation Tool ---
    # Adjust these values to test different configurations
    BATCH_SIZE = 32
    IMG_SIZE = 256
    IN_CHANNELS = 1
    MODEL_NAME = 'vit_small_patch16_224' # Options: convnext_tiny, swinv2_tiny_window16_256, vit_small_patch16_224

    model = get_backbone(model_name=MODEL_NAME, in_channels=IN_CHANNELS)
    summary(model, 
            input_size=(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            verbose=1)