import torch
import torchvision
from torchinfo import summary
import timm


model = torchvision.models.convnext_tiny()
print("ConvNeXt Tiny Model Summary:")
summary(model, input_size=(1, 3, 224, 224)) 

model = torchvision.models.swin_v2_t()
print("Swin V2 Tiny Model Summary:")
summary(model, input_size=(1, 3, 224, 224))
print("ResNet50 Model Summary:")
model = timm.create_model('resnet50', pretrained=False)
summary(model, input_size=(1, 3, 224, 224))
print("ViT Tiny Model Summary:")
model = timm.create_model('vit_small_patch16_224', pretrained=False)
summary(model, input_size=(1, 3, 224, 224))

