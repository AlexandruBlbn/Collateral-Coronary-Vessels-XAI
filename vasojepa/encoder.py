import sys
sys.path.append(".")
from utils.helpers import *
import timm
import segmentation_models_pytorch as smp

#trebuie sa fac ca sa returnez modelul, sa incerc sa extrag featureuri de pe diferite layere si sa returnez de pe layerele embeddigurile

class Encoder(nn.Module):
    def __init__(self, model='tiny_vit_21m_224', pretrain=False):
        super(Encoder, self).__init__()
        
        self.model = timm.create_model(
            model_name=model,
            pretrained=pretrain,
            cache_dir='/models',
            num_classes=0,
            in_chans=1,
            features_only=True
          )
        
    def forward(self, x):
        features = self.model(x)
        features0 = features[0]
        features1 = features[1]
        features2 = features[2]
        features3 = features[3]
        
        B, C0, H0, W0 = features0.shape
        B, C1, H1, W1 = features1.shape
        B, C2, H2, W2 = features2.shape
        B, C3, H3, W3 = features3.shape
        
        f_f0 = features0.permute(0, 2, 3, 1).reshape(B, H0*W0, C0) #C H W B -> B H*W C
        f_f1 = features1.permute(0, 2, 3, 1).reshape(B, H1*W1, C1) #C H W B -> B H*W C
        f_f2 = features2.permute(0, 2, 3, 1).reshape(B, H2*W2, C2) #C H W B -> B H*W C
        f_f3 = features3.permute(0, 2, 3, 1).reshape(B, H3*W3, C3) #C H W B -> B H*W C
        
        return f_f0, f_f1, f_f2, f_f3 
        #
    
    def get_structure(self):
        x = torch.randn(1, 1, 224, 224)
        features = self.model(x)
        for i, feature in enumerate(features):
            print(f"Stage {i}: {feature.shape}")
        

# model = Encoder(model='tiny_vit_21m_224', pretrain=False)
# print(model.get_structure())
# x = torch.randn(1, 1, 224, 224)
# features = model(x)
# for i in range(len(features)):
#     print(f"f_stage_{i}: {features[i].shape}")

# Stage 0: torch.Size([1, 96, 56, 56])
# Stage 1: torch.Size([1, 192, 28, 28])
# Stage 2: torch.Size([1, 384, 14, 14])
# Stage 3: torch.Size([1, 576, 7, 7])
# None
# f_stage_0: torch.Size([1, 3136, 96])
# f_stage_1: torch.Size([1, 784, 192])
# f_stage_2: torch.Size([1, 196, 384])
# f_stage_3: torch.Size([1, 49, 576])