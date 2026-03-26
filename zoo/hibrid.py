import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

# ==========================================
# 1. COMPONENTE DE BAZĂ (UTILITIES)
# ==========================================

class LayerNorm2d(nn.Module):
    """ Implementare LayerNorm pentru tensori CNN (B, C, H, W). """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNextBlock(nn.Module):
    """ Bloc Inverted Residual flexibil pentru Encoder. """
    def __init__(self, dim, kernel_size=7, drop_path=0., mlp_ratio=4):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, mlp_ratio * dim, kernel_size=1) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(mlp_ratio * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

# ==========================================
# 2. ENCODER-UL (ANGIO PROGRESSIVE STEM)
# ==========================================

class AngioProgressiveStem(nn.Module):
    """ Stem Convoluțional Progresiv, complet customizabil. """
    def __init__(self, in_chans=1, dims=[32, 64, 128, 256], depths=[2, 2, 2], kernel_size=7, drop_path_rate=0.0):
        super().__init__()
        assert len(dims) == len(depths) + 1, "Lista 'dims' trebuie să aibă un element în plus față de 'depths'."
        self.dims = dims
        self.depths = depths

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, padding=1, bias=False),
            LayerNorm2d(dims[0]),
            nn.GELU()
        )
        
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur_dp = 0

        for i in range(len(depths)):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1, bias=False)
            )
            self.downsample_layers.append(downsample)
            
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(ConvNextBlock(dim=dims[i+1], kernel_size=kernel_size, drop_path=dp_rates[cur_dp]))
                cur_dp += 1
            self.stages.append(nn.Sequential(*stage_blocks))
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x) 
        skip_connections = [x]
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i < len(self.stages) - 1:
                skip_connections.append(x)
        return x, skip_connections

# ==========================================
# 3. DECODORUL (LIGHTWEIGHT PROGRESSIVE)
# ==========================================

class DepthwiseSeparableConv(nn.Module):
    """ Bloc ultra-ușor pentru fuziunea trăsăturilor în Decodor. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convoluție spațială (cost foarte mic)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.norm = LayerNorm2d(in_channels)
        self.act = nn.GELU()
        # Convoluție pe canale (pointwise) care reduce/ajustează dimensiunea tensorului
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pwconv(self.act(self.norm(self.dwconv(x))))

class DecoderUpBlock(nn.Module):
    """ Bloc de Upsampling + Concatenare Skip Connection + Fuziune """
    def __init__(self, in_channels_from_up, in_channels_from_skip, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Canalele se adună datorită concatenării pe axa 1 (C)
        total_in_channels = in_channels_from_up + in_channels_from_skip
        
        # Folosim 2 blocuri separabile pentru a rafina marginile vaselor
        self.fusion = nn.Sequential(
            DepthwiseSeparableConv(total_in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        # Concatenăm tensorul upsampled cu harta de înaltă rezoluție din encoder
        x = torch.cat([x, skip_feature], dim=1) 
        x = self.fusion(x)
        return x

class LightweightProgressiveDecoder(nn.Module):
    """
    Reconstruiește imaginea de la 64x64 înapoi la 512x512, 
    folosind piramida de skip connections.
    """
    def __init__(self, encoder_dims, num_classes=1):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        
        # Inversăm lista de dimensiuni a encoder-ului pentru a construi decodorul simetric
        # Ex pt standard: [32, 64, 128, 256] -> decodorul începe cu 256.
        enc_dims = list(reversed(encoder_dims)) 
        
        # enc_dims[0] = 256 (bottleneck), enc_dims[1] = 128 (skip 3), etc.
        for i in range(len(enc_dims) - 1):
            in_ch_up = enc_dims[i]
            in_ch_skip = enc_dims[i+1]
            out_ch = enc_dims[i+1] # Reducem numărul de canale treptat
            
            self.up_blocks.append(
                DecoderUpBlock(in_channels_from_up=in_ch_up, 
                               in_channels_from_skip=in_ch_skip, 
                               out_channels=out_ch)
            )

        # Capul de predicție final: transformă din canalele de bază (ex. 32) în num_classes (ex. 1)
        self.prediction_head = nn.Conv2d(enc_dims[-1], num_classes, kernel_size=1)

    def forward(self, bottleneck, skip_connections):
        x = bottleneck
        # Parcurgem blocurile de upsampling și le cuplăm cu skip connection-ul corespunzător
        # Atenție: skip_connections vin în ordinea [512, 256, 128], deci trebuie inversate și ele!
        skips_reversed = list(reversed(skip_connections))
        
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skips_reversed[i])
            
        x = self.prediction_head(x)
        return x

# ==========================================
# 4. ANSAMBLUL COMPLET (ENCODER + DECODER)
# ==========================================

class AngioSegmenter(nn.Module):
    """
    Modelul complet care leagă Encoder-ul Progresiv de Decodorul Ușor.
    Acesta este modulul pe care îl vei folosi în bucla de antrenament.
    """
    def __init__(self, in_chans=1, num_classes=1, dims=[32, 64, 128, 256], depths=[2, 2, 2], drop_path_rate=0.1):
        super().__init__()
        
        self.encoder = AngioProgressiveStem(
            in_chans=in_chans, 
            dims=dims, 
            depths=depths, 
            drop_path_rate=drop_path_rate
        )
        
        self.decoder = LightweightProgressiveDecoder(
            encoder_dims=dims, 
            num_classes=num_classes
        )

    def forward(self, x):
        # 1. Extragerea trăsăturilor
        bottleneck, skips = self.encoder(x)
        
        # 2. Reconstrucția
        logits = self.decoder(bottleneck, skips)
        
        # Returnăm tensorul brut (logits). 
        # Funcția de activare (Sigmoid) se va aplica de obicei în loss function (ex: BCEWithLogitsLoss).
        return logits

# ==========================================
# 5. TESTARE ȘI REZUMAT
# ==========================================
if __name__ == "__main__":
    # Inițializăm modelul complet (configurația standard)
    model = AngioSegmenter(in_chans=1, num_classes=1, dims=[32, 64, 128, 256], depths=[2, 2, 2])
    
    # Numărarea parametrilor
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = enc_params + dec_params
    
    print(f"Parametri Encoder: {enc_params / 1e6:.2f} M")
    print(f"Parametri Decoder: {dec_params / 1e6:.2f} M (Extrem de ușor!)")
    print(f"Total Parametri:   {total_params / 1e6:.2f} M")
    
    # Testarea unui Forward Pass complet
    print("\nSimulare Forward Pass cu Angiografie 512x512:")
    dummy_input = torch.randn(2, 1, 512, 512) # Batch 2, 1 Canal, 512x512
    
    # Executăm inferența
    output_mask = model(dummy_input)
    
    print(f"Intrare originală: {dummy_input.shape}")
    print(f"Ieșire predicție (Logits): {output_mask.shape}")
    assert output_mask.shape == (2, 1, 512, 512), "Rezoluția măștii nu se potrivește cu intrarea!"