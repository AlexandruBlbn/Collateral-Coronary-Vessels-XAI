import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class HierarchicalDualStreamStage(nn.Module):
    """
    A single reusable stage that processes spatial and context streams in parallel,
    fuses them using PixelShuffle, and transitions (downsamples) for the next stage.
    """
    def __init__(self, in_channels, out_channels, attn_dim=384, num_heads=6):
        super().__init__()
        
        # --- 1. SPATIAL BRANCH ---
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # --- 2. CONTEXT BRANCH (ViT) ---
        self.patch_size = 8
        # Dynamically patchifies based on current spatial resolution
        self.patch_embed = nn.Conv2d(in_channels, attn_dim, kernel_size=8, stride=8)
        
        # Native PyTorch Transformer (Automatically uses Flash Attention in PyTorch 2.0+)
        self.attn_block = nn.TransformerEncoderLayer(
            d_model=attn_dim, 
            nhead=num_heads, 
            dim_feedforward=attn_dim * 4, 
            batch_first=True, 
            activation="gelu",
            norm_first=True
        )
        
        # --- 3. FUSION (PixelShuffle) ---
        self.unfolded_features = 16 
        prep_channels = self.unfolded_features * (self.patch_size ** 2) # 16 * 64 = 1024
        
        self.channel_prep = nn.Sequential(
            nn.Conv2d(attn_dim, prep_channels, kernel_size=1),
            nn.BatchNorm2d(prep_channels),
            nn.ReLU(inplace=True)
        )
        self.pixel_shuffle = nn.PixelShuffle(self.patch_size)
        
        # --- 4. MIXER ---
        self.mix_conv = nn.Sequential(
            nn.Conv2d(in_channels + self.unfolded_features, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # --- 5. TRANSITION (Replaces MaxPool) ---
        # Strided convolution learns the downsampling while doubling the channel capacity
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x shape: (B, C_in, H, W)
        
        # Spatial Pass
        spatial_out = self.double_conv(x)
        
        # Context Pass
        patches = self.patch_embed(x) 
        B, C_attn, H_grid, W_grid = patches.shape
        
        # Flatten for attention: (B, C, H, W) -> (B, H*W, C)
        seq = patches.view(B, C_attn, -1).transpose(1, 2)
        attn_seq = self.attn_block(seq)
        
        # Unflatten back to grid: (B, H*W, C) -> (B, C, H, W)
        attn_grid = attn_seq.transpose(1, 2).view(B, C_attn, H_grid, W_grid)
        
        # Fusion Pass
        attn_upsampled = self.pixel_shuffle(self.channel_prep(attn_grid))
        fused = torch.cat([spatial_out, attn_upsampled], dim=1)
        mixed = self.mix_conv(fused)
        
        # Transition Pass (Downsample)
        out_next_stage = self.downsample(mixed) 
        
        # Return both the downsampled tensor (for the next stage) 
        # and the mixed high-res tensor (for the U-Net Decoder skip connections later)
        return out_next_stage, mixed


class CoronaryEncoder(nn.Module):
    """
    The full hierarchical encoder tying the stages together.
    """
    def __init__(self, in_channels=3, base_channels=64, num_stages=4):
        super().__init__()
        
        # Entry Stem: Gets the raw RGB/Grayscale image ready for Stage 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dynamically build the hierarchical stages
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for _ in range(num_stages):
            next_channels = current_channels * 2 # Double the channels at each step
            stage = HierarchicalDualStreamStage(
                in_channels=current_channels, 
                out_channels=next_channels,
                attn_dim=384 
            )
            self.stages.append(stage)
            current_channels = next_channels

    def forward(self, x):
        # Extract initial spatial features
        x = self.stem(x)
        
        skips = []
        for i, stage in enumerate(self.stages):
            # Pass through the stage
            x, skip_connection = stage(x)
            skips.append(skip_connection)
            
        # Returns the deep bottleneck tensor and the high-res skip connections
        return x, skips

def print_parameter_breakdown(model):
    """
    Calculates and prints the trainable parameters for the entire network
    and breaks it down by hierarchical stages.
    """
    print("\n" + "="*45)
    print(f"{'ARCHITECTURE PARAMETER BREAKDOWN':^45}")
    print("="*45)
    
    total_params = 0
    
    # 1. Count Stem Parameters
    stem_params = sum(p.numel() for p in model.stem.parameters() if p.requires_grad)
    print(f"{'Entry Stem':<25} | {stem_params:>15,} params")
    total_params += stem_params
    
    # 2. Count Each Stage's Parameters
    for i, stage in enumerate(model.stages):
        stage_params = sum(p.numel() for p in stage.parameters() if p.requires_grad)
        print(f"{f'Stage {i+1} (Dual-Stream)':<25} | {stage_params:>15,} params")
        total_params += stage_params
        
    print("-" * 45)
    
    # 3. Print Total
    print(f"{'TOTAL TRAINABLE PARAMS':<25} | {total_params:>15,} params")
    print("="*45 + "\n")
    
    # Quick VRAM Estimation for bfloat16
    # 1 parameter in bfloat16 = 2 bytes. 
    # AdamW optimizer states require ~4 bytes per parameter.
    # Total memory footprint per parameter during training ≈ 6 bytes.
    est_vram_model_mb = (total_params * 6) / (1024 ** 2)
    print(f"[Hardware Check] Estimated Base VRAM for Model + Optimizer (bf16): ~{est_vram_model_mb:.2f} MB")
    print("*(Note: This does not include the memory required for the activation tensors during the forward/backward pass, which will consume the bulk of your 32GB GPU).*")



import torch
from torchinfo import summary
# Assuming CoronaryEncoder and HierarchicalDualStreamStage are already defined above

if __name__ == '__main__':
    print("Initializing Dual-Stream Hierarchical Encoder...")
    model = CoronaryEncoder(in_channels=3, base_channels=64, num_stages=4)
    
    # Define the batch size and image dimensions you plan to train with
    batch_size = 4
    input_shape = (batch_size, 3, 512, 512)
    
    # Generate the comprehensive summary table
    print("\nGenerating Architecture Summary...")
    summary(
        model, 
        input_size=input_shape,
        # Display the shapes going in, coming out, parameter counts, and FLOPs
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        # Depth controls how far down into the nested blocks it prints. 
        # Depth=4 will show the internal layers of your Transformer and Double Convs.
        depth=4,
        # Formats the large numbers with commas for readability
        row_settings=["var_names"] 
    )
    
    # Initialize the model
    model = CoronaryEncoder(in_channels=3, base_channels=64, num_stages=4)
    model.eval() 
    
    # Print the Parameter Breakdown
    print_parameter_breakdown(model)
    
    # Create a dummy tensor simulating a 512x512 angiogram batch (Batch=1)
    dummy_input = torch.randn(1, 3, 512, 512)
    print(f"Input Image Shape: {dummy_input.shape}\n")
    
    # Forward Pass
    with torch.no_grad():
        bottleneck, skips = model(dummy_input)
        
    # Print the Mathematical Flow
    print("--- TENSOR SHAPE TRACKER ---")
    for i, skip in enumerate(skips):
        print(f"Stage {i+1} Output (Skip Connection): {skip.shape}")
    print(f"Final Bottleneck Output: {bottleneck.shape}\n")
    
    # Plotting the Activations
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Dual-Stream Hierarchical Encoder Activations (Channel Mean)", fontsize=16)
    
    # Plot Input
    axes[0].imshow(dummy_input[0, 0].numpy(), cmap='gray')
    axes[0].set_title("Input Image (512x512)")
    axes[0].axis('off')
    
    # Plot Stages
    for i, skip in enumerate(skips):
        # Take the mean across the channel dimension [Batch, Channels, H, W] -> [H, W]
        activation_map = skip[0].mean(dim=0).numpy()
        
        axes[i+1].imshow(activation_map, cmap='viridis')
        axes[i+1].set_title(f"Stage {i+1} Fused Features\n{skip.shape[2]}x{skip.shape[3]} (Channels: {skip.shape[1]})")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.show()