import sys
import os
import torch
import yaml
import time
import importlib.util

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Model dynamically
model_path = os.path.join(project_root, 'zoo', 'UNetX-S.py')
spec = importlib.util.spec_from_file_location("UNetX_S_Module", model_path)
unext_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unext_module)
UNeXt_S = unext_module.UNeXt_S

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def benchmark():
    config_path = os.path.join(project_root, 'config', 'UNet_config.yaml')
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Benchmarking on: {device}")
    
    # Initialize Model with config parameters
    model = UNeXt_S(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        base_channels=config['model']['base_channels'],
        depths=config['model']['depths'],
        mlp_ratio=config['model']['mlp_ratio'],
        drop_rate=0.0 # No dropout during inference
    ).to(device)
    model.eval()
    
    # Dummy Input (Batch size 1 for single image inference)
    input_size = config['model']['input_size']
    # Create a random tensor of shape (1, 1, 256, 256)
    dummy_input = torch.randn(1, 1, input_size, input_size).to(device)
    
    # Warmup GPU (first few passes are always slower due to initialization)
    print("--> Warming up GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    iterations = 200
    print(f"--> Running {iterations} iterations...")
    
    # Synchronize CUDA to get accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    fps = 1 / avg_time
    
    print(f"\nResults for {config['model']['name']}:")
    print(f"  Average Latency: {avg_time*1000:.2f} ms per image")
    print(f"  Throughput:      {fps:.2f} FPS (Frames Per Second)")

if __name__ == "__main__":
    benchmark()