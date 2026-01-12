import torch
import sys
import os

sys.path.append('/home/tehreem/nn-gpt/out/nngpt/llm/epoch/A0/synth_nn/B0')

try:
    import new_nn
    model = new_nn.load(device='cpu')
    print("Model loaded successfully.")
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            print(f"Parameter: {name}")
            print(f"  Layout: {param.layout}")
            print(f"  Is Sparse: {param.is_sparse}")
            print(f"  Format: {param.layout}")
            # Check if it has zeros
            sparsity = (param == 0).float().mean().item()
            print(f"  Actual Zero Sparsity: {sparsity:.2%}")
            break
except Exception as e:
    print(f"Error: {e}")
