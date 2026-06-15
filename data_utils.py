import torch
import torch.nn.functional as F
import numpy as np

def encode_grid(raw_grid):
    # Convert input array to a PyTorch long tensor
    grid_tensor = torch.tensor(raw_grid, dtype=torch.long)
    
    # Perform one-hot encoding across 16 color dimensions (0-15)
    one_hot = F.one_hot(grid_tensor, num_classes=16)
    
    # Permute from (64, 64, 16) to (16, 64, 64) for spatial network layers
    return one_hot.permute(2, 0, 1).float()