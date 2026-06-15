import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label

def encode_grid(raw_grid):
    # Convert input array to a PyTorch long tensor
    grid_tensor = torch.tensor(raw_grid, dtype=torch.long)
    
    # Perform one-hot encoding across 16 color dimensions (0-15)
    one_hot = F.one_hot(grid_tensor, num_classes=16)
    
    # Permute from (64, 64, 16) to (16, 64, 64) for spatial network layers
    return one_hot.permute(2, 0, 1).float()

def get_action_mask(raw_grid, background_color=0):
    # Create a base mask of 4104 zeros (8 discrete + 4096 spatial)
    mask = torch.zeros(4104, dtype=torch.bool)
    
    # Allow the 8 discrete actions by default (indices 0 to 7)
    mask[:8] = True 
    
    # Find contiguous color clusters ignoring the background
    labeled_grid, num_features = label(raw_grid != background_color)
    
    for i in range(1, num_features + 1):
        # Get all coordinates for this specific object cluster
        coords = np.argwhere(labeled_grid == i)
        if len(coords) > 0:
            # Choose the first coordinate (top-left) as the representative click point
            rep_y, rep_x = coords[0]
            # Map (y, x) to the flattened 4096 spatial index (offset by the 8 discrete nodes)
            flat_index = 8 + (rep_y * 64 + rep_x)
            mask[flat_index] = True
            
    return mask

def unify_action(action_id, x=0, y=0):
    # Create the empty 136-node vector (8 discrete + 64 X + 64 Y)
    vector = torch.zeros(136, dtype=torch.float32)
    
    # 1. One-hot encode the chosen discrete action (Indices 0-7)
    vector[action_id] = 1.0
    
    # 2. One-hot encode spatial coordinates ONLY if Action 6 is chosen
    if action_id == 6:
        # X coordinate offset begins at index 8
        vector[8 + x] = 1.0
        
        # Y coordinate offset begins at index 72 (8 + 64)
        vector[72 + y] = 1.0
        
    return vector