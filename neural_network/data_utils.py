import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label

def encode_grid(raw_grid):
    grid_tensor = torch.tensor(raw_grid, dtype=torch.long)
    one_hot = F.one_hot(grid_tensor, num_classes=16).float()
    
    y_coords = torch.linspace(-1.0, 1.0, 64).view(64, 1).expand(64, 64).unsqueeze(-1)
    x_coords = torch.linspace(-1.0, 1.0, 64).view(1, 64).expand(64, 64).unsqueeze(-1)
    
    # --- DYNAMIC CLUSTER MAPPING ---
    # Group contiguous pixels of the same color into strictly logical sets
    grid_np = np.array(raw_grid)
    cluster_ids = np.zeros_like(grid_np, dtype=np.int32)
    current_id = 0
    
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for c in np.unique(grid_np):
        color_mask = (grid_np == c).astype(int)
        labeled, num_features = label(color_mask, structure=structure)
        for i in range(1, num_features + 1):
            cluster_ids[labeled == i] = current_id
            current_id += 1
            
    cluster_tensor = torch.tensor(cluster_ids, dtype=torch.float32).view(64, 64).unsqueeze(-1)
    
    # 18 channels (Physics) + 1 channel (Cluster ID) = 19 channels
    combined = torch.cat([one_hot, x_coords, y_coords, cluster_tensor], dim=-1)
    return combined.view(-1, 19)

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
    vector = torch.zeros(10, dtype=torch.float32)
    if action_id < 8:
        vector[action_id] = 1.0
        
    # Normalize spatial clicks from 0-63 to -1.0 to 1.0
    vector[8] = (x / 63.0) * 2.0 - 1.0 if action_id == 6 else 0.0
    vector[9] = (y / 63.0) * 2.0 - 1.0 if action_id == 6 else 0.0
    return vector