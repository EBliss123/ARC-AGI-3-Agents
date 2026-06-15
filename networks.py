import torch
import torch.nn as nn

class PlannerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flattened Input (16 colors * 64 * 64) -> 256 hidden nodes
        self.layer1 = nn.Linear(65536, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        
        # 4,104 output nodes (8 discrete actions + 4096 spatial coordinates)
        self.output_layer = nn.Linear(256, 4104)

    def forward(self, grid_tensor, action_mask):
        # Flatten the (Batch, 16, 64, 64) grid into a 65,536-node vector
        x = grid_tensor.view(grid_tensor.size(0), -1) 
        
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        
        # Get raw action probabilities
        raw_logits = self.output_layer(x)
        
        # Apply the dynamic guardrail mask (sets invalid moves to negative infinity)
        masked_logits = raw_logits.masked_fill(~action_mask, float('-inf'))
        
        # Return proper probabilities where invalid moves are now exactly 0%
        return torch.softmax(masked_logits, dim=-1)