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
    
class ActionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 136 input nodes (8 discrete + 64 X + 64 Y)
        self.layer1 = nn.Linear(136, 256)
        self.layer2 = nn.Linear(256, 256)
        
        # 2,048 output nodes (4 target layers * 256 gamma + 4 target layers * 256 beta)
        self.output_layer = nn.Linear(256, 2048)

    def forward(self, action_vector):
        x = torch.relu(self.layer1(action_vector))
        x = torch.relu(self.layer2(x))
        
        # Raw linear output for the modulators (no activation function at the end)
        return self.output_layer(x)
    
class PredictorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flattened Input (16 colors * 64 * 64) -> 256 hidden nodes
        self.layer1 = nn.Linear(65536, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        
        # Final output predicting the 64x64x16 grid (65,536 nodes)
        self.output_layer = nn.Linear(256, 65536)

    def forward(self, grid_tensor, film_params):
        # Flatten the grid
        x = grid_tensor.view(grid_tensor.size(0), -1)
        
        # Split the 2,048-node film_params into 4 chunks of 512
        # Each chunk contains 256 gamma (multipliers) and 256 beta (shifts)
        params = torch.split(film_params, 512, dim=-1)
        
        # Layer 1 Processing & FiLM Injection
        gamma1, beta1 = torch.split(params[0], 256, dim=-1)
        x = torch.relu(self.layer1(x))
        x = (x * gamma1) + beta1  
        
        # Layer 2 Processing & FiLM Injection
        gamma2, beta2 = torch.split(params[1], 256, dim=-1)
        x = torch.relu(self.layer2(x))
        x = (x * gamma2) + beta2
        
        # Layer 3 Processing & FiLM Injection
        gamma3, beta3 = torch.split(params[2], 256, dim=-1)
        x = torch.relu(self.layer3(x))
        x = (x * gamma3) + beta3
        
        # Layer 4 Processing & FiLM Injection
        gamma4, beta4 = torch.split(params[3], 256, dim=-1)
        x = torch.relu(self.layer4(x))
        x = (x * gamma4) + beta4
        
        # Output raw logits (CrossEntropyLoss will handle the probabilities)
        return self.output_layer(x)