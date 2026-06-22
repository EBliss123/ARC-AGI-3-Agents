import torch
import torch.nn as nn

class PlannerNetwork(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # Project the 18-channel pixel nodes into the mathematical transformer space
        self.pixel_embedding = nn.Linear(18, d_model)
        
        # The Self-Attention Brain (The Tactician)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4,104 output nodes (8 discrete actions + 4096 spatial coordinates)
        self.output_layer = nn.Linear(d_model, 4104)

    def forward(self, grid_nodes, action_mask):
        # grid_nodes shape: (Batch, 4096, 18)
        x = self.pixel_embedding(grid_nodes)
        
        # Self-Attention allows every pixel to analyze every other pixel instantly
        x = self.transformer(x)
        
        # Pool the 4096 grid nodes into a single global state vector by averaging them
        global_state = x.mean(dim=1) 
        
        # Predict the best action
        raw_logits = self.output_layer(global_state)
        masked_logits = raw_logits.masked_fill(~action_mask, float('-inf'))
        return torch.softmax(masked_logits, dim=-1)

class PredictorNetwork(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # Project both node types into the exact same mathematical dimension
        self.pixel_embedding = nn.Linear(18, d_model)
        self.action_embedding = nn.Linear(10, d_model)
        
        # The Self-Attention Brain (The Physics Observer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- THE DUAL-HEAD PHYSICS ENGINE ---
        # Head 1: Predicts if the pixel's condition was met to trigger a change (0 = No, 1 = Yes)
        self.change_mask_head = nn.Linear(d_model, 2)  
        
        # Head 2: Predicts what the new color will be (the 16 color channels)
        self.color_target_head = nn.Linear(d_model, 16) 

    def forward(self, grid_nodes, action_vector):
        # grid_nodes shape: (Batch, 4096, 18)
        # action_vector shape: (Batch, 10) -> unsqueeze to (Batch, 1, 10) so it acts as a sequence node
        action_node = action_vector.unsqueeze(1)
        
        # Embed both into the d_model dimension (64)
        p_emb = self.pixel_embedding(grid_nodes)
        a_emb = self.action_embedding(action_node)
        
        # Concatenate the Global Action Node (Node 0) with the 4096 Pixel Nodes
        # Total Sequence length becomes 4097 nodes
        sequence = torch.cat([a_emb, p_emb], dim=1)
        
        # Let all 4097 nodes mathematically query each other
        transformed = self.transformer(sequence)
        
        # Slice out just the 4096 pixel nodes to get their physical reactions 
        # (We drop Index 0 because that was the Action Node)
        pixel_outputs = transformed[:, 1:, :]
        
        # Calculate the direct conditional events
        change_logits = self.change_mask_head(pixel_outputs)
        color_logits = self.color_target_head(pixel_outputs)
        
        return change_logits, color_logits