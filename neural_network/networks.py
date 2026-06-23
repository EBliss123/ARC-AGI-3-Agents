import torch
import torch.nn as nn

class PlannerNetwork(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.pixel_projection = nn.Linear(18, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, 4104)

    def forward(self, grid_nodes, action_mask):
        batch_size = grid_nodes.size(0)
        
        # Extract pure 18-channel physics data and 1-channel cluster IDs
        pixel_data = grid_nodes[:, :, :18]
        cluster_ids = grid_nodes[:, :, 18].long()
        
        # 1. High-Dimensional Fingerprinting
        p_emb = self.pixel_projection(pixel_data)
        
        # 2. Summation (The Cryptographic Hash via Set Embedding)
        max_clusters = int(cluster_ids.max().item()) + 1
        embed_dim = p_emb.size(2)
        
        expanded_ids = cluster_ids.unsqueeze(-1).expand(-1, -1, embed_dim)
        object_nodes = torch.zeros(batch_size, max_clusters, embed_dim, device=p_emb.device)
        object_nodes.scatter_add_(1, expanded_ids, p_emb)
        
        # 3. The Brain (Self-Attention dynamically sized to exact object count)
        x = self.transformer(object_nodes)
        
        # Pool the object nodes into a single global state vector
        global_state = x.mean(dim=1) 
        
        raw_logits = self.output_layer(global_state)
        masked_logits = raw_logits.masked_fill(~action_mask, float('-inf'))
        return torch.softmax(masked_logits, dim=-1)

class PredictorNetwork(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        # High-dimensional spatial fingerprinting (18 -> 128)
        self.pixel_projection = nn.Linear(18, d_model)
        self.action_embedding = nn.Linear(10, d_model)
        
        # The narrow "neck" of the hourglass
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer that maps the combined broadcast back to pixels
        self.broadcast_fusion = nn.Linear(d_model * 2, d_model)
        
        self.change_mask_head = nn.Linear(d_model, 2)  
        self.color_target_head = nn.Linear(d_model, 16) 

    def forward(self, grid_nodes, action_vector):
        batch_size = grid_nodes.size(0)
        
        # Extract pure 18-channel physics data and 1-channel cluster IDs
        pixel_data = grid_nodes[:, :, :18]
        cluster_ids = grid_nodes[:, :, 18].long()
        
        # 1. High-Dimensional Fingerprinting (Batch, 4096, 128)
        p_emb = self.pixel_projection(pixel_data) 
        
        # 2. Summation (The Cryptographic Hash via Set Embedding)
        max_clusters = int(cluster_ids.max().item()) + 1
        embed_dim = p_emb.size(2)
        
        expanded_ids = cluster_ids.unsqueeze(-1).expand(-1, -1, embed_dim)
        object_nodes = torch.zeros(batch_size, max_clusters, embed_dim, device=p_emb.device)
        
        # Flawlessly sums the 128-D exact coordinate strings into dynamic object nodes
        object_nodes.scatter_add_(1, expanded_ids, p_emb)
        
        # Embed the Action Node
        action_node = self.action_embedding(action_vector.unsqueeze(1))
        
        # 3. The Brain (Self-Attention dynamically sized to exact object count)
        sequence = torch.cat([action_node, object_nodes], dim=1)
        transformed_seq = self.transformer(sequence)
        
        # Extract the updated Object Nodes
        updated_objects = transformed_seq[:, 1:, :]
        
        # 4. The Broadcast (Expanding back down to 4096 using advanced indexing)
        broadcasted_data = torch.gather(updated_objects, 1, expanded_ids)
            
        # 5. Fusion: Each pixel combines its exact original fingerprint with the object's physics conclusion
        fused_pixels = torch.relu(self.broadcast_fusion(torch.cat([p_emb, broadcasted_data], dim=-1)))
        
        # 6. Final Pixel-Level Predictions based strictly on exact maintained coordinates
        change_logits = self.change_mask_head(fused_pixels)
        color_logits = self.color_target_head(fused_pixels)
        
        return change_logits, color_logits