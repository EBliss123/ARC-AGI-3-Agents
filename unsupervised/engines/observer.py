import torch

class DeltaObserver:
    def observe(self, grid_before, frames_after):
        """
        Takes the 2D grid before the action, and the 3D tensor of animation frames after.
        Returns a list of unified [x, y, old_color, new_color] tensors representing every visual step.
        """
        transitions = []
        
        # Pre-calculate our X and Y coordinates mathematically
        height, width = grid_before.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        
        # Flatten the coordinates so they are flat 1D arrays
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        
        # Attach the before_grid to the front of the animation frames 
        sequence = torch.cat([grid_before.unsqueeze(0), frames_after], dim=0)
        
        # Generate the mathematical delta for each consecutive step in the cascade
        for i in range(sequence.shape[0] - 1):
            old_grid = sequence[i].flatten()
            new_grid = sequence[i+1].flatten()
            
            # Instantly bind them into our [x, y, old, new] schema
            state_tensor = torch.stack([x_flat, y_flat, old_grid, new_grid], dim=1)
            transitions.append(state_tensor)
            
        return transitions