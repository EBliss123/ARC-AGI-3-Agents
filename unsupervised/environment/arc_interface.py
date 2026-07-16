import torch
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

class ARCInterface:
    def __init__(self, task_id="ls20", render_mode="terminal-fast"):
        """
        Initializes the ARC-AGI-3 environment.
        render_mode="terminal-fast" runs the environment headlessly for maximum speed.
        """
        self.task_id = task_id
        
        # Initialize the arcade and make the specific environment
        self.arc = Arcade()
        self.env = self.arc.make(self.task_id, render_mode=render_mode)

    def step(self, action_name):
        """
        Executes an action and extracts the resulting animation frames as a PyTorch tensor.
        action_name should be a string like "ACTION1", "ACTION2", etc.
        """
        # Convert string to the official GameAction enum
        action_enum = getattr(GameAction, action_name)
        
        # Step the environment
        obs = self.env.step(action_enum)
        
        # If the environment returns None, the game might be over or failed to step
        if obs is None:
            return None
            
        # The observation contains the frame data. 
        # According to the ARC docs, this can contain multiple frames if animations occurred.
        # We assume obs.frame is the 3D array (integer[][][])
        raw_frames = obs.frame
        np_frames = np.array(raw_frames, dtype=np.int32)
        tensor_frames = torch.tensor(np_frames, dtype=torch.int32)
        
        if len(tensor_frames.shape) == 2:
            tensor_frames = tensor_frames.unsqueeze(0)
            
        return tensor_frames
        
    def reset(self):
        """Resets the environment and returns the initial state."""
        obs = self.env.reset()
        if obs is None:
             return None
             
        # Convert the initial frame (which might just be a 2D array or 3D with 1 frame)
        raw_frames = obs.frame
        np_frames = np.array(raw_frames, dtype=np.int32)
        tensor_frames = torch.tensor(np_frames, dtype=torch.int32)
        
        if len(tensor_frames.shape) == 2:
            tensor_frames = tensor_frames.unsqueeze(0)
            
        return tensor_frames