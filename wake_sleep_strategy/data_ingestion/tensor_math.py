import torch
from typing import Dict, Any, List

def process_transitions_to_tensors(transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes raw parsed transitions and converts grids into PyTorch tensors.
    Computes the delta between frames and generates the Dynamic and Static masks.
    """
    tensor_data = []
    
    for turn in transitions:
        s_t_raw = turn.get("s_t")
        s_next_raw = turn.get("s_next")
        
        # Skip gracefully if the parser handed us a frame missing its grid
        if s_t_raw is None or s_next_raw is None:
            continue
            
        # Cast raw nested lists to 2D PyTorch tensors (int8 for memory efficiency)
        s_t_tensor = torch.tensor(s_t_raw, dtype=torch.int8)
        s_next_tensor = torch.tensor(s_next_raw, dtype=torch.int8)
        
        # Calculate the math delta (isolating exactly what changed)
        delta_s = s_next_tensor - s_t_tensor
        
        # Generate the twin boolean masks
        dynamic_mask = (delta_s != 0)
        static_mask = (delta_s == 0)
        
        # Package the processed tensors back into a clean dictionary
        tensor_data.append({
            "step": turn.get("step"),
            "action_id": turn.get("action_id"),
            "s_t": s_t_tensor,
            "s_next": s_next_tensor,
            "delta_s": delta_s,
            "dynamic_mask": dynamic_mask,
            "static_mask": static_mask,
            "is_win": turn.get("is_win")
        })
        
    return tensor_data