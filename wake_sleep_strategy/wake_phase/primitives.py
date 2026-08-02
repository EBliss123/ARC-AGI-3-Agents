import torch
from typing import Tuple, List, Dict

def get_pixel_state(tensor: torch.Tensor, z: int, y: int, x: int) -> int:
    """Returns the absolute integer color value at a specific coordinate."""
    return tensor[z, y, x].item()

def get_deltas(s_t: torch.Tensor, s_next: torch.Tensor) -> List[Dict[str, tuple]]:
    """
    Extracts the absolute truth of what changed between two states.
    Returns a list of dictionaries containing the coordinate and the algebraic transition.
    """
    # Find all coordinates where the tensors mathematically differ
    difference_mask = s_t != s_next
    changed_indices = torch.nonzero(difference_mask)
    
    deltas = []
    for idx in changed_indices:
        z, y, x = idx.tolist()
        color_before = s_t[z, y, x].item()
        color_after = s_next[z, y, x].item()
        
        deltas.append({
            "coord": (z, y, x),
            "transition": (color_before, color_after)
        })
        
    return deltas

def calculate_distance(coord_a: Tuple[int, int, int], coord_b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Calculates the strict algebraic distance (dz, dy, dx) between two coordinates.
    This provides the raw geometric scaffolding for future learning algorithms.
    """
    z_a, y_a, x_a = coord_a
    z_b, y_b, x_b = coord_b
    
    dz = z_b - z_a
    dy = y_b - y_a
    dx = x_b - x_a
    
    return (dz, dy, dx)

if __name__ == "__main__":
    # Quick mathematical verification of the primitives
    t1 = torch.zeros((1, 3, 3), dtype=torch.int8)
    t2 = torch.zeros((1, 3, 3), dtype=torch.int8)
    
    # Simulate a single pixel changing color from 0 to 4
    t2[0, 1, 1] = 4
    
    print("--- Testing Primitives ---")
    changes = get_deltas(t1, t2)
    print(f"Deltas detected: {changes}")
    
    if changes:
        origin = (0, 0, 0)
        target = changes[0]["coord"]
        dist = calculate_distance(origin, target)
        print(f"Distance from origin to changed pixel: {dist}")