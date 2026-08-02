import torch
from typing import List, Dict, Tuple, Optional
from wake_phase.primitives import calculate_distance, get_pixel_state, get_deltas

def build_active_graph(deltas: List[Dict[str, tuple]]) -> List[Dict]:
    """
    Calculates the pairwise geometric relationships between all pixels that changed.
    This provides the Sleep Phase with the raw shape of moving/changing elements
    without assuming any objects exist.
    """
    graph = []
    for i, node_a in enumerate(deltas):
        relations = []
        for j, node_b in enumerate(deltas):
            if i == j:
                continue
            # Calculate strict dx, dy, dz distance between the two changing pixels
            dist = calculate_distance(node_a["coord"], node_b["coord"])
            relations.append({
                "target_index": j,
                "dz_dy_dx": dist
            })
        
        graph.append({
            "node": node_a,
            "relations": relations
        })
        
    return graph

def probe_offset(tensor: torch.Tensor, origin: Tuple[int, int, int], dz: int, dy: int, dx: int) -> Optional[int]:
    """
    Lazy evaluation API. Allows the agent to query the static environment
    at a specific offset without pre-computing the entire grid.
    Returns None if the probe hits the edge of the board.
    """
    z, y, x = origin
    target_z = z + dz
    target_y = y + dy
    target_x = x + dx
    
    # Verify the probe doesn't go out of bounds
    max_z, max_y, max_x = tensor.shape
    if (0 <= target_z < max_z) and (0 <= target_y < max_y) and (0 <= target_x < max_x):
        return get_pixel_state(tensor, target_z, target_y, target_x)
    
    # Return None if the agent tries to look past the edge of the grid
    return None

if __name__ == "__main__":
    # Test block to verify the Active Graph and Probing API
    t1 = torch.zeros((1, 4, 4), dtype=torch.int8)
    t2 = torch.zeros((1, 4, 4), dtype=torch.int8)
    
    # Draw a static wall (color 8) at x=2
    t1[0, :, 2] = 8
    t2[0, :, 2] = 8
    
    # Simulate a pixel (color 4) moving from x=0 to x=1
    t1[0, 1, 0] = 4
    t2[0, 1, 1] = 4
    
    print("--- Testing Extractor ---")
    changes = get_deltas(t1, t2)
    
    # 1. Test the Active Graph
    graph = build_active_graph(changes)
    print(f"Active Graph generated with {len(graph)} nodes.")
    for node in graph:
        print(f"Node at {node['node']['coord']} has {len(node['relations'])} relation(s).")
        
    # 2. Test the Probing API
    # The agent tests a hypothesis: "If I am at (0,1,1), is there something exactly 1 unit to my right?"
    probe_coord = (0, 1, 1)
    wall_color = probe_offset(t2, probe_coord, dz=0, dy=0, dx=1)
    
    print(f"Probe at dx=1 from {probe_coord} returned color: {wall_color}")