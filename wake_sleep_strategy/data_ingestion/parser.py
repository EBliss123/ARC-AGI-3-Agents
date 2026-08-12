import json
from pathlib import Path
from typing import Dict, Any, List, Iterator

def stream_jsonl(file_path: Path) -> Iterator[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def extract_frame_transitions(file_path: Path) -> List[Dict[str, Any]]:
    # Use the streaming generator we proved works for this file format
    frames = [f for f in stream_jsonl(file_path) if f.get("data", {}).get("frame") is not None]
        
    transitions = []
    
    # Milestone 1.3: Flatten the 3D frame arrays into a continuous 2D timeline to capture animations
    timeline = []
    for step_idx, f in enumerate(frames):
        data = f.get("data", {})
        action_input = data.get("action_input")
        action_id = action_input.get("id") if action_input else None
        is_win = (data.get("levels_completed", 0) > 0)
        
        c_frame = data.get("frame")
        grids = c_frame.get("grid") if isinstance(c_frame, dict) else c_frame
        
        # Iterate through the animation sequence
        for j, grid in enumerate(grids):
            timeline.append({
                "step": step_idx,
                "grid": grid,
                "action_id": action_id if j == 0 else None,
                "is_win": is_win if j == len(grids) - 1 else False
            })
            
    # Build the S_t -> S_next tuples
    for i in range(len(timeline) - 1):
        transitions.append({
            "step": timeline[i]["step"],
            "s_t": timeline[i]["grid"],
            "action_id": timeline[i]["action_id"],
            "s_next": timeline[i+1]["grid"],
            "is_win": timeline[i+1]["is_win"]
        })
        
    return transitions

def extract_level_1_transitions(file_path: Path) -> List[Dict[str, Any]]:
    """Isolates only the frames belonging to the first level (up to the first WIN state)."""
    all_transitions = extract_frame_transitions(file_path)
    level_1_frames = []
    
    for turn in all_transitions:
        level_1_frames.append(turn)
        if turn.get("is_win"):
            break  # Stop extracting once Level 1 is solved
            
    return level_1_frames

if __name__ == "__main__":
    # Test block to verify the parser works locally
    test_path = Path(r"C:\Users\Easton\ARC-AGI-3-Agents\wake_sleep_strategy\ar25-2a854897-cb79-48f4-92e1-0288df2cf6a9.json")
    
    if test_path.exists():
        data = extract_frame_transitions(test_path)
        print(f"Successfully extracted {len(data)} atomic frame transitions.")
    else:
        print(f"Test file not found at: {test_path}")