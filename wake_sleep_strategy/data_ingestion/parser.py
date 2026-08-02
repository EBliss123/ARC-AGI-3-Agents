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
    frames = list(stream_jsonl(file_path))
        
    transitions = []
    
    for i in range(len(frames) - 1):
        current_data = frames[i].get("data", {})
        next_data = frames[i + 1].get("data", {})
        
        # Safely extract the grid whether it is a dictionary containing "grid" or the 2D list itself
        c_frame = current_data.get("frame")
        s_t = c_frame.get("grid") if isinstance(c_frame, dict) else c_frame
        
        n_frame = next_data.get("frame")
        s_next = n_frame.get("grid") if isinstance(n_frame, dict) else n_frame
        
        # Fallback just in case the key was literally called "grid" all along
        if not s_t: s_t = current_data.get("grid")
        if not s_next: s_next = next_data.get("grid")
        
        # Skip intro/outro frames that do not contain a valid game board
        if s_t is None or s_next is None:
            continue
            
        # Extract action ID if a human pressed a key on this frame; otherwise None
        action_input = current_data.get("action_input")
        action_id = action_input.get("id") if action_input else None
        
        # Extract environment status flags (checking if the next frame reached a WIN state)
        is_win = (next_data.get("state") == "WIN")
        
        # Store as an atomic state transition block
        transitions.append({
            "step": i,
            "s_t": s_t,
            "action_id": action_id,
            "s_next": s_next,
            "is_win": is_win
        })
        
    return transitions

if __name__ == "__main__":
    # Test block to verify the parser works locally
    test_path = Path(r"C:\Users\Easton\ARC-AGI-3-Agents\wake_sleep_strategy\ar25-2a854897-cb79-48f4-92e1-0288df2cf6a9.json")
    
    if test_path.exists():
        data = extract_frame_transitions(test_path)
        print(f"Successfully extracted {len(data)} atomic frame transitions.")
    else:
        print(f"Test file not found at: {test_path}")