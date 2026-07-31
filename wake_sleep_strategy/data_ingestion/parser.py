import json
from pathlib import Path
from typing import Dict, Any, List, Iterator

def stream_jsonl(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Yields parsed JSON objects line-by-line to minimize memory footprint."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def extract_frame_transitions(file_path: Path) -> List[Dict[str, Any]]:
    """
    Reads human telemetry and pairs contiguous frames (S_t, S_next).
    Captures both action-driven transitions and autonomous animations.
    """
    frames = list(stream_jsonl(file_path))
    transitions = []
    
    for i in range(len(frames) - 1):
        current_frame = frames[i]
        next_frame = frames[i + 1]
        
        # Extract the raw 2D grid arrays
        s_t = current_frame.get("grid")
        s_next = next_frame.get("grid")
        
        # Extract action ID if a human pressed a key on this frame; otherwise None
        action_input = current_frame.get("action_input")
        action_id = action_input.get("id") if action_input else None
        
        # Extract environment status flags from the resulting frame
        is_win = next_frame.get("level_cleared", False)
        
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
    test_path = Path(r"C:\Users\Easton\ARC-AGI-3-Agents\wake_sleep_strategy\sample_replay.jsonl")
    
    if test_path.exists():
        data = extract_frame_transitions(test_path)
        print(f"Successfully extracted {len(data)} atomic frame transitions.")
    else:
        print(f"Test file not found at: {test_path}")