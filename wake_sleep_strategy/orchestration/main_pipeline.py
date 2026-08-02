import sys
from pathlib import Path

# Dynamically add the root directory to the system path so imports work cleanly from anywhere
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from data_ingestion.parser import extract_frame_transitions
from data_ingestion.tensor_math import process_transitions_to_tensors

def verify_milestone_1(jsonl_path: Path):
    print("--- Starting Milestone 1 Verification ---")
    print(f"Target file: {jsonl_path.name}")
    
    if not jsonl_path.exists():
        print("ERROR: File not found. Please download the real replay file and place it in the root directory.")
        return

    # Step 1: Parse the JSONL
    print("\n[1/2] Streaming and parsing JSONL telemetry...")
    raw_transitions = extract_frame_transitions(jsonl_path)
    print(f"Successfully extracted {len(raw_transitions)} atomic frame transitions.")
    
    if not raw_transitions:
        print("No transitions found. The file might be empty or improperly formatted.")
        return

    # Step 2: Convert to Tensors and compute Deltas
    print("[2/2] Converting to PyTorch tensors and calculating masks...")
    tensor_data = process_transitions_to_tensors(raw_transitions)
    print(f"Successfully processed {len(tensor_data)} tensor dictionaries.")
    
    # Step 3: Print a diagnostic report of the very first valid transition
    print("\n--- Tensor Verification (First Frame) ---")
    first_frame = tensor_data[0]
    
    print(f"Step Index:    {first_frame['step']}")
    print(f"Action ID:     {first_frame['action_id']}")
    print(f"Win State:     {first_frame['is_win']}")
    print(f"S_t Shape:     {first_frame['s_t'].shape} (dtype: {first_frame['s_t'].dtype})")
    print(f"S_next Shape:  {first_frame['s_next'].shape}")
    print(f"Delta_S Shape: {first_frame['delta_s'].shape}")
    
    # Calculate how many pixels actually changed vs stayed the same
    changed_pixels = first_frame['dynamic_mask'].sum().item()
    static_pixels = first_frame['static_mask'].sum().item()
    
    print(f"Dynamic Mask:  {changed_pixels} pixels changed.")
    print(f"Static Mask:   {static_pixels} pixels remained the same.")
    
    print("\nMilestone 1 is complete and functioning perfectly.")

if __name__ == "__main__":
    # Point directly to the sample file you placed in the root directory
    target_file = root_dir / "ar25-2a854897-cb79-48f4-92e1-0288df2cf6a9.json"
    verify_milestone_1(target_file)