import sys
from pathlib import Path
from wake_phase.primitives import get_deltas
from wake_phase.evolution_engine import evolve, evolve_win_condition

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
    
    print("\n--- Starting Milestone 2 (Wake Phase) ---")
        
    # Explicitly grab the tensors for the first frame 
    first_frame = tensor_data[0]
    s_t = first_frame["s_t"]
    s_next = first_frame["s_next"]
    
    # Extract the deltas. s_t and s_next contain the entire grid, 
    # so all static pixels are included for the agent to probe later.
    raw_deltas = get_deltas(s_t, s_next)
    
    print(f"Evolving rules based on {len(raw_deltas)} dynamic pixels...")
    best_rule = evolve(s_t, s_next, raw_deltas, generations=3)
    
    print(f"Winning Rule Fitness Score: {best_rule.fitness_score:.2f}")
    print(f"Winning Rule Complexity: {best_rule.complexity}")
    print(f"Predictions preserved: {len(best_rule.proposed_deltas)}")
    print("Milestone 2 is officially complete!")

    print("\n--- Micro-Step 2.5: Evolving the Win Condition ---")
    is_win = first_frame["is_win"]
    win_rule = evolve_win_condition(s_t, is_win, generations=5)

    print(f"Goal Rule Fitness Score: {win_rule.fitness_score:.2f}")
    print(f"Goal Rule Complexity: {win_rule.ast_tree.get_complexity()}")
    print(f"Best Goal Equation: {win_rule.ast_tree}")
    print("Milestone 2 is officially complete!")

if __name__ == "__main__":
    # Point directly to the sample file you placed in the root directory
    target_file = root_dir / "ar25-2a854897-cb79-48f4-92e1-0288df2cf6a9.json"
    verify_milestone_1(target_file)