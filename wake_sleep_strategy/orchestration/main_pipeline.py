import sys
from pathlib import Path
from wake_phase.primitives import get_deltas
from wake_phase.evolution_engine import evolve, evolve_win_condition
from sleep_phase.memory_cache import HierarchicalCache
from relational_engine.cross_game_matrix import RelationalMatrix
from relational_engine.intra_game_tracker import TrajectoryTracker
from relational_engine.seed_generator import generate_smart_population
from wake_phase.primitives import Constant
from data_ingestion.parser import extract_level_1_transitions
from data_ingestion.tensor_math import process_transitions_to_tensors
from wake_phase.fitness import apply_proposed_deltas

# Dynamically add the root directory to the system path so imports work cleanly from anywhere
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from data_ingestion.parser import extract_frame_transitions
from data_ingestion.tensor_math import process_transitions_to_tensors

# The Global Brain: Persists across all games
global_cache = HierarchicalCache()
global_matrix = RelationalMatrix()

def process_level(game_id: str, file_path: Path, level_id: int, game_tracker: TrajectoryTracker):
    """The Grand Loop for a single level."""
    # 1. Pre-Check: Ask the Tracker and Matrix for hints
    hypotheses = game_tracker.predict_next_level_goal(game_id)
    if not hypotheses:
        active_physics = [name for name, fn in global_cache.functions.items() if fn.is_active]
        hypotheses = global_matrix.query_historical_goals(active_physics)
        
    # 2. Build the Seed Population
    starting_population = generate_smart_population(hypotheses, population_size=10)
    
    # 3. Data Ingestion (Goal 1 Focus)
    raw_level_1 = extract_level_1_transitions(file_path)
    tensor_frames = process_transitions_to_tensors(raw_level_1)
    
    # 4. Process all frames with cross-frame contradiction verification
    level_rules = []
    active_rule_library = []

    for frame in tensor_frames:
        action_desc = f"Action {frame['action_id']}" if frame['action_id'] is not None else "Animation"
        step_idx = frame['step']
        changed_count = frame['dynamic_mask'].sum().item()
        print(f"  Step {step_idx} [{action_desc}]: {changed_count} changing pixels.")
        
        # 5. First: Check if an existing trajectory rule perfectly explains this frame
        matched_rule = None
        for candidate_rule in active_rule_library:
            pred_mask = apply_proposed_deltas(frame["s_t"], [], ast_tree=candidate_rule.ast_tree)
            if (pred_mask == frame["dynamic_mask"].int()).all():
                matched_rule = candidate_rule
                break
                
        if matched_rule is not None:
            print(f"    [Reused Trajectory Law]: {matched_rule.ast_tree} (Complexity: {matched_rule.complexity})")
            level_rules.append(matched_rule)
            continue

        # 6. Evolve minimal relational rule if not yet covered
        best_rule = evolve(
            frame["s_t"],
            frame["s_next"],
            dynamic_mask=frame["dynamic_mask"],
            action_id=frame["action_id"]
        )
        
        # 7. Cross-validate against all static / zero-delta frames to ensure zero contradictions
        is_consistent = True
        for check_frame in tensor_frames:
            if check_frame['dynamic_mask'].sum().item() == 0 and check_frame['action_id'] == frame['action_id']:
                check_pred = apply_proposed_deltas(check_frame["s_t"], [], ast_tree=best_rule.ast_tree)
                if check_pred.any():
                    is_consistent = False
                    break
                    
        if is_consistent:
            active_rule_library.append(best_rule)
            
        print(f"    Winning AST: {best_rule.ast_tree} (Complexity: {best_rule.complexity})")
        level_rules.append(best_rule)
        
    if not level_rules:
        print(f"  No movement detected in {game_id} Level {level_id}.")
        return Constant(1), []
    
    # [Placeholder]: Sleep Phase caching and Win Condition evolution will hook in here next.
    mock_winning_ast = Constant(1) 
    mock_active_physics = ["fn_0"]
    
    return mock_winning_ast, mock_active_physics
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
    dynamic_mask = first_frame["dynamic_mask"]
    action_id = first_frame["action_id"]
    
    print(f"Evolving relational rules on {dynamic_mask.sum().item()} changing coordinates...")
    best_rule = evolve(s_t, s_next, dynamic_mask=dynamic_mask, action_id=action_id)
    
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

def play_game(game_id: str, file_path_str: str, num_levels: int):
    print(f"--- Booting Game: {game_id} ---")
    game_tracker = TrajectoryTracker()
    target_file = Path(file_path_str)
    
    if not target_file.exists():
        print(f"Error: Could not find file at {target_file}")
        return
    
    for level in range(1, num_levels + 1):
        print(f"  Simulating Level {level}...")
        winning_goal, active_physics = process_level(game_id, target_file, level, game_tracker)
        game_tracker.record_level_goal(game_id, winning_goal)
        
    global_matrix.record_game_solution(game_id, active_physics, winning_goal)
    print(f"--- Game Mastered! Matrix Updated. ---\n")

if __name__ == "__main__":
    # Goal 1: Process the first level of ls20
    ls20_path = r"C:\Users\Easton\ARC-AGI-3-Agents\wake_sleep_strategy\ls20-34d098c6-df52-458c-b5ad-19cefeb75981.json"
    
    # We set num_levels=1 because extract_level_1_transitions strictly isolates Level 1
    play_game("ls20", ls20_path, num_levels=1)