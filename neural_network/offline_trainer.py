import os
import torch
import numpy as np
import copy

# Import your existing architecture
from main import AGI_Architecture
from data_utils import encode_grid, unify_action, get_action_mask
from meta_loop import setup_game_clone, run_single_turn_adaptation, average_successful_weights

def calculate_human_action_index(action_id, x, y):
    """Translates the human's action back into the 4104-index array."""
    if action_id == 6:
        # Spatial action: offset by 8 discrete nodes
        return 8 + (y * 64) + x
    return action_id

def train_basecamp_offline():
    print("Booting ARC-AGI-3 Offline Basecamp Trainer...")
    
    # 1. Initialize the Global Master Model
    global_model = AGI_Architecture()
    
    # Try to load existing basecamp if you have one
    checkpoint_path = os.path.join('__pycache__', 'arc_agi_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        # STRICLY load into the Planner. The Predictor stays blank.
        global_model.planner.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print("-> Loaded Expert Planner Basecamp weights.")
    else:
        print("-> No existing checkpoint found. Starting fresh Basecamp.")

    data_dir = 'human_data'
    if not os.path.exists(data_dir):
        print(f"Error: Could not find directory '{data_dir}'.")
        return

    # Grab all your saved .pt files
    trajectory_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    if not trajectory_files:
        print("No human data files found to train on.")
        return

    print(f"-> Found {len(trajectory_files)} completed games. Beginning simulated inner loops...\n")
    
    successful_weights = []

    for file_name in trajectory_files:
        file_path = os.path.join(data_dir, file_name)
        human_trajectory = torch.load(file_path, weights_only=False)
        
        print(f"Processing {file_name} ({len(human_trajectory)} moves)...")
        
        # Spawn the temporary clone for this specific game
        clone_model, physics_optimizer, policy_optimizer = setup_game_clone(global_model, inner_lr=0.05)
        
        life_step = 0
        game_won = False
        
        for move in human_trajectory:
            # Skip manual UI resets
            if move["action_id"] == -1:
                life_step = 0
                continue
                
            raw_state = move["state"]
            raw_next_state = move["next_state"]
            action_id = move["action_id"]
            x = move["click_x"]
            y = move["click_y"]
            status = move["terminal_status"]

            # 1. Prepare Data
            grid_tensor = encode_grid(raw_state)
            action_vector = unify_action(action_id, x, y)
            target_next_frame = torch.tensor(raw_next_state, dtype=torch.long)
            
            # Generate the action mask directly from the raw grid
            action_mask = get_action_mask(raw_state)
            
            # 2. Extract Clone's prediction for the human's exact action
            # We must pass it through the planner so PyTorch can calculate the gradient
            grid_batch = grid_tensor.unsqueeze(0)
            mask_batch = action_mask.unsqueeze(0)
            
            # Set planner to eval briefly just to get the probabilities cleanly
            clone_model.planner.train() 
            action_probs = clone_model.planner(grid_batch, mask_batch)
            
            chosen_index = calculate_human_action_index(action_id, x, y)
            log_prob = torch.log(action_probs[0, chosen_index] + 1e-10)
            
            # 3. Reward Translator & Logic Flags
            valid_change_flag = -1.0 if np.array_equal(raw_state, raw_next_state) else 1.0
            
            terminal_reward = 0.0
            if status == "WIN":
                terminal_reward = 100.0
                game_won = True
            elif status == "GAME_OVER":
                terminal_reward = -10.0
                
            # 4. Execute the Simulated Turn
            turn_data = (grid_tensor, action_vector, target_next_frame, log_prob, valid_change_flag, life_step, terminal_reward)
            
            clone_model, final_reward, _ = run_single_turn_adaptation(clone_model, physics_optimizer, policy_optimizer, turn_data)
            
            # Handle step logic
            life_step += 1
            if status == "GAME_OVER":
                life_step = 0
                
        # If the human trajectory ended in a win, commit the knowledge immediately
        if game_won:
            # ONLY the Planner continually inherits the newly adapted heuristics
            global_model.planner.load_state_dict(clone_model.planner.state_dict())
            print(f"   -> Heuristics extracted from {file_name}. Basecamp updated.")
        else:
            print(f"   -> Skipped {file_name} (Trajectory did not contain a WIN state)")

    # Finally, average the global basecamp
    if successful_weights:
        print(f"\nAveraging {len(successful_weights)} successful minds into the Universal Basecamp...")
        average_successful_weights(global_model, successful_weights)
        
        os.makedirs('__pycache__', exist_ok=True)
        torch.save(global_model.planner.state_dict(), checkpoint_path)
        print("★ Basecamp Successfully Updated and Saved! ★")
    else:
        print("\nNo wins were processed. Basecamp remains unchanged.")

if __name__ == "__main__":
    train_basecamp_offline()