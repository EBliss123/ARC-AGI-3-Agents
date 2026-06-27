import torch
import torch.nn as nn
import torch.optim as optim
import arc_agi
from arcengine import GameAction, GameState
import random
import os
import csv
import numpy as np
from scipy.ndimage import label

# Import our custom modules
from networks import PlannerNetwork, PredictorNetwork
from data_utils import encode_grid, get_action_mask, unify_action
import copy
from meta_loop import setup_game_clone, run_single_turn_adaptation, average_successful_weights

class AGI_Architecture(nn.Module):
    def __init__(self):
        super().__init__()
        # The architecture is now beautifully reduced to just two decentralized brains
        self.planner = PlannerNetwork()
        self.predictor = PredictorNetwork()

def get_cluster_representatives(raw_grid):
    grid = np.array(raw_grid)
    if grid.ndim != 2:
        return []
        
    representatives = []
    # Structure defines the 4-way connectivity (up, down, left, right)
    structure = np.array([[0, 1, 0], 
                          [1, 1, 1], 
                          [0, 1, 0]])
                          
    for c in np.unique(grid):
        # Create a binary mask for just this color
        binary_mask = (grid == c).astype(int)
        labeled_array, num_features = label(binary_mask, structure=structure)
        
        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled_array == i)
            if len(coords) > 0:
                y, x = coords[0]  # Grab the first coordinate of this specific blob
                representatives.append((x, y))
                
    return representatives

def play_game_turn(obs, clone_model):
    # 1. Prepare the data
    raw_grid = obs.frame[-1] if isinstance(obs.frame, list) else obs.frame
    grid_tensor = encode_grid(raw_grid)
    
    # Start with a clean mask
    action_mask = torch.zeros(4104, dtype=torch.bool)
    
    # --- DYNAMIC GUARDRAIL: Official Emulator Actions ---
    for action in obs.available_actions:
        # If it's an Enum, use .value; if it's an int, use it directly
        action_val = action.value if hasattr(action, 'value') else action
        if action_val < 8:
            action_mask[action_val] = True
            
    # --- DYNAMIC SPATIAL CLUSTER MASKING ---
    # Only enable spatial clicking if index 6 (ACTION6) is in available_actions
    if any((a.value if hasattr(a, 'value') else a) == 6 for a in obs.available_actions):
        valid_coords = get_cluster_representatives(raw_grid)
        for x, y in valid_coords:
            spatial_index = 8 + (y * 64) + x
            if spatial_index < 4104:
                action_mask[spatial_index] = True
        
    # 2. Query the Planner Network (Adding .unsqueeze(0) to create a Batch dimension for the Transformer)
    grid_batch = grid_tensor.unsqueeze(0)
    mask_batch = action_mask.unsqueeze(0)
    action_probs = clone_model.planner(grid_batch, mask_batch)
    
    # 3. Select action
    chosen_index = torch.multinomial(action_probs, 1).item()
    
    # Capture the mathematical log probability
    log_prob = torch.log(action_probs[0, chosen_index] + 1e-10)
    
    if chosen_index < 8:
        action_id = chosen_index
        x, y = 0, 0
    else:
        action_id = 6 # Map all spatial indices to the generic ACTION6 enum
        spatial_index = chosen_index - 8
        y = spatial_index // 64
        x = spatial_index % 64
        
    action_vector = unify_action(action_id, x, y)
    emulator_action = next(a for a in GameAction if a.value == action_id)
    emulator_coords = {"x": x, "y": y} if action_id == 6 else None
    
    return grid_tensor, action_vector, emulator_action, emulator_coords, log_prob
    
if __name__ == "__main__":
    # 1. Initialize the Global Master Model
    global_model = AGI_Architecture()
    
    # Load the Expert Planner Basecamp (The Predictor intentionally remains random)
    checkpoint_path = os.path.join('__pycache__', 'arc_agi_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        global_model.planner.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print("-> Loaded Expert Planner Basecamp weights.")

    epochs = 1
    best_loss = float('inf')

    # Initialize the CSV log file locally
    with open('training_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average_Meta_Loss'])

    print("Booting ARC-AGI-3 Meta-Learning Engine...")

    # Initialize the Kaggle Arcade Emulator
    arc = arc_agi.Arcade()
    public_games = arc.get_environments()

    for epoch in range(epochs):
        successful_weights = []
        
        # Grab a random batch of 5 games
        batch_of_games = random.sample(public_games, 5)
        
        for game in batch_of_games:
            # A. Spawn the temporary clone and its optimizers
            clone_model, physics_optimizer, policy_optimizer = setup_game_clone(global_model, inner_lr=0.05)
            
            # Boot up the live emulator for this specific game
            env = arc.make(game.game_id)
            obs = env.reset()  # <--- FIX 1: Kaggle only returns 1 value
            
            train_states = []
            action_history = []  # Tracks the exact buttons pressed this game
            done = False
            
            # Helper to safely extract the 64x64 grid from the Kaggle object
            def extract_grid(observation):
                grid_data = observation.frame if hasattr(observation, 'frame') else observation["grid"]
                # Handle animations: if it returns a list of frames, grab the final one
                return grid_data[-1] if isinstance(grid_data, list) else grid_data

            step = 0
            life_step = 0
            max_steps = 1500
            
            # B. Run the Test-Time Training (Inner Loop) until solved or capped
            while step < max_steps:
                raw_grid = extract_grid(obs)
                 
                # Ask the Planner for a move
                grid_tensor, action_vector, em_action, em_coords, log_prob = play_game_turn(obs, clone_model)
                
                if em_coords:
                    action_history.append(f"{em_action.name}({em_coords['x']},{em_coords['y']})")
                else:
                    action_history.append(em_action.name)
                
                # Submit the move to the live emulator
                next_obs = env.step(em_action, data=em_coords)
                
                # --- VERIFICATION & ADVERSARIAL CURIOSITY FLAG ---
                next_raw_grid = extract_grid(next_obs)
                
                # We flag whether the board changed or not
                if np.array_equal(raw_grid, next_raw_grid):
                    action_history[-1] += " (NO_CHANGE)"
                    valid_change_flag = -1.0  # Wasted click
                else:
                    valid_change_flag = 1.0   # Trigger the adversarial surprise calculation
                    
                # --- EXTRINSIC TERMINAL REWARDS ---
                terminal_reward = 0.0
                if hasattr(next_obs, 'state'):
                    if next_obs.state == GameState.WIN:
                        terminal_reward = 100.0  # The Ultimate Jackpot
                    elif next_obs.state == GameState.GAME_OVER:
                        terminal_reward = -10.0  # The Death Penalty
                
                # Store the true outcome for the final exam backprop
                target_next_frame = torch.tensor(next_raw_grid, dtype=torch.long)
                
                # Bundle the step count and terminal reward so the meta_loop can apply them
                turn_data = (grid_tensor, action_vector, target_next_frame, log_prob, valid_change_flag, life_step, terminal_reward)
                train_states.append(turn_data)
                
                # --- REAL-TIME LEARNING: Update the brain instantly after this single move ---
                clone_model, final_reward, phys_loss = run_single_turn_adaptation(clone_model, physics_optimizer, policy_optimizer, turn_data)
                
                # --- REAL-TIME TERMINAL LOGGING ---
                mode = "EXPLORE" if phys_loss > 0.5 else "EXPLOIT"
                print(f"\r   ... Step {step+1} | Mode: {mode} (Loss: {phys_loss:.2f}) | Action: {action_history[-1]}    ", end="", flush=True)
                
                obs = next_obs
                
                # --- CHECK GAME STATE ---
                if hasattr(obs, 'state'):
                    if obs.state == GameState.GAME_OVER:
                        obs = env.reset()
                        action_history.append("---DIED_AND_RESET---")
                        life_step = 0
                    elif obs.state == GameState.WIN:
                        break  
                        
                step += 1
                life_step += 1
                
            done = (obs.state == GameState.WIN) if hasattr(obs, 'state') else True
                    
            # --- VERIFICATION LOG ---
            final_status = obs.state.name if hasattr(obs, 'state') else "ALIVE_BUT_CAPPED"
            action_str = "[" + ", ".join(action_history) + "]"
            print(f"   -> Played {game.game_id}: {len(train_states)} moves | Outcome: {final_status}")
            print(f"      Actions: {action_str}")
            # ------------------------
                    
            # The clone is already fully adapted because it learned move-by-move!
            adapted_clone = clone_model
            
            # C. Save the adapted clone's weights if it won
            if done:
                successful_weights.append(copy.deepcopy(adapted_clone.state_dict()))
            
        # 3. The Outer Loop (Universal Basecamp Update)
        if successful_weights:
            average_successful_weights(global_model, successful_weights)
            print(f"Epoch {epoch+1} | Basecamp updated from {len(successful_weights)} wins!")
            
            # Checkpoint the model
            os.makedirs('__pycache__', exist_ok=True)
            torch.save(global_model.state_dict(), os.path.join('__pycache__', 'arc_agi_checkpoint.pt'))
            print(f"   -> New Basecamp checkpoint saved to __pycache__.")
                
        else:
            print(f"Epoch {epoch+1} | Skipped Update (No games were won)")