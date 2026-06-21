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
from networks import PlannerNetwork, ActionNetwork, PredictorNetwork
from data_utils import encode_grid, get_action_mask, unify_action
from meta_loop import setup_game_clone, run_fast_adaptation, transition_to_next_level, run_outer_update

class AGI_Architecture(nn.Module):
    def __init__(self):
        super().__init__()
        # Bring the three networks under one global umbrella
        self.planner = PlannerNetwork()
        self.action_network = ActionNetwork()
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

def play_game_turn(obs, model_clone):
    # 1. Prepare the data
    raw_grid = obs.frame[-1] if isinstance(obs.frame, list) else obs.frame
    grid_tensor = encode_grid(raw_grid).unsqueeze(0)
    
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
        
    # 2. The Planner makes a decision
    action_probs = model_clone.planner(grid_tensor, action_mask)
    
    # 3. Select action
    chosen_index = torch.multinomial(action_probs, 1).item()
    
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
    
    return grid_tensor, action_vector, emulator_action, emulator_coords
    
if __name__ == "__main__":
    # 1. Initialize the Global Master Model
    global_model = AGI_Architecture()

    # 2. Setup the Global Outer Optimizer (Adam for slow, stable learning)
    outer_optimizer = optim.Adam(global_model.parameters(), lr=0.0005)

    epochs = 30
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
        meta_test_data = []
        
        # Grab a random batch of 5 games
        batch_of_games = random.sample(public_games, 5)
        
        for game in batch_of_games:
            # A. Spawn the temporary clone and its fast optimizer
            clone_model, inner_optimizer = setup_game_clone(global_model, inner_lr=0.05)
            
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

            # B. Run the 30-step Test-Time Training (Inner Loop)
            for step in range(30):
                raw_grid = extract_grid(obs)
                
                # Ask the Planner for a move
                grid_tensor, action_vector, em_action, em_coords = play_game_turn(obs, clone_model)
                
                # Record the human-readable action name, plus coordinates if it's a spatial click
                if em_coords:
                    action_history.append(f"{em_action.name}({em_coords['x']},{em_coords['y']})")
                else:
                    action_history.append(em_action.name)
                
                # Submit the move to the live emulator
                next_obs = env.step(em_action, data=em_coords)
                
                # --- VERIFICATION: Did the action actually change the board? ---
                next_raw_grid = extract_grid(next_obs)
                if np.array_equal(raw_grid, next_raw_grid):
                    # Append a warning to the last recorded action in our log
                    action_history[-1] += " (NO_CHANGE)"
                
                # Store the true outcome for the backprop
                target_next_frame = torch.tensor(next_raw_grid, dtype=torch.long)
                train_states.append((grid_tensor, action_vector, target_next_frame))
                
                obs = next_obs
                
                # FIX 3: Check if the game is over using the explicit GameState enum
                if obs and obs.state in [GameState.WIN, GameState.GAME_OVER]:
                    done = True
                    break
                    
            # --- VERIFICATION LOG ---
            final_status = obs.state.name if hasattr(obs, 'state') else "ALIVE_BUT_CAPPED"
            action_str = "[" + ", ".join(action_history) + "]"
            print(f"   -> Played {game.game_id}: {len(train_states)} moves | Outcome: {final_status}")
            print(f"      Actions: {action_str}")
            # ------------------------
                    
            # Run the actual fast adaptation on those captured turns
            adapted_clone = run_fast_adaptation(clone_model, inner_optimizer, train_states, steps=len(train_states))
            
            # C. Save the adapted clone and an unseen frame for the final exam
            if not done:
                test_grid, test_action_vector, test_em_action, test_coords = play_game_turn(obs, adapted_clone)
                next_obs = env.step(test_em_action, data=test_coords)
                
                true_test_frame = torch.tensor(extract_grid(next_obs), dtype=torch.long)
                meta_test_data.append((adapted_clone, test_grid, test_action_vector, true_test_frame))
            
        # 3. The Outer Loop (Gradient-through-a-gradient)
        if meta_test_data:
            epoch_loss = run_outer_update(global_model, outer_optimizer, meta_test_data)
            print(f"Epoch {epoch+1} | Average Meta-Loss: {epoch_loss:.4f}")
            
            # Log the metric to the CSV
            with open('training_log.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, epoch_loss])
                
            # Checkpoint the model if it hits a new record
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                os.makedirs('__pycache__', exist_ok=True)
                torch.save(global_model.state_dict(), os.path.join('__pycache__', 'arc_agi_checkpoint.pt'))
                print(f"   -> New best loss! Checkpoint saved to __pycache__.")
                
        else:
            print(f"Epoch {epoch+1} | Skipped Update (All games ended early)")