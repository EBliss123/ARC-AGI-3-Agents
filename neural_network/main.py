import torch
import torch.nn as nn
import torch.optim as optim
import arc_agi
from arcengine import GameAction, GameState
import random
import os
import csv

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

def play_game_turn(raw_grid, model_clone):
    # 1. Prepare the data
    grid_tensor = encode_grid(raw_grid).unsqueeze(0)
    action_mask = get_action_mask(raw_grid)

    valid_action_ids = [a.value for a in GameAction]
    
    for i in range(8):
        if i not in valid_action_ids:
            action_mask[i] = False
            
    if 6 not in valid_action_ids:
        action_mask[8:] = False

    # Hard-disable the RESET button so the agent never intentionally restarts
    for a in GameAction:
        if a.name == 'RESET':
            action_mask[a.value] = False
    
    # --- DYNAMIC GUARDRAIL: Restrict to valid actions for the current game ---
    valid_action_ids = [a.value for a in GameAction]
    
    # Disable any discrete buttons (0-7) that don't exist in this game
    for i in range(8):
        if i not in valid_action_ids:
            action_mask[i] = False
            
    # If ACTION6 (Spatial click) is not allowed, disable all 4,096 spatial nodes
    if 6 not in valid_action_ids:
        action_mask[8:] = False
        
    # 2. The Planner makes a decision (Forward Pass)
    # Add a batch dimension to the grid_tensor for the network (1, 16, 64, 64)
    action_probs = model_clone.planner(grid_tensor, action_mask)
    
    # 3. Select the action (Using multinomial to sample based on probabilities)
    chosen_index = torch.multinomial(action_probs, 1).item()
    
    # 4. Translate the flat index back into an action type and coordinates
    if chosen_index < 8:
        action_id = chosen_index
        x, y = 0, 0
    else:
        action_id = 6
        spatial_index = chosen_index - 8
        y = spatial_index // 64
        x = spatial_index % 64
        
    # 5. Build the exact unified vector for the Action Network
    action_vector = unify_action(action_id, x, y)
    
    # 6. Format for the Kaggle Emulator
    emulator_action = next(a for a in GameAction if a.value == action_id)
    emulator_coords = {"x": x, "y": y} if action_id == 6 else None
    
    return grid_tensor, action_vector, emulator_action, emulator_coords
    
if __name__ == "__main__":
    # 1. Initialize the Global Master Model
    global_model = AGI_Architecture()

    # 2. Setup the Global Outer Optimizer (Adam for slow, stable learning)
    outer_optimizer = optim.Adam(global_model.parameters(), lr=0.0005)

    epochs = 100
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
                grid_tensor, action_vector, em_action, em_coords = play_game_turn(raw_grid, clone_model)
                
                # Record the human-readable action name, plus coordinates if it's a spatial click
                if em_coords:
                    action_history.append(f"{em_action.name}({em_coords['x']},{em_coords['y']})")
                else:
                    action_history.append(em_action.name)
                
                # Submit the move to the live emulator
                next_obs = env.step(em_action, data=em_coords)
                
                # Store the true outcome for the backprop
                target_next_frame = torch.tensor(extract_grid(next_obs), dtype=torch.long)
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
                test_grid_raw = extract_grid(obs)
                test_grid, test_action_vector, test_em_action, test_coords = play_game_turn(test_grid_raw, adapted_clone)
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
                torch.save(global_model.state_dict(), 'arc_agi_checkpoint.pt')
                print(f"   -> New best loss! Checkpoint saved to disk.")
                
        else:
            print(f"Epoch {epoch+1} | Skipped Update (All games ended early)")