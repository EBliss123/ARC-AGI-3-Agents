import torch
import torch.nn as nn
import torch.optim as optim

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

def play_game_turn(raw_grid, model_clone, target_next_frame):
    # 1. Prepare the data
    grid_tensor = encode_grid(raw_grid)
    action_mask = get_action_mask(raw_grid)
    
    # 2. The Planner makes a decision (Forward Pass)
    # Add a batch dimension to the grid_tensor for the network (1, 16, 64, 64)
    action_probs = model_clone.planner(grid_tensor.unsqueeze(0), action_mask)
    
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
    
    # Return the exact tuple required by our run_fast_adaptation function
    return (grid_tensor, action_vector, target_next_frame)
    
if __name__ == "__main__":
    # 1. Initialize the Global Master Model
    global_model = AGI_Architecture()

    # 2. Setup the Global Outer Optimizer (Adam for slow, stable learning)
    outer_optimizer = optim.Adam(global_model.parameters(), lr=0.0005)

    epochs = 1000
    print("Booting ARC-AGI-3 Meta-Learning Engine...")

    for epoch in range(epochs):
        # We will plug your ARC JSON data loader in here to grab a batch of games.
        # This list will hold the final exam data for the Batched Outer Loop.
        meta_test_data = []
        
        # --- THE BATCH LOOP (Placeholder) ---
        # for game in batch_of_games:
            
        #     # A. Spawn the temporary clone and its fast optimizer
        #     clone_model, inner_optimizer = setup_game_clone(global_model, inner_lr=0.05)
            
        #     # B. Run the 5-step Test-Time Training (Inner Loop)
        #     # train_states = game.get_first_5_turns()
        #     adapted_clone = run_fast_adaptation(clone_model, inner_optimizer, train_states, steps=5)
            
        #     # C. Save the adapted clone and an unseen frame for the final exam
        #     # test_grid, test_action, true_test_frame = game.get_unseen_test_turn()
        #     meta_test_data.append((adapted_clone, test_grid, test_action, true_test_frame))
            
        # --- THE META UPDATE ---
        # 3. The Outer Loop (Gradient-through-a-gradient)
        # epoch_loss = run_outer_update(global_model, outer_optimizer, meta_test_data)
        # print(f"Epoch {epoch+1} | Average Meta-Loss: {epoch_loss:.4f}")