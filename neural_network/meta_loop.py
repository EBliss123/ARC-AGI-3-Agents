import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

def setup_game_clone(global_model, inner_lr=0.05):
    # 1. Spawn a completely independent temporary clone of the entire network
    clone_model = copy.deepcopy(global_model)
    
    # 2. Freeze the Predictor Trunk (Slow Weights)
    for param in clone_model.predictor.parameters():
        param.requires_grad = False
        
    # 3. Ensure the Action Network AND Planner are active and ready to learn
    for param in clone_model.action_network.parameters():
        param.requires_grad = True
    for param in clone_model.planner.parameters():
        param.requires_grad = True
        
    # 4. DECOUPLE OPTIMIZERS: One for Physics (Action), One for Policy (Planner)
    physics_optimizer = optim.SGD(clone_model.action_network.parameters(), lr=inner_lr)
    policy_optimizer = optim.SGD(clone_model.planner.parameters(), lr=0.01)
    
    return clone_model, physics_optimizer, policy_optimizer

def run_single_turn_adaptation(clone_model, physics_optimizer, policy_optimizer, turn_data):
    # Set the clone to training mode to enable gradient tracking
    clone_model.train()
    
    # Unpack the exact data from the move that just happened
    grid_tensor, action_vector, target_next_frame, log_prob, reward = turn_data
    
    # --- 1. PHYSICS UPDATE (Learn the mechanics of what just changed) ---
    film_params = clone_model.action_network(action_vector)
    predicted_frame = clone_model.predictor(grid_tensor, film_params)
    
    pixel_losses = F.cross_entropy(
        predicted_frame.reshape(16, -1).unsqueeze(0), 
        target_next_frame.reshape(-1).unsqueeze(0),
        reduction='none'
    )
    
    original_flat = grid_tensor.argmax(dim=1).reshape(-1)
    target_flat = target_next_frame.reshape(-1)
    change_mask = (original_flat != target_flat).float()
    
    weight_map = torch.where(change_mask > 0, 50.0, 1.0)
    physics_loss = (pixel_losses.squeeze() * weight_map).mean()
    
    physics_optimizer.zero_grad()
    physics_loss.backward()
    physics_optimizer.step()
    
    # --- 2. POLICY UPDATE (Instantly update the Planner based on the reward) ---
    # Because we do this immediately, PyTorch's memory stays perfectly clean!
    policy_loss = -(log_prob * reward)
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
        
    return clone_model

def transition_to_next_level(clone_model, boost_lr=0.08):
    # Keep the adapted fast weights intact, but assign a new, highly aggressive 
    # inner optimizer to prevent the physics from getting rigidly locked in.
    new_inner_optimizer = optim.SGD(clone_model.action_network.parameters(), lr=boost_lr)
    
    return clone_model, new_inner_optimizer

def run_outer_update(global_model, outer_optimizer, meta_test_data):
    # Ensure the global model is active to receive the slow weight updates
    global_model.train()
    outer_optimizer.zero_grad()
    
    total_meta_loss = 0.0
    
    # meta_test_data is a list of tuples: (adapted_clone, test_grid, test_action, true_test_frame)
    for adapted_clone, test_grid, test_action, true_test_frame in meta_test_data:
        
        # 1. The clone predicts an unseen frame using its adapted fast weights
        film_params = adapted_clone.action_network(test_action)
        predicted_frame = adapted_clone.predictor(test_grid, film_params)
        
        # 2. Calculate the Meta-Loss with the Delta-Focus Multiplier
        pixel_losses = F.cross_entropy(
            predicted_frame.reshape(16, -1).unsqueeze(0), 
            true_test_frame.reshape(-1).unsqueeze(0),
            reduction='none'
        )
        
        original_flat = test_grid.argmax(dim=1).reshape(-1)
        target_flat = true_test_frame.reshape(-1)
        change_mask = (original_flat != target_flat).float()
        
        weight_map = torch.where(change_mask > 0, 50.0, 1.0)
        weighted_loss = (pixel_losses.squeeze() * weight_map).mean()
        
        total_meta_loss += weighted_loss
        
    # 3. Average the loss across the batch of randomized games
    avg_meta_loss = total_meta_loss / len(meta_test_data)
    
    # 4. The Gradient-through-a-Gradient update
    # This physically alters the global model's slow weights and baseline fast weights
    avg_meta_loss.backward()
    outer_optimizer.step()
    
    return avg_meta_loss.item()