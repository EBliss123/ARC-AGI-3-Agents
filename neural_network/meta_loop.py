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
        
    # 3. Ensure the Action Network (Fast Weights) is active and ready to learn
    for param in clone_model.action_network.parameters():
        param.requires_grad = True
        
    # 4. Bind the Inner Optimizer strictly to the Action Network
    inner_optimizer = optim.SGD(clone_model.action_network.parameters(), lr=inner_lr)
    
    return clone_model, inner_optimizer

def run_fast_adaptation(clone_model, inner_optimizer, game_states, steps=5):
    # Set the clone to training mode to enable gradient tracking
    clone_model.train()
    
    for step in range(steps):
        # Unpack the specific turn data (Input Frame, Action Taken, Actual Next Frame)
        grid_tensor, action_vector, target_next_frame = game_states[step]
        
        # 1. Generate the physics modifiers from the Action Network
        film_params = clone_model.action_network(action_vector)
        
        # 2. Predict the next frame using the frozen Predictor + active modifiers
        predicted_frame = clone_model.predictor(grid_tensor, film_params)
        
        # 3. Calculate "Surprise" (Categorical Cross-Entropy Loss)
        # Flatten the spatial dimensions for the loss calculation
        loss = F.cross_entropy(
            predicted_frame.view(16, -1).unsqueeze(0), 
            target_next_frame.view(-1).unsqueeze(0)
        )
        
        # 4. Backpropagate the error directly into the Action Network
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
        
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
        
        # 2. Calculate the Meta-Loss (Surprise on the unseen frame)
        loss = F.cross_entropy(
            predicted_frame.view(16, -1).unsqueeze(0), 
            true_test_frame.view(-1).unsqueeze(0)
        )
        total_meta_loss += loss
        
    # 3. Average the loss across the batch of randomized games
    avg_meta_loss = total_meta_loss / len(meta_test_data)
    
    # 4. The Gradient-through-a-Gradient update
    # This physically alters the global model's slow weights and baseline fast weights
    avg_meta_loss.backward()
    outer_optimizer.step()
    
    return avg_meta_loss.item()