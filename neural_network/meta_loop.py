import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

def setup_game_clone(global_model, inner_lr=0.05):
    # 1. Spawn a completely independent temporary clone of the entire network
    clone_model = copy.deepcopy(global_model)
    
    # 2. Both Planner and Predictor are now ACTIVE learners in the inner loop!
    for param in clone_model.predictor.parameters():
        param.requires_grad = True
    for param in clone_model.planner.parameters():
        param.requires_grad = True
        
    # 3. DECOUPLE OPTIMIZERS: 
    # Physics Engine gets the fast lr (0.05), Decision Engine gets a slower, stable lr (0.001)
    physics_optimizer = optim.SGD(clone_model.predictor.parameters(), lr=inner_lr)
    policy_optimizer = optim.SGD(clone_model.planner.parameters(), lr=0.001)
    
    return clone_model, physics_optimizer, policy_optimizer

def run_single_turn_adaptation(clone_model, physics_optimizer, policy_optimizer, turn_data):
    clone_model.train()
    
    # Unpack the exact data, including the change flag and the current step count
    grid_nodes, action_vector, target_next_frame, log_prob, valid_change_flag, step = turn_data
    
    grid_batch = grid_nodes.unsqueeze(0)
    action_batch = action_vector.unsqueeze(0)
    
    # --- 1. PHYSICS UPDATE (Observer) ---
    change_logits, color_logits = clone_model.predictor(grid_batch, action_batch)
    
    change_logits = change_logits.squeeze(0)  
    color_logits = color_logits.squeeze(0)    
    
    target_colors = target_next_frame.view(-1)
    
    original_colors = grid_nodes[:, :16].argmax(dim=1)
    actual_change_mask = (original_colors != target_colors).long()
    
    change_loss = F.cross_entropy(change_logits, actual_change_mask)
    pixel_color_losses = F.cross_entropy(color_logits, target_colors, reduction='none')
    weight_map = torch.where(actual_change_mask > 0, 50.0, 1.0)
    weighted_color_loss = (pixel_color_losses * weight_map).mean()
    
    # The Physics Error is the raw measure of "Surprise"
    physics_loss = change_loss + weighted_color_loss
    
    # --- 2. ADVERSARIAL REWARD CALCULUS ---
    if valid_change_flag < 0:
        # Heavily punish wasted clicks
        final_reward = -1.0
    else:
        # Surprise Metric: Scale the physics loss down so it's a manageable reward
        surprise_bonus = min(physics_loss.item() * 0.1, 2.0)
        
        # Growing Time Penalty: Starts at -0.01 and gets slightly worse every move
        time_penalty = 0.01 + (0.0001 * step)
        
        # The agent gets dopamine for being confused, but suffers a tax for taking too long
        final_reward = surprise_bonus - time_penalty
    
    # --- 3. EXECUTE GRADIENTS ---
    # Update the Physics Engine (Reduce Surprise)
    physics_optimizer.zero_grad()
    physics_loss.backward()
    physics_optimizer.step()
    
    # Update the Decision Engine (Maximize Reward / Seek Surprise)
    policy_loss = -(log_prob * final_reward)
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
        
    return clone_model, final_reward

def run_outer_update(global_model, outer_optimizer, meta_test_data):
    global_model.train()
    outer_optimizer.zero_grad()
    
    total_meta_loss = 0.0
    
    for adapted_clone, test_grid, test_action, true_test_frame in meta_test_data:
        
        grid_batch = test_grid.unsqueeze(0)
        action_batch = test_action.unsqueeze(0)
        
        change_logits, color_logits = adapted_clone.predictor(grid_batch, action_batch)
        
        change_logits = change_logits.squeeze(0)
        color_logits = color_logits.squeeze(0)
        target_colors = true_test_frame.view(-1)
        
        original_colors = test_grid[:, :16].argmax(dim=1)
        actual_change_mask = (original_colors != target_colors).long()
        
        change_loss = F.cross_entropy(change_logits, actual_change_mask)
        pixel_color_losses = F.cross_entropy(color_logits, target_colors, reduction='none')
        weight_map = torch.where(actual_change_mask > 0, 50.0, 1.0)
        weighted_color_loss = (pixel_color_losses * weight_map).mean()
        
        total_meta_loss += (change_loss + weighted_color_loss)
        
    avg_meta_loss = total_meta_loss / len(meta_test_data)
    avg_meta_loss.backward()
    outer_optimizer.step()
    
    return avg_meta_loss.item()