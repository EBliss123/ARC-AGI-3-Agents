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
    predicted_change_mask = change_logits.argmax(dim=1)
    
    # Dynamic active mask: what actually changed OR what the network thought would change
    active_mask = (actual_change_mask > 0) | (predicted_change_mask > 0)
    active_area = active_mask.sum().float().clamp(min=1.0) # clamp prevents division by zero
    
    change_loss = F.cross_entropy(change_logits, actual_change_mask)
    pixel_color_losses = F.cross_entropy(color_logits, target_colors, reduction='none')
    
    # Isolate color loss to the active area and apply square root scaling
    active_color_loss = pixel_color_losses[active_mask].sum() / torch.sqrt(active_area)
    
    # The Physics Error is the raw measure of "Surprise"
    physics_loss = change_loss + active_color_loss
    
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
    torch.nn.utils.clip_grad_norm_(clone_model.predictor.parameters(), max_norm=1.0)
    physics_optimizer.step()
    
    # Update the Decision Engine (Maximize Reward / Seek Surprise)
    policy_loss = -(log_prob * final_reward)
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(clone_model.planner.parameters(), max_norm=1.0)
    policy_optimizer.step()
        
    return clone_model, final_reward

def average_successful_weights(global_model, successful_weights_list):
    avg_state_dict = global_model.state_dict()
    for key in avg_state_dict.keys():
        stacked = torch.stack([w[key] for w in successful_weights_list])
        # .float() ensures we don't get mathematical errors if averaging integer tracking layers
        avg_state_dict[key] = torch.mean(stacked.float(), dim=0).to(stacked.dtype)
    global_model.load_state_dict(avg_state_dict)