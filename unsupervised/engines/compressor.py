import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CompressorNet(nn.Module):
    def __init__(self, num_sets=16):
        super().__init__()
        # Ingests [x, y, c_initial, c_final] for all 4096 pixels
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, num_sets)

    def forward(self, x):
        # x shape: [4096, 4]
        out = F.relu(self.fc1(x))
        logits = self.fc2(out) # shape: [4096, num_sets]
        # Return the most probable Set ID for every single coordinate
        return torch.argmax(logits, dim=-1)

class AlgebraicCompressor:
    def __init__(self):
        pass # The neural network is removed; grouping is now purely algorithmic

    def process_transition(self, action, transition_tensor, experience_buffer, rulebook, ledger):
        experience_buffer.add_transition(action, transition_tensor)
        
        # 1. Capture the Full Board Baseline
        pixel_equations = []
        for row in transition_tensor:
            x, y, old_c, new_c = row.tolist()
            if old_c != new_c:
                pixel_equations.append({"trigger": action, "x": x, "y": y, "c_initial": old_c, "c_final": new_c})
        rulebook.set_baseline(pixel_equations)
        
        # 2. Algorithmic Grouping (Collapse by raw commonalities)
        ledger.wipe_ledger()
        grouped_sets = {}
        
        for row in transition_tensor:
            x, y, old_c, new_c = row.tolist()
            trans_key = (old_c, new_c)
            
            if trans_key not in grouped_sets:
                grouped_sets[trans_key] = []
            grouped_sets[trans_key].append({"x": x, "y": y, "c_initial": old_c, "c_final": new_c})
            
        # 3. Generate Abstracted State Rules 
        state_changes = []
        for trans_key, pixels in grouped_sets.items():
            old_c, new_c = trans_key
            if old_c == new_c:
                continue # Skip ledger registration and rules for pure static background pixels
                
            # Only add to ledger if it is actually going to be used in a rule
            ledger_id = ledger.add_set(pixels)
                
            abstract_rule = {
                "affected_ledger_id": ledger_id,
                "logic": "state_mapping",
                "conditions": [{"c_initial": old_c, "c_final": new_c}]
            }
            state_changes.append(abstract_rule)
            
        # 4. Wrap under a single universal trigger
        if state_changes:
            unified_action_rule = {
                "trigger": action,
                "state_changes": state_changes
            }
            rulebook.set_compressed([unified_action_rule])
        else:
            rulebook.set_compressed([])
            
        return len(pixel_equations), len(rulebook.get_active_rules())