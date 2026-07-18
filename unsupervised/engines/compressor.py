import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # The neural engine that infers the pattern groupings
        self.net = CompressorNet()

    def process_transition(self, transition_tensor, experience_buffer, rulebook, ledger):
        experience_buffer.add_transition(transition_tensor)
        
        # 1. Capture the Full Board Baseline
        pixel_equations = []
        for row in transition_tensor:
            x, y, old_c, new_c = row.tolist()
            if old_c != new_c: # Only store actual changes in the baseline list
                pixel_equations.append({"x": x, "y": y, "c_initial": old_c, "c_final": new_c})
        rulebook.set_baseline(pixel_equations)
        
        # 2. Run the Neural Network to assign Set IDs across the raw history
        # Convert tensor data to float for the linear layers
        input_data = transition_tensor.float()
        set_assignments = self.net(input_data) # Returns a Set ID for all 4096 pixels
        
        # 3. State-Mapping Translator (Group pixels by Set ID)
        ledger.wipe_ledger()
        grouped_sets = {}
        for idx, set_id_tensor in enumerate(set_assignments):
            set_id = int(set_id_tensor.item())
            x, y, old_c, new_c = transition_tensor[idx].tolist()
            
            if set_id not in grouped_sets:
                grouped_sets[set_id] = []
            grouped_sets[set_id].append({"x": x, "y": y, "c_initial": old_c, "c_final": new_c})
            
        # 4. Generate Abstracted State Rules based on the network's sets
        compressed_rules = []
        for set_id, pixels in grouped_sets.items():
            # Register full coordinate and color info inside the ledger
            ledger_id = ledger.add_set(pixels)
            
            # Find which pixels in this set actually changed colors
            changes_in_set = [p for p in pixels if p["c_initial"] != p["c_final"]]
            
            if not changes_in_set:
                continue # If no pixels changed in this set, it's a pure static object (no rule needed)
                
            # Create a compact state-mapping rule pointing to this ledger set
            abstract_rule = {
                "target_set": ledger_id,
                "logic": "state_mapping",
                "conditions": [{"c_initial": p["c_initial"], "c_final": p["c_final"]} for p in changes_in_set[:1]]
            }
            compressed_rules.append(abstract_rule)
            
        rulebook.set_compressed(compressed_rules)
        return len(pixel_equations), len(compressed_rules)