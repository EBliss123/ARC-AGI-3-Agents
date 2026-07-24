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
        # 1. Log the new experience permanently into the timeline
        experience_buffer.add_transition(action, transition_tensor)
        
        # 2. Universal Re-evaluation: Rebuild Baseline from ALL history
        pixel_equations = []
        for record in experience_buffer.history:
            hist_action = record["action"]
            hist_tensor = record["tensor"]
            for row in hist_tensor:
                x, y, old_c, new_c = row.tolist()
                if old_c != new_c:
                    pixel_equations.append({"trigger": hist_action, "x": x, "y": y, "c_initial": old_c, "c_final": new_c})
        rulebook.set_baseline(pixel_equations)
        
        # 3. Algorithmic Grouping across ALL history
        ledger.wipe_ledger()
        grouped_sets = {}
        
        for record in experience_buffer.history:
            hist_action = record["action"]
            hist_tensor = record["tensor"]
            for row in hist_tensor:
                x, y, old_c, new_c = row.tolist()
                # Group by the action AND the color transition to map behaviors across turns
                trans_key = (hist_action, old_c, new_c) 
                
                if trans_key not in grouped_sets:
                    grouped_sets[trans_key] = []
                grouped_sets[trans_key].append({"x": x, "y": y, "c_initial": old_c, "c_final": new_c})
                
        # 4. Generate Abstracted State Rules 
        action_rules_map = {}
        for trans_key, pixels in grouped_sets.items():
            hist_action, old_c, new_c = trans_key
            if old_c == new_c:
                continue # Skip ledger registration and rules for pure static background pixels
                
            # Deduplicate coordinates in case pixels are static in some turns and moving in others
            unique_pixels = []
            seen_coords = set()
            for p in pixels:
                coord = (p['x'], p['y'])
                if coord not in seen_coords:
                    seen_coords.add(coord)
                    unique_pixels.append(p)

            # Extract the physical footprint of the current group
            current_coords = set((p['x'], p['y']) for p in unique_pixels)
            
            # Generate the Relational Signature (Shape & Color Orientation)
            # Sort by Y then X to reliably find the top-left-most actual pixel in the shape
            sorted_pixels = sorted(unique_pixels, key=lambda p: (p['y'], p['x']))
            anchor_x, anchor_y = sorted_pixels[0]['x'], sorted_pixels[0]['y']
            
            # Map every single pixel's delta and color relative to the anchor
            shape_signature = tuple(
                (p['x'] - anchor_x, p['y'] - anchor_y, p['c_initial']) 
                for p in sorted_pixels
            )
            
            existing_ledger_id = None
            
            # Scan the existing universe to see if this object already has a Set ID
            for set_id, existing_pixels in ledger.sets.items():
                existing_coords = set((p['x'], p['y']) for p in existing_pixels)
                if current_coords == existing_coords:
                    existing_ledger_id = set_id
                    break
                    
            # Reuse the existing ID if found, otherwise register a new object
            if existing_ledger_id:
                ledger_id = existing_ledger_id
            else:
                ledger_id = ledger.add_set(unique_pixels)
            
            abstract_rule = {
                "affected_ledger_id": ledger_id,
                "logic": "state_mapping",
                "conditions": [{"c_initial": old_c, "c_final": new_c}]
            }
            
            if hist_action not in action_rules_map:
                action_rules_map[hist_action] = []
            action_rules_map[hist_action].append(abstract_rule)
            
        # 5. Wrap under universal triggers
        compressed_rules = []
        for hist_action, state_changes in action_rules_map.items():
            compressed_rules.append({
                "trigger": hist_action,
                "state_changes": state_changes
            })
            
        rulebook.set_compressed(compressed_rules)
            
        return len(pixel_equations), len(rulebook.get_active_rules())