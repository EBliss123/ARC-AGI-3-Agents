"""
Purpose: The Central Orchestrator.
Responsibilities:
1. Initializes the environment, memory, and engines.
2. Runs the main execution loop (Predict -> Act -> Observe -> Verify).
3. Passes tensors between modules. Contains NO math or logic of its own.
"""
from environment.arc_interface import ARCInterface
from engines.observer import DeltaObserver
from engines.compressor import AlgebraicCompressor
from memory.object_ledger import ObjectLedger
from memory.physics_rulebook import PhysicsRulebook
from memory.experience_buffer import ExperienceBuffer
import json

def main():
    print("Initializing ARC-AGI-3 Environment for 'ls20'...")
    interface = ARCInterface(task_id="ls20")
    observer = DeltaObserver()
    ledger = ObjectLedger()
    rulebook = PhysicsRulebook()
    buffer = ExperienceBuffer()
    compressor = AlgebraicCompressor()
    
    # 1. Start the game and get the initial board state (grid_before)
    initial_frames = interface.reset()
    if initial_frames is None:
        print("Failed to load environment.")
        return
        
    # initial_frames is a 3D tensor [Frames, Y, X]. We just need the very last frame 
    # to serve as our "before" grid for the upcoming action.
    grid_before = initial_frames[-1] 
    print(f"Game Started. Initial Board Shape: {grid_before.shape}")

    # 2. Execute multiple actions to trigger animations
    actions_to_take = ["ACTION1", "ACTION2"]
    current_grid = initial_frames[-1]
    
    for step_idx, action in enumerate(actions_to_take):
        print(f"\n=============================================")
        print(f"Executing {action} (Step {step_idx + 1})...")
        frames_after = interface.step(action)
        
        if frames_after is None:
            print("Action failed or game ended unexpectedly.")
            break
            
        print(f"Action resulted in {frames_after.shape[0]} animation frame(s).")
        
        # 3. Pass the data to the math engine (The Observer)
        transitions = observer.observe(current_grid, frames_after)
        
        for transition_tensor in transitions:
            bloat_count, compressed_count = compressor.process_transition(action, transition_tensor, buffer, rulebook, ledger)
            
            print(f"ExperienceBuffer logged raw tensor of shape: {transition_tensor.shape}")
            print(f"Universal Baseline (Bloat) Memory: {bloat_count} raw equations.")
            print(f"Universal Compressed Memory: {compressed_count} abstract rule(s).")
            
        # Print state of theories after this action
        print("\n--- Active Theories & Sets ---")
        print("Active Physics Rules:")
        for rule in rulebook.get_active_rules():
            print(json.dumps(rule, indent=4))
            
        print("\nObject Ledger Definitions:")
        for set_id, pixels in ledger.sets.items():
            changed = len([p for p in pixels if p['c_initial'] != p['c_final']])
            coords = [(p['x'], p['y']) for p in pixels]
            print(f"  {set_id}: {len(pixels)} total pixels ({changed} changed).")
            print(f"    Coords: {coords}")

        # Update the current_grid for the next action in the loop
        current_grid = frames_after[-1]

if __name__ == "__main__":
    main()