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

    # 2. Execute an action to trigger an animation
    action_to_take = "ACTION1"
    print(f"\nExecuting {action_to_take}...")
    frames_after = interface.step(action_to_take)
    
    if frames_after is None:
        print("Action failed or game ended unexpectedly.")
        return
        
    print(f"Action resulted in {frames_after.shape[0]} animation frame(s).")
    
    # 3. Pass the data to the math engine (The Observer)
    print("\nRouting data to the DeltaObserver...")
    transitions = observer.observe(grid_before, frames_after)
    
    # 4. Read the Ground Truth (Test Output)
    print(f"\nObserver successfully generated {len(transitions)} transition tensor(s)!")
    
    for step_idx, transition_tensor in enumerate(transitions):
        print(f"\n--- Animation Step {step_idx + 1} ---")
        
        bloat_count, compressed_count = compressor.process_transition(action_to_take, transition_tensor, buffer, rulebook, ledger)
        
        print(f"ExperienceBuffer logged raw tensor of shape: {transition_tensor.shape}")
        print(f"Baseline (Bloat) Memory: {bloat_count} raw equations.")
        print(f"Compressed Working Memory: {compressed_count} abstract rule(s).")
        
        print("\n--- Active Theories & Sets ---")
        print("Active Physics Rules:")
        for rule in rulebook.get_active_rules():
            print(json.dumps(rule, indent=4))
            
        print("\nObject Ledger Definitions:")
        for set_id, pixels in ledger.sets.items():
            changed = len([p for p in pixels if p['c_initial'] != p['c_final']])
            # Extract just the x, y tuples for a clean console print
            coords = [(p['x'], p['y']) for p in pixels]
            print(f"  {set_id}: {len(pixels)} total pixels ({changed} changed).")
            print(f"    Coords: {coords}")

if __name__ == "__main__":
    main()