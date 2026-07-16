"""
Purpose: The Central Orchestrator.
Responsibilities:
1. Initializes the environment, memory, and engines.
2. Runs the main execution loop (Predict -> Act -> Observe -> Verify).
3. Passes tensors between modules. Contains NO math or logic of its own.
"""
from environment.arc_interface import ARCInterface
from engines.observer import DeltaObserver

def main():
    print("Initializing ARC-AGI-3 Environment for 'ls20'...")
    interface = ARCInterface(task_id="ls20")
    observer = DeltaObserver()
    
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
        # Filter the tensor to only show rows where the old color doesn't match the new color
        changed_pixels = transition_tensor[transition_tensor[:, 2] != transition_tensor[:, 3]]
        
        print(f"\n--- Animation Step {step_idx + 1} ---")
        if len(changed_pixels) == 0:
            print("No pixels changed in this frame.")
        else:
            print(f"Found {len(changed_pixels)} changed pixels [x, y, old_color, new_color]:")
            print(changed_pixels)

if __name__ == "__main__":
    main()