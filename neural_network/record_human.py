import torch
import arc_agi
from arcengine import GameAction
import matplotlib.pyplot as plt
import numpy as np
import os

# Master list to hold our human data trajectories
human_data = []

def extract_grid(obs):
    """Safely extracts the grid from the Kaggle observation."""
    grid_data = obs.frame if hasattr(obs, 'frame') else obs["grid"]
    return np.array(grid_data[-1] if isinstance(grid_data, list) else grid_data)

def print_available_actions(obs):
    """Prints out the current valid actions to the terminal."""
    print("\n" + "="*40)
    print("--- VALID AVAILABLE ACTIONS ---")
    for i, action in enumerate(obs.available_actions):
        # Handle Enum versus raw int attributes smoothly
        action_name = action.name if hasattr(action, 'name') else f"ACTION_{action}"
        action_val = action.value if hasattr(action, 'value') else action
        print(f" [{action_val}] -> {action_name}")
    print("="*40)
    print("-> Click on the grid to make a spatial click (ACTION6),")
    print("   OR go to your terminal and enter a valid numeric action ID for a discrete move.")

def play_and_record(game_id):
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    obs = env.reset()
    
    current_grid = extract_grid(obs)
    
    # Setup the visual UI
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title(f"Playing: {game_id} (Check terminal for valid moves menu)")
    img_display = ax.imshow(current_grid, cmap='tab20', vmin=0, vmax=15)
    
    # Show the initial action menu
    print_available_actions(obs)

    def process_and_log_action(action_id, coords=None):
        """Helper to process an action, step the emulator, and log tensors."""
        nonlocal current_grid, obs
        
        em_action = next(a for a in GameAction if a.value == action_id)
        next_obs = env.step(em_action, data=coords)
        next_grid = extract_grid(next_obs)
        
        # --- THE SNAPSHOT ---
        # Capture exactly what the network will train on later
        human_data.append({
            "state": current_grid.copy(),
            "action_id": action_id,
            "click_x": coords["x"] if coords else 0,
            "click_y": coords["y"] if coords else 0,
            "next_state": next_grid.copy()
        })
        
        # Update state and display
        current_grid = next_grid
        img_display.set_data(current_grid)
        fig.canvas.draw_idle()
        obs = next_obs
        
        action_str = f"{em_action.name}({coords['x']},{coords['y']})" if coords else em_action.name
        print(f"\n[RECORDED] Executed: {action_str}")
        
        # Check game state and offer the reset option
        if hasattr(obs, 'state'):
            print(f"Current Status: {obs.state.name}")
            if obs.state.name in ["GAME_OVER", "WIN"]:
                print("\n" + "!"*45)
                print(f"--- LEVEL ENDED: {obs.state.name} ---")
                print("-> Type 'r' and press Enter to RESET the board.")
                print("-> OR close the image window to SAVE and exit.")
                print("!"*45)
            
        # Reprint menu for the next turn if still alive
        if not hasattr(obs, 'state') or obs.state.name not in ["GAME_OVER", "WIN"]:
            print_available_actions(obs)

    def on_click(event):
        """Handles grid click interactions (ACTION6)."""
        if event.inaxes != ax: return
        
        # Verify ACTION6 is actually allowed right now
        valid_vals = [(a.value if hasattr(a, 'value') else a) for a in obs.available_actions]
        if 6 not in valid_vals:
            print("\n[WARNING] Spatial clicking (ACTION6) is not currently allowed in this state!")
            return
            
        x = max(0, min(63, int(round(event.xdata))))
        y = max(0, min(63, int(round(event.ydata))))
        
        process_and_log_action(6, {"x": x, "y": y})

    # Hook up the mouse event listener
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Run a secondary terminal loop to let you type discrete commands without breaking the window
    while plt.fignum_exists(fig.number):
        plt.pause(0.1) # Keep window reactive
        
        try:
            user_input = input("\nEnter Action ID (or 'r' to reset, press Enter for mouse): ").strip().lower()
            
            if user_input == 'r':
                print("\n[SYSTEM] Resetting the environment...")
                obs = env.reset()
                
                # Update visual board
                current_grid = extract_grid(obs)
                img_display.set_data(current_grid)
                fig.canvas.draw_idle()
                
                # Log the reset as a special trajectory step (-1)
                human_data.append({
                    "state": current_grid.copy(),
                    "action_id": -1, 
                    "click_x": 0,
                    "click_y": 0,
                    "next_state": current_grid.copy()
                })
                
                print_available_actions(obs)
                continue

            if user_input:
                action_choice = int(user_input)
                valid_vals = [(a.value if hasattr(a, 'value') else a) for a in obs.available_actions]
                
                if action_choice in valid_vals:
                    if action_choice == 6:
                        print("Use mouse clicks on the window to submit spatial actions.")
                    else:
                        process_and_log_action(action_choice, coords=None)
                else:
                    print(f"Invalid choice. {action_choice} is not an authorized action right now.")
        except ValueError:
            print("Please enter a valid integer ID.")
        except (KeyboardInterrupt, EOFError):
            break

import random # Add this to your imports at the top if it isn't there

if __name__ == "__main__":
    print("Initializing ARC Human Auto-Queue Recorder...")
    
    # 1. Boot the arcade to get the master list of all games
    temp_arc = arc_agi.Arcade()
    all_game_ids = [game.game_id for game in temp_arc.get_environments()]
    
    # 2. Check the hard drive for games you already beat
    os.makedirs('human_data', exist_ok=True)
    saved_files = os.listdir('human_data')
    completed_games = [f.replace('_trajectory.pt', '') for f in saved_files if f.endswith('_trajectory.pt')]
    
    # 3. Filter the master list down to only unplayed games
    unplayed_games = [g for g in all_game_ids if g not in completed_games]
    
    if not unplayed_games:
        print("Incredible! You have recorded data for every single game in the arcade.")
    else:
        # Pick a random unplayed game to keep your data diverse
        target_game = random.choice(unplayed_games)
        
        print(f"-> Found {len(completed_games)} completed games on disk.")
        print(f"-> {len(unplayed_games)} games remaining.")
        print(f"-> Auto-selecting new game: {target_game}\n")
        
        try:
            play_and_record(target_game)
        except Exception as e:
            print(f"Error running recorder session: {e}")
            
        # 4. Save the trajectory
        if len(human_data) > 0:
            save_path = f"human_data/{target_game}_trajectory.pt"
            torch.save(human_data, save_path)
            print(f"\nSuccessfully saved {len(human_data)} human moves to {save_path}!")
        else:
            print("\nSession exited. No moves were recorded.")