import argparse
import time
import sys
import os
import numpy as np
import traceback

# --- Import the Toolkit Correctly ---
import arc_agi 

# --- Import your Agent ---
sys.path.insert(0, os.getcwd())
from agents.obml_agi_3 import ObmlAgi3Agent
from agents.structs import GameAction, GameState
from agents.agent import FrameData

def parse_arguments():
    parser = argparse.ArgumentParser(description="High-Speed Local Runner for ARC-AGI")
    parser.add_argument("-g", "--game", type=str, required=True, help="Game ID (e.g. 'ls20')")
    parser.add_argument("-a", "--agent", type=str, help="Ignored (compatibility flag).")
    return parser.parse_args()

def get_available_actions():
    return [
        GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, 
        GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6
    ]

def translate_action_to_env(agent_output):
    """
    Returns a tuple: (action_id_int, data_dict)
    This allows us to pass them separately to env.step()
    """
    if not agent_output: return (0, {})
    
    action_enum = agent_output
    data = {}
    if isinstance(agent_output, tuple):
        action_enum = agent_output[0]
        if len(agent_output) > 1: data = agent_output[1]
    elif hasattr(agent_output, 'data'):
        data = agent_output.data

    name = action_enum.name if hasattr(action_enum, 'name') else str(action_enum)

    if name == 'ACTION1': return (1, {})
    if name == 'ACTION2': return (2, {})
    if name == 'ACTION3': return (3, {})
    if name == 'ACTION4': return (4, {})
    if name == 'ACTION5': return (5, {})
    
    if name == 'ACTION6':
        # Prepare the data dictionary expected by ActionInput
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        # Note: The toolkit likely expects 'x' and 'y' in the data dict for clicks
        return (6, {'x': x, 'y': y})
        
    return (0, {})

def run_fast():
    args = parse_arguments()
    print(f"--- INITIALIZING HIGH-SPEED RUNNER FOR: {args.game} ---")

    try:
        arcade = arc_agi.Arcade()
        env = arcade.make(args.game, render_mode="terminal")
    except Exception as e:
        print(f"Error initializing ARC Environment: {e}")
        return

    agent = ObmlAgi3Agent(
        card_id="local-fast-run",
        game_id=args.game,
        agent_name="ScientificAgent",
        ROOT_URL="local",
        record=False
    )
    
    # Silence Input/Output Logs
    agent.debug_channels['PERCEPTION'] = False
    agent.debug_channels['CONTEXT_DETAILS'] = False
    agent.debug_channels['ACTION_SCORE'] = True
    agent.debug_channels['CHANGES'] = True
    agent.debug_channels['HYPOTHESIS'] = True 

    # --- RESET ---
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs = reset_result
        info = {'score': getattr(obs, 'levels_completed', 0)}
    
    total_steps = 0
    start_time = time.time()
    
    try:
        while True:
            # --- 1. EXTRACT GRID ---
            grid = None
            if hasattr(obs, 'frame'): grid = obs.frame
            elif isinstance(obs, dict):
                if 'frame' in obs: grid = obs['frame']
                elif 'image' in obs: grid = obs['image']
            elif isinstance(obs, np.ndarray): grid = obs
            
            if grid is None:
                print("CRITICAL ERROR: No grid found in observation.")
                break

            # --- 2. CONVERT TO PYTHON LIST ---
            if isinstance(grid, np.ndarray): grid = grid.tolist()
            elif isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], np.ndarray):
                grid = [row.tolist() for row in grid]

            # --- 3. BUILD AGENT FRAME ---
            current_frame = FrameData(
                frame=grid,
                state=GameState.NOT_FINISHED,
                available_actions=get_available_actions(),
                score=info.get('score', 0)
            )
            
            # --- 4. AGENT DECIDES ---
            action = agent.choose_action([current_frame], current_frame)
            
            # --- 5. STEP ENVIRONMENT (THE FIX) ---
            # Unpack the ID and Data separately
            action_id, action_data = translate_action_to_env(action)
            
            # Pass them as separate arguments: step(id, data=...)
            step_result = env.step(action_id, data=action_data)
            
            # --- 6. UNPACK RESULT ---
            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                elif len(step_result) == 4:
                    obs, reward, terminated, info = step_result
                    truncated = False
                else:
                    obs = step_result[0]
                    terminated, truncated = False, False
                    info = {}
            else:
                obs = step_result
                state_val = getattr(obs, 'state', 'NOT_FINISHED')
                state_str = str(state_val)
                terminated = "NOT_FINISHED" not in state_str
                truncated = False
                info = {'score': getattr(obs, 'levels_completed', 0)}

            total_steps += 1
            if total_steps % 1000 == 0:
                elapsed = time.time() - start_time
                fps = total_steps / elapsed
                print(f"Step {total_steps} | Score: {info.get('score', 0)} | FPS: {fps:.0f}")
            
            if terminated or truncated:
                print(f"--- GAME OVER ---")
                print(f"Final Score: {info.get('score', 0)}")
                print(f"Total Steps: {total_steps}")
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nRuntime Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_fast()