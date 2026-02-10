import argparse
import time
import sys
import os
import json
import numpy as np
import traceback
import gymnasium as gym
import requests
from dotenv import load_dotenv

# --- Load Environment Variables (Match main.py setup) ---
load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)

if (SCHEME == "http" and str(PORT) == "80") or (SCHEME == "https" and str(PORT) == "443"):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"

# --- Import the Toolkit Correctly ---
import arc_agi 

# --- Import your Agent ---
sys.path.insert(0, os.getcwd())
from agents.obrl_agi3 import ObrlAgi3Agent
from agents.structs import GameAction, GameState
from agents.agent import FrameData

def parse_arguments():
    parser = argparse.ArgumentParser(description="High-Speed Local Runner for ARC-AGI")
    parser.add_argument("-g", "--game", type=str, help="Game ID (e.g. 'ls20'). If omitted, runs ALL discovered games.")
    parser.add_argument("-a", "--agent", type=str, help="Ignored (compatibility flag).")
    return parser.parse_args()

def get_game_list(specific_game=None):
    if specific_game:
        return [specific_game]
    
    print("--- Discovering ARC-AGI games ---")
    games = []
    
    # --- STRATEGY 1: Fetch from API Server (Source of Truth) ---
    print(f"Attempting to fetch game list from {ROOT_URL}/api/games ...")
    try:
        r = requests.get(f"{ROOT_URL}/api/games", timeout=2)
        if r.status_code == 200:
            api_games = [g["game_id"] for g in r.json()]
            print(f"-> API Server returned {len(api_games)} games.")
            for g in api_games:
                if g not in games: games.append(g)
        else:
            print(f"-> API request failed: {r.status_code}")
    except Exception as e:
        print(f"-> Could not connect to API server ({e}). Assuming offline mode.")

    # --- STRATEGY 2: Check Gym Registry (Local Fallback) ---
    print("Scanning local Gym registry...")
    for env_id in gym.envs.registry.keys():
        if "ARC-AGI/" in env_id and "-v" in env_id:
            raw_id = env_id.split("/")[1].split("-v")[0]
            if raw_id not in games:
                games.append(raw_id)

    if not games:
        print("Warning: No games found via API or Registry.")
        # Fallback to known common training sets if totally empty
        #games = ["ls20", "vc33", "ft09"]
        print(f"GAMES NOT FOUND. EXITING")
        
    print(f"Total Games Scheduled: {len(games)}")
    return games

def get_dynamic_actions(env):
    valid_actions = []
    space = env.action_space
    
    if isinstance(space, gym.spaces.Discrete):
        n = space.n
        if n >= 2: valid_actions.append(GameAction.ACTION1)
        if n >= 3: valid_actions.append(GameAction.ACTION2)
        if n >= 4: valid_actions.append(GameAction.ACTION3)
        if n >= 5: valid_actions.append(GameAction.ACTION4)
        if n >= 6: valid_actions.append(GameAction.ACTION5)

    elif isinstance(space, gym.spaces.Tuple) or isinstance(space, gym.spaces.Dict):
        valid_actions = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6
        ]
    
    if not valid_actions:
        return [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, 
            GameAction.ACTION4, GameAction.ACTION5
        ]
        
    return valid_actions

def translate_action_to_env(agent_output):
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
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        return (6, {'x': x, 'y': y})
        
    return (0, {})

def run_single_game(game_id):
    print(f"\n>>> STARTING GAME: {game_id} <<<")
    
    result = {
        "game_id": game_id,
        "score": 0,
        "steps": 0,
        "status": "FAILED",
        "error": None
    }

    try:
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id, render_mode="terminal")
    except Exception as e:
        print(f"Skipping {game_id}: Error initializing environment ({e})")
        result["error"] = str(e)
        return result

    agent = ObrlAgi3Agent(
        card_id="local-run",
        game_id=game_id,
        agent_name="ScientificAgent",
        ROOT_URL="local",
        record=False
    )
    
    # Quiet Logs
    agent.debug_channels['PERCEPTION'] = False
    agent.debug_channels['CONTEXT_DETAILS'] = False
    agent.debug_channels['ACTION_SCORE'] = True
    agent.debug_channels['CHANGES'] = True
    agent.debug_channels['HYPOTHESIS'] = True 

    valid_actions = get_dynamic_actions(env)

    # --- RESET ---
    try:
        reset_result = env.reset()
    except Exception as e:
        print(f"Reset failed for {game_id}: {e}")
        result["error"] = f"Reset failed: {e}"
        return result

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs = reset_result
        info = {'score': getattr(obs, 'levels_completed', 0)}
    
    total_steps = 0
    
    try:
        while True:
            if total_steps >= agent.MAX_ACTIONS:
                print(f"--- MAX ACTIONS REACHED ({agent.MAX_ACTIONS}) ---")
                result["status"] = "TIMEOUT"
                break

            agent.action_counter = total_steps

            # 1. Extract Grid
            grid = None
            if hasattr(obs, 'frame'): grid = obs.frame
            elif isinstance(obs, dict):
                if 'frame' in obs: grid = obs['frame']
                elif 'image' in obs: grid = obs['image']
            elif isinstance(obs, np.ndarray): grid = obs
            
            if grid is None:
                print("CRITICAL ERROR: No grid found.")
                result["error"] = "No grid found in observation"
                break

            # 2. Convert to List
            if isinstance(grid, np.ndarray): grid = grid.tolist()
            elif isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], np.ndarray):
                grid = [row.tolist() for row in grid]

            # 3. Build FrameData
            current_frame = FrameData(
                frame=grid,
                state=GameState.NOT_FINISHED,
                available_actions=valid_actions,
                score=info.get('score', 0)
            )
            
            # 4. Agent Decides
            action = agent.choose_action([current_frame], current_frame)
            
            # 5. Step
            action_id, action_data = translate_action_to_env(action)
            step_result = env.step(action_id, data=action_data)
            
            # 6. Unpack
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
                print(f"Step {total_steps} | Score: {info.get('score', 0)}")
            
            if terminated or truncated:
                final_score = info.get('score', 0)
                print(f"--- GAME OVER: {game_id} ---")
                print(f"Final Score: {final_score}")
                
                result["score"] = final_score
                result["status"] = "COMPLETED"
                break
                
    except KeyboardInterrupt:
        raise 
    except Exception as e:
        print(f"\nRuntime Error in {game_id}: {e}")
        traceback.print_exc()
        result["error"] = str(e)
    # env.close() # Removed to prevent crashes
        
    result["steps"] = total_steps
    return result

def run_fast():
    args = parse_arguments()
    
    # 1. Discover Games
    games_to_play = get_game_list(args.game)
    
    total_start = time.time()
    all_results = []
    
    # 2. Run Loop
    for i, game_id in enumerate(games_to_play):
        print(f"\n[{i+1}/{len(games_to_play)}] Running {game_id}...")
        try:
            res = run_single_game(game_id)
            all_results.append(res)
        except KeyboardInterrupt:
            print("\nGlobal Stop requested.")
            break
    
    elapsed = time.time() - total_start
    
    # 3. Report
    print("\n" + "="*40)
    print(f" FAST RUN SUMMARY (Time: {elapsed:.2f}s)")
    print("="*40)
    print(f"{'GAME ID':<20} | {'SCORE':<5} | {'STEPS':<5} | {'STATUS'}")
    print("-" * 40)
    
    total_score = 0
    for r in all_results:
        print(f"{r['game_id']:<20} | {r['score']:<5} | {r['steps']:<5} | {r['status']}")
        total_score += r.get('score', 0)
        
    print("-" * 40)
    print(f"TOTAL SCORE: {total_score}")
    print("="*40)
    
    # Save to JSON
    try:
        with open("fast_run_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("Results saved to 'fast_run_results.json'")
    except Exception as e:
        print(f"Could not save JSON results: {e}")

if __name__ == "__main__":
    run_fast()