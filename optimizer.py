import optuna
import subprocess
import json
import argparse
import sys
import os

def run_agent_and_get_score(params: dict) -> float:
    """
    Launches the main agent script directly using the correct Python executable.
    """
    # Build the command as a LIST of arguments, starting with the exact Python executable.
    # This is the most robust way to run a subprocess from an activated environment.
    command = [sys.executable, 'main.py', '-a', 'obrlagi3agent']
    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    print(f"\n--- Starting Trial with command: {' '.join(command)} ---")
    
    try:
        # Pass the command as a list and set shell=False
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=1800,
            env=os.environ,
            shell=False # Important: shell=False when passing a list
        )
        
        for line in result.stdout.strip().split('\n'):
            if line.startswith("FINAL_RESULT:"):
                parts = line.replace("FINAL_RESULT: ", "").split(',')
                score = int(parts[0].split('=')[1])
                actions = int(parts[1].split('=')[1])
                
                objective_score = 0.0
                if score > 0:
                    efficiency_penalty = actions / score
                    objective_score = (score * 1000) - efficiency_penalty
                else:
                    objective_score = -actions
                
                print(f"--- Trial Complete. Score: {score}, Actions: {actions}. Objective Score: {objective_score:.2f} ---")
                return objective_score

        print("--- Trial Warning: FINAL_RESULT line not found in agent output. ---")
        print("--- Agent's Standard Output (stdout) ---")
        print(result.stdout) # Print stdout to see what happened
        return -float('inf')

    except subprocess.CalledProcessError as e:
        print(f"--- Trial FAILED with a crash (non-zero exit code). ---")
        print("--- Agent's Standard Output (stdout) ---")
        print(e.stdout)
        print("\n--- Agent's Standard Error (stderr) ---")
        print(e.stderr)
        return -float('inf')
    except subprocess.TimeoutExpired:
        print(f"--- Trial FAILED: Timed out. ---")
        return -float('inf')

def objective(trial: optuna.Trial) -> float:
    """
    Defines the search space and executes one trial.
    """
    # 1. Define the search space for each hyperparameter you want to tune.
    # Optuna will intelligently pick a value from this range for each trial.
    params = {
        # Learning Algorithm Parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
        'discount_factor': trial.suggest_float('discount_factor', 0.8, 0.99, step=0.01),

        # Reward & Penalty Magnitudes
        'reward_win': trial.suggest_float('reward_win', 100.0, 1000.0, step=0.01),
        'reward_novelty_multiplier': trial.suggest_float('reward_novelty_multiplier', 5.0, 250.0, step=0.01),
        'reward_new_effect_pattern': trial.suggest_float('reward_new_effect_pattern', 1.0, 50.0, step=0.01),
        'penalty_unexpected_failure': trial.suggest_float('penalty_unexpected_failure', 5.0, 150.0, step=0.01),
        'penalty_repeated_effect': trial.suggest_float('penalty_repeated_effect', 5.0, 500.0, step=0.01),
        'penalty_boring_move': trial.suggest_float('penalty_boring_move', 5.0, 500.0, step=0.01),
        'penalty_predicted_failure': trial.suggest_float('penalty_predicted_failure', 50.0, 1000.0, step=0.01),
        'penalty_blacklist_base': trial.suggest_float('penalty_blacklist_base', 1000.0, 10000.0, step=0.01),
        'penalty_blacklist_scaler': trial.suggest_float('penalty_blacklist_scaler', 50.0, 200.0, step=0.01),
        'drought_increment': trial.suggest_float('drought_increment', 5.0, 50.0, step=0.01),

        # Exploration & Goal Bonuses
        'bonus_action_exp': trial.suggest_float('bonus_action_exp', 10.0, 500.0, step=0.01),
        'bonus_state_exp_unknown': trial.suggest_float('bonus_state_exp_unknown', 25.0, 750.0, step=0.01),
        'bonus_state_exp_known_scaler': trial.suggest_float('bonus_state_exp_known_scaler', 10.0, 500.0, step=0.01),
        'bonus_goal_seeking': trial.suggest_float('bonus_goal_seeking', 50.0, 300.0, step=0.01),

        # Heuristic Weights & Thresholds
        'weight_novelty_ratio': trial.suggest_float('weight_novelty_ratio', 2.0, 75.0, step=0.01),
        'planning_confidence_threshold': trial.suggest_float('planning_confidence_threshold', 0.75, 0.99, step=0.01),
        'recent_effect_patterns_maxlen': trial.suggest_int('recent_effect_patterns_maxlen', 3, 50),
    }

    # 2. Run the agent with these parameters and get its performance score.
    return run_agent_and_get_score(params)


# --- The Main Tuning Process ---
if __name__ == "__main__":
    # Create a "study," which is a single tuning session.
    # The 'storage' argument saves results to a database file, so you can stop
    # and resume the tuning process at any time.
    study = optuna.create_study(
        direction='maximize',
        study_name='obrl_agi3_tuning_6', #change name for different studies
        storage='sqlite:///tuning_study_6.db', #change name for different studies
        load_if_exists=True
    )

    # Start the optimization. Optuna will call the 'objective' function N times.
    # 'n_jobs=-1' tells Optuna to run trials in parallel using all available CPU cores.
    try:
        study.optimize(objective, n_trials=8, n_jobs=-1) #trials is total changes, jobs uses all cores
    except KeyboardInterrupt:
        print("--- Tuning interrupted by user. Study progress has been saved. ---")

    # Print the results
    print("\n--- TUNING COMPLETE ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Objective Score: ", trial.value)
    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")