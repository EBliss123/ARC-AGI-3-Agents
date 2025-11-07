import optuna
import subprocess
import json
import argparse
import sys
import os

def run_agent_and_get_score(params: dict) -> float:
    """
    Launches the main agent script, streams its output, and captures the final score.
    """
    # Build the command as a LIST of arguments
    command = [sys.executable, 'main.py', '-a', 'obrlagi3agent']
    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    print(f"\n--- Starting Trial with command: {' '.join(command)} ---")
    
    final_result_line = None
    
    try:
        # Use Popen to get real-time access to the output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge error output with standard output
            text=True,
            env=os.environ
        )

        # Read the output line by line as it is generated
        for line in process.stdout:
            stripped_line = line.strip()

            # --- Filtered Printing ---
            # Only print the lines we care about for debugging
            if "DEBUG" in stripped_line or "WARNING" in stripped_line or "ERROR" in stripped_line:
                print(stripped_line)
            
            # Always check for the final result line to get the score
            if stripped_line.startswith("FINAL_RESULT:"):
                final_result_line = stripped_line
        
        process.wait() # Wait for the subprocess to fully complete

        if final_result_line:
            parts = final_result_line.replace("FINAL_RESULT: ", "").split(',')
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

        print("--- Trial Warning: FINAL_RESULT line not found. Trial likely failed. ---")
        return -float('inf')

    except Exception as e:
        print(f"--- Trial FAILED with an unexpected error: {e} ---")
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

        # Reward & Penalty Magnitudes (Integers, Log Scale)
        'reward_win': trial.suggest_int('reward_win', 100, 1000, log=True),
        'reward_novelty_multiplier': trial.suggest_int('reward_novelty_multiplier', 5, 250, log=True),
        'reward_new_effect_pattern': trial.suggest_int('reward_new_effect_pattern', 1, 50, log=True),
        'penalty_unexpected_failure': trial.suggest_int('penalty_unexpected_failure', 5, 150, log=True),
        'penalty_repeated_effect': trial.suggest_int('penalty_repeated_effect', 5, 500, log=True),
        'penalty_boring_move': trial.suggest_int('penalty_boring_move', 5, 500, log=True),
        'penalty_predicted_failure': trial.suggest_int('penalty_predicted_failure', 50, 1000, log=True),
        'penalty_blacklist_base': trial.suggest_int('penalty_blacklist_base', 1000, 10000, log=True),
        'penalty_blacklist_scaler': trial.suggest_int('penalty_blacklist_scaler', 50, 200, log=True),
        'drought_increment': trial.suggest_int('drought_increment', 5, 50, log=True),

        # Exploration & Goal Bonuses (Integers, Log Scale)
        'bonus_action_exp': trial.suggest_int('bonus_action_exp', 10, 500, log=True),
        'bonus_state_exp_unknown': trial.suggest_int('bonus_state_exp_unknown', 25, 750, log=True),
        'bonus_state_exp_known_scaler': trial.suggest_int('bonus_state_exp_known_scaler', 10, 500, log=True),
        'bonus_goal_seeking': trial.suggest_int('bonus_goal_seeking', 50, 500, log=True),
        'reward_goal_proximity': trial.suggest_int('reward_goal_proximity', 50, 500, log=True),

        # Heuristic Weights & Thresholds
        'weight_novelty_ratio': trial.suggest_int('weight_novelty_ratio', 2, 75, log=True),
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
        study_name='obrl_agi3_tuning_14', #change name for different studies
        storage='sqlite:///tuning_study_14.db', #change name for different studies
        load_if_exists=True
    )

    # Start the optimization. Optuna will call the 'objective' function N times.
    # 'n_jobs=-1' tells Optuna to run trials in parallel using all available CPU cores.
    try:
        study.optimize(objective, n_trials=2, n_jobs=2) #trials is total changes, jobs is how many cores (should be 2)
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