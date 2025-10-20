import optuna
import subprocess
import argparse
import sys

def run_agent_and_get_score(params: dict) -> float:
    """
    Launches the main agent script with a given set of hyperparameters,
    captures its output, and parses the final score.
    """
    # ---
    # PREREQUISITE 1: Your main entrypoint script (e.g., main.py) must be modified
    # to accept command-line arguments and pass them to the agent's constructor.
    # An example of how to do this is provided in the comments at the bottom of this file.
    # ---
    
    # This command assumes your main script is named 'main.py'.
    # Change 'main.py' if your entrypoint script has a different name.
    command = ['python', 'main.py']
    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    print(f"\n--- Starting Trial with command: {' '.join(command)} ---")
    
    try:
        # Run the agent script and wait for it to complete.
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True, # This will raise an error if the script fails
            timeout=1800  # Set a timeout (e.g., 30 minutes) to kill trials that get stuck
        )
        
        # ---
        # PREREQUISITE 2: Your main script must print a machine-readable final line
        # upon completion. For example: "FINAL_RESULT: WINS=4,STEPS=150"
        # ---
        for line in result.stdout.strip().split('\n'):
            if line.startswith("FINAL_RESULT:"):
                parts = line.replace("FINAL_RESULT: ", "").split(',')
                wins = int(parts[0].split('=')[1])
                steps = int(parts[1].split('=')[1])
                
                # This is our objective score: reward wins, penalize steps.
                # A higher score is better.
                score = (wins * 1000) - steps
                print(f"--- Trial Complete. Wins: {wins}, Steps: {steps}. Objective Score: {score} ---")
                return score

        print("--- Trial Warning: FINAL_RESULT line not found in agent output. ---")
        return -float('inf') # Return a very bad score if the result line isn't found

    except subprocess.CalledProcessError as e:
        print(f"--- Trial FAILED with an error. ---")
        print(e.stderr) # Print the error for debugging
        return -float('inf') # Return a very bad score for failed/crashed runs
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
        'discount_factor': trial.suggest_float('discount_factor', 0.85, 0.99),

        # Reward & Penalty Magnitudes
        'reward_win': trial.suggest_float('reward_win', 100.0, 500.0),
        'reward_novelty_multiplier': trial.suggest_float('reward_novelty_multiplier', 5.0, 50.0),
        'reward_new_effect_pattern': trial.suggest_float('reward_new_effect_pattern', 10.0, 50.0),
        'penalty_unexpected_failure': trial.suggest_float('penalty_unexpected_failure', 10.0, 50.0),
        'penalty_repeated_effect': trial.suggest_float('penalty_repeated_effect', 1.0, 20.0),
        'penalty_boring_move': trial.suggest_float('penalty_boring_move', 20.0, 200.0),
        'penalty_predicted_failure': trial.suggest_float('penalty_predicted_failure', 50.0, 1000.0),
        'penalty_blacklist_base': trial.suggest_float('penalty_blacklist_base', 1000.0, 10000.0),
        'penalty_blacklist_scaler': trial.suggest_float('penalty_blacklist_scaler', 50.0, 200.0),
        'drought_increment': trial.suggest_float('drought_increment', 5.0, 25.0),

        # Exploration & Goal Bonuses
        'bonus_action_exp': trial.suggest_float('bonus_action_exp', 10.0, 50.0),
        'bonus_state_exp_unknown': trial.suggest_float('bonus_state_exp_unknown', 25.0, 150.0),
        'bonus_state_exp_known_scaler': trial.suggest_float('bonus_state_exp_known_scaler', 10.0, 100.0),
        'bonus_goal_seeking': trial.suggest_float('bonus_goal_seeking', 50.0, 200.0),

        # Heuristic Weights & Thresholds
        'weight_novelty_ratio': trial.suggest_float('weight_novel_ratio', 2.0, 20.0),
        'planning_confidence_threshold': trial.suggest_float('planning_confidence_threshold', 0.7, 0.99),
        'recent_effect_patterns_maxlen': trial.suggest_int('recent_effect_patterns_maxlen', 10, 50),
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
        study_name='obrl_agi3_tuning',
        storage='sqlite:///tuning_study.db', # This file will be created
        load_if_exists=True
    )

    # Start the optimization. Optuna will call the 'objective' function N times.
    # 'n_jobs=-1' tells Optuna to run trials in parallel using all available CPU cores.
    try:
        study.optimize(objective, n_trials=100, n_jobs=-1)
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