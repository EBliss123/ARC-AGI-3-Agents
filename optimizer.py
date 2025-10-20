import optuna
import subprocess
import json
import argparse
import sys

def run_agent_and_get_score(params: dict) -> float:
    """
    Launches the main agent script, captures its final output,
    and returns a single objective score to maximize.
    """
    command = ['python', 'main.py']
    for key, value in params.items():
        command.extend([f'--{key}', str(value)])

    print(f"\n--- Starting Trial with command: {' '.join(command)} ---")

    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=1800
        )

        # Find the final result line in the agent's output
        for line in result.stdout.strip().split('\n'):
            if line.startswith("FINAL_RESULT:"):
                parts = line.replace("FINAL_RESULT: ", "").split(',')
                score = int(parts[0].split('=')[1])
                actions = int(parts[1].split('=')[1])

                # --- NEW OBJECTIVE SCORE CALCULATION ---
                # Prioritize score, then efficiency (actions/score).
                objective_score = 0.0
                if score > 0:
                    efficiency_penalty = actions / score
                    objective_score = (score * 1000) - efficiency_penalty
                else:
                    # If score is 0 or less, just penalize the number of actions.
                    objective_score = -actions

                print(f"--- Trial Complete. Score: {score}, Actions: {actions}. Objective Score: {objective_score:.2f} ---")
                return objective_score

        print("--- Trial Warning: FINAL_RESULT line not found in agent output. ---")
        return -float('inf')

    except subprocess.CalledProcessError as e:
        print(f"--- Trial FAILED with an error. ---")
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