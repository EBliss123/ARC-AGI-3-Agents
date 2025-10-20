# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import os
import signal
import sys
import threading
from functools import partial
from types import FrameType
from typing import Optional
from agents.my_custom_agent import MyCustomAgent

import requests

from agents import AVAILABLE_AGENTS, Swarm
from agents.tracing import initialize as init_agentops
from agents.my_custom_agent import MyCustomAgent 

logger = logging.getLogger()

SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)

# Hide standard ports in URL
if (SCHEME == "http" and str(PORT) == "80") or (
    SCHEME == "https" and str(PORT) == "443"
):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"
HEADERS = {
    "X-API-Key": os.getenv("ARC_API_KEY", ""),
    "Accept": "application/json",
}


def run_agent(swarm: Swarm) -> None:
    swarm.main()
    os.kill(os.getpid(), signal.SIGINT)


def cleanup(
    swarm: Swarm,
    signum: Optional[int],
    frame: Optional[FrameType],
) -> None:
    logger.info("Received SIGINT, exiting...")
    card_id = swarm.card_id
    if card_id:
        scorecard = swarm.close_scorecard(card_id)
        if scorecard:
            logger.info("--- EXISTING SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
            swarm.cleanup(scorecard)

        # Provide web link to scorecard
        if card_id:
            scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
            logger.info(f"View your scorecard online: {scorecard_url}")

    sys.exit(0)


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("***%(asctime)s | %(levelname)s | %(message)s***")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("logs.log", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # logging.getLogger("requests").setLevel(logging.CRITICAL)
    # logging.getLogger("werkzeug").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="ARC-AGI-3-Agents")
    parser.add_argument(
        "-a",
        "--agent",
        choices=AVAILABLE_AGENTS.keys(),
        help="Choose which agent to run.",
    )
    parser.add_argument(
        "-g",
        "--game",
        help="Choose a specific game_id for the agent to play. If none specified, an agent swarm will play all available games.",
    )
    parser.add_argument(
        "-t",
        "--tags",
        type=str,
        help="Comma-separated list of tags for the scorecard (e.g., 'experiment,v1.0')",
        default=None,
    )

    # --- Hyperparameters for the Optimizer ---
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--discount_factor', type=float)
    parser.add_argument('--reward_win', type=float)
    parser.add_argument('--reward_novelty_multiplier', type=float)
    parser.add_argument('--reward_new_effect_pattern', type=float)
    parser.add_argument('--penalty_unexpected_failure', type=float)
    parser.add_argument('--penalty_repeated_effect', type=float)
    parser.add_argument('--penalty_boring_move', type=float)
    parser.add_argument('--penalty_predicted_failure', type=float)
    parser.add_argument('--penalty_blacklist_base', type=float)
    parser.add_argument('--penalty_blacklist_scaler', type=float)
    parser.add_argument('--drought_increment', type=float)
    parser.add_argument('--bonus_action_exp', type=float)
    parser.add_argument('--bonus_state_exp_unknown', type=float)
    parser.add_argument('--bonus_state_exp_known_scaler', type=float)
    parser.add_argument('--bonus_goal_seeking', type=float)
    parser.add_argument('--weight_novelty_ratio', type=float)
    parser.add_argument('--planning_confidence_threshold', type=float)
    parser.add_argument('--recent_effect_patterns_maxlen', type=int)

    args = parser.parse_args()

    if not args.agent:
        logger.error("An Agent must be specified")
        return

    print(f"{ROOT_URL}/api/games")

    # Get the list of games from the API
    full_games = []
    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            r = session.get(f"{ROOT_URL}/api/games", timeout=10)

        if r.status_code == 200:
            try:
                full_games = [g["game_id"] for g in r.json()]
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse games response: {e}")
                logger.error(f"Response content: {r.text[:200]}")
        else:
            logger.error(
                f"API request failed with status {r.status_code}: {r.text[:200]}"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to API server: {e}")

    # For playback agents, we can derive the game from the recording filename
    if not full_games and args.agent and args.agent.endswith(".recording.jsonl"):
        from agents.recorder import Recorder

        game_prefix = Recorder.get_prefix_one(args.agent)
        full_games = [game_prefix]
        logger.info(
            f"Using game '{game_prefix}' derived from playback recording filename"
        )
    games = full_games[:]
    if args.game:
        filters = args.game.split(",")
        games = [
            gid
            for gid in full_games
            if any(gid.startswith(prefix) for prefix in filters)
        ]

    logger.info(f"Game list: {games}")

    if not games:
        if full_games:
            logger.error(
                f"The specified game '{args.game}' does not exist or is not available with your API key. Please try a different game."
            )
        else:
            logger.error(
                "No games available to play. Check API connection or recording file."
            )
        return

    # Start with Empty tags, "agent" and agent name will be added by the Swarm later
    tags = []

    # Append user-provided tags if any
    if args.tags:
        user_tags = [tag.strip() for tag in args.tags.split(",")]
        tags.extend(user_tags)

    # Initialize AgentOps client
    init_agentops(api_key=os.getenv("AGENTOPS_API_KEY"), log_level=log_level)

    # --- Collect Hyperparameters for the Agent ---
    # Create a dictionary of only the hyperparameters that were passed via command line
    hyperparams_to_pass = {
        'learning_rate': args.learning_rate,
        'discount_factor': args.discount_factor,
        'reward_win': args.reward_win,
        'reward_novelty_multiplier': args.reward_novelty_multiplier,
        'reward_new_effect_pattern': args.reward_new_effect_pattern,
        'penalty_unexpected_failure': args.penalty_unexpected_failure,
        'penalty_repeated_effect': args.penalty_repeated_effect,
        'penalty_boring_move': args.penalty_boring_move,
        'penalty_predicted_failure': args.penalty_predicted_failure,
        'penalty_blacklist_base': args.penalty_blacklist_base,
        'penalty_blacklist_scaler': args.penalty_blacklist_scaler,
        'drought_increment': args.drought_increment,
        'bonus_action_exp': args.bonus_action_exp,
        'bonus_state_exp_unknown': args.bonus_state_exp_unknown,
        'bonus_state_exp_known_scaler': args.bonus_state_exp_known_scaler,
        'bonus_goal_seeking': args.bonus_goal_seeking,
        'weight_novelty_ratio': args.weight_novelty_ratio,
        'planning_confidence_threshold': args.planning_confidence_threshold,
        'recent_effect_patterns_maxlen': args.recent_effect_patterns_maxlen,
    }
    # Filter out any that were not set, so we don't pass `None`
    cli_params = {key: value for key, value in hyperparams_to_pass.items() if value is not None}

    swarm = Swarm(
        args.agent,
        ROOT_URL,
        games,
        tags=tags,
        params=cli_params,  # Pass the collected hyperparameters
    )
    agent_thread = threading.Thread(target=partial(run_agent, swarm))
    agent_thread.daemon = True  # die when the main thread dies
    agent_thread.start()

    signal.signal(signal.SIGINT, partial(cleanup, swarm))  # handler for Ctrl+C

    try:
        # Wait for the agent thread to complete
        while agent_thread.is_alive():
            agent_thread.join(timeout=5)  # Check every 5 second
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main thread")
        cleanup(swarm, signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Unexpected error in main thread: {e}")
        cleanup(swarm, None, None)
if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()
