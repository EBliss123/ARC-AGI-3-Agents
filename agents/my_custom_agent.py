import random
import time
import numpy as np
from typing import Any
from enum import Enum, auto

from .agent import Agent
from .structs import FrameData, GameAction, GameState

class AgentState(Enum):
    """Defines the different states the agent can be in."""
    ANALYZING_ACTIONS = auto()
    IDENTIFYING_GAME_TYPE = auto()
    EXPLORING_MAP = auto()
    SOLVING = auto()

class MyCustomAgent(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        self._reset_memory()

    def _reset_memory(self) -> None:
        """Resets all the agent's memory attributes to their initial state."""
        # State Management
        self.agent_state = AgentState.ANALYZING_ACTIONS

        # Memory for Action Analysis
        self.actions_to_test = [a for a in GameAction if a is not GameAction.RESET]
        self.action_results = {}
        self.effective_actions = []

        # Memory for Action Execution
        self.grid_before_action = None
        self.last_action_tested = None
        self.score_before_action = None

        # Memory for Game Understanding
        self.game_type = "unknown" # 'avatar' or 'static'
        self.avatar_info = {}
        self.action_effects = {} # e.g. {'UP': {'dy': -1, 'dx': 0}}
        self.wall_colors = set()
        self.floor_colors = set()
        self.avatar_test_actions = []

    @property
    def name(self) -> str:
        return "MyCustomAgent"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """The main brain of the agent, using a state-driven approach."""
        # --- Preamble: Handle Game Reset & Prepare Action ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._reset_memory()
            return GameAction.RESET

        def prep_and_return(action, reasoning=""):
            action.reasoning = f"State: {self.agent_state.name} - {reasoning}"
            self.grid_before_action = np.array(latest_frame.frame)
            self.score_before_action = latest_frame.score
            self.last_action_tested = action
            return action

        # --- Universal: Analyze the result of the previous action ---
        if self.grid_before_action is not None and self.last_action_tested:
            did_anything_change = not np.array_equal(self.grid_before_action, np.array(latest_frame.frame)) or self.score_before_action != latest_frame.score
            self.action_results[self.last_action_tested.name] = did_anything_change

        # --- State-Driven Action Selection ---

        # 1. ANALYZING_ACTIONS: Test every action to see if it has an effect.
        if self.agent_state is AgentState.ANALYZING_ACTIONS:
            if self.actions_to_test:
                action = self.actions_to_test.pop(0)
                return prep_and_return(action, f"Testing action '{action.name}'.")
            else:
                # Analysis complete, let's see what we learned
                self.effective_actions = [GameAction[name] for name, changed in self.action_results.items() if changed]
                print(f"Effective actions found: {[a.name for a in self.effective_actions]}")

                # Transition to the next state
                self.agent_state = AgentState.IDENTIFYING_GAME_TYPE
                # Fall-through to the next state in the same frame

        # 2. IDENTIFYING_GAME_TYPE: Figure out if we control an avatar.
        if self.agent_state is AgentState.IDENTIFYING_GAME_TYPE:
            # On the first run of this state, populate the list of actions to test.
            if not self.avatar_test_actions:
                simple_moves = [a for a in self.effective_actions if a.is_simple()]
                if not simple_moves:
                    # No effective simple moves, must be a static game.
                    self.game_type = "static"
                    print("No effective simple moves. Classifying as a static game.")
                    return prep_and_return(GameAction.RESET, "Static game detected.")
                self.avatar_test_actions = simple_moves

            # Analyze the result of the LAST action, if there was one.
            if self.last_action_tested and self.last_action_tested in self.effective_actions:
                grid_before = self.grid_before_action
                grid_after = np.array(latest_frame.frame)
                before_2d = grid_before.sum(axis=2)
                after_2d = grid_after.sum(axis=2)

                disappeared_mask = (before_2d > 0) & (after_2d == 0)
                appeared_mask = (before_2d == 0) & (after_2d > 0)

                # Heuristic for finding an avatar: a small, equal number of pixels disappeared and appeared.
                is_avatar_move = (np.sum(disappeared_mask) == np.sum(appeared_mask)) and (0 < np.sum(appeared_mask) < 25)

                if is_avatar_move:
                    print("Avatar identified!")
                    self.game_type = "avatar"
                    pos_before = np.argwhere(disappeared_mask).mean(axis=0)
                    pos_after = np.argwhere(appeared_mask).mean(axis=0)
                    dy, dx = round(pos_after[0] - pos_before[0]), round(pos_after[1] - pos_before[1])

                    self.action_effects[self.last_action_tested.name] = {"dy": dy, "dx": dx}
                    self.avatar_info['position'] = tuple(map(int, pos_after))

                    # For now, let's stop here. We'll implement mapping next.
                    return prep_and_return(GameAction.RESET, f"Avatar found at {self.avatar_info['position']}.")

            # If we are here, it means the last action didn't find an avatar. Let's try the next one.
            if self.avatar_test_actions:
                action_to_test = self.avatar_test_actions.pop(0)
                return prep_and_return(action_to_test, f"Testing for avatar with {action_to_test.name}.")

            # If we've run out of actions to test and haven't found an avatar, it must be a static game.
            else:
                self.game_type = "static"
                print("Tried all simple moves; no avatar found. Classifying as a static game.")
                return prep_and_return(GameAction.RESET, "Static game detected.")

        # 3. EXPLORING_MAP: Systematically explore the world.
        if self.agent_state is AgentState.EXPLORING_MAP:
            return prep_and_return(GameAction.RESET, "Logic not implemented.")

        # 4. SOLVING: Use knowledge to solve the puzzle.
        if self.agent_state is AgentState.SOLVING:
            return prep_and_return(GameAction.RESET, "Logic not implemented.")

        # Fallback action
        return prep_and_return(GameAction.RESET, "Reached end of logic.")