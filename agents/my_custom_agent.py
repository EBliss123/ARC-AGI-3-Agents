import random
import time
import numpy as np
from typing import Any

from .agent import Agent
from .structs import FrameData, GameAction, GameState


class MyCustomAgent(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

        # --- Agent Memory ---
        # Phase 1: Action Exploration
        self.actions_to_test = [a for a in GameAction if a is not GameAction.RESET]
        self.action_results = {}
        self.grid_before_action = None
        self.last_action_tested = None

        # Phase 2: Discovery
        self.game_type = "unknown"  # Can be 'unknown', 'avatar', or 'static'
        self.avatar_info = {}       # To store avatar color, size, etc. if found

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
        """Determines game type, then chooses an action."""
        # --- Phase 0: Handle Game Reset ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        # --- Universal: Analyze the result of the previous action ---
        if self.grid_before_action is not None and self.last_action_tested is not None:
            grid_after = np.array(latest_frame.frame)
            grid_before = self.grid_before_action

            # 1. Record if the action had an effect
            did_grid_change = not np.array_equal(grid_before, grid_after)
            self.action_results[self.last_action_tested.name] = did_grid_change

            # 2. If in Discovery Phase, analyze the type of change
            if self.game_type == "unknown" and self.last_action_tested.is_simple() and did_grid_change:
                # Convert to 2D to find changed pixels
                before_2d = grid_before.sum(axis=2)
                after_2d = grid_after.sum(axis=2)

                # Find where pixels appeared and disappeared
                disappeared_mask = (before_2d > 0) & (after_2d == 0)
                appeared_mask = (before_2d == 0) & (after_2d > 0)
                
                # Heuristic for finding an avatar: a small, equal number of pixels moved.
                if np.sum(disappeared_mask) == np.sum(appeared_mask) and 0 < np.sum(appeared_mask) < 25:
                    self.game_type = "avatar"
                    # We would add more logic here to store avatar color, shape, etc.
                else:
                    self.game_type = "static"

            self.grid_before_action = None # Clear the stored grid

        # --- Action Selection based on current phase ---

        # Phase 1: Still exploring actions?
        if len(self.actions_to_test) > 0:
            action = self.actions_to_test.pop(0)
            self.grid_before_action = np.array(latest_frame.frame)
            self.last_action_tested = action
            action.reasoning = f"Exploring: Testing action '{action.name}'."
            return action

        # Phase 2: Ready to discover game type?
        elif self.game_type == "unknown":
            effective_simple = [GameAction[name] for name, changed in self.action_results.items() if changed and GameAction[name].is_simple()]
            if effective_simple:
                # Trigger the discovery by taking the first simple, effective action
                action = effective_simple[0]
                self.grid_before_action = np.array(latest_frame.frame)
                self.last_action_tested = action
                action.reasoning = "Discovery: Triggering analysis with a simple move."
                return action
            else:
                # No simple actions had any effect, must be a static game
                self.game_type = "static"

        # Phase 3: Solving Mode (now informed by game type)
        if self.game_type == "avatar":
            action = GameAction.RESET # Placeholder for future avatar logic
            action.reasoning = "Solving: Game type is AVATAR. (Logic not implemented yet)."
            return action
        
        elif self.game_type == "static":
            # Fallback to our old find-and-click logic for static games
            grid_2d = np.array(latest_frame.frame).sum(axis=2)
            object_coords = np.argwhere(grid_2d > 0)
            if len(object_coords) > 0:
                y, x = object_coords[0]
                complex_actions = [GameAction[name] for name, changed in self.action_results.items() if changed and GameAction[name].is_complex()]
                if complex_actions:
                    action = complex_actions[0]
                    action.set_data({"x": int(x), "y": int(y)})
                    action.reasoning = f"Solving: STATIC game. Trying complex action on object at ({x},{y})."
                else:
                    action = GameAction.RESET
                    action.reasoning = "Solving: STATIC game but no effective complex actions found."
            else:
                action = GameAction.RESET
                action.reasoning = "Solving: STATIC game but grid is empty."
            return action
            
        return GameAction.RESET # Default fallback