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
        self.score_before_action = None

        # Phase 2: Discovery
        self.game_type = "unknown"  # Can be 'unknown', 'avatar', or 'static'
        self.avatar_info = {}       # To store avatar color, size, etc. if found
        self.action_effects = {}

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
        """Determines game type, then chooses an action with a 'no reset' policy."""
        # --- Phase 0: Handle Game Reset (This is the only mandatory reset) ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        # --- Universal: Analyze the result of the previous action ---
        if self.grid_before_action is not None and self.last_action_tested is not None:
            grid_after = np.array(latest_frame.frame)
            grid_before = self.grid_before_action
            score_after = latest_frame.score
            score_before = self.score_before_action

            did_grid_change = not np.array_equal(grid_before, grid_after)
            did_score_change = score_after != score_before
            did_anything_change = did_grid_change or did_score_change
            
            self.action_results[self.last_action_tested.name] = did_anything_change

            if self.game_type == "unknown" and self.last_action_tested.is_simple() and did_grid_change:
                before_2d = grid_before.sum(axis=2)
                after_2d = grid_after.sum(axis=2)
                disappeared_mask = (before_2d > 0) & (after_2d == 0)
                appeared_mask = (before_2d == 0) & (after_2d > 0)
                
                if np.sum(disappeared_mask) == np.sum(appeared_mask) and 0 < np.sum(appeared_mask) < 25:
                    self.game_type = "avatar"
                    appeared_coords = np.argwhere(appeared_mask)
                    disappeared_coords = np.argwhere(disappeared_mask)
                    pos_before = disappeared_coords.mean(axis=0)
                    pos_after = appeared_coords.mean(axis=0)
                    dy = round(pos_after[0] - pos_before[0])
                    dx = round(pos_after[1] - pos_before[1])
                    self.action_effects[self.last_action_tested.name] = {"dy": dy, "dx": dx}
                    self.avatar_info['position'] = pos_after
                    self.avatar_info['size'] = len(appeared_coords)
                else:
                    self.game_type = "static"

            self.grid_before_action = None

        # --- Action Selection based on current phase ---

        if len(self.actions_to_test) > 0:
            action = self.actions_to_test.pop(0)
            self.grid_before_action = np.array(latest_frame.frame)
            self.score_before_action = latest_frame.score
            self.last_action_tested = action
            action.reasoning = f"Exploring: Testing action '{action.name}'."
            return action

        elif self.game_type == "unknown":
            effective_simple = [GameAction[name] for name, changed in self.action_results.items() if changed and GameAction[name].is_simple()]
            if effective_simple:
                action = effective_simple[0]
                self.grid_before_action = np.array(latest_frame.frame)
                self.score_before_action = latest_frame.score
                self.last_action_tested = action
                action.reasoning = "Discovery: Triggering analysis with a simple move."
                return action
            else:
                self.game_type = "static"

        # --- Phase 3: Solving Mode ---
        if self.game_type == "avatar":
            known_move_actions = list(self.action_effects.keys())
            if known_move_actions:
                action_name = random.choice(known_move_actions)
                action = GameAction[action_name]
                action.reasoning = f"Solving (Avatar): Randomly exploring with action {action.name}."
            else:
                # This is now the ONLY reset condition during avatar solving
                action = GameAction.RESET
                action.reasoning = "Solving (Avatar): STUCK! No known move actions."
            return action
        
        elif self.game_type == "static":
            effective_simple = [GameAction[name] for name, changed in self.action_results.items() if changed and GameAction[name].is_simple()]
            effective_complex = [GameAction[name] for name, changed in self.action_results.items() if changed and GameAction[name].is_complex()]

            grid_2d = np.array(latest_frame.frame).sum(axis=2)
            object_coords = np.argwhere(grid_2d > 0)

            if len(object_coords) > 0 and effective_complex:
                # Primary strategy: use a complex action on an object
                y, x = object_coords[0]
                action = effective_complex[0]
                action.set_data({"x": int(x), "y": int(y)})
                action.reasoning = f"Solving (Static): Trying complex action on object at ({x},{y})."
            elif effective_simple:
                # Fallback strategy: No object or no complex action, try a simple one
                action = effective_simple[0]
                action.reasoning = "Solving (Static): No object/complex action. Trying a simple effective action."
            else:
                # Last resort: No effective actions of any kind were ever found.
                action = GameAction.RESET
                action.reasoning = "Solving (Static): STUCK! No effective actions known."
            return action
            
        # This fallback should now rarely, if ever, be reached.
        return GameAction.RESET