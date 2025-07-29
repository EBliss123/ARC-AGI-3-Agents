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

        # --- Exploration Memory ---
        # This now programmatically gets all actions except RESET.
        self.actions_to_test = [a for a in GameAction if a is not GameAction.RESET]
        
        # A dictionary to store the results of our tests.
        self.action_results = {}
        # A place to store the grid state before we take an action.
        self.grid_before_action = None
        # A way to remember the last action we tried.
        self.last_action_tested = None

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
        """Choose action via an exploration phase, then a solving phase."""
        # --- Phase 0: Handle Game Reset ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        # --- Phase 1: Check the result of our LAST action ---
        if self.grid_before_action is not None and self.last_action_tested is not None:
            new_grid = np.array(latest_frame.frame) # Change this line
            did_grid_change = not np.array_equal(self.grid_before_action, new_grid)
            self.action_results[self.last_action_tested.name] = did_grid_change
            self.grid_before_action = None

        # --- Phase 2: Decide Mode (Explore or Solve) ---
        if len(self.actions_to_test) > 0:
            # --- EXPLORATION MODE ---
            action = self.actions_to_test.pop(0)

            self.grid_before_action = np.array(latest_frame.frame) # And this line
            self.last_action_tested = action
            action.reasoning = f"Exploring: Testing action '{action.name}'."
            return action
        else:
            # --- SOLVING MODE ---
            # Exploration is complete! We can now use our knowledge.
            grid_3d = np.array(latest_frame.frame)

            # Convert the 3D color grid to a 2D grid.
            # We do this by summing the color channels (the 3rd dimension, axis=2).
            # Black pixels (0,0,0) will have a sum of 0. Colored pixels will have a sum > 0.
            grid_2d = grid_3d.sum(axis=2)
            
            # Now find coordinates on our new 2D grid.
            object_coordinates = np.argwhere(grid_2d > 0)

            if len(object_coordinates) > 0:
                y, x = object_coordinates[0]
                action = GameAction.CLICK
                action.set_data({"x": int(x), "y": int(y)})
                action.reasoning = f"Solving: Exploration done. Found object at ({x}, {y})."
            else:
                # If there are no objects, let's try a default non-CLICK action.
                # We'll use the first action we found during exploration that caused a change.
                effective_actions = [name for name, changed in self.action_results.items() if changed]
                if effective_actions:
                    action = GameAction[effective_actions[0]]
                    action.reasoning = f"Solving: No objects to click. Using first effective action: {action.name}"
                else:
                    # If no actions did anything, we are likely stuck.
                    action = GameAction.RESET
                    action.reasoning = "Solving: No objects and no effective actions found. Resetting."

            return action