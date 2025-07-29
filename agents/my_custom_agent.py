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
        # A list of simple actions we want to test.
        self.actions_to_test = [
            GameAction.UP,
            GameAction.DOWN,
            GameAction.LEFT,
            GameAction.RIGHT,
        ]
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
            # If the game is over, we'll want to reset the exploration state too,
            # but we'll handle that logic later. For now, just reset the game.
            return GameAction.RESET

        # --- Phase 1: Check the result of our LAST action ---
        # If we have a grid saved from the last turn, let's see if our action changed it.
        if self.grid_before_action is not None and self.last_action_tested is not None:
            new_grid = np.array(latest_frame.grid)
            # np.array_equal checks if two arrays are identical.
            did_grid_change = not np.array_equal(self.grid_before_action, new_grid)
            # Store the result in our memory.
            self.action_results[self.last_action_tested.name] = did_grid_change
            # Clear the stored grid so we don't re-check this next turn.
            self.grid_before_action = None

        # --- Phase 2: Decide Mode (Explore or Solve) ---
        if len(self.actions_to_test) > 0:
            # --- EXPLORATION MODE ---
            # If we still have actions to test, let's test one.
            action = self.actions_to_test.pop(0)  # Get the next action from the front of the list.

            # Store the current grid and the action we're about to take.
            self.grid_before_action = np.array(latest_frame.grid)
            self.last_action_tested = action
            action.reasoning = f"Exploring: Testing action '{action.name}'."
            return action
        else:
            # --- SOLVING MODE ---
            # Exploration is complete! We can now use our knowledge.
            # For now, we'll just use our old "find and click" logic.
            # In the future, we could use the self.action_results data here.

            grid = np.array(latest_frame.grid)
            object_coordinates = np.argwhere(grid > 0)

            if len(object_coordinates) > 0:
                y, x = object_coordinates[0]
                action = GameAction.CLICK
                action.set_data({"x": int(x), "y": int(y)})
                action.reasoning = f"Solving: Exploration done. Found object at ({x}, {y})."
            else:
                action = GameAction.UP
                action.reasoning = "Solving: Exploration done. Grid is empty, moving UP."

            return action