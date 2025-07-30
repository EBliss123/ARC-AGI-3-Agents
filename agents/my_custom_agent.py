import numpy as np
import logging
from collections import defaultdict

# Imports from the game framework's files
from .agent import Agent
from .structs import FrameData, GameAction, GameState

class GameObserver:
    """Analyzes the game grid to identify objects and changes."""
    def analyze_grid(self, grid):
        """Scans the grid to find all objects and their properties."""
        objects = defaultdict(list)
        for r_idx, row in enumerate(grid):
            for c_idx, cell in enumerate(row):
                color_key = tuple(cell)
                objects[color_key].append((r_idx, c_idx))
        return dict(objects)

class ActionController:
    """Manages the agent's actions and decision-making."""
    pass

class WorldMapper:
    """Builds and maintains an internal map of the game world."""
    pass

class RuleEngine:
    """Formulates and tests hypotheses about game rules."""
    pass

class MyCustomAgent(Agent):
    def __init__(self, *args, **kwargs):
        """Initializes the agent."""
        super().__init__(*args, **kwargs)
        
        # Long-term components
        self.observer = GameObserver()
        self.controller = ActionController()
        self.mapper = WorldMapper()
        self.engine = RuleEngine()

        # Episode-specific state
        self.turn_count = 0
        self.phase = "discovery"
        self.last_grid = np.array([])
        self.actions_to_test = [1, 2, 3, 4, 5]
        self.discovered_actions = []
        self.last_action_taken = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        # The framework does a pre-check. We must return False to start the game.
        if self.action_counter == 0:
            return False

        current_state = latest_frame.state

        # The game is finished if the state is WIN or GAME_OVER.
        return current_state == GameState.WIN or current_state == GameState.GAME_OVER

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose which action the Agent should take."""
        grid = latest_frame.frame

        # --- Analyze the result of the LAST turn ---
        if self.action_counter > 0 and not np.array_equal(grid, self.last_grid):
            logging.info(f"--- Change detected! Action {self.last_action_taken} is effective. ---")
            if self.last_action_taken not in self.discovered_actions:
                self.discovered_actions.append(self.last_action_taken)

        # --- Decide the CURRENT action's number ---
        action_num = 0  # Default to no-op
        if self.phase == "discovery":
            if self.actions_to_test:
                action_num = self.actions_to_test.pop(0)
                logging.info(f"--- Turn {self.action_counter}: Testing action {action_num} ---")
            else:
                logging.info(f"--- Simple action discovery complete. Discovered actions: {self.discovered_actions} ---")
                self.phase = "mapping"

        # --- Create the action object using the official GameAction class ---
        action_obj = GameAction.from_id(action_num)
        action_obj.reasoning = f"Phase: {self.phase}. Taking action {action_num}."
        
        # --- Update state for the next turn ---
        self.last_grid = np.copy(grid)
        self.last_action_taken = action_num
        
        return action_obj