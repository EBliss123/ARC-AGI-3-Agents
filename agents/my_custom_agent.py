import numpy as np
import logging
from collections import defaultdict

# Imports from the game framework's files
from .agent import Agent
from .structs import FrameData, GameAction, GameState, ComplexAction

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
        # ... (other components can be added here later) ...

        # Episode-specific state
        self.phase = "discovery"
        self.last_grid = np.array([])
        self.last_action_taken = 0
        
        # State for discovery phases
        self.actions_to_test = [1, 2, 3, 4, 5]
        self.discovered_actions = []
        self.initial_objects = None
        self.click_targets = []


    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        if self.action_counter == 0:
            return False
        
        current_state = latest_frame.state
        return current_state == GameState.WIN or current_state == GameState.GAME_OVER

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose which action the Agent should take."""

        # --- On the very first turn, RESET to get the GUID and initial state ---
        if self.action_counter == 0:
            logging.info("--- Turn 0: Resetting the environment. ---")
            self.last_grid = np.copy(latest_frame.frame)
            # Analyze the initial grid to find all objects to click later
            self.initial_objects = self.observer.analyze_grid(latest_frame.frame)
            self.last_action_taken = 0
            return GameAction.from_id(0)

        grid = latest_frame.frame

        # --- Analyze the result of the LAST turn ---
        if not np.array_equal(grid, self.last_grid):
            logging.info(f"--- Change detected! Action {self.last_action_taken} was effective. ---")
            # We can build more complex rule logic here later
            if self.last_action_taken not in self.discovered_actions:
                self.discovered_actions.append(self.last_action_taken)

        action_num = 0  # Default to no-op
        action_data = None

        # --- Agent Logic: Decide action based on current phase ---
        if self.phase == "discovery":
            if self.actions_to_test:
                action_num = self.actions_to_test.pop(0)
                logging.info(f"--- Phase 1: Testing simple action {action_num} ---")
            else:
                logging.info("--- Phase 1 Complete. Moving to Click Discovery. ---")
                self.phase = "click_discovery"

        if self.phase == "click_discovery":
            # First time in this phase? Prepare the list of click targets.
            if not self.click_targets:
                all_coords = []
                for color, coords_list in self.initial_objects.items():
                    all_coords.extend(coords_list)
                self.click_targets = all_coords
                logging.info(f"--- Phase 2: Prepared {len(self.click_targets)} coordinates to click. ---")

            if self.click_targets:
                action_num = 6 # Click Action
                y, x = self.click_targets.pop(0)
                action_data = {'x': x, 'y': y}
                logging.info(f"--- Phase 2: Testing click at (x={x}, y={y}) ---")
            else:
                logging.info("--- Phase 2 Complete. All objects clicked. Moving to Mapping. ---")
                self.phase = "mapping"

        # --- Create and return the final action object ---
        action_obj = GameAction.from_id(action_num)
        if action_data:
            action_obj.set_data(action_data)
        action_obj.reasoning = f"Phase: {self.phase}. Action: {action_num}."
        
        self.last_grid = np.copy(grid)
        self.last_action_taken = action_num
        
        return action_obj