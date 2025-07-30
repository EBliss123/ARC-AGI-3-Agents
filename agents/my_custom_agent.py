import numpy as np
from collections import defaultdict
from .agent import Agent
import logging

class GameObserver:
    """Analyzes the game grid to identify objects and changes."""
    def analyze_grid(self, grid):
        """Scans the grid to find all objects and their properties."""
        objects = defaultdict(list)
        for r_idx, row in enumerate(grid):
            for c_idx, cell in enumerate(row):
                # Using a tuple of the cell value as a key
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
        """Initializes the agent's long-term components."""
        logging.warning("--- MyCustomAgent.__init__ CALLED ---")
        super().__init__(*args, **kwargs)
        # These components are created once and persist across games.
        self.observer = GameObserver()
        self.controller = ActionController()
        self.mapper = WorldMapper()
        self.engine = RuleEngine()

        # --- NEW DEBUGGING CODE ---
        logging.warning("--- AGENT BASE CLASS INSPECTION ---")
        try:
            # Get the base class (should be 'Agent')
            base_class = MyCustomAgent.__bases__[0]
            logging.warning(f"Inspecting base class: {base_class.__name__}")

            # List all non-private attributes and methods
            for attr_name in dir(base_class):
                if not attr_name.startswith('__'):
                    attr = getattr(base_class, attr_name)
                    if callable(attr):
                        logging.warning(f"  - Found Method: {attr_name}")
                    else:
                        logging.warning(f"  - Found Property: {attr_name}")
            logging.warning("--- END OF BASE CLASS INSPECTION ---")
        except Exception as e:
            logging.error(f"Error during agent inspection: {e}")

    def reset(self, obs, info):
        """Resets the agent's state for a new game episode."""
        logging.warning("--- MyCustomAgent.reset CALLED ---")
        logging.info("--- Agent Resetting for New Game ---")

        # These variables are reset at the start of each game.
        self.turn_count = 0
        self.phase = "discovery"
        self.last_grid = np.copy(obs['grid'])

        # Action discovery state
        self.actions_to_test = [1, 2, 3, 4, 5]
        self.discovered_actions = []
        self.last_action_taken = 0

    def choose_action(self, obs, reward, done, info):
        """The main method called by the game on each turn."""
        logging.warning("--- MyCustomAgent.choose_action CALLED ---")
        grid = obs['grid']

        # --- Step 1: Analyze the result of the LAST turn ---
        if not np.array_equal(grid, self.last_grid):
            # The grid has changed, so the last action we took was effective.
            logging.info(f"--- Change detected! Action {self.last_action_taken} is effective. ---")
            if self.last_action_taken not in self.discovered_actions:
                self.discovered_actions.append(self.last_action_taken)

        # --- Step 2: Decide the CURRENT action based on our phase ---
        action_to_take = 0  # Default to no-op
        if self.phase == "discovery":
            if self.actions_to_test:
                # If we have actions left to test, pop the next one.
                action_to_take = self.actions_to_test.pop(0)
                logging.info(f"--- Turn {self.turn_count}: Testing action {action_to_take} ---")
            else:
                # No more actions to test, discovery is over.
                logging.info("--- Simple action discovery complete. Discovered actions: "
                            f"{self.discovered_actions} ---")
                self.phase = "mapping"

        # --- Step 3: Update state for the next turn ---
        self.last_grid = np.copy(grid)
        self.last_action_taken = action_to_take
        self.turn_count += 1
        return action_to_take

    def is_done(self, obs, done) -> bool:
        logging.warning("--- MyCustomAgent.is_done CALLED ---")
        return done