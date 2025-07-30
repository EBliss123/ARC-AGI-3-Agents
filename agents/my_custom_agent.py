import numpy as np
from collections import defaultdict
from .agent import Agent

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
        """Initializes the agent and its components."""
        super().__init__(*args, **kwargs)

        self.turn_count = 0
        self.phase = "discovery" # Phases: discovery, mapping, rule_testing, solving

        # Initialize helper components based on the architecture
        self.observer = GameObserver()
        self.controller = ActionController()
        self.mapper = WorldMapper()
        self.engine = RuleEngine()

        # Agent's memory
        self.initial_objects = None
        self.last_grid = None

    def choose_action(self, obs, reward, done, info):
        """The main method called by the game on each turn."""
        grid = obs['grid']

        # --- Phase 1: Initial Observation ---
        if self.turn_count == 0:
            print("--- Turn 0: Performing initial scan. ---")
            # The first step is to take note of every object in the starting grid.
            self.initial_objects = self.observer.analyze_grid(grid)
            print(f"Initial objects found: {self.initial_objects}")
            self.last_grid = np.copy(grid)

        self.turn_count += 1

        # For now, we will take no action and just observe.
        # Action 0 is a no-op action.
        return 0
    
    def is_done(self, obs, done) -> bool:
        return done