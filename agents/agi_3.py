# ARC-AGI-3 Main Script
import random
from .agent import Agent
from .structs import FrameData, GameAction, GameState

# --- Game Environment Classes ---
# These classes will hold the state and logic for each specific game.

class LS20_Game:
    """Environment for the LS20 game."""
    pass

class FT09_Game:
    """Environment for the FT09 game."""
    pass

class VC33_Game:
    """Environment for the VC33 game."""
    pass

# --- Core AGI Logic ---

class AGI3(Agent):
    """The general agent that learns to play the games."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"Custom AGI initialized for game: {self.game_id}")
        
        # --- New variables for discovery and learning ---
        self.learned_rules = {}
        self.last_action_taken = None
        self.previous_grid = []
        
        # A list of simple actions to try during the discovery phase
        self.actions_to_try = [
            GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5
        ]
        self.simple_actions_produced_change = False

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        # The agent stops this attempt if it wins the level.
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        current_grid = latest_frame.frame
        if not current_grid:
            return GameAction.RESET

        # 1. Detect changes and see if the last action worked
        if self.detect_pixel_changes(current_grid):
            self.simple_actions_produced_change = True

        # 2. --- Decision Making ---
        action = GameAction.RESET
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        elif self.actions_to_try: # Still have simple actions to test
            action = self.actions_to_try.pop(0)
            print(f"Discovery Mode: Trying simple action {action.name}")
        elif not self.simple_actions_produced_change: # Simple actions failed, try ACTION6
            print("Discovery Mode: Simple actions had no effect. Trying ACTION6.")
            action = GameAction.ACTION6
            # We need to give it random coordinates to click
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            # We'll set the flag to True to prevent trying ACTION6 again
            self.simple_actions_produced_change = True 
        else: # Discovery is complete
            print("Discovery complete. Acting randomly.")
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        # 3. Remember state for next turn
        self.previous_grid = current_grid
        self.last_action_taken = action
        return action
        
    def detect_pixel_changes(self, current_grid):
        if not self.previous_grid or self.last_action_taken is None:
            return False

        if len(self.previous_grid) != len(current_grid):
            return False

        changes = []
        # ... (the for loops for checking pixels remain the same) ...
        for y in range(len(current_grid)):
            for x in range(len(current_grid[0])):
                if self.previous_grid[y][x] != current_grid[y][x]:
                    # ... (the change detection logic is the same) ...
                    changes.append(f"Pixel at ({x},{y}) changed")

        if changes:
            print(f"--- [{self.game_id}] Change Detected After {self.last_action_taken.name} ---")
            # We'll just print a summary for readability
            print(f"  - {len(changes)} pixels changed.")
            print("-------------------------------------------------")
            return True # Return True because changes were found

        return False # Return False if no changes were found

    # --- Methods from your original plan ---

    def perceive(self, latest_frame: FrameData):
        """Receives raw game data and creates a structured model."""
        # Process latest_frame.grid here.
        pass

    def discover_actions(self):
        """Tries actions and logs the changes they cause."""
        pass

    def synthesize_rules(self):
        """Creates object-based rules from observed actions."""
        pass
    
    def explore(self):
        """Uses curiosity to explore new game states."""
        pass