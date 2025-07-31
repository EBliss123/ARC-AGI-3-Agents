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
        # Custom agent initializations can go here.
        self.previous_frame = None
        self.changed_pixels = []
        self.static_pixels = []

        print(f"Custom AGI initialized for game: {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        # The agent stops this attempt if it wins the level.
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """This is the main decision-making method for the AGI."""
        
        # --- AGI LOGIC PIPELINE ---
        # The logic from your plan will be called from here.
        
        # 1. Perception: Create a structured model from the raw grid data.
        self.perceive(latest_frame)

        # 2. Object Segmentation: Identify all objects on the grid.
        self.segment_objects(latest_frame)

        # 3. Action Discovery & Rule Synthesis will be used here.
        # 4. Curiosity-Driven Exploration will guide the action choice.

        # For now, we will return a random action as a placeholder
        # so the agent is runnable.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET
        else:
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])
            return action

    # --- Methods from your original plan ---

    def perceive(self, latest_frame: FrameData):
        """Receives raw game data and creates a structured model.
        Compares the current frame with the previous one to detect changes."""
        current_frame = latest_frame.frame

        # The frame can be empty on the initial NOT_PLAYED state. Do nothing.
        if not current_frame:
            return

        # On the first real frame, we can't do a comparison yet, so just store it.
        if self.previous_frame is None:
            self.previous_frame = tuple(list(row) for row in current_frame)
            return

        # Get dimensions of both frames to check for changes in grid size.
        current_height = len(current_frame)
        current_width = len(current_frame[0])
        prev_height = len(self.previous_frame)
        prev_width = len(self.previous_frame[0])

        # If dimensions have changed, we can't do a direct comparison.
        # We'll assume everything changed and reset for the next frame.
        if current_height != prev_height or current_width != prev_width:
            self.changed_pixels = [list(row) for row in current_frame]
            self.static_pixels = []  # Or an empty grid of the new size
            self.previous_frame = tuple(list(row) for row in current_frame)
            return

        # If we get here, dimensions are the same. Proceed with comparison.
        self.changed_pixels = [[0 for _ in range(current_width)] for _ in range(current_height)]
        self.static_pixels = [[0 for _ in range(current_width)] for _ in range(current_height)]

        for r in range(current_height):
            for c in range(current_width):
                if self.previous_frame[r][c] == current_frame[r][c]:
                    # If the pixel is the same, record it in static_pixels.
                    self.static_pixels[r][c] = current_frame[r][c]
                else:
                    # If the pixel changed, record its new value in changed_pixels.
                    self.changed_pixels[r][c] = current_frame[r][c]

        # Save the current frame for the next frame's comparison.
        self.previous_frame = tuple(list(row) for row in current_frame)

    def segment_objects(self, latest_frame: FrameData):
        """Scans the grid to find and define all objects."""
        # Identify objects from the grid here.
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