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

class GameObject:
    """A blueprint for storing all attributes of a found object."""
    def __init__(self, obj_id, color, pixels):
        self.id = obj_id
        self.color = color
        self.pixels = set(pixels)
        self.size = len(self.pixels)
        
        # Find the top-left corner to define position
        min_x = min(p[0] for p in self.pixels)
        min_y = min(p[1] for p in self.pixels)
        self.position = (min_x, min_y)

    def __repr__(self):
        return f"GameObject(id={self.id}, color={self.color}, size={self.size}, pos={self.position})"

# --- Core AGI Logic ---

class AGI3(Agent):
    """The general agent that learns to play the games."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom agent initializations can go here.
        print(f"Custom AGI initialized for game: {self.game_id}")
        self.objects = []

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
        """Receives raw game data and creates a structured model."""
        # Process latest_frame.grid here.
        pass

    def segment_objects(self, latest_frame: FrameData):
        """Scans the grid to find and define all objects."""
        grid = latest_frame.frame  # Use .frame instead of .grid

        # Add a check to handle cases where the frame is empty
        if not grid:
            self.objects = []
            return

        height = len(grid)
        width = len(grid[0])
        self.objects = []
        visited = set()
        object_id_counter = 0

        for y in range(height):
            for x in range(width):
                if (x, y) in visited or grid[y][x] == 0:
                    continue

                object_pixels = self._flood_fill_search(grid, x, y, visited)

                if object_pixels:
                    color = grid[y][x]
                    new_object = GameObject(object_id_counter, color, object_pixels)
                    self.objects.append(new_object)
                    object_id_counter += 1

        if self.objects:
            print(f"Found {len(self.objects)} objects: {self.objects}")

    def _flood_fill_search(self, grid, start_x, start_y, visited):
        """Performs a search to find all connected pixels of the same color."""
        pixels = []
        target_color = grid[start_y][start_x]
        q = [(start_x, start_y)] # A queue for our search
        visited.add((start_x, start_y))

        while q:
            x, y = q.pop(0)
            pixels.append((x, y))

            # Check all four neighbors (up, down, left, right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                # Ensure the neighbor is within the grid and part of the same object
                if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and \
                (nx, ny) not in visited and grid[ny][nx] == target_color:

                    visited.add((nx, ny))
                    q.append((nx, ny))

        return pixels

    def discover_actions(self):
        """Tries actions and logs the changes they cause."""
        pass

    def synthesize_rules(self):
        """Creates object-based rules from observed actions."""
        pass
    
    def explore(self):
        """Uses curiosity to explore new game states."""
        pass