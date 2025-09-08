import random
from .agent import Agent, FrameData
from .structs import GameAction, GameState
from collections import deque

class ObrlAgi3Agent(Agent):
    """
    An agent for the ARC-AGI-3 challenge using object-based reinforcement learning.
    """

    def __init__(self, **kwargs):
        """
        The constructor for the agent.
        """
        super().__init__(**kwargs)
        # A flag to make sure we only print the actions once.
        self.actions_printed = False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        This method is called by the game to get the next action.
        """
        # If the game is over or hasn't started, the correct action is to reset.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.actions_printed = False  # Reset the print flag for the new game.
            return GameAction.RESET
        
        # On every turn, perceive the objects on the screen.
        object_summary = self._perceive_objects(latest_frame)
        if object_summary:
            print(f"--- Frame Summary ---")
            for obj in object_summary:
                obj_id = obj['id'].replace('obj_', 'id_')
                size_str = f"{obj['size'][0]}x{obj['size'][1]}"
                print(
                    f"- Object {obj_id}: Found a {size_str} object of color {obj['color']} "
                    f"at position {obj['position']} with {obj['pixels']} pixels "
                    f"and shape fingerprint {obj['fingerprint']}."
                )

        # This is the REAL list of actions for this specific game on this turn.
        game_specific_actions = latest_frame.available_actions

        # If we just discovered the game-specific actions, print them once.
        if game_specific_actions and not self.actions_printed:
            print(f"Discovered game-specific actions: {[action.name for action in game_specific_actions]}")
            self.actions_printed = True

        # If the game-specific list is available, choose from it.
        if game_specific_actions:
            return random.choice(game_specific_actions)

        # FALLBACK: If the specific list is empty (e.g., first turn),
        # choose from the master list of all actions to avoid a crash.
        fallback_options = [a for a in GameAction if a is not GameAction.RESET]
        return random.choice(fallback_options)


    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """
        This method is called by the game to see if the agent thinks it is done.
        """
        return False
    
    def _perceive_objects(self, frame: FrameData) -> list[dict]:
        """Scans the pixel grid to find all contiguous objects."""
        # The pixel grid is a 2D list of color numbers.
        grid = frame.frame[0]
        if not grid:
            return []

        height = len(grid)
        width = len(grid[0])
        visited = set()
        objects = []
        object_id_counter = 0

        for r in range(height):
            for c in range(width):
                if (r, c) in visited:
                    continue

                color = grid[r][c]

                # Start of a new object, begin flood-fill (BFS)
                q = deque([(r, c)])
                visited.add((r, c))
                object_pixels = []
                min_r, max_r, min_c, max_c = r, r, c, c

                while q:
                    row, col = q.popleft()
                    object_pixels.append((row, col))

                    # Update bounding box
                    min_r, max_r = min(min_r, row), max(max_r, row)
                    min_c, max_c = min(min_c, col), max(max_c, col)

                    # Check neighbors (up, down, left, right)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = row + dr, col + dc

                        if (0 <= nr < height and 0 <= nc < width and
                                (nr, nc) not in visited and grid[nr][nc] == color):
                            visited.add((nr, nc))
                            q.append((nr, nc))

                # Flood-fill is complete, describe the object
                object_id_counter += 1
                # Normalize pixel coordinates to be relative to the top-left corner.
                # We use a frozenset to make the shape hashable.
                normalized_pixels = frozenset((r - min_r, c - min_c) for r, c in object_pixels)
                shape_fingerprint = hash(normalized_pixels)

                obj = {
                    'id': f'obj_{object_id_counter}',
                    'color': color,
                    'pixels': len(object_pixels),
                    'position': (min_r, min_c),  # Top-left corner
                    'size': (max_r - min_r + 1, max_c - min_c + 1),
                    'fingerprint': shape_fingerprint
                }
                objects.append(obj)

        return objects