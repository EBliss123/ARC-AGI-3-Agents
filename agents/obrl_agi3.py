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
        self.actions_printed = False
        self.last_object_summary = []

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        This method is called by the game to get the next action.
        """
        # If the game is over or hasn't started, the correct action is to reset.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.actions_printed = False  # Reset the print flag for the new game.
            return GameAction.RESET
        
        current_summary = self._perceive_objects(latest_frame)

        # If this is the first scan (last summary is empty), print the full summary.
        if not self.last_object_summary:
            print("--- Initial Frame Summary ---")
            if not current_summary:
                print("No objects found.")
            for obj in current_summary:
                obj_id = obj['id'].replace('obj_', 'id_')
                size_str = f"{obj['size'][0]}x{obj['size'][1]}"
                print(
                    f"- Object {obj_id}: Found a {size_str} object of color {obj['color']} "
                    f"at position {obj['position']} with {obj['pixels']} pixels "
                    f"and shape fingerprint {obj['fingerprint']}."
                )
        # On subsequent turns, print only the changes.
        else:
            changes = self._log_changes(self.last_object_summary, current_summary)
            if changes:
                print("--- Change Log ---")
                for change in changes:
                    print(change)

        # Update the memory for the next turn.
        self.last_object_summary = current_summary

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
    
    def _log_changes(self, old_summary: list[dict], new_summary: list[dict]) -> list[str]:
        """Compares summaries by identifying recolors, moves, and new/removed objects."""
        if not old_summary and not new_summary:
            return []

        changes = []
        
        # --- Pass 1: Identify stable and recolored objects ---
        # Key: (position, fingerprint, size) -> Value: object dict
        old_map_by_pos_shape = {(o['position'], o['fingerprint'], o['size']): o for o in old_summary}
        new_map_by_pos_shape = {(n['position'], n['fingerprint'], n['size']): n for n in new_summary}

        explained_old_keys = set()
        explained_new_keys = set()

        for pos_shape_key, old_obj in old_map_by_pos_shape.items():
            if pos_shape_key in new_map_by_pos_shape:
                new_obj = new_map_by_pos_shape[pos_shape_key]
                # Mark both as explained so they aren't processed in later passes
                explained_old_keys.add(pos_shape_key)
                explained_new_keys.add(pos_shape_key)
                
                if old_obj['color'] != new_obj['color']:
                    size_str = f"{old_obj['size'][0]}x{old_obj['size'][1]}"
                    changes.append(
                        f"- RECOLORED: A {size_str} object at {old_obj['position']} "
                        f"changed color from {old_obj['color']} to {new_obj['color']}."
                    )
        
        # --- Pass 2: Identify moved objects from the remaining pool ---
        # A stable ID for moving objects (location-independent)
        def get_stable_id(obj):
            return (obj['fingerprint'], obj['color'], obj['size'], obj['pixels'])
        
        # Create maps of unexplained objects grouped by their stable move ID
        unexplained_old_map = {}
        for key, obj in old_map_by_pos_shape.items():
            if key not in explained_old_keys:
                stable_id = get_stable_id(obj)
                if stable_id not in unexplained_old_map:
                    unexplained_old_map[stable_id] = []
                unexplained_old_map[stable_id].append(obj)

        unexplained_new_map = {}
        for key, obj in new_map_by_pos_shape.items():
            if key not in explained_new_keys:
                stable_id = get_stable_id(obj)
                if stable_id not in unexplained_new_map:
                    unexplained_new_map[stable_id] = []
                unexplained_new_map[stable_id].append(obj)

        # Find common IDs that could have moved
        movable_ids = set(unexplained_old_map.keys()) & set(unexplained_new_map.keys())

        for stable_id in movable_ids:
            old_instances = unexplained_old_map[stable_id]
            new_instances = unexplained_new_map[stable_id]
            
            # Simple matching: pair up old and new instances to log them as moved
            num_matches = min(len(old_instances), len(new_instances))
            for _ in range(num_matches):
                old_inst = old_instances.pop()
                new_inst = new_instances.pop()
                if old_inst['position'] != new_inst['position']:
                    changes.append(f"- MOVED: Object with ID {stable_id} moved from {old_inst['position']} to {new_inst['position']}.")

        # --- Pass 3: Log remaining objects as removed or new ---
        # Anything left in unexplained_old_map was removed
        for stable_id, remaining_objs in unexplained_old_map.items():
            for obj in remaining_objs:
                changes.append(f"- REMOVED: Object with ID {stable_id} at {obj['position']} has disappeared.")

        # Anything left in unexplained_new_map is new
        for stable_id, remaining_objs in unexplained_new_map.items():
            for obj in remaining_objs:
                changes.append(f"- NEW: Object with ID {stable_id} has appeared at {obj['position']}.")

        return changes