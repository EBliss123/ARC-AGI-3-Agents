import random
from .agent import Agent, FrameData
from .structs import GameAction, GameState
from collections import deque
import copy

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

        # --- Build a list of possible move DESCRIPTIONS ---
        possible_moves = []

        # Find the generic click action.
        click_action_template = next((action for action in game_specific_actions if action.name == 'ACTION6'), None)

        # Add descriptions for all non-click actions.
        for action in game_specific_actions:
            if action.name != 'ACTION6':
                possible_moves.append({'type': action, 'object': None})

        # Add descriptions for potential click actions, linking them to target objects.
        if click_action_template and current_summary:
            for obj in current_summary:
                possible_moves.append({'type': click_action_template, 'object': obj})

        # If we have any moves, choose one, set data for it, and return it.
        if possible_moves:
            choice = random.choice(possible_moves)
            action_template = choice['type']
            chosen_object = choice['object']

            # If the chosen move is a click, use the set_data method.
            if chosen_object:
                pos = chosen_object['position']
                # Position is (row, column). Let's use the standard x=col, y=row.
                click_y = pos[0]  # row
                click_x = pos[1]  # column

                obj_id = chosen_object['id'].replace('obj_', 'id_')
                print(f"Setting data for CLICK on object {obj_id} with dict: {{'x': {click_x}, 'y': {click_y}}}")

                # Call the set_data method with a dictionary of coordinates.
                action_template.set_data({'x': click_x, 'y': click_y})
                return action_template
            else:
                # For simple actions, return the template itself.
                print(f"Agent chose action: {action_template.name}")
                return action_template

        # Fallback logic (if game_specific_actions only had ACTION6 but no objects were found)
        if click_action_template:
            print("Agent chose generic ACTION6 (no objects to click).")
            return click_action_template

        # Final fallback if no actions could be determined.
        fallback_options = [a for a in GameAction if a is not GameAction.RESET]
        chosen_action = random.choice(fallback_options)
        print(f"Agent chose fallback action: {chosen_action.name}")
        return chosen_action

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
        """Compares summaries by identifying in-place changes, moves, and new/removed objects."""
        if not old_summary and not new_summary:
            return []

        changes = []
        old_unexplained = list(old_summary)
        new_unexplained = list(new_summary)

        # --- Pass 1: Identify in-place changes (recolor, shape change, etc.) ---
        matches_to_remove = []
        processed_new_objs_in_pass1 = set()

        for old_obj in old_unexplained:
            # Find a new object at the same position that hasn't been matched yet
            for new_obj in new_unexplained:
                if id(new_obj) in processed_new_objs_in_pass1:
                    continue
                
                if old_obj['position'] == new_obj['position']:
                    color_changed = old_obj['color'] != new_obj['color']
                    shape_changed = (old_obj['fingerprint'] != new_obj['fingerprint'] or
                                     old_obj['size'] != new_obj['size'] or
                                     old_obj['pixels'] != new_obj['pixels'])

                    # If anything changed, it's an event. If not, it's stable.
                    # In either case, we've explained this pair of objects.
                    if color_changed and shape_changed:
                        changes.append(
                            f"- TRANSFORMED: Object at {old_obj['position']} changed shape and color "
                            f"(from C:{old_obj['color']} to C:{new_obj['color']})."
                        )
                    elif color_changed:
                        size_str = f"{old_obj['size'][0]}x{old_obj['size'][1]}"
                        changes.append(
                            f"- RECOLORED: A {size_str} object at {old_obj['position']} "
                            f"changed color from {old_obj['color']} to {new_obj['color']}."
                        )
                    elif shape_changed:
                        changes.append(
                            f"- SHAPE_CHANGED: The object at {old_obj['position']} (Color: {old_obj['color']}) changed its shape."
                        )

                    matches_to_remove.append((old_obj, new_obj))
                    processed_new_objs_in_pass1.add(id(new_obj))
                    break  # Move to the next old_obj, as its position is now explained
        
        # Remove the explained objects before the next pass
        for old_match, new_match in matches_to_remove:
            old_unexplained.remove(old_match)
            new_unexplained.remove(new_match)

        # --- Pass 2: Identify moved objects from the remaining pool ---
        def get_stable_id(obj):
            return (obj['fingerprint'], obj['color'], obj['size'], obj['pixels'])
        
        old_map_by_id = {}
        for obj in old_unexplained:
            stable_id = get_stable_id(obj)
            old_map_by_id.setdefault(stable_id, []).append(obj)
        
        new_map_by_id = {}
        for obj in new_unexplained:
            stable_id = get_stable_id(obj)
            new_map_by_id.setdefault(stable_id, []).append(obj)

        movable_ids = set(old_map_by_id.keys()) & set(new_map_by_id.keys())
        
        moves_to_remove = []
        for stable_id in movable_ids:
            old_instances = old_map_by_id[stable_id]
            new_instances = new_map_by_id[stable_id]
            
            num_matches = min(len(old_instances), len(new_instances))
            for _ in range(num_matches):
                old_inst = old_instances.pop()
                new_inst = new_instances.pop()
                # A move is only a move if the position is different.
                if old_inst['position'] != new_inst['position']:
                    changes.append(f"- MOVED: Object with ID {stable_id} moved from {old_inst['position']} to {new_inst['position']}.")
                moves_to_remove.append((old_inst, new_inst))
        
        for old_match, new_match in moves_to_remove:
            old_unexplained.remove(old_match)
            new_unexplained.remove(new_match)

        # --- Pass 3: Log remaining objects as removed or new ---
        for obj in old_unexplained:
            stable_id = get_stable_id(obj)
            changes.append(f"- REMOVED: Object with ID {stable_id} at {obj['position']} has disappeared.")

        for obj in new_unexplained:
            stable_id = get_stable_id(obj)
            changes.append(f"- NEW: Object with ID {stable_id} has appeared at {obj['position']}.")

        return sorted(changes)