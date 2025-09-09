import random
from .agent import Agent, FrameData
from .structs import GameAction, GameState
from collections import deque
import copy
import ast

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
        self.last_action_context = None  # Will store a tuple of (action_name, coords_dict)
        self.rule_hypotheses = {}
        self.failure_contexts = {}

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
        # On subsequent turns, analyze the outcome of the previous action.
        else:
            prev_summary = self.last_object_summary
            changes = self._log_changes(prev_summary, current_summary)

            if self.last_action_context:
                prev_action_name, prev_coords = self.last_action_context
                learning_key = self._get_learning_key(prev_action_name, prev_coords)

                if changes:
                    # --- Success Path ---
                    print("--- Change Log ---")
                    for change in changes:
                        print(change)
                    self._analyze_and_report(learning_key, changes)
                else:
                    # --- Failure Path ---
                    if learning_key in self.rule_hypotheses:
                        print(f"\n--- Failure Detected for Action {learning_key} ---")
                        print("Action produced no changes. Recording world state to learn failure conditions.")
                        self.failure_contexts.setdefault(learning_key, []).append(prev_summary)
                        self._analyze_failures(learning_key)

            elif changes:
                # Fallback for when changes happen without a known previous action
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

        action_to_return = None
        coords_for_context = None  # Variable to hold click coordinates for our internal memory

        # If we have any moves, choose one and prepare to return it.
        if possible_moves:
            choice = random.choice(possible_moves)
            action_template = choice['type']
            chosen_object = choice['object']

            if chosen_object:
                pos = chosen_object['position']
                click_y, click_x = pos[0], pos[1]
                obj_id = chosen_object['id'].replace('obj_', 'id_')
                print(f"Setting data for CLICK on object {obj_id} with dict: {{'x': {click_x}, 'y': {click_y}}}")
                action_template.set_data({'x': click_x, 'y': click_y})
                action_to_return = action_template
                
                # Capture the coordinates at the moment we decide on them
                coords_for_context = {'x': click_x, 'y': click_y}
            else:
                print(f"Agent chose action: {action_template.name}")
                action_to_return = action_template
        
        # Fallback logic if no primary action was chosen
        if not action_to_return:
            if click_action_template:
                print("Agent chose generic ACTION6 (no objects to click).")
                action_to_return = click_action_template
            else:
                fallback_options = [a for a in GameAction if a is not GameAction.RESET]
                action_to_return = random.choice(fallback_options)
                print(f"Agent chose fallback action: {action_to_return.name}")

        # Before returning, store the context of the chosen action for the next turn's analysis.
        self.last_action_context = (action_to_return.name, coords_for_context)
        return action_to_return

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """
        This method is called by the game to see if the agent thinks it is done.
        """
        return False
    
    def _get_learning_key(self, action_name: str, coords: dict | None) -> str:
        """Generates a unique key for learning, specific to coordinates for CLICK actions."""
        if action_name == 'ACTION6' and coords:
            return f"{action_name}_({coords['x']},{coords['y']})"
        return action_name
    
    def _parse_change_logs_to_events(self, changes: list[str]) -> list[dict]:
        """Parses a list of human-readable change logs into a list of structured event dictionaries."""
        events = []
        for log_str in changes:
            try:
                change_type, details = log_str.replace('- ', '', 1).split(': ', 1)
                event = {'type': change_type}

                if change_type == 'MOVED':
                    parts = details.split(' moved from ')
                    id_tuple = ast.literal_eval(parts[0].replace('Object with ID ', ''))
                    pos_parts = parts[1].replace('.', '').split(' to ')
                    start_pos, end_pos = ast.literal_eval(pos_parts[0]), ast.literal_eval(pos_parts[1])
                    
                    event.update({
                        'vector': (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]),
                        'fingerprint': id_tuple[0], 'color': id_tuple[1],
                        'size': id_tuple[2], 'pixels': id_tuple[3]
                    })
                    events.append(event)

                elif change_type == 'RECOLORED':
                    # "A 1x1 object at (2, 2) changed color from 15 to 3."
                    size_str = details.split(' object at ')[0].split(' ')[-1]
                    position_str = details.split(' at ')[1].split(' changed color')[0]
                    from_color_str = details.split(' from ')[1].split(' to ')[0]
                    to_color_str = details.split(' to ')[1].replace('.', '')

                    event.update({
                        'size': ast.literal_eval(size_str.replace('x', ',')),
                        'position': ast.literal_eval(position_str),
                        'from_color': int(from_color_str),
                        'to_color': int(to_color_str)
                    })
                    events.append(event)
                
                elif change_type == 'SHAPE_CHANGED':
                    # "The object at (8, 8) (Color: 3) changed shape (fingerprint: ... -> ...)."
                    position_str = details.split(' (Color:')[0].split(' at ')[1]
                    color_str = details.split('(Color: ')[1].split(')')[0]
                    fp_part = details.split('fingerprint: ')[1]
                    from_fp_str, to_fp_str = fp_part.replace(').','').split(' -> ')

                    event.update({
                        'position': ast.literal_eval(position_str),
                        'color': int(color_str),
                        'from_fingerprint': int(from_fp_str),
                        'to_fingerprint': int(to_fp_str)
                    })
                    events.append(event)
            
            except (ValueError, IndexError, SyntaxError):
                # This will catch any log string that doesn't match our expected formats
                # and allow the agent to continue without crashing.
                continue
        return events

    def _analyze_and_report(self, action_key: str, changes: list[str]):
        """Builds a hypothesis from raw changes, refining from a 'case file' to an 'abstract rule'."""
        
        def create_blueprint_from_events(events: list[dict]) -> dict:
            """Creates an abstract summary blueprint from a list of raw event dictionaries."""
            blueprint = {
                'event_counts': {}, 'move_vectors': set(), 'moved_object_colors': set(),
                'moved_object_sizes': set(), 'moved_object_pixels': set()
            }
            event_types = [e['type'] for e in events]
            blueprint['event_counts'] = {t: event_types.count(t) for t in set(event_types)}

            for event in events:
                if event['type'] == 'MOVED':
                    blueprint['move_vectors'].add(event['vector'])
                    blueprint['moved_object_colors'].add(event['color'])
                    blueprint['moved_object_sizes'].add(event['size'])
                    blueprint['moved_object_pixels'].add(event['pixels'])
            return blueprint

        # --- Step 1: Parse the current turn's changes into structured events ---
        new_events = self._parse_change_logs_to_events(changes)
        if not new_events: return

        # --- Step 2: Retrieve existing hypothesis and update it ---
        existing_hypothesis = self.rule_hypotheses.get(action_key)

        if not existing_hypothesis:
            # --- Stage 1: First Observation -> Create the "Case File" ---
            self.rule_hypotheses[action_key] = new_events
            print(f"\n--- Initial Case File for {action_key} ---")
            for event in new_events:
                print(f"- Observed {event['type']} with details: { {k:v for k,v in event.items() if k != 'type'} }")
        
        elif isinstance(existing_hypothesis, list):
            # --- Stage 2: Second Observation -> From "Case File" to "Abstract Rule" ---
            old_blueprint = create_blueprint_from_events(existing_hypothesis)
            new_blueprint = create_blueprint_from_events(new_events)

            # Intersect the two blueprints to form the first abstract rule
            for key in old_blueprint:
                if isinstance(old_blueprint[key], set):
                    old_blueprint[key].intersection_update(new_blueprint[key])
                elif isinstance(old_blueprint[key], dict):
                     old_blueprint[key] = {
                         k: v for k, v in old_blueprint[key].items() if new_blueprint[key].get(k) == v
                     }
            
            self.rule_hypotheses[action_key] = old_blueprint # Promote the hypothesis to a dict
            print(f"\n--- First Abstract Rule for {action_key} (from 2 observations) ---")
            for key, value in old_blueprint.items():
                if value: print(f"- {key.replace('_', ' ').title()}: {value}")

        elif isinstance(existing_hypothesis, dict):
            # --- Stage 3: Subsequent Observations -> Refine the "Abstract Rule" ---
            new_blueprint = create_blueprint_from_events(new_events)
            
            # Intersect the new blueprint with the existing abstract rule
            for key in existing_hypothesis:
                if isinstance(existing_hypothesis[key], set):
                    existing_hypothesis[key].intersection_update(new_blueprint[key])
                elif isinstance(existing_hypothesis[key], dict):
                    existing_hypothesis[key] = {
                        k: v for k, v in existing_hypothesis[key].items() if new_blueprint[key].get(k) == v
                    }
            
            print(f"\n--- Refined Hypothesis for {action_key} ---")
            found_any_rules = False
            for key, value in existing_hypothesis.items():
                if value:
                    found_any_rules = True
                    print(f"- {key.replace('_', ' ').title()}: {value}")
            if not found_any_rules:
                print("No consistent properties remain in the hypothesis.")

    def _analyze_failures(self, action_name: str):
        """Analyzes the contexts of an action's failures to find common preconditions."""
        failure_summaries = self.failure_contexts.get(action_name, [])
        num_failures = len(failure_summaries)

        if num_failures < 2:
            # We need at least two failures to find a pattern.
            return

        # --- Step 1: Create a "Grid Fingerprint" for each failure state ---
        fingerprints = []
        for summary in failure_summaries:
            fingerprint = {'object_count': len(summary), 'colors_present': set()}
            color_counts = {}
            for obj in summary:
                color = obj['color']
                fingerprint['colors_present'].add(color)
                color_counts[color] = color_counts.get(color, 0) + 1
            fingerprint['color_counts'] = color_counts
            fingerprints.append(fingerprint)

        # --- Step 2: Intersect the fingerprints to find common conditions ---
        # Start with the first failure as the base hypothesis
        consistent_conditions = fingerprints[0]
        for i in range(1, num_failures):
            next_fp = fingerprints[i]
            # Intersect simple values (like object_count)
            if consistent_conditions['object_count'] != next_fp['object_count']:
                consistent_conditions['object_count'] = None # Invalidate if not consistent
            
            # Intersect sets of present colors
            consistent_conditions['colors_present'].intersection_update(next_fp['colors_present'])
            
            # Intersect dictionaries of color counts
            consistent_conditions['color_counts'] = {
                k: v for k, v in consistent_conditions['color_counts'].items()
                if next_fp['color_counts'].get(k) == v
            }

        # --- Step 3: Report the findings ---
        print(f"Analyzing {num_failures} failure instances for action '{action_name}'...")
        found_any_conditions = False
        for key, value in consistent_conditions.items():
            if value is not None and value: # Check for invalidated properties or empty sets/dicts
                print(f"- Common Failure Precondition: {key.replace('_', ' ').title()} -> {value}")
                found_any_conditions = True
        
        if not found_any_conditions:
            print("No consistent failure preconditions identified yet.")

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
    
    def _get_stable_id(self, obj):
        """Creates a hashable, stable ID for an object based on its intrinsic properties."""
        return (obj['fingerprint'], obj['color'], obj['size'], obj['pixels'])
    
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
                            f"- TRANSFORMED: Object at {old_obj['position']} changed shape "
                            f"(fingerprint: {old_obj['fingerprint']} -> {new_obj['fingerprint']}) "
                            f"and color (from C:{old_obj['color']} to C:{new_obj['color']})."
                        )
                    elif color_changed:
                        size_str = f"{old_obj['size'][0]}x{old_obj['size'][1]}"
                        changes.append(
                            f"- RECOLORED: A {size_str} object at {old_obj['position']} "
                            f"changed color from {old_obj['color']} to {new_obj['color']}."
                        )
                    elif shape_changed:
                        changes.append(
                            f"- SHAPE_CHANGED: The object at {old_obj['position']} (Color: {old_obj['color']}) changed shape "
                            f"(fingerprint: {old_obj['fingerprint']} -> {new_obj['fingerprint']})."
                        )

                    matches_to_remove.append((old_obj, new_obj))
                    processed_new_objs_in_pass1.add(id(new_obj))
                    break  # Move to the next old_obj, as its position is now explained
        
        # Remove the explained objects before the next pass
        for old_match, new_match in matches_to_remove:
            old_unexplained.remove(old_match)
            new_unexplained.remove(new_match)

        # --- Pass 2: Identify moved objects from the remaining pool ---
        old_map_by_id = {}
        for obj in old_unexplained:
            stable_id = self._get_stable_id(obj)
            old_map_by_id.setdefault(stable_id, []).append(obj)
        
        new_map_by_id = {}
        for obj in new_unexplained:
            stable_id = self._get_stable_id(obj)
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

        # --- Pass 3: Fuzzy Matching for GROWTH and SHRINK events ---
        if old_unexplained and new_unexplained:
            potential_pairs = []
            for old_obj in old_unexplained:
                for new_obj in new_unexplained:
                    pos1 = old_obj['position']
                    pos2 = new_obj['position']
                    # Calculate Manhattan distance between corners
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    potential_pairs.append({'old': old_obj, 'new': new_obj, 'dist': distance})
            
            # Sort by distance to find the closest pairs first
            potential_pairs.sort(key=lambda p: p['dist'])
            
            matched_old = set()
            matched_new = set()
            for pair in potential_pairs:
                old_obj, new_obj = pair['old'], pair['new']
                # If neither object in a close pair has been matched yet, match them.
                if id(old_obj) not in matched_old and id(new_obj) not in matched_new:
                    old_pixels = old_obj['pixels']
                    new_pixels = new_obj['pixels']
                    
                    if new_pixels > old_pixels:
                        event_type = "GROWTH"
                    elif old_pixels > new_pixels:
                        event_type = "SHRINK"
                    else:
                        event_type = "TRANSFORM" # Same pixel count, but other properties changed

                    pixel_diff = abs(new_obj['pixels'] - old_obj['pixels'])
                    old_size_str = f"{old_obj['size'][0]}x{old_obj['size'][1]}"
                    new_size_str = f"{new_obj['size'][0]}x{new_obj['size'][1]}"
                    
                    if event_type == "GROWTH":
                        details = f"grew by {pixel_diff} pixels (from {old_size_str} to {new_size_str})"
                    elif event_type == "SHRINK":
                        details = f"shrank by {pixel_diff} pixels (from {old_size_str} to {new_size_str})"
                    else: # TRANSFORM
                        details = f"transformed (size {old_size_str} -> {new_size_str})"

                    changes.append(f"- {event_type}: Object at {old_obj['position']} {details}, now at {new_obj['position']}.")

                    matched_old.add(id(old_obj))
                    matched_new.add(id(new_obj))
            
            # Remove the newly matched objects from the unexplained lists
            old_unexplained = [obj for obj in old_unexplained if id(obj) not in matched_old]
            new_unexplained = [obj for obj in new_unexplained if id(obj) not in matched_new]

        # --- Pass 3: Log remaining objects as removed or new ---
        for obj in old_unexplained:
            stable_id = self._get_stable_id(obj)
            changes.append(f"- REMOVED: Object with ID {stable_id} at {obj['position']} has disappeared.")

        for obj in new_unexplained:
            stable_id = self._get_stable_id(obj)
            changes.append(f"- NEW: Object with ID {stable_id} has appeared at {obj['position']}.")

        return sorted(changes)