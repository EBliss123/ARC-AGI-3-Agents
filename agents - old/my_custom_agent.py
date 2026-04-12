import numpy as np
import logging
import random
from collections import defaultdict
from collections import deque

# Imports from the game framework's files
from .agent import Agent
from .structs import FrameData, GameAction, GameState, ComplexAction

class GameObserver:
    """Analyzes the game grid to identify all distinct, contiguous objects."""

    def _find_contiguous_clusters(self, coords_list):
        """Uses BFS to find separate groups of contiguous pixels."""
        all_coords_set = set(coords_list)
        visited = set()
        clusters = []

        for r_start, c_start in coords_list:
            if (r_start, c_start) in visited:
                continue

            # Found the start of a new, unvisited object
            new_cluster = set()
            q = deque([(r_start, c_start)])
            visited.add((r_start, c_start))

            while q:
                r, c = q.popleft()
                new_cluster.add((r, c))

                # Check neighbors (up, down, left, right)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (r + dr, c + dc)
                    if neighbor in all_coords_set and neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)

            clusters.append(frozenset(new_cluster))
        return clusters

    def analyze_and_describe_objects(self, grid):
        """Scans the grid to find and describe all distinct objects."""
        if grid.ndim < 2:
            logging.warning(f"Grid is not 2-dimensional (shape: {grid.shape}). Cannot analyze objects.")
            return []

        # Step 1: Group all pixels by their color
        objects_by_color = defaultdict(list)
        for r_idx, row in enumerate(grid):
            for c_idx, cell in enumerate(row):
                color_key = tuple(cell) if hasattr(cell, '__iter__') else (cell,)
                # Ignore black background pixels entirely
                if color_key != (0, 0, 0, 255) and color_key != (0,):
                    objects_by_color[color_key].append((r_idx, c_idx))

        # Step 2: For each color, find the distinct contiguous objects
        final_objects = []
        for color, coords_list in objects_by_color.items():
            clusters = self._find_contiguous_clusters(coords_list)
            for cluster_coords in clusters:
                min_r, max_r = min(r for r, c in cluster_coords), max(r for r, c in cluster_coords)
                min_c, max_c = min(c for r, c in cluster_coords), max(c for r, c in cluster_coords)

                final_objects.append({
                    'color': color,
                    'coords': cluster_coords,
                    'size': len(cluster_coords),
                    'center': (int(round((min_r + max_r) / 2)), int(round((min_c + max_c) / 2))),
                    'height': max_r - min_r + 1,
                    'width': max_c - min_c + 1,
                    'shape_key': frozenset((r - min_r, c - min_c) for r, c in cluster_coords)
                })

        return final_objects 
    
    def analyze_and_describe_objects(self, grid):
        """Scans the grid, describes all objects, and merges objects near the border."""
        # Add a safety check for the grid's dimensions
        if grid.ndim < 2:
            logging.warning(f"Grid is not 2-dimensional (shape: {grid.shape}). Cannot analyze objects.")
            return []

        # Step 1: Find all objects based on color
        objects_by_color = defaultdict(list)
        grid_height, grid_width = grid.shape[:2]

        for r_idx, row in enumerate(grid):
            for c_idx, cell in enumerate(row):
                color_key = tuple(cell) if hasattr(cell, '__iter__') else (cell,)
                objects_by_color[color_key].append((r_idx, c_idx))

        # Step 2: Get initial descriptions for all colored objects
        initial_descriptions = []
        for color, coords in objects_by_color.items():
            if not coords: continue
            min_r, max_r = min(r for r, c in coords), max(r for r, c in coords)
            min_c, max_c = min(c for r, c in coords), max(c for r, c in coords)
            shape_key = frozenset((r - min_r, c - min_c) for r, c in coords)
            initial_descriptions.append({
                'color': color, 'coords': frozenset(coords), 'size': len(coords),
                'center': (int(round((min_r + max_r) / 2)), int(round((min_c + max_c) / 2))),
                'height': max_r - min_r + 1, 'width': max_c - min_c + 1, 'shape_key': shape_key
            })

        # Step 3: Identify and separate border objects from main objects
        border_margin = 2
        border_object_coords = set()
        main_objects = []

        for obj in initial_descriptions:
            is_border_obj = all(
                r < border_margin or r >= grid_height - border_margin or
                c < border_margin or c >= grid_width - border_margin
                for r, c in obj['coords']
            )
            if is_border_obj:
                border_object_coords.update(obj['coords'])
            else:
                main_objects.append(obj)

        final_objects = main_objects
        # Step 4: If any border objects were found, create a single merged object
        if border_object_coords:
            min_r, max_r = min(r for r, c in border_object_coords), max(r for r, c in border_object_coords)
            min_c, max_c = min(c for r, c in border_object_coords), max(c for r, c in border_object_coords)

            merged_obj = {
                'color': (-1, -1, -1, -1),
                'coords': frozenset(border_object_coords),
                'size': len(border_object_coords),
                'center': (int(round((min_r + max_r) / 2)), int(round((min_c + max_c) / 2))),
                'height': max_r - min_r + 1, 'width': max_c - min_c + 1,
                'shape_key': frozenset((r - min_r, c - min_c) for r, c in border_object_coords)
            }
            final_objects.append(merged_obj)

        return final_objects

class ActionController:
    """Manages the agent's actions and decision-making."""
    pass

class WorldMapper:
    """Builds and maintains an internal map of the game world."""
    pass

class RuleEngine:
    """Formulates and tests hypotheses about game rules."""
    def __init__(self):
        # The agent's memory of what happened after each action
        # Format: { action_id: [list of changes seen on turn_1, list of changes seen on turn_2, ...]}
        self.observations = defaultdict(list)

    def record_observation(self, action_id, changes):
        """Records the observed changes for a given action."""
        if changes: # Only record if something actually happened
            self.observations[action_id].append(changes)
            logging.info(f"RuleEngine: Logged {len(changes)} changes for action {action_id}.")

    def find_consistent_rules(self, action_id, confidence_threshold=0.9):
        """Analyzes observations for an action and returns high-confidence rules."""
        rules = []
        action_observations = self.observations.get(action_id, [])
        total_observations = len(action_observations)

        if total_observations < 3: # Don't bother analyzing with too little data
            return []

        # Tally up all the effects we've seen for this action
        effect_counts = defaultdict(int)
        for turn_changes in action_observations:
            for change in turn_changes:
                # Create a unique key for each effect on each object
                effect_key = (change['id'], change['type'])
                effect_counts[effect_key] += 1

        # Check which effects are consistent
        for (obj_id, effect_type), count in effect_counts.items():
            confidence = count / total_observations
            if confidence >= confidence_threshold:
                rules.append(f"Rule found: Action {action_id} consistently causes '{effect_type}' on object with color {obj_id} (Confidence: {confidence:.0%})")

        return rules

class MyCustomAgent(Agent):
    def __init__(self, *args, **kwargs):
        """Initializes the agent."""
        super().__init__(*args, **kwargs)
        
        # Long-term components
        self.observer = GameObserver()
        self.rule_engine = RuleEngine()
        self.world_mapper = WorldMapper()
        self.action_controller = ActionController()

        # Episode-specific state
        self.phase = "discovery"
        self.last_grid = np.array([])
        self.last_action_taken = 0

        # State for discovery phases
        self.actions_to_test = [1, 2, 3, 4, 5]
        self.discovered_actions = []
        self.initial_objects = None
        self.click_targets = []
        self.click_discovery_prepared = False
        self.rules_analyzed = False
        self.player_identified = False
        self.last_player_pos = None

    def _analyze_grid_changes(self, last_desc, current_desc):
        """Compares two lists of described objects to find all changes."""
        changes = []

        # Create maps for easy lookup using a stable ID (color + shape)
        last_obj_map = {(obj['color'], obj['shape_key']): obj for obj in last_desc}
        current_obj_map = {(obj['color'], obj['shape_key']): obj for obj in current_desc}

        # Find objects that moved
        for obj_id, last_obj in last_obj_map.items():
            if obj_id in current_obj_map:
                current_obj = current_obj_map[obj_id]
                if last_obj['center'] != current_obj['center']:
                    changes.append({'type': 'MOVE', 'id': obj_id, 'from_center': last_obj['center'], 'to_center': current_obj['center']})

        # Find objects that transformed (changed shape/color but not ID)
        # This part requires a more complex comparison, so we'll add it later.
        # For now, focusing on moves gives us the most important information.

        return changes

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        if self.action_counter == 0:
            return False
        
        current_state = latest_frame.state
        return current_state == GameState.WIN or current_state == GameState.GAME_OVER

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose which action the Agent should take."""

        # --- Turn 0: RESET is always the first action ---
        if self.action_counter == 0:
            logging.info("--- Turn 0: Resetting the environment. ---")
            self.last_action_taken = 0
            return GameAction.from_id(0)

        # --- Post-Reset Logic (from Turn 1 onwards) ---
        grid = latest_frame.frame
        grid = np.array(grid)

        # On Turn 1, analyze the TRUE initial grid state (after RESET)
        if self.action_counter == 1:
            self.initial_objects = self.observer.analyze_and_describe_objects(grid)
            logging.info("--- Turn 1: Initial Object Analysis ---")
            for obj in self.initial_objects:
                # Create a clean string for logging, converting frozenset for readability
                shape_str = str(sorted(list(obj['shape_key'])))
                logging.info(f"Found Object: color={obj['color']}, size={obj['size']}, center={obj['center']}, shape={shape_str}")
            logging.info("------------------------------------")

        # Analyze the result of the LAST turn
        if not np.array_equal(grid, self.last_grid):
            logging.info(f"--- Change detected! Action {self.last_action_taken} was effective. ---")
            if self.last_action_taken not in self.discovered_actions:
                self.discovered_actions.append(self.last_action_taken)

            # Analyze all changes and log them in the RuleEngine
            last_descriptions = self.observer.analyze_and_describe_objects(self.last_grid)
            current_descriptions = self.observer.analyze_and_describe_objects(grid)
            changes = self._analyze_grid_changes(last_descriptions, current_descriptions)
            self.rule_engine.record_observation(self.last_action_taken, changes)


        action_num = 0
        action_data = None
        reasoning = f"Phase: {self.phase}"

        # --- Phase-Based Action Selection ---
        if self.phase == "discovery":
            if self.actions_to_test:
                action_num = self.actions_to_test.pop(0)
                reasoning += f". Testing simple action {action_num}."
            else:
                logging.info("--- Phase 1 Complete. Moving to Click Discovery. ---")
                self.phase = "click_discovery"

        elif self.phase == "click_discovery":
            if not self.click_discovery_prepared:
                targets = []
                # Define background color as a tuple, matching the object's color format
                background_color = (0, 0, 0, 255) # Assuming RGBA

                if self.initial_objects:
                    for obj in self.initial_objects:
                        # Skip the background object
                        if obj['color'] == background_color:
                            continue
                        # The center is already calculated for us in the object description
                        center_y, center_x = obj['center']
                        targets.append((center_y, center_x))

                self.click_targets = targets
                self.click_discovery_prepared = True
                logging.info(f"--- Phase 2: Prepared {len(self.click_targets)} object centers to click. ---")

            if self.click_targets:
                action_num = 6
                y, x = self.click_targets.pop(0)
                action_data = {'x': x, 'y': y}
                reasoning += f". Testing click at (x={x}, y={y})."
            else:
                logging.info("--- Phase 2 Complete. Moving to Mapping. ---")
                self.phase = "mapping"

        elif self.phase == "mapping":
            # When first entering the mapping phase, analyze the data gathered so far.
            if not self.rules_analyzed:
                logging.info("--- Analyzing observations to find consistent rules... ---")
                all_actions = self.discovered_actions + [6] # Check move actions and clicks
                for action_id in all_actions:
                    rules = self.rule_engine.find_consistent_rules(action_id)
                    for rule in rules:
                        logging.info(rule) # Print any found rules to the console
                self.rules_analyzed = True

            reasoning += ". Exploring the map based on discovered rules."
            # Filter out RESET (action 0) and CLICK (action 6) from the list of actions to use for exploring.
            exploring_actions = [a for a in self.discovered_actions if a not in [0, 6]]

            if exploring_actions:
                action_num = random.choice(exploring_actions)
            else:
                logging.warning("No effective movement actions found to explore with.")
                action_num = 0 # Fallback to RESET if no move actions were found


        # --- Create and return the final action object ---
        action_obj = GameAction.from_id(action_num)
        if action_data:
            action_obj.set_data(action_data)
        action_obj.reasoning = reasoning

        self.last_grid = np.copy(grid)
        self.last_action_taken = action_num

        return action_obj