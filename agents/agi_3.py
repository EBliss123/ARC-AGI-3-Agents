# ARC-AGI-3 Main Script
import random
import copy
from enum import Enum
from .agent import Agent
from .structs import FrameData, GameAction, GameState
from collections import Counter

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

class AgentState(Enum):
    """Represents the agent's current operational state."""
    DISCOVERY = 1
    RANDOM_ACTION = 2

# --- Core AGI Logic ---

class AGI3(Agent):
    """The general agent that learns to play the games."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom agent initializations can go here.
        self.previous_frame = None
        self.changed_pixels = []
        self.static_pixels = []
        self.debug_counter = 0
        self.visited_grids = set() # Stores previously seen grid states
        self.ignore_for_state_hash = set()
        self.state_graph = {} # Stores stateA -> action -> stateB
        self.last_grid_tuple = None

        # --- State Management ---
        self.agent_state = AgentState.DISCOVERY
        self.discovery_runs = 0
        self.last_action = None
        self.action_effects = {} # Will store actions and all their resulting changes
        self.action_failures = {}
        self.ineffective_actions = [] # Tracks actions that had no effect since the last success
        self.level_start_frame = None
        self.level_start_score = 0
        self.resource_indicator_candidates = {}
        self.confirmed_resource_indicator = None
        self.RESOURCE_CONFIDENCE_THRESHOLD = 3 # Actions in a row to confirm

        # --- Object & Shape Tracking ---
        self.observed_object_shapes = {} # Maps shape tuple -> count
        self.last_known_objects = [] # Stores full object descriptions from the last frame
        self.world_model = {
            'player_signature': None,
            'floor_color': None,
            'action_map': {} # Will store confirmed action -> effect mappings
        }
        # This will store hypotheses like {(obj_signature, color): confidence_count}
        self.player_floor_hypothesis = {}
        self.agent_move_hypothesis = {} # Tracks how many times a shape has moved
        self.floor_hypothesis = {} # Tracks how many times a color has been identified as floor
        self.action_effect_hypothesis = {} # Tracks action -> effect hypotheses
        self.CONCEPT_CONFIDENCE_THRESHOLD = 3 # Number of times a pattern must be seen to be learned

        # --- Generic Action Groups ---
        # Get all possible actions, excluding RESET, to create generic groups.
        all_discoverable_actions = [a for a in GameAction if a is not GameAction.RESET]
        
        # Group 1-5 are primary; 6 is secondary.
        self.primary_actions = all_discoverable_actions[:5]
        
        # Safely get the 6th action if it exists.
        if len(all_discoverable_actions) > 5:
            self.secondary_actions = [all_discoverable_actions[5]]
        else:
            self.secondary_actions = []

        # --- Discovery Phase Tracking ---
        # Start the first discovery run with a shuffled list of primary actions.
        actions_for_first_run = self.primary_actions.copy()
        random.shuffle(actions_for_first_run)
        self.actions_to_try = actions_for_first_run
        self.discovery_sub_phase = 'PRIMARY' # Can be 'PRIMARY' or 'SECONDARY'
        self.discovered_in_current_run = False
        self.level_knowledge_is_learned = False

        print(f"Custom AGI initialized for game: {self.game_id}")

    def _end_discovery_run(self):
        """Helper method to finalize a discovery run and set up for the next."""
        self.discovery_runs += 1
        print(f"--- Discovery Run {self.discovery_runs} Complete ---")

        if self.discovery_runs >= 3:
            print("--- All discovery runs complete. Switching to RANDOM_ACTION state. ---")
            self.agent_state = AgentState.RANDOM_ACTION
            self.level_knowledge_is_learned = True
        else:
            # Set up the next run with a new, randomly shuffled sequence.
            next_run_actions = self.primary_actions.copy()
            random.shuffle(next_run_actions)
            self.actions_to_try = next_run_actions
            self.discovery_sub_phase = 'PRIMARY'
            self.discovered_in_current_run = False

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        # The agent stops this attempt if it wins the level.
        return latest_frame.state is GameState.WIN
    
    def _reset_for_new_attempt(self):
        """Resets the agent's state for a new life or attempt without printing."""
        self.previous_frame = None
        self.last_action = None
        self.last_known_objects = []

        if self.level_knowledge_is_learned:
            self.agent_state = AgentState.RANDOM_ACTION
        else:
            self.agent_state = AgentState.DISCOVERY
            self.discovery_runs = 0
            # Reset to the first discovery run with a fresh shuffled sequence.
            initial_actions = self.primary_actions.copy()
            random.shuffle(initial_actions)
            self.actions_to_try = initial_actions
            self.discovery_sub_phase = 'PRIMARY'
            self.discovered_in_current_run = False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """This is the main decision-making method for the AGI."""
        # --- 1. Store initial level state if not already set ---
        if self.level_start_frame is None:
            print("--- New Level Detected (Initial Frame). Storing start state. ---")
            self.level_start_frame = copy.deepcopy(latest_frame.frame)
            self.level_start_score = latest_frame.score
        
        # To check for new states, create a version of the grid that ignores noisy coordinates.
        grid_for_hashing = copy.deepcopy(latest_frame.frame)

        # Mask all coordinates that are flagged to be ignored (e.g., resource indicators).
        # This ensures states are considered identical even if ignored UI elements change.
        if self.ignore_for_state_hash:
            for row_index in self.ignore_for_state_hash:
                # Ensure the coordinate is within the current grid's bounds before masking.
                # The grid is effectively frame[0], so we check if the index is valid for that list.
                if row_index < len(grid_for_hashing[0]):
                    grid_for_hashing[0][row_index] = [-1] # Use a constant, non-game value.

        # Convert the (potentially masked) grid to a hashable format for memory storage.
        grid_tuple = tuple(tuple(tuple(p) for p in row) for row in grid_for_hashing)

        # Check if this masked grid state is new before adding it to memory.
        if grid_tuple not in self.visited_grids:
            print(f"🔎 New grid state discovered! Total unique states seen: {len(self.visited_grids) + 1}")
            self.visited_grids.add(grid_tuple)

        # If we have a previous state and action, record the transition in our graph.
        if self.last_grid_tuple and self.last_action:
            # Ensure the 'from' state is in the graph.
            if self.last_grid_tuple not in self.state_graph:
                self.state_graph[self.last_grid_tuple] = {}
            # Record that last_action from last_state leads to the current state.
            self.state_graph[self.last_grid_tuple][self.last_action] = grid_tuple
        
        # --- 1. Check for Win/Loss State First (Highest Priority) ---
        if latest_frame.state is GameState.WIN:
            print("🏆 Level Solved! Awaiting next level. 🏆")
            # We don't reset here, just wait for the new level to load.
            return GameAction.NOOP # Do nothing until the next level starts
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # If the whole game is new/over, reset everything.
            if latest_frame.state == GameState.NOT_PLAYED:
                print("--- New Game Detected. Resetting all knowledge. ---")
                self.level_knowledge_is_learned = False
            else: # GameState.GAME_OVER
                print("--- Game Over Detected. Resetting attempt. ---")
            
            self._reset_for_new_attempt()
            return GameAction.RESET

        # --- 2. Perception & Consequence of Last Action ---
        novel_changes_found, known_changes_found, change_descriptions, structured_changes = self.perceive(latest_frame)

        # --- Special Handling for Dimension Changes (New Level or Lost Life) ---
        if novel_changes_found and change_descriptions == ["Frame dimensions changed"]:
            new_grid = latest_frame.frame
            new_score = latest_frame.score

            # A new level is detected if the grid changes AND the score increases.
            is_new_level = (new_grid != self.level_start_frame and new_score > self.level_start_score)

            if is_new_level:
                print(f"--- New Level Detected! Score increased to {new_score}. ---")
                # Update the baseline for the new level
                self.level_start_frame = copy.deepcopy(new_grid)
                self.level_start_score = new_score
                # Attribute success to the last action and describe it as a level advance.
                change_descriptions = [f"Advanced to a new level with score {new_score}."]
            else:
                # If the state matches the start of the level, it's a lost life. 
                if self.level_knowledge_is_learned:
                    print("--- Lost a Life (Frame reset). Knowledge retained, resuming... ---")
                else:
                    print("--- Lost a Life (Frame reset). Resetting discovery process. ---")
                
                self._reset_for_new_attempt()
                return GameAction.RESET

        # Process the result of the last action.
        if self.last_action:
            # A "successful" action is one that causes a novel, non-indicator change.
            if novel_changes_found:
                # --- Action SUCCEEDED (caused a novel change) ---
                was_cleared = len(self.ineffective_actions) > 0
                self.ineffective_actions.clear()
                clear_message = " Clearing ineffective actions list." if was_cleared else ""

                if self.agent_state == AgentState.DISCOVERY:
                    self.action_effects[self.last_action] = change_descriptions
                    self.discovered_in_current_run = True
                    print(f"Action {self.last_action.name} caused {len(change_descriptions)} novel changes. Storing success.{clear_message}")
                else: # RANDOM_ACTION state
                    print(f"Known action {self.last_action.name} succeeded, causing {len(change_descriptions)} novel changes.{clear_message}")

                for description in change_descriptions[:10]:
                    print(description)
                if len(change_descriptions) > 10:
                    print("  - ...and more.")

            else:
                # --- Action FAILED (caused no novel change) ---
                # It might have only changed the resource indicator, which we treat as an ineffective action.
                if known_changes_found:
                    print("💧 Resource level changed, but no other effects were observed.")

                if self.last_action not in self.ineffective_actions:
                    self.ineffective_actions.append(self.last_action)

                if self.agent_state == AgentState.RANDOM_ACTION:
                    print(f"Action {self.last_action.name} had no novel effect. Ineffective actions: {[a.name for a in self.ineffective_actions]}")
                    context = copy.deepcopy(self.previous_frame)
                    if self.last_action not in self.action_failures:
                        self.action_failures[self.last_action] = []
                    self.action_failures[self.last_action].append(context)

            # Still run indicator tracking if ANY change happened, to keep it updated.
            if novel_changes_found or known_changes_found:
                # We pass `structured_changes` here, which contains ALL changes (novel and known).
                self._update_resource_indicator_tracking(structured_changes, self.last_action)

        # --- Object Finding and Tracking ---
        if novel_changes_found:
            # --- Separate UI changes from game-world changes for analysis ---
            object_logic_changes = []
            if self.confirmed_resource_indicator:
                indicator_row = self.confirmed_resource_indicator['row_index']
                indicator_changes = []

                for change in structured_changes:
                    if change['row_index'] == indicator_row:
                        indicator_changes.append(change)
                    else:
                        object_logic_changes.append(change)

                # If changes happened on the indicator, log them in the desired format.
                if indicator_changes:
                    print("--- UI Resource Indicator Update ---")
                    for change in indicator_changes:
                        row = change['row_index']
                        for px_change in change['changes']:
                            print(f"🎨 [RESOURCE INDICATOR]: Indicator at ({row}, {px_change['index']}) changed from {px_change['from']} to {px_change['to']}.")
            else:
                # If no indicator is confirmed, all changes are for game-world object logic.
                object_logic_changes = structured_changes

            # 1. Find and describe all objects in the current frame
            current_objects = self._find_and_describe_objects(object_logic_changes, latest_frame.frame)

            # 2. Track objects from the last frame to the current one
            if self.last_known_objects: # Can only track if we have a "before" state
                tracking_logs = self._track_objects(current_objects, self.last_known_objects, latest_frame.frame, object_logic_changes, self.last_action)
                if tracking_logs:
                    print(f"--- Object Tracking Report (Action: {self.last_action.name}) ---")
                    for log in tracking_logs:
                        print(log)

            # 3. Update memory for the next turn by merging static and changed objects
            new_object_memory = []
            
            # First, add all static objects from the previous turn's memory
            if self.last_known_objects:
                # Get the set of coordinates that changed in this turn
                changed_coords = set()
                for change in object_logic_changes:
                    for px_change in change['changes']:
                        changed_coords.add((change['row_index'], px_change['index']))
                
                # Add any object from memory that was NOT in a changed location
                for obj in self.last_known_objects:
                    # Purge any object from memory that is on the confirmed indicator row.
                    if self.confirmed_resource_indicator:
                        indicator_row = self.confirmed_resource_indicator['row_index']
                        obj_rows = range(obj['top_row'], obj['top_row'] + obj['height'])
                        if indicator_row in obj_rows:
                            continue # Skip this object, it's on the UI row.
                    is_static = True
                    for r in range(obj['top_row'], obj['top_row'] + obj['height']):
                        if any((r, c) in changed_coords for c in range(obj['left_index'], obj['left_index'] + obj['width'])):
                            is_static = False
                            break
                    if is_static:
                        new_object_memory.append(obj)
            
            # Second, add the new, updated objects from the current turn
            # (The 'current_objects' list only contains objects from changed areas)
            new_object_memory.extend(current_objects)

            # Finally, set the agent's memory for the next turn
            self.last_known_objects = new_object_memory

        # --- 3. Choose a New Action to Take ---
        if self.agent_state == AgentState.DISCOVERY:
            if not self.actions_to_try:
                if self.discovery_sub_phase == 'PRIMARY':
                    if self.discovered_in_current_run or not self.secondary_actions:
                        self._end_discovery_run()
                    else:
                        print("--- Primary actions yielded no results. Trying secondary actions. ---")
                        self.discovery_sub_phase = 'SECONDARY'
                        self.actions_to_try = self.secondary_actions.copy()
                else: 
                    self._end_discovery_run()
            
            if self.agent_state == AgentState.DISCOVERY and self.actions_to_try:
                action = self.actions_to_try.pop(0)
                self.last_action = action
                return action

        # If discovery is over, use the state graph to explore intelligently.
        if self.agent_state == AgentState.RANDOM_ACTION:
            print("--- Choosing Action Based on State Graph ---")
            
            # 1. Identify all possible actions.
            base_actions = list(self.action_effects.keys())
            available_actions = [a for a in base_actions if a not in self.ineffective_actions]
            if not available_actions and base_actions:
                print("--- All actions were ineffective. Resetting list. ---")
                self.ineffective_actions.clear()
                available_actions = base_actions

            # 2. Categorize actions based on the state graph.
            novel_actions = []
            boring_actions = []
            known_transitions = self.state_graph.get(grid_tuple, {})

            for act in available_actions:
                if act not in known_transitions:
                    # This action has not been tried from this specific grid state. It's novel.
                    novel_actions.append(act)
                else:
                    # We know where this action leads. We'll consider it "boring" to prioritize novelty.
                    boring_actions.append(act)

            print(f"Novel actions from this state: {[a.name for a in novel_actions]}")
            print(f"Boring actions from this state: {[a.name for a in boring_actions]}")

            # 3. Prioritize novel actions to maximize discovery.
            if novel_actions:
                action = random.choice(novel_actions)
            elif boring_actions:
                action = random.choice(boring_actions)
            else:
                action = GameAction.NOOP # Fallback if no actions are available

        self.last_grid_tuple = grid_tuple
        self.last_action = action
        return action

    # --- Methods from your original plan ---

    def perceive(self, latest_frame: FrameData) -> tuple[bool, bool, list[str], list]:
        """Compares frames, separating novel changes from known indicator changes."""
        current_frame = latest_frame.frame
        novel_changes_found = False
        known_changes_found = False
        novel_change_descriptions = []
        all_structured_changes = []

        if not current_frame:
            return False, False, [], []

        if self.previous_frame is None:
            self.previous_frame = copy.deepcopy(current_frame)
            return False, False, [], []

        # The "grid" is a list containing one list of rows. The number of rows is the "width".
        num_rows = len(current_frame[0])
        prev_num_rows = len(self.previous_frame[0])

        if num_rows != prev_num_rows:
            print("--- Frame dimensions changed (number of rows)! Analyzing... ---")
            self.previous_frame = copy.deepcopy(current_frame)
            return True, False, ["Frame dimensions changed"], []

        # Iterate through the list of rows. The "coordinate" is just the row_index.
        for row_index in range(num_rows):
            old_row_data = self.previous_frame[0][row_index]
            new_row_data = current_frame[0][row_index]

            if old_row_data != new_row_data:
                # First, create structured data for ALL changes.
                pixel_level_changes = []
                if len(old_row_data) == len(new_row_data):
                    for i in range(len(old_row_data)):
                        if old_row_data[i] != new_row_data[i]:
                            pixel_level_changes.append({'index': i, 'from': old_row_data[i], 'to': new_row_data[i]})
                if pixel_level_changes:
                    all_structured_changes.append({'row_index': row_index, 'changes': pixel_level_changes})

                # Second, check if it's a known indicator change or a novel one.
                if self.confirmed_resource_indicator and row_index == self.confirmed_resource_indicator['row_index']:
                    known_changes_found = True
                else:
                    novel_changes_found = True
                    novel_change_descriptions.append(f"  - Changes at row {row_index}:")
                    if len(old_row_data) == len(new_row_data):
                        for change in pixel_level_changes:
                            novel_change_descriptions.append(f"    - Pixel {change['index']}: From {change['from']} to {change['to']}")
                    else:
                        novel_change_descriptions.append(f"    - Data lists changed length.")

        self.previous_frame = copy.deepcopy(current_frame)
        return novel_changes_found, known_changes_found, novel_change_descriptions, all_structured_changes

    def _find_and_describe_objects(self, structured_changes: list, latest_frame: list) -> list[dict]:
        """Finds objects by grouping changed pixels by their new color first, then clustering."""
        changed_coords = set()
        for change in structured_changes:
            row_idx = change['row_index']
            for pixel_change in change['changes']:
                changed_coords.add((row_idx, pixel_change['index']))
        
        if not structured_changes:
            return []

        # 1. Group all changed points by their new color
        points_by_color = {}
        for change in structured_changes:
            row_idx = change['row_index']
            for pixel_change in change['changes']:
                new_color = latest_frame[0][row_idx][pixel_change['index']]
                if new_color not in points_by_color:
                    points_by_color[new_color] = set()
                points_by_color[new_color].add((row_idx, pixel_change['index']))

        # 2. Run flood-fill on each color group to find monochromatic object parts
        monochromatic_parts = []
        for color, points in points_by_color.items():
            visited = set()
            for point in points:
                if point not in visited:
                    component_points = set()
                    q = [point]
                    visited.add(point)
                    while q:
                        p = q.pop(0)
                        component_points.add(p)
                        r, p_idx = p
                        for dr, dp_idx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            neighbor = (r + dr, p_idx + dp_idx)
                            if neighbor in points and neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                    monochromatic_parts.append(component_points)

        # 3. For now, we treat each part as a separate object.
        # A future improvement could merge touching parts of different colors.
        final_objects = []
        grid = latest_frame[0]
        grid_height = len(grid)
        grid_width = len(grid[0]) if grid_height > 0 else 0

        for obj_points in monochromatic_parts:
            if not obj_points: continue

            # --- Background Verification Step ---
            # Check if this changed patch is connected to a static area of the same color.
            is_part_of_background = False
            sample_point = next(iter(obj_points))
            obj_color = grid[sample_point[0]][sample_point[1]]

            for r, p_idx in obj_points:
                for dr, dp_idx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (r + dr, p_idx + dp_idx)
                    
                    # Check if neighbor is within bounds, didn't change, and has the same color.
                    if (0 <= neighbor[0] < grid_height and
                        0 <= neighbor[1] < grid_width and
                        neighbor not in changed_coords and
                        grid[neighbor[0]][neighbor[1]] == obj_color):
                        
                        is_part_of_background = True
                        break
                if is_part_of_background:
                    break
            
            if is_part_of_background:
                # This event is a strong clue for what the floor is.
                log_message = f"🕵️‍♀️ A change at {sample_point} revealed what may be the background (Color: {obj_color})."

                # If the floor color is not yet known, this is a learning opportunity.
                if self.world_model['floor_color'] is None:
                    self.floor_hypothesis[obj_color] = self.floor_hypothesis.get(obj_color, 0) + 1
                    confidence = self.floor_hypothesis[obj_color]
                    print(f"🕵️‍♀️ Floor Hypothesis: A change at {sample_point} revealed color {obj_color} (Confidence: {confidence}).")

                    # Check for confirmation and print the one-time confirmation message.
                    if confidence >= self.CONCEPT_CONFIDENCE_THRESHOLD:
                        self.world_model['floor_color'] = obj_color
                        print(f"✅ [FLOOR] Confirmed: Color {obj_color} is the floor.")

                # If the floor is already known, log any event where it is revealed again.
                elif obj_color == self.world_model['floor_color']:
                    print(f"🕵️‍♀️ [FLOOR]: A change at {sample_point} revealed the known floor color ({obj_color}).")

                continue # Always skip creating an object from this background change.

            # --- If it's a real object, proceed with description ---
            min_row = min(r for r, _ in obj_points)
            max_row = max(r for r, _ in obj_points)
            min_idx = min(p_idx for _, p_idx in obj_points)
            max_idx = max(p_idx for _, p_idx in obj_points)
            height, width = max_row - min_row + 1, max_idx - min_idx + 1

            # The background check from the last update is still useful for context.
            border_colors = []
            background_color = None
            border_min_row = max(0, min_row - 1)
            border_max_row = min(grid_height - 1, max_row + 1)
            border_min_idx = max(0, min_idx - 1)
            border_max_idx = min(grid_width - 1, max_idx + 1)

            for r in range(border_min_row, border_max_row + 1):
                for p_idx in range(border_min_idx, border_max_idx + 1):
                    is_on_border = (r < min_row or r > max_row or p_idx < min_idx or p_idx > max_idx)
                    if is_on_border:
                        border_colors.append(grid[r][p_idx])
            
            if border_colors and all(c == border_colors[0] for c in border_colors):
                background_color = border_colors[0]

            data_map = tuple(tuple(latest_frame[0][r][p] for p in range(min_idx, max_idx + 1)) for r in range(min_row, max_row + 1))

            final_objects.append({
                'height': height, 'width': width, 'top_row': min_row,
                'left_index': min_idx, 'data_map': data_map,
                'background_color': background_color
            })

        return final_objects
    
    def _track_objects(self, current_objects: list, last_objects: list, latest_grid: list, structured_changes: list, action: GameAction) -> list[str]:
        """Compares current objects to last known objects to track movement and changes."""
        log_messages = []
        unmatched_current = list(current_objects)
        unmatched_last = list(last_objects)

        # --- Stage 1: Match by same position & shape (for Recolor) ---
        # This is the highest priority. We use a safe while loop to handle list modification.
        i = 0
        while i < len(unmatched_current):
            curr_obj = unmatched_current[i]
            match_found = False
            j = 0
            while j < len(unmatched_last):
                last_obj = unmatched_last[j]
                if (curr_obj['top_row'] == last_obj['top_row'] and
                    curr_obj['left_index'] == last_obj['left_index'] and
                    curr_obj['height'] == last_obj['height'] and
                    curr_obj['width'] == last_obj['width']):
                    
                    if curr_obj['data_map'] != last_obj['data_map']:
                        log_messages.append(f"🎨 RECOLOR: Object at ({curr_obj['top_row']}, {curr_obj['left_index']}) changed its data.")
                    
                    # Pair found. Remove both from their lists and stop searching for this curr_obj.
                    unmatched_current.pop(i)
                    unmatched_last.pop(j)
                    match_found = True
                    break # Exit the inner (j) loop
                else:
                    j += 1
            
            if not match_found:
                # If no match was found for curr_obj, move to the next one.
                i += 1

        # --- Stage 2: Move Detection & Composite Grouping ---
        move_matched_pairs = []
        i = 0
        while i < len(unmatched_current):
            curr_obj = unmatched_current[i]
            match_found = False
            j = 0
            while j < len(unmatched_last):
                last_obj = unmatched_last[j]
                if curr_obj['data_map'] == last_obj['data_map']:
                    move_matched_pairs.append((curr_obj, last_obj))
                    unmatched_current.pop(i)
                    unmatched_last.pop(j)
                    match_found = True
                    break
                else:
                    j += 1
            if not match_found:
                i += 1

        # --- Group moves by vector ---
        moves_by_vector = {}
        for curr, last in move_matched_pairs:
            vector = (curr['top_row'] - last['top_row'], curr['left_index'] - last['left_index'])
            if vector not in moves_by_vector:
                moves_by_vector[vector] = []
            moves_by_vector[vector].append((curr, last))

        # --- Analyze groups for composite objects and identify true moves ---
        processed_pairs = []
        true_moves = [] # Will store structured info about each move.

        for vector, pairs in moves_by_vector.items():
            if len(pairs) < 2: continue

            unclustered = list(pairs)
            while unclustered:
                cluster = [unclustered.pop(0)]
                while True:
                    new_neighbor_found = False
                    for neighbor_pair in list(unclustered):
                        is_adjacent = any(self._are_objects_adjacent(neighbor_pair[0], member_pair[0]) for member_pair in cluster)
                        if is_adjacent:
                            cluster.append(neighbor_pair)
                            unclustered.remove(neighbor_pair)
                            new_neighbor_found = True
                    if not new_neighbor_found:
                        break

                if len(cluster) > 1:
                    min_row = min(p[0]['top_row'] for p in cluster)
                    max_row = max(p[0]['top_row'] + p[0]['height'] for p in cluster)
                    min_col = min(p[0]['left_index'] for p in cluster)
                    max_col = max(p[0]['left_index'] + p[0]['width'] for p in cluster)

                    comp_h, comp_w = max_row - min_row, max_col - min_col
                    signature = (comp_h, comp_w)
                    # Generate a simpler log if the moved object is the confirmed agent.
                    if signature == self.world_model.get('player_signature'):
                        log_messages.append(f"🧠 [AGENT] moved by vector {vector}.")
                    else:
                        log_messages.append(f"🧠 COMPOSITE MOVE: Object [{comp_h}x{comp_w}] moved by vector {vector}.")
                true_moves.append({'type': 'composite', 'signature': signature, 'parts': cluster, 'vector': vector})
                for pair in cluster: processed_pairs.append(pair)

        # Log individual moves and collect their structured info.
        for curr, last in move_matched_pairs:
            if (curr, last) not in processed_pairs:
                signature = (curr['height'], curr['width'])
                # Generate a simpler log if the moved object is the confirmed agent.
                if signature == self.world_model.get('player_signature'):
                    log_messages.append(f"🧠 [AGENT] moved from ({last['top_row']}, {last['left_index']}) to ({curr['top_row']}, {curr['left_index']}).")
                else:
                    log_messages.append(f"🧠 MOVE: Object [{curr['height']}x{curr['width']}] moved from ({last['top_row']}, {last['left_index']}) to ({curr['top_row']}, {curr['left_index']}).")

                vector = (curr['top_row'] - last['top_row'], curr['left_index'] - last['left_index'])
                true_moves.append({'type': 'individual', 'signature': signature, 'last_obj': last, 'vector': vector})

        # --- Concept Learning from Movement (Agent and Floor) ---
        if not true_moves: return log_messages # No moves, nothing to learn.

        # 1. Identify the Agent
        if self.world_model['player_signature'] is None:
            for move in true_moves:
                signature = move['signature']
                self.agent_move_hypothesis[signature] = self.agent_move_hypothesis.get(signature, 0) + 1
                confidence = self.agent_move_hypothesis[signature]
                log_messages.append(f"🕵️‍♂️ Agent Hypothesis: Signature {signature} has moved {confidence} time(s).")

                if confidence >= self.CONCEPT_CONFIDENCE_THRESHOLD:
                    self.world_model['player_signature'] = signature
                    log_messages.append(f"✅ Confirmed Agent Signature: {signature}.")
                    break # Stop after confirming.

        # 2. Identify the Floor (can only happen after agent is known)
        if self.world_model['player_signature'] is not None and self.world_model['floor_color'] is None:
            for move in true_moves:
                if move['signature'] == self.world_model['player_signature']:
                    # The agent moved. Find the color of the ground it was on.
                    background_colors = set()
                    if move['type'] == 'individual':
                        color = move['last_obj'].get('background_color')
                        if color is not None: background_colors.add(color)
                    else: # Composite move
                        for _, last_part in move['parts']:
                            color = last_part.get('background_color')
                            if color is not None: background_colors.add(color)

                    # If we found one consistent background color for all parts, it's a candidate.
                    if len(background_colors) == 1:
                        floor_candidate = background_colors.pop()
                        self.floor_hypothesis[floor_candidate] = self.floor_hypothesis.get(floor_candidate, 0) + 1
                        confidence = self.floor_hypothesis[floor_candidate]
                        log_messages.append(f"🕵️‍♀️ Floor Hypothesis: Color {floor_candidate} is a candidate (Confidence: {confidence}).")

                        if confidence >= self.CONCEPT_CONFIDENCE_THRESHOLD:
                            self.world_model['floor_color'] = floor_candidate
                            log_messages.append(f"✅ Confirmed Floor Color: {floor_candidate}.")
                            break # Stop after confirming.

        # --- Action Effect Learning ---
        # Learn how actions affect the agent.
        if self.world_model['player_signature'] is not None and action:
            for move in true_moves:
                if move['signature'] == self.world_model['player_signature']:
                    # The agent moved. This is an effect of the last action.
                    effect_vector = move['vector']

                    # We only learn about actions that aren't already confirmed.
                    if action not in self.world_model['action_map']:
                        # Initialize hypothesis dict for this action if it doesn't exist.
                        if action not in self.action_effect_hypothesis:
                            self.action_effect_hypothesis[action] = {}

                        hypo_dict = self.action_effect_hypothesis[action]
                        hypo_dict[effect_vector] = hypo_dict.get(effect_vector, 0) + 1
                        confidence = hypo_dict[effect_vector]

                        log_messages.append(f"🕵️‍♀️ Action Hypothesis: {action.name} -> move by {effect_vector} (Confidence: {confidence}).")

                        if confidence >= self.CONCEPT_CONFIDENCE_THRESHOLD:
                            self.world_model['action_map'][action] = {'move_vector': effect_vector}
                            log_messages.append(f"✅ Confirmed Action Effect: {action.name} consistently moves the agent by vector {effect_vector}.")
                            del self.action_effect_hypothesis[action] # Clean up memory

                    break # Agent's move found, no need to check other moves.

        return log_messages
    
    def _are_objects_adjacent(self, obj1: dict, obj2: dict) -> bool:
        """Checks if two objects' bounding boxes are touching."""
        # Check for horizontal adjacency
        if obj1['top_row'] == obj2['top_row'] and obj1['height'] == obj2['height']:
            if obj1['left_index'] + obj1['width'] == obj2['left_index'] or \
               obj2['left_index'] + obj2['width'] == obj1['left_index']:
                return True
        # Check for vertical adjacency
        if obj1['left_index'] == obj2['left_index'] and obj1['width'] == obj2['width']:
            if obj1['top_row'] + obj1['height'] == obj2['top_row'] or \
               obj2['top_row'] + obj2['height'] == obj1['top_row']:
                return True
        return False
    
    def _update_resource_indicator_tracking(self, structured_changes: list, action: GameAction):
        """Analyzes changes to find a resource indicator, which depletes on any action."""
        if self.confirmed_resource_indicator or not action:
            return

        changed_rows = {change['row_index'] for change in structured_changes}

        # 1. Prune candidates that were expected to change but didn't.
        keys_to_remove = []
        for row_idx in self.resource_indicator_candidates:
            if row_idx not in changed_rows:
                print(f"📉 Candidate at row {row_idx} was inconsistent (did not change), removing.")
                keys_to_remove.append(row_idx)

        for key in keys_to_remove:
            del self.resource_indicator_candidates[key]

        # 2. Check all changes for potential indicator patterns
        for change in structured_changes:
            row_idx = change['row_index']
            if len(change['changes']) != 1:
                continue

            detail = change['changes'][0]
            current_index, old_val, new_val = detail['index'], detail['from'], detail['to']

            if not isinstance(new_val, (int, float)) or not isinstance(old_val, (int, float)):
                continue

            value_direction = 'inc' if new_val > old_val else 'dec'

            if row_idx in self.resource_indicator_candidates:
                candidate = self.resource_indicator_candidates[row_idx]

                if candidate['value_direction'] == value_direction:
                    index_direction = 'inc' if current_index > candidate['last_index'] else 'dec'

                    if candidate.get('index_direction') is None:
                        candidate['index_direction'] = index_direction
                        candidate['confidence'] += 1
                    elif candidate['index_direction'] == index_direction:
                        candidate['confidence'] += 1
                    else:
                        del self.resource_indicator_candidates[row_idx]
                        continue

                    candidate['last_index'] = current_index
                    print(f"📈 Resource candidate at row {row_idx} confidence is now {candidate['confidence']}.")

                    if candidate['confidence'] >= self.RESOURCE_CONFIDENCE_THRESHOLD:
                        self.confirmed_resource_indicator = {'row_index': row_idx, **candidate}
                        self.ignore_for_state_hash.add(row_idx)
                        print(f"✅ Confirmed resource indicator at row {row_idx}! It will now be ignored for state uniqueness checks.")
                        self.resource_indicator_candidates.clear()
                        return
                else:
                    del self.resource_indicator_candidates[row_idx]
            else:
                self.resource_indicator_candidates[row_idx] = {
                    'confidence': 1,
                    'last_index': current_index,
                    'value_direction': value_direction,
                    'index_direction': None
                }
                print(f"🤔 New resource candidate found at row {row_idx}.")

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