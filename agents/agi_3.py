# ARC-AGI-3 Main Script
import random
import copy
import math
import heapq
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

class CellType(Enum):
    """Represents the classification of a single grid cell."""
    UNKNOWN = 0
    FLOOR = 1
    WALL = 2
    POTENTIALLY_INTERACTABLE = 3 # For identified but not fully understood objects
    PLAYER = 4
    CONFIRMED_INTERACTABLE = 5
    RESOURCE = 6

class ExplorationPhase(Enum):
    """Manages the agent's goal-oriented exploration strategy."""
    INACTIVE = 0
    BUILDING_MAP = 1
    SEEKING_TARGET = 2
    EXECUTING_PLAN = 3

class AgentState(Enum):
    """Represents the agent's current operational state."""
    DISCOVERY = 1
    RANDOM_ACTION = 2
    AWAITING_STABILITY = 3

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
        self.ignored_areas = []
        self.state_graph = {} # Stores stateA -> action -> stateB
        self.last_grid_tuple = None

        # --- State Management ---
        self.agent_state = AgentState.DISCOVERY
        self.MASSIVE_CHANGE_THRESHOLD = 4000 # Num changes to trigger wait
        self.discovery_runs = 0
        self.last_action = None
        self.action_effects = {} # Will store actions and all their resulting changes
        self.action_failures = {}
        self.ineffective_actions = [] # Tracks actions that had no effect since the last success
        self.level_start_frame = None
        self.level_start_score = 0
        self.resource_indicator_candidates = {}
        self.confirmed_resource_indicator = None
        self.resource_bar_full_state = None
        self.resource_bar_empty_state = None
        self.RESOURCE_CONFIDENCE_THRESHOLD = 3 # Actions in a row to confirm
        self.level_knowledge_is_learned = False
        self.wait_action = GameAction.ACTION6 # Use a secondary action for waiting

        # --- Move Tracking ---
        self.max_moves = 0
        self.current_moves = 0
        self.resource_pixel_color = None
        self.resource_empty_color = None
        self.resource_bar_indices = []

        # --- Goal State Tracking ---
        self.active_patterns = [] # Stores a list of currently active patterns on the grid.

        # --- Object & Shape Tracking ---
        self.observed_object_shapes = {} # Maps shape tuple -> count
        self.last_known_objects = [] # Stores full object descriptions from the last frame
        self.world_model = {
            'player_signature': None,
            'floor_color': None,
            'wall_colors': set(), # Use a set for multiple possible wall colors
            'action_map': {} # Will store confirmed action -> effect mappings
        }
        self.world_model['life_indicator_object'] = None
        self.player_floor_hypothesis = {}
        self.agent_move_hypothesis = {} # Tracks how many times a shape has moved
        self.floor_hypothesis = {} # Tracks how many times a color has been identified as floor
        self.action_effect_hypothesis = {} # Tracks action -> effect hypotheses
        self.wall_hypothesis = {} # Tracks wall color candidates
        self.last_known_player_obj = None # Stores the full player object from the last frame
        self.CONCEPT_CONFIDENCE_THRESHOLD = 3 # Number of times a pattern must be seen to be learned

        # --- Exploration & Pathfinding ---
        self.exploration_phase = ExplorationPhase.INACTIVE
        self.tile_map = {} # Stores (tile_x, tile_y) -> CellType for the macro grid
        self.tile_size = None # Stores the grid size (e.g., 8)
        self.exploration_target = None # Stores (row, col) of the current target
        self.exploration_plan = [] # Stores a list of GameActions to execute
        self.inverse_action_map = {} # e.g., {(0, 1): GameAction.RIGHT} for pathfinding
        self.reachable_floor_area = set() # Stores all floor tiles connected to the player
        self.just_vacated_tile = None

        # --- Interaction Learning ---
        self.observing_interaction_for_tile = None # Stores the coords of the tile being observed
        self.interaction_hypotheses = {} # signature -> {'immediate_effect': [], 'aftermath_effect': [], 'confidence': 0}
        self.static_level_objects = []
        self.has_summarized_interactions = False
        self.awaiting_final_summary = False
        self.final_tile_of_level = None
        self.level_goal_hypotheses = []
        self.consumed_tiles_this_life = set()

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

    def _get_grid_state_tuple(self, frame_data: list) -> tuple:
        """Creates a hashable tuple of the grid, ignoring specific rectangular areas."""
        if not frame_data or not frame_data[0]:
            return tuple()

        # Create a mutable copy of the grid to modify.
        grid_copy = [list(row) for row in frame_data[0]]

        # "Neutralize" the pixels within each ignored rectangle.
        for area in self.ignored_areas:
            start_row, end_row = area['top_row'], area['top_row'] + area['height']
            start_col, end_col = area['left_index'], area['left_index'] + area['width']

            for r in range(start_row, end_row):
                if 0 <= r < len(grid_copy):
                    for c in range(start_col, end_col):
                        if 0 <= c < len(grid_copy[r]):
                            grid_copy[r][c] = -1 # Use a neutral "ignored" value.
        
        # Convert the modified grid to an immutable, hashable tuple.
        return tuple(tuple(row) for row in grid_copy)

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
        # --- Core Resets for any attempt ---
        self.previous_frame = None
        self.level_start_frame = None
        self.last_action = None
        self.last_known_objects = []
        self.last_known_player_obj = None # Crucial: Force agent to re-find itself.

        # --- Reset Level-Specific Layout and Plans ---
        self.has_summarized_interactions = False
        self.awaiting_final_summary = False
        self.exploration_phase = ExplorationPhase.INACTIVE
        self.exploration_plan = []
        self.exploration_target = None
        self.observing_interaction_for_tile = None
        self.active_patterns.clear()
        self.consumed_tiles_this_life.clear()

        # --- Reset Agent's Action State ---
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

        # --- NEW: Refresh consumable objects on the map for the new life ---
        print("🔄 Refreshing knowledge of consumable objects for new life.")
        refreshed_count = 0
        for signature, hypothesis in self.interaction_hypotheses.items():
            if hypothesis.get('is_consumable') is True:
                # Extract tile coordinates from a signature like "tile_pos_(10, 5)"
                try:
                    tile_str = signature.replace("tile_pos_(", "").replace(")", "")
                    tile_coords = tuple(map(int, tile_str.split(',')))
                    
                    # If this tile was marked as floor, restore its interactable status.
                    if self.tile_map.get(tile_coords) == CellType.FLOOR:
                        # If we know it provides a resource, mark it as such. Otherwise, CONFIRMED.
                        if hypothesis.get('provides_resource'):
                            self.tile_map[tile_coords] = CellType.RESOURCE
                        else:
                            self.tile_map[tile_coords] = CellType.CONFIRMED_INTERACTABLE
                        refreshed_count += 1
                except (ValueError, IndexError):
                    continue # Signature was not in the expected format
        if refreshed_count > 0:
            print(f"-> Refreshed {refreshed_count} consumed tile(s) back to an interactable state.")

    def _reset_for_new_level(self):
        """Resets all level-specific knowledge for a new level, preserving core learned concepts."""
        # Record the agent's last position to correctly label the summary.
        if self.last_known_player_obj and self.tile_size:
            self.final_tile_of_level = (self.last_known_player_obj['top_row'] // self.tile_size, self.last_known_player_obj['left_index'] // self.tile_size)

        # --- NEW: Print the summary of the level that was just completed ---
        # Check if a summary wasn't already printed at the end of the level.
        if not self.has_summarized_interactions:
            print("\n--- Final Summary of Previous Level ---")
            self._review_and_summarize_interactions()

        # A new level is a more thorough version of a new attempt.
        self._reset_for_new_attempt()

        # Additionally, reset knowledge that is strictly tied to a level's design.
        print("🧹 Wiping interaction hypotheses and active patterns for the new level.")
        self.interaction_hypotheses.clear()
        self.active_patterns.clear()
        self.has_summarized_interactions = False

        # Since the world model is preserved, we can skip discovery.
        print("🧠 Knowledge preserved. Skipping discovery and entering action state.")
        self.agent_state = AgentState.RANDOM_ACTION

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """This is the main decision-making method for the AGI."""
        frame_before_perception = copy.deepcopy(self.previous_frame)
        # --- 1. Store initial level state if not already set ---
        if self.level_start_frame is None:
            # Only store the initial frame if it actually contains data.
            if latest_frame.frame:
                print("--- New Level Detected. Storing initial valid frame and score. ---")
                self.level_start_frame = copy.deepcopy(latest_frame.frame)
                self.level_start_score = latest_frame.score
                if self.confirmed_resource_indicator:
                    indicator_row_index = self.confirmed_resource_indicator['row_index']
                    self.resource_bar_full_state = copy.deepcopy(latest_frame.frame[0][indicator_row_index])
                    print(f"-> Captured the 'full' state of the resource bar at row {indicator_row_index}.")
            else:
                # If the frame is blank, print a message but do nothing else.
                # This allows the normal action-selection logic below to run,
                # preventing the VALIDATION_ERROR. We'll try to store the frame
                # again on the next turn.
                print("--- Ignoring blank starting frame. Waiting for a valid one... ---")
        
        # --- Handle screen transitions before any other logic ---
        if self.agent_state == AgentState.AWAITING_STABILITY:
            # We perceive here to check if the screen has stopped changing.
            novel_changes_found, _, _, _ = self.perceive(latest_frame)

            # If there are NO changes from the last frame, the screen is stable.
            if not novel_changes_found:
                print("✅ Screen is stable (no change from last frame). Analyzing outcome...")
                new_score = latest_frame.score

                if new_score > self.level_start_score:
                    print(f"--- New Level Detected! Score increased to {new_score}. ---")
                    self.level_start_frame = copy.deepcopy(latest_frame.frame)
                    self.level_start_score = new_score
                    self._reset_for_new_level() 
                    return self.wait_action
                else:
                    print("--- Lost a Life (Score did not increase). Analyzing reset state... ---")
                    if self.level_start_frame is not None:
                        print("-> Comparing current grid with the grid from the start of the level...")

                        # Safety check to prevent crashing on an empty frame
                        if not self.level_start_frame or not latest_frame.frame:
                            print("-> Analysis skipped: A frame needed for comparison is empty.")
                        else:
                            # This logic is now protected from the crash
                            grid_start = self.level_start_frame[0]
                            grid_current = latest_frame.frame[0]
                            
                            # Step 1: Collect all pixel changes into a structured format.
                            structured_life_changes = []
                            if len(grid_start) == len(grid_current):
                                for i in range(len(grid_start)):
                                    row_start, row_current = grid_start[i], grid_current[i]
                                    if row_start != row_current and len(row_start) == len(row_current):
                                        pixel_changes = [{'index': j, 'from': row_start[j], 'to': row_current[j]} for j in range(len(row_start)) if row_start[j] != row_current[j]]
                                        if pixel_changes:
                                            structured_life_changes.append({'row_index': i, 'changes': pixel_changes})

                            # Step 2: Use the agent's own object finder on these changes.
                            if structured_life_changes:
                                print("-> Analysis result: Found differences from the level start.")
                                life_indicator_objects = self._find_and_describe_objects(structured_life_changes, latest_frame.frame)
                                
                                if life_indicator_objects:
                                    print("-> These changes form the following object(s):")
                                    for i, obj in enumerate(life_indicator_objects):
                                        pos = (obj['top_row'], obj['left_index'])
                                        size = (obj['height'], obj['width'])
                                        tile_pos = (pos[0] // self.tile_size, pos[1] // self.tile_size)
                                        print(f"  - Object {i+1}: A {size[0]}x{size[1]} object at pixel {pos} (tile {tile_pos}).")

                                        # --- Life Indicator Learning Logic ---
                                        # If the indicator isn't known, learn it from this first observation.
                                        if not self.world_model.get('life_indicator_object'):
                                            old_color = obj.get('original_color')
                                            new_color = obj['color']
                                            
                                            if old_color is not None:
                                                # The signature includes the full color transition.
                                                signature = (obj['height'], obj['width'], old_color, new_color)
                                                
                                                # Immediately confirm the concept and store it.
                                                self.world_model['life_indicator_object'] = signature
                                                print(f"✅ [LIFE INDICATOR] Confirmed: A color change from {signature[2]} to {signature[3]} on a ({signature[0]}x{signature[1]}) object is the life indicator.")
                                                
                                                # Add the object's specific rectangle to the ignore list.
                                                new_area = {
                                                    'top_row': obj['top_row'],
                                                    'left_index': obj['left_index'],
                                                    'height': obj['height'],
                                                    'width': obj['width']
                                                }
                                                self.ignored_areas.append(new_area)
                                                print(f"-> Ignoring the {new_area['height']}x{new_area['width']} area at ({new_area['top_row']}, {new_area['left_index']}) for uniqueness checks.")
                                                    
                                else:
                                    print("-> The changes did not form a distinct object (likely a background reveal).")
                            else:
                                print("-> Analysis result: The grid has returned to the exact visual state as the start of the level.")

                    else:
                        print("-> Could not perform analysis: The start-of-level frame was not stored.")
                    
                    self._reset_for_new_attempt()
                    # After handling the transition, wait for the next turn to act.
                    return self.wait_action
            else:
                # If the screen is still changing, just wait.
                return self.wait_action
            
        agent_part_fingerprints = set()
        novel_changes_found, known_changes_found, change_descriptions, structured_changes = self.perceive(latest_frame)    
        
        # --- Update and display move count ---
        self._analyze_and_update_moves(latest_frame)
        if self.max_moves > 0:
            print(f"-> Moves Remaining: {self.current_moves}/{self.max_moves}")

        # --- NEW: Check for and analyze the "aftermath" of an interaction ---
        if self.observing_interaction_for_tile is not None:
            # Get the agent's CURRENT tile position to see if it successfully moved.
            current_player_tile = None
            if self.last_known_player_obj and self.tile_size:
                current_player_tile = (self.last_known_player_obj['top_row'] // self.tile_size, self.last_known_player_obj['left_index'] // self.tile_size)

            # If the player is still on the tile we're observing, their move failed.
            if current_player_tile == self.observing_interaction_for_tile:
                print(f"-> Aftermath check for tile {self.observing_interaction_for_tile} paused: Agent's move failed.")
                # We do nothing else; the flag remains, and we'll re-check after the next move attempt.
            else:
                # The agent has successfully moved away. Now we can analyze the aftermath.
                print("-> Stepped away from observed tile. Analyzing aftermath...")
                self._analyze_consumable_aftermath(latest_frame.frame)

                # End the full observation cycle and return to normal exploration.
                self.observing_interaction_for_tile = None
                self.exploration_phase = ExplorationPhase.BUILDING_MAP # Resume normal exploration
            

        # --- Trigger queued summary AFTER aftermath is processed ---
        if self.awaiting_final_summary:
            self._review_and_summarize_interactions()
            self.awaiting_final_summary = False # Reset flag
            self.exploration_phase = ExplorationPhase.INACTIVE # Officially end exploration
        
        # --- State Graph Update ---
        # If we have a previous state and an action that led to the current state,
        # update the state graph to record the transition.
        grid_tuple = self._get_grid_state_tuple(latest_frame.frame)
        if self.last_grid_tuple is not None and self.last_action is not None:
            if self.last_grid_tuple not in self.state_graph:
                self.state_graph[self.last_grid_tuple] = {}
            self.state_graph[self.last_grid_tuple][self.last_action] = grid_tuple

        # ---

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
        # Create a hashable representation of the grid for state graph tracking.
        grid_tuple = self._get_grid_state_tuple(latest_frame.frame)

        # Create a hashable representation of the grid for state graph tracking.
        # We already did this above for the state graph, so we just use the variable.
        
        # Check if the current state is one we've seen before.
        if grid_tuple in self.visited_grids:
            print(f"-> State already visited ({len(self.visited_grids)} unique states known).")
        else:
            print(f"✅ New unique state discovered! ({len(self.visited_grids) + 1} total)")
            self.visited_grids.add(grid_tuple)

        # Check for massive changes indicating a transition
        if novel_changes_found:
            is_dimension_change = "Frame dimensions changed" in change_descriptions
            if len(change_descriptions) > self.MASSIVE_CHANGE_THRESHOLD or is_dimension_change:
                print(f"💥 Massive change detected ({len(change_descriptions)} changes). Likely a level transition.")

                # --- Capture the frame BEFORE the massive change as the potential 'empty' state ---
                if self.confirmed_resource_indicator and frame_before_perception:
                    # Check if score has NOT increased, indicating a death, not a win.
                    if latest_frame.score <= self.level_start_score:
                        indicator_row_index = self.confirmed_resource_indicator['row_index']
                        # frame_before_perception holds the grid state right before the death animation.
                        self.resource_bar_empty_state = copy.deepcopy(frame_before_perception[0][indicator_row_index])
                        print(f"-> Captured potential 'empty' state of the resource bar from the frame before transition.")

                print("-> Waiting for stability...")
                self.agent_state = AgentState.AWAITING_STABILITY
                self.stability_counter = 0
                self.previous_frame = None # Invalidate frame to ensure fresh perception after stability
                return self.wait_action

        # Process the result of the last action, but only if it wasn't a wait action.
        if self.last_action and self.last_action is not self.wait_action:
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
                # --- An action failed to produce a novel change ---
                # If this was a KNOWN action, it's a learning opportunity.
                if self.last_action in self.world_model['action_map']:
                    self._learn_from_interaction_failure(self.last_action, self.previous_frame)

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
                for change in structured_changes:
                    if change['row_index'] != indicator_row:
                        object_logic_changes.append(change)
            else:
                object_logic_changes = structured_changes

            # 1. Find all new objects based on pixel changes.
            current_objects = self._find_and_describe_objects(object_logic_changes, latest_frame.frame)

            # NEW: Store the tile the agent is on before tracking its move.
            self.just_vacated_tile = None
            if self.last_known_player_obj and self.tile_size:
                self.just_vacated_tile = (self.last_known_player_obj['top_row'] // self.tile_size, self.last_known_player_obj['left_index'] // self.tile_size)

            moved_agent_this_turn = None
            
            # 2. Track objects and identify the agent's parts FIRST.
            if self.last_known_objects:
                tracking_logs, moved_agent_this_turn, agent_part_fingerprints = self._track_objects(current_objects, self.last_known_objects, latest_frame.frame, object_logic_changes, self.last_action)
                if tracking_logs:
                    print(f"--- Object Tracking Report (Action: {self.last_action.name}) ---")
                    for log in tracking_logs:
                        print(log)

            # 3. Filter out agent parts to get a list of true environmental changes.
            non_agent_objects = [obj for obj in current_objects if obj.get('fingerprint') not in agent_part_fingerprints]
            if len(current_objects) != len(non_agent_objects):
                print(f"🕵️‍♂️ Goal check is ignoring {len(current_objects) - len(non_agent_objects)} object(s) identified as agent parts.")

            # 4. Check for goal patterns on the REMAINING non-agent objects.
            if non_agent_objects:
                self._find_and_update_patterns(non_agent_objects, latest_frame.frame)
            if self.active_patterns:
                print(f"--- {len(self.active_patterns)} Active Pattern(s) on Grid ---")
                for i, pattern in enumerate(self.active_patterns):
                    dk = pattern['dynamic_key']
                    sk = pattern['static_key']
                    dk_tile = (dk['top_row'] // self.tile_size, dk['left_index'] // self.tile_size) if self.tile_size else (dk['top_row'], dk['left_index'])
                    sk_tile = (sk['top_row'] // self.tile_size, sk['left_index'] // self.tile_size) if self.tile_size else (sk['top_row'], sk['left_index'])
                    print(f"  - Pattern {i+1}: Dynamic object at {dk_tile} matches static object at {sk_tile}.")
           
            # 5. Update memory for the next turn.
            self.last_known_objects = current_objects

            # 6. Update the player object's known position for the next turn.
            if moved_agent_this_turn:
                self.last_known_player_obj = moved_agent_this_turn

                        # --- NEW: Check for resource refill events ---
            resource_event = None
            if self.confirmed_resource_indicator:
                indicator_row = self.confirmed_resource_indicator['row_index']
                for change in structured_changes:
                    if change['row_index'] == indicator_row:
                        for pixel in change['changes']:
                            if isinstance(pixel.get('to'), (int, float)) and isinstance(pixel.get('from'), (int, float)):
                                if pixel['to'] > pixel['from']:
                                    resource_event = "REFILLED"
                                    break
                    if resource_event:
                        break
            
            if resource_event == "REFILLED":
                print(f"✅ [RESOURCE] Resource bar was refilled!")
                if self.last_known_player_obj and self.tile_size:
                    player_tile = (self.last_known_player_obj['top_row'] // self.tile_size, self.last_known_player_obj['left_index'] // self.tile_size)
                    print(f"-> Agent is on tile {player_tile}. Classifying as a resource.")
                    self.tile_map[player_tile] = CellType.RESOURCE
                    # Update interaction hypothesis for this tile
                    signature = f"tile_pos_{player_tile}"
                    if signature not in self.interaction_hypotheses:
                        self.interaction_hypotheses[signature] = {'immediate_effect': [], 'aftermath_effect': [], 'confidence': 0}
                    self.interaction_hypotheses[signature]['provides_resource'] = True

        # --- NEW, SIMPLIFIED NEW-LEVEL DETECTION ---
        # If the score has increased at any point, assume it's a new level and reset.
        if latest_frame.score > self.level_start_score:
            print(f"--- New Level Detected (Score Increased)! Analyzing final move before reset. ---")
            
            # --- Manually trigger interaction analysis for the goal tile ---
            # This ensures we learn from the winning move before the summary is printed.
            if self.last_known_player_obj and self.tile_size:
                goal_tile = (self.last_known_player_obj['top_row'] // self.tile_size, self.last_known_player_obj['left_index'] // self.tile_size)
                print(f"-> Analyzing winning interaction with tile {goal_tile}...")
                self.observing_interaction_for_tile = goal_tile
                self._analyze_and_log_interaction_effect(structured_changes, 'immediate_effect', latest_frame.frame, self.last_known_objects, agent_part_fingerprints)
                self.observing_interaction_for_tile = None # Clear immediately after use.

            # Now that the final interaction is logged, proceed with the reset.
            self.level_start_frame = copy.deepcopy(latest_frame.frame)
            self.level_start_score = latest_frame.score
            self._summarize_level_goal()
            self._reset_for_new_level()
            return self.wait_action

        # --- Intelligent Exploration Logic ---
        can_explore = (self.world_model.get('player_signature') and
               self.world_model.get('floor_color') and
               self.world_model.get('action_map') and
               self.last_known_player_obj and
               self.tile_size) 

        if can_explore and self.exploration_phase == ExplorationPhase.INACTIVE:
            print("🤖 World model is sufficiently complete. Activating exploration phase.")
            self.exploration_phase = ExplorationPhase.BUILDING_MAP

        if self.exploration_phase != ExplorationPhase.INACTIVE:
            if self.exploration_phase == ExplorationPhase.EXECUTING_PLAN:
                if self.exploration_plan:
                    action = self.exploration_plan.pop(0)
                    print(f"🗺️ Executing plan: {action.name}. {len(self.exploration_plan)} steps remaining.")
                    self.last_action = action
                    self.last_grid_tuple = grid_tuple
                    return action
                else:
                    print("✅ Plan complete. Beginning interaction observation.")
                    self._print_debug_map()
                    if self.exploration_target and self.tile_size:
                        target_tile = (self.exploration_target[0] // self.tile_size, self.exploration_target[1] // self.tile_size)
                        if self.tile_map.get(target_tile) == CellType.POTENTIALLY_INTERACTABLE:
                            self.tile_map[target_tile] = CellType.CONFIRMED_INTERACTABLE
                            print(f"✅ Target at {target_tile} confirmed as interactable.")
                        self.observing_interaction_for_tile = target_tile
                        self._analyze_and_log_interaction_effect(structured_changes, 'immediate_effect', latest_frame.frame, self.last_known_objects, agent_part_fingerprints)
                    self.exploration_phase = ExplorationPhase.BUILDING_MAP

            if self.exploration_phase == ExplorationPhase.BUILDING_MAP:
                print("🗺️ Building/updating the level map...")
                self._build_level_map(latest_frame.frame)
                self.just_vacated_tile = None # Clear the one-time flag after use.
                self.exploration_phase = ExplorationPhase.SEEKING_TARGET

            if self.exploration_phase == ExplorationPhase.SEEKING_TARGET:
                print("🗺️ Seeking a new exploration target...")
                target_found = self._find_target_and_plan()
                if target_found:
                    target_tile_coords = (self.exploration_target[0] // self.tile_size, self.exploration_target[1] // self.tile_size)
                    print(f"🎯 New target acquired at pixel {self.exploration_target} (tile {target_tile_coords}). Plan created with {len(self.exploration_plan)} steps.")
                    self.exploration_phase = ExplorationPhase.EXECUTING_PLAN
                    # Execute the first step of the new plan immediately.
                    if self.exploration_plan:
                        action = self.exploration_plan.pop(0)
                        print(f"🗺️ Executing plan: {action.name}. {len(self.exploration_plan)} steps remaining.")
                        self.last_action = action
                        self.last_grid_tuple = grid_tuple
                        return action
                else:
                    # If we just finished exploring and queued a summary, we should pause.
                    # This allows the summary to print and the aftermath of the last action to be processed.
                    if self.awaiting_final_summary:
                        print("🧐 Pausing exploration to generate summary and process interaction aftermath.")
                        self.last_action = self.wait_action
                        self.last_grid_tuple = grid_tuple
                        return self.wait_action
                    else:
                        # If there are no targets AND a summary isn't pending, exploration is truly done.
                        print("🧐 No more targets found. Reverting to state graph exploration.")
                        self.exploration_phase = ExplorationPhase.INACTIVE
                        # Fall through to the default state graph action state

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
                action = self.wait_action # Fallback if no actions are available

        self.last_grid_tuple = grid_tuple
        self.last_action = action
        return action

    def _build_level_map(self, grid: list):
        """
        Creates a refined tile-based map, aware of the player's current and previous positions.
        """
        if not self.tile_size or not grid:
            return
        
        confirmed_interactables = {pos for pos, cell_type in self.tile_map.items() if cell_type == CellType.CONFIRMED_INTERACTABLE}

        temp_tile_map = {}
        grid_data = grid[0]
        grid_height = len(grid_data)
        grid_width = len(grid_data[0]) if grid_height > 0 else 0
        floor_color = self.world_model['floor_color']
        wall_colors = self.world_model['wall_colors']

        player_tile_coords = None
        if self.last_known_player_obj and self.tile_size:
            player_pixel_pos = (self.last_known_player_obj['top_row'], self.last_known_player_obj['left_index'])
            player_tile_coords = (player_pixel_pos[0] // self.tile_size, player_pixel_pos[1] // self.tile_size)

        for r in range(0, grid_height, self.tile_size):
            for c in range(0, grid_width, self.tile_size):
                tile_coords = (r // self.tile_size, c // self.tile_size)

                # --- NEW: Preserve confirmed knowledge ---
                # If we already have a confirmed interaction for this tile,
                # preserve its type and don't re-evaluate it based on color.
                existing_type = self.tile_map.get(tile_coords)
                if existing_type in [CellType.CONFIRMED_INTERACTABLE, CellType.RESOURCE]:
                    temp_tile_map[tile_coords] = existing_type
                    continue
                
                if tile_coords == player_tile_coords or tile_coords == self.just_vacated_tile:
                    if self.tile_map.get(tile_coords) == CellType.CONFIRMED_INTERACTABLE:
                        temp_tile_map[tile_coords] = CellType.CONFIRMED_INTERACTABLE
                    else:
                        temp_tile_map[tile_coords] = CellType.FLOOR
                    continue

               # Get all unique colors within the tile's bounds
                tile_pixels = [
                    grid_data[row][col]
                    for row in range(r, r + self.tile_size)
                    for col in range(c, c + self.tile_size)
                    if 0 <= row < grid_height and 0 <= col < grid_width
                ]
                
                unique_colors = set(tile_pixels)

                # A tile is only one type if all its pixels are that one color.
                if len(unique_colors) == 1:
                    single_color = unique_colors.pop()
                    if single_color == floor_color:
                        temp_tile_map[tile_coords] = CellType.FLOOR
                    elif single_color in wall_colors:
                        temp_tile_map[tile_coords] = CellType.WALL
                    else:
                        # It's a uniform tile of an unknown type
                        temp_tile_map[tile_coords] = CellType.POTENTIALLY_INTERACTABLE
                else:
                    # Mixed pixels mean it's definitely something to investigate
                    temp_tile_map[tile_coords] = CellType.POTENTIALLY_INTERACTABLE
        
        self.tile_map.update(temp_tile_map)

        if not self.reachable_floor_area:
            print("🗺️ No reachable area found. Map will not be refined.")

        refined_tile_map = {}
        for tile_coords in self.tile_map.keys():
            # --- FIX: Preserve any tile that has a known function ---
            # This prevents the refinement logic below from overwriting this knowledge.
            existing_type = self.tile_map.get(tile_coords)
            if existing_type in [CellType.CONFIRMED_INTERACTABLE, CellType.RESOURCE]:
                refined_tile_map[tile_coords] = existing_type
                continue

            is_in_or_adjacent_to_reachable = tile_coords in self.reachable_floor_area or any((tile_coords[0] + dr, tile_coords[1] + dc) in self.reachable_floor_area for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)])
            if is_in_or_adjacent_to_reachable:
                # Trust the initial, more thorough classification for adjacent tiles.
                # This prevents single-pixel errors during refinement.
                initial_type = temp_tile_map.get(tile_coords)
                if initial_type is not None:
                    refined_tile_map[tile_coords] = initial_type
            else:
                # Preserve the tile's existing type if it's not near the reachable area.
                # This prevents erasing memory of explored but currently unreachable parts of the map.
                if self.tile_map.get(tile_coords) is not None:
                    refined_tile_map[tile_coords] = self.tile_map.get(tile_coords)

        for tile_coords, cell_type in list(refined_tile_map.items()):
            if cell_type in [CellType.POTENTIALLY_INTERACTABLE, CellType.CONFIRMED_INTERACTABLE]:
                # --- HYBRID CLEANUP LOGIC ---
                # 1. Use a robust, full-tile check for Floors.
                tile_pixels = [
                    grid_data[row][col]
                    for row in range(tile_coords[0] * self.tile_size, (tile_coords[0] + 1) * self.tile_size)
                    for col in range(tile_coords[1] * self.tile_size, (tile_coords[1] + 1) * self.tile_size)
                    if 0 <= row < grid_height and 0 <= col < grid_width
                ]
                unique_colors = set(tile_pixels)
                is_uniform_floor = (len(unique_colors) == 1 and floor_color in unique_colors)

                if is_uniform_floor:
                    refined_tile_map[tile_coords] = CellType.FLOOR
                else:
                    # 2. If not floor, use the original, single-pixel check for Walls.
                    sample_color = grid_data[tile_coords[0] * self.tile_size][tile_coords[1] * self.tile_size]
                    if wall_colors and sample_color in wall_colors:
                        refined_tile_map[tile_coords] = CellType.WALL

        self.tile_map = refined_tile_map

        self.reachable_floor_area = self._find_reachable_floor_tiles()

        # For the log, summarize the composition of the *reachable* area for consistency.
        reachable_tile_types = [self.tile_map[tile] for tile in self.reachable_floor_area if tile in self.tile_map]
        counts = Counter(reachable_tile_types)
        total_reachable_tiles = len(self.reachable_floor_area)

        # Build a dynamic string for the counts to avoid printing zero-count categories.
        counts_parts = []
        if counts[CellType.FLOOR]:
            counts_parts.append(f"{counts[CellType.FLOOR]} floor")
        if counts[CellType.POTENTIALLY_INTERACTABLE]:
            counts_parts.append(f"{counts[CellType.POTENTIALLY_INTERACTABLE]} potential")
        if counts[CellType.CONFIRMED_INTERACTABLE]:
            counts_parts.append(f"{counts[CellType.CONFIRMED_INTERACTABLE]} interactable")
        if counts[CellType.RESOURCE]:
            counts_parts.append(f"{counts[CellType.RESOURCE]} resource")
        # Note: Walls should not be in the reachable area, but include as a safety check.
        if counts[CellType.WALL]:
            counts_parts.append(f"{counts[CellType.WALL]} wall")
            
        counts_str = ", ".join(counts_parts)
        
        # This new print statement is for our test.
        print(f"✅ TEST LOG ({total_reachable_tiles} reachable tiles): {counts_str}.")
    
    def _find_target_and_plan(self) -> bool:
        """
        Finds the best interactable tile by pathfinding to all available targets
        and picking the one with the shortest path.
        """

        self.exploration_target = None
        self.exploration_plan = []
        if not self.last_known_player_obj or not self.tile_size:
            return False

        player_pixel_pos = (self.last_known_player_obj['top_row'], self.last_known_player_obj['left_index'])
        player_tile_pos = (player_pixel_pos[0] // self.tile_size, player_pixel_pos[1] // self.tile_size)
        
        # --- PRIORITY 1: Explore all potentially interactable objects ---
        print("🎯 Activating PRIORITY 1: Seeking all potentially interactable tiles.")
        # First, calculate the visible "display area," just as the debug map does.
        display_area = set(self.reachable_floor_area)
        for r_tile, c_tile in self.reachable_floor_area:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r_tile + dr, c_tile + dc)
                if neighbor in self.tile_map:
                    display_area.add(neighbor)

        # Now, search for targets ONLY within this visible area.
        potential_targets = [pos for pos, type in self.tile_map.items() 
                            if type == CellType.POTENTIALLY_INTERACTABLE and pos in display_area]
        
        # Before planning, review potential targets for any pre-existing knowledge.
        known_interactables = []
        for tile_pos in potential_targets:
            signature = f"tile_pos_{tile_pos}"
            if signature in self.interaction_hypotheses:
                print(f"🧠 Pre-existing knowledge found for tile {tile_pos}. Reclassifying as CONFIRMED_INTERACTABLE.")
                self.tile_map[tile_pos] = CellType.CONFIRMED_INTERACTABLE
                known_interactables.append(tile_pos)
        
        # Remove the now-known interactables from the list of potential targets.
        potential_targets = [p for p in potential_targets if p not in known_interactables]
        
        # --- End of Priority 1 Logic ---

        if not potential_targets:
            # If there are no more potentially interactable tiles, queue the summary immediately.
            if not self.has_summarized_interactions and not self.awaiting_final_summary:
                    print("✅ All potential interactables discovered. Queuing knowledge summary for the next turn.")
                    self.awaiting_final_summary = True
            
            # --- PRIORITY 2: Test mysterious objects while a pattern is active ---
            if self.active_patterns:
                print("🎯 Activating PRIORITY 2: Pattern is active. Seeking interactables with no known effect.")
                
                interactables_with_no_effect = []
                all_confirmed_interactables = [pos for pos, type in self.tile_map.items() if type == CellType.CONFIRMED_INTERACTABLE]

                for tile_pos in all_confirmed_interactables:
                    if tile_pos in self.consumed_tiles_this_life:
                        continue
                    signature = f"tile_pos_{tile_pos}"
                    hypothesis = self.interaction_hypotheses.get(signature)
                    # "No effect observed" means a hypothesis exists, but it recorded no effects.
                    if hypothesis and not hypothesis.get('immediate_effect') and not hypothesis.get('aftermath_effect') and not hypothesis.get('provides_resource'):
                        interactables_with_no_effect.append(tile_pos)

                if interactables_with_no_effect:
                    print(f"-> Found {len(interactables_with_no_effect)} interactable(s) with no effect to re-test.")
                    # If we found some, they become our new potential targets for this turn.
                    potential_targets = interactables_with_no_effect
                else:
                    print("-> No mysterious objects to test. All interactables have an observed effect.")
            # --- End of Priority 2 Logic ---

            else:
                # --- PRIORITY 3: Use known tools to create a pattern ---
                print("🎯 Activating PRIORITY 3: No patterns active. Seeking interactables with a known function.")

                interactables_with_known_function = []
                all_confirmed_interactables = [pos for pos, type in self.tile_map.items() if type == CellType.CONFIRMED_INTERACTABLE]

                for tile_pos in all_confirmed_interactables:
                    if tile_pos in self.consumed_tiles_this_life:
                        continue
                    signature = f"tile_pos_{tile_pos}"
                    hypothesis = self.interaction_hypotheses.get(signature)
                    # "Known function" means a hypothesis exists and it recorded some kind of effect.
                    if hypothesis and (hypothesis.get('immediate_effect') or hypothesis.get('aftermath_effect')):
                        interactables_with_known_function.append(tile_pos)
                
                if interactables_with_known_function:
                    print(f"-> Found {len(interactables_with_known_function)} interactable(s) with known functions to test.")
                    potential_targets = interactables_with_known_function
                else:
                    print("-> No interactable objects with known functions found.")
            # --- End of Priority 3 Logic ---

            # If we still have no targets after P1 and P2, then exploration is complete for now.
            if not potential_targets:
                return False

        # 2. Find paths to all potential targets and store the ones that are reachable.
        reachable_targets = []
        for target_pos in potential_targets:
            if target_pos == player_tile_pos:
                continue
                
            path = self._find_path_to_target(player_tile_pos, target_pos)
            if path:
                reachable_targets.append({'pos': target_pos, 'path': path})

        if not reachable_targets:
            print("🧐 All interactable targets are currently unreachable.")
            potential_targets = []

            if not reachable_targets:
                return False # If we still have no reachable targets after all checks, now we can exit.

        # 3. From the list of reachable targets, select the one with the shortest path.
        best_target = min(reachable_targets, key=lambda t: len(t['path']))
        
        # --- NEW: Resource Safety Check ---
        # Before committing to the plan, check if we need to get resources first.
        nearest_resource_info = self._find_nearest_resource(player_tile_pos)
        if nearest_resource_info:
            distance_to_resource = len(nearest_resource_info['path'])
            # Add a small safety buffer. If current moves are just enough to get there (or less),
            # the agent must prioritize survival.
            safety_buffer = 1
            if self.current_moves <= distance_to_resource + safety_buffer:
                print(f"⚠️ Low on moves ({self.current_moves})! Resource is {distance_to_resource} steps away. Prioritizing survival.")
                # Override the original plan with the resource-gathering plan.
                best_target = nearest_resource_info
            else:
                print(f"-> Moves ({self.current_moves}) sufficient. Nearest resource is {distance_to_resource} steps away.")
        # --- End of Safety Check ---

        # 4. Set the exploration plan based on the best target found.
        self.exploration_target = (best_target['pos'][0] * self.tile_size, best_target['pos'][1] * self.tile_size)
        self.exploration_plan = best_target['path']
        print(f"🎯 New target acquired at {best_target['pos']}. Plan created with {len(self.exploration_plan)} steps.")
        
        # Calculate the reachable area just before printing the map to give the most current view.
        self.reachable_floor_area = self._find_reachable_floor_tiles()
        self._print_debug_map()

        return True

    def _find_nearest_resource(self, player_tile_pos: tuple) -> dict | None:
        """Finds the closest reachable resource tile."""
        resource_tiles = [pos for pos, type in self.tile_map.items() if type == CellType.RESOURCE and pos not in self.consumed_tiles_this_life]
        if not resource_tiles:
            return None

        reachable_resources = []
        for resource_pos in resource_tiles:
            path = self._find_path_to_target(player_tile_pos, resource_pos, ignore_move_cost=True)
            if path:
                reachable_resources.append({'pos': resource_pos, 'path': path})

        if not reachable_resources:
            return None

        # Return the resource with the shortest path
        return min(reachable_resources, key=lambda r: len(r['path']))

    def _find_path_to_target(self, start_tile: tuple, target_tile: tuple, ignore_move_cost: bool = False) -> list:
        """
        Finds the shortest, resource-aware path to a target using the A* algorithm.
        This path considers the agent's current moves and potential resource pickups.
        """
        # 1. Build a fresh map from TILE vectors to actions.
        tile_vector_to_action = {}
        if self.tile_size and self.world_model.get('action_map'):
            for action, effect in self.world_model['action_map'].items():
                if 'move_vector' in effect:
                    px_vec = effect['move_vector']
                    tile_vec = (px_vec[0] // self.tile_size, px_vec[1] // self.tile_size)
                    if tile_vec != (0, 0):
                        tile_vector_to_action[tile_vec] = action

        if not tile_vector_to_action:
            return []

        # 2. A* Implementation
        def heuristic(a, b):
            """Manhattan distance heuristic for A*."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # The priority queue stores: (f_score, g_score, tile, remaining_moves, path)
        open_set = []
        initial_f_score = heuristic(start_tile, target_tile)
        heapq.heappush(open_set, (initial_f_score, 0, start_tile, self.current_moves, []))

        # g_scores tracks the cost to reach a state: g_scores[(tile, moves)] = cost
        g_scores = {(start_tile, self.current_moves): 0}

        # Visited set to avoid re-processing states
        visited_states = set()

        while open_set:
            _, g, current_tile, current_moves, path = heapq.heappop(open_set)

            if current_tile == target_tile:
                return path  # Path found!

            state = (current_tile, current_moves)
            if state in visited_states:
                continue
            visited_states.add(state)

            for tile_vec, action in tile_vector_to_action.items():
                neighbor_tile = (current_tile[0] + tile_vec[0], current_tile[1] + tile_vec[1])

                tile_type = self.tile_map.get(neighbor_tile)
                can_move_to = tile_type in [CellType.FLOOR, CellType.POTENTIALLY_INTERACTABLE, CellType.CONFIRMED_INTERACTABLE, CellType.RESOURCE]

                if not can_move_to:
                    continue

                # Calculate the state for the next step in the path
                next_moves = current_moves - 1
                if not ignore_move_cost and next_moves < 0:
                    continue  # This path ran out of resources

                # If the neighbor is a resource, refill the moves for the next state
                if self.tile_map.get(neighbor_tile) == CellType.RESOURCE:
                    next_moves = self.max_moves

                tentative_g_score = g + 1
                neighbor_state = (neighbor_tile, next_moves)

                # Check if this new path to the neighbor is better than any previous one
                if tentative_g_score < g_scores.get(neighbor_state, float('inf')):
                    g_scores[neighbor_state] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor_tile, target_tile)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor_tile, next_moves, path + [action]))

        return []  # No path found
    
    def _find_reachable_floor_tiles(self) -> set:
        """
        Performs a BFS/flood-fill from the player's position to find all connected floor tiles.
        """
        # Ensure we have the necessary information to start.
        if not self.last_known_player_obj or not self.tile_size or not self.tile_map:
            print("⚠️ Cannot find reachable floor tiles: Missing player, tile size, or map.")
            return set()

        # 1. Get the player's starting position in tile coordinates.
        player_pixel_pos = (self.last_known_player_obj['top_row'], self.last_known_player_obj['left_index'])
        start_tile = (player_pixel_pos[0] // self.tile_size, player_pixel_pos[1] // self.tile_size)

        # The player might be on an interactable tile, which is also a valid starting point.
        start_tile_type = self.tile_map.get(start_tile)
        if start_tile_type not in [CellType.FLOOR, CellType.POTENTIALLY_INTERACTABLE, CellType.CONFIRMED_INTERACTABLE, CellType.RESOURCE]:
            print(f"⚠️ Player starting tile {start_tile} is not on a known FLOOR, INTERACTABLE, or RESOURCE. Aborting flood fill.")
            return set()
        # 2. Initialize BFS data structures.
        q = [start_tile]
        visited = {start_tile}

        # 3. Perform the flood-fill.
        while q:
            current_tile = q.pop(0)

            # Explore neighbors (up, down, left, right in tile space).
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_tile = (current_tile[0] + dr, current_tile[1] + dc)

                # We can only traverse through tiles classified as FLOOR.
                if self.tile_map.get(neighbor_tile) in [CellType.FLOOR, CellType.POTENTIALLY_INTERACTABLE, CellType.CONFIRMED_INTERACTABLE, CellType.RESOURCE] and neighbor_tile not in visited:
                    visited.add(neighbor_tile)
                    q.append(neighbor_tile)
        
        print(f"🗺️ Found {len(visited)} reachable tiles connected to the player.")
        return visited
    
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

    def _create_normalized_fingerprint(self, obj_datamap: tuple) -> tuple:
        """
        Creates a scale-invariant fingerprint of an object's data map.
        This allows matching objects of different sizes that have the same proportions and pattern.
        """
        if not obj_datamap or not obj_datamap[0]:
            return None, None # Return None if datamap is empty

        height = len(obj_datamap)
        width = len(obj_datamap[0])

        # 1. Find the greatest common divisor (GCD) to get the base aspect ratio.
        divisor = math.gcd(height, width)
        base_h = height // divisor
        base_w = width // divisor

        # The scale factor tells us how many pixels in the original map
        # correspond to one pixel in the normalized map.
        scale_h = divisor
        scale_w = divisor

        # 2. Create the normalized map by sampling pixels.
        normalized_map_list = []
        for r in range(base_h):
            new_row = []
            for c in range(base_w):
                # Sample the top-left pixel of the corresponding block in the original map.
                original_row = r * scale_h
                original_col = c * scale_w
                sampled_pixel = obj_datamap[original_row][original_col]
                new_row.append(sampled_pixel)
            normalized_map_list.append(tuple(new_row))
        
        normalized_map_tuple = tuple(normalized_map_list)

        # 3. Generate the final fingerprint using a hash of the tuple.
        fingerprint = hash(normalized_map_tuple)
        
        # Return the base shape, the fingerprint, AND the normalized map itself for debugging.
        return ((base_h, base_w), fingerprint, normalized_map_tuple)

    
    def _find_static_candidates_by_color(self, color: int, grid_data: list) -> list[dict]:
        """Scans the entire grid to find and describe all objects of a specific color."""
        if not grid_data or not grid_data[0]:
            return []

        candidates = []
        grid_height = len(grid_data)
        grid_width = len(grid_data[0])
        visited_coords = set()

        for r in range(grid_height):
            for c in range(grid_width):
                if (r, c) not in visited_coords and grid_data[r][c] == color:
                    # Found a starting point for a potential candidate object.
                    # Use flood-fill to find all its connected pixels.
                    component_points = set()
                    q = [(r, c)]
                    visited_coords.add((r, c))
                    
                    while q:
                        p = q.pop(0)
                        component_points.add(p)
                        row, col = p
                        
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            neighbor = (row + dr, col + dc)
                            if (0 <= neighbor[0] < grid_height and 
                                0 <= neighbor[1] < grid_width and 
                                neighbor not in visited_coords and 
                                grid_data[neighbor[0]][neighbor[1]] == color):
                                
                                visited_coords.add(neighbor)
                                q.append(neighbor)
                    
                    # If an object was found, describe it.
                    if component_points:
                        min_row = min(r for r, _ in component_points)
                        max_row = max(r for r, _ in component_points)
                        min_idx = min(p_idx for _, p_idx in component_points)
                        max_idx = max(p_idx for _, p_idx in component_points)
                        height, width = max_row - min_row + 1, max_idx - min_idx + 1

                        data_map = tuple(tuple(grid_data[r][p] if (r, p) in component_points else None for p in range(min_idx, max_idx + 1)) for r in range(min_row, max_row + 1))
                        base_shape, fingerprint, norm_map = self._create_normalized_fingerprint(data_map)

                        candidates.append({
                            'height': height, 'width': width, 'top_row': min_row,
                            'left_index': min_idx, 'color': color,
                            'data_map': data_map, 'fingerprint': fingerprint, 'base_shape': base_shape, 'normalized_map': norm_map
                        })
        return candidates

    def _find_and_describe_objects(self, structured_changes: list, latest_frame: list, is_interaction_analysis: bool = False) -> list[dict]:
        """Finds objects by grouping changed pixels by their new color first, then clustering."""
        # --- Create a lookup map for original colors ---
        from_color_map = {}
        for change in structured_changes:
            row_idx = change['row_index']
            for pixel_change in change['changes']:
                from_color_map[(row_idx, pixel_change['index'])] = pixel_change['from']

        changed_coords = set()
        for change in structured_changes:
            row_idx = change['row_index']
            for pixel_change in change['changes']:
                changed_coords.add((row_idx, pixel_change['index']))
        
        if not structured_changes:
            return []

        # 1. Group points by object color, handling both additions and removals.
        floor_color = self.world_model.get('floor_color')
        wall_colors = self.world_model.get('wall_colors', set())
        background_colors = {floor_color} | wall_colors
        background_colors.discard(None)

        points_by_color = {} # This is the dictionary the original function expects.
        grid = latest_frame[0]
        grid_height = len(grid)
        grid_width = len(grid[0]) if grid_height > 0 else 0

        for change in structured_changes:
            r = change['row_index']
            for px_change in change['changes']:
                c = px_change['index']
                from_color = px_change['from']
                to_color = px_change['to']
                coord = (r, c)

                # Case 1: A new object color appears. The changed pixel itself is the seed.
                if to_color not in background_colors:
                    if to_color not in points_by_color:
                        points_by_color[to_color] = set()
                    points_by_color[to_color].add(coord)

                # Case 2: An old object color disappears. Seeds must be neighbors
                # that are still the original color.
                if from_color not in background_colors and to_color in background_colors:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = (r + dr, c + dc)
                        if not (0 <= neighbor[0] < grid_height and 0 <= neighbor[1] < grid_width):
                            continue
                        
                        if grid[neighbor[0]][neighbor[1]] == from_color:
                            if from_color not in points_by_color:
                                points_by_color[from_color] = set()
                            points_by_color[from_color].add(neighbor)

        # 2. Run a more comprehensive flood-fill starting from each changed pixel
        #    to find the full extent of each object, including its static parts.
        monochromatic_parts = []
        visited_coords = set() # Use one visited set for the whole grid, including static pixels explored.
        grid = latest_frame[0]
        grid_height = len(grid)
        grid_width = len(grid[0]) if grid_height > 0 else 0

        # We still iterate through the changed points to find starting locations for our search.
        for color, points in points_by_color.items():
            for point in points:
                if point not in visited_coords:
                    # Start a new search for a component.
                    component_points = set()
                    q = [point]
                    visited_coords.add(point) # Mark as visited to avoid redundant searches.
                    
                    while q:
                        p = q.pop(0)
                        component_points.add(p)
                        r, p_idx = p
                        
                        # Explore all four neighbors.
                        for dr, dp_idx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            neighbor = (r + dr, p_idx + dp_idx)

                            # Check if neighbor is within grid boundaries.
                            if not (0 <= neighbor[0] < grid_height and 0 <= neighbor[1] < grid_width):
                                continue
                            
                            # The crucial change: If the neighbor is the same color and we haven't
                            # visited it yet, add it to the queue. This allows the search to
                            # expand across both changed and static pixels.
                            if neighbor not in visited_coords and grid[neighbor[0]][neighbor[1]] == color:
                                visited_coords.add(neighbor)
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

            # --- NEW: Filter for incidentally-discovered static objects ---
            if not is_interaction_analysis:
                # An object is static if it was only detected because it was adjacent to a
                # change, but none of its own pixels actually changed this frame.
                changed_pixels_in_component = changed_coords.intersection(obj_points)
                if not changed_pixels_in_component:
                    print(f"🕵️‍♀️ Ignoring static object discovered adjacent to a change.")
                    continue

            # --- Background Verification Step ---
            # An "object" is likely a background reveal if it's the known floor color,
            # or if the number of pixels that actually changed is tiny compared to its total size.
            sample_point = next(iter(obj_points))
            obj_color = grid[sample_point[0]][sample_point[1]]
            known_floor_color = self.world_model.get('floor_color')

            # If we have confirmed the floor color, the check is simple and reliable.
            if known_floor_color is not None:
                if obj_color == known_floor_color:
                    # This is just the floor being revealed. Log it and skip creating an object.
                    print(f"🕵️‍♀️ [FLOOR]: A change revealed the known floor color ({obj_color}). Ignoring as object.")
                    continue
                # Otherwise, if it's not the floor color, it must be a real object.

            else:
                # If we DON'T know the floor color yet, we use a heuristic to learn it.
                # A background reveal happens when a small change connects to a massive static area.
                changed_pixels_in_component = changed_coords.intersection(obj_points)

                # Heuristic: If the component is large and the changed part is small, it's probably background.
                # This prevents the entire 64x64 background from being treated as an object.
                is_background_candidate = False
                if len(obj_points) > 50: # Must be a reasonably large area
                    # Avoid division by zero if obj_points is somehow empty, though we check earlier.
                    if len(obj_points) > 0:
                        ratio = len(changed_pixels_in_component) / len(obj_points)
                        if ratio < 0.5: # The changed part is less than half the total size
                            is_background_candidate = True

                if is_background_candidate:
                    self.floor_hypothesis[obj_color] = self.floor_hypothesis.get(obj_color, 0) + 1
                    confidence = self.floor_hypothesis[obj_color]
                    print(f"🕵️‍♀️ Floor Hypothesis: A small change revealed a large area of color {obj_color} (Confidence: {confidence}).")

                    if confidence >= self.CONCEPT_CONFIDENCE_THRESHOLD:
                        self.world_model['floor_color'] = obj_color
                        print(f"✅ [FLOOR] Confirmed: Color {obj_color} is the floor.")
                        # Future improvement: could add map cleanup logic here.

                    # While learning what the floor is, we skip creating an object from background reveals.
                    continue

            # --- If it's a real object, proceed with description ---
            min_row = min(r for r, _ in obj_points)
            max_row = max(r for r, _ in obj_points)
            min_idx = min(p_idx for _, p_idx in obj_points)
            max_idx = max(p_idx for _, p_idx in obj_points)
            height, width = max_row - min_row + 1, max_idx - min_idx + 1

            # --- NEW: Hard-coded filter for the 64x64 background object ---
            # If an object is found that spans the entire grid, it's the background/wall canvas.
            # We explicitly ignore it to prevent it from being treated as a dynamic game object.
            if height >= 64 and width >= 64 and min_row == 0 and min_idx == 0:
                print(f"🕵️‍♀️ Ignoring the {height}x{width} canvas object found at ({min_row}, {min_idx}).")
                continue # Skip to the next monochromatic part

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

            data_map = tuple(tuple(grid[r][p] if (r, p) in obj_points else None for p in range(min_idx, max_idx + 1)) for r in range(min_row, max_row + 1))
            base_shape, fingerprint, norm_map = self._create_normalized_fingerprint(data_map)

            original_color = from_color_map.get(sample_point) # Get original color from our map

            final_objects.append({
                'height': height, 'width': width, 'top_row': min_row,
                'left_index': min_idx, 'color': obj_color,
                'original_color': original_color,
                'data_map': data_map,
                'background_color': background_color,
                'fingerprint': fingerprint, 
                'base_shape': base_shape,
                'normalized_map': norm_map
            })

        return final_objects
    
    def _are_objects_adjacent(self, obj1: dict, obj2: dict) -> bool:
        """Checks if two objects' bounding boxes are touching or overlapping."""
        # Define the bounding box edges for obj1
        obj1_left = obj1['left_index']
        obj1_right = obj1['left_index'] + obj1['width']
        obj1_top = obj1['top_row']
        obj1_bottom = obj1['top_row'] + obj1['height']

        # Define the bounding box edges for obj2
        obj2_left = obj2['left_index']
        obj2_right = obj2['left_index'] + obj2['width']
        obj2_top = obj2['top_row']
        obj2_bottom = obj2['top_row'] + obj2['height']

        # Check for no overlap. Two rectangles do NOT overlap if one is entirely
        # to the left, right, above, or below the other.
        if (obj1_right < obj2_left or obj2_right < obj1_left or
            obj1_bottom < obj2_top or obj2_bottom < obj1_top):
            return False
        
        # If they are not completely separate, they must be adjacent or overlapping.
        return True
    
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
                if curr_obj.get('fingerprint') and curr_obj['fingerprint'] == last_obj.get('fingerprint'):
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
                    
                    # --- NEW: Build a full datamap for the composite object ---
                    composite_datamap_list = [[None for _ in range(comp_w)] for _ in range(comp_h)]
                    for curr_part, _ in cluster:
                        part_h, part_w = curr_part['height'], curr_part['width']
                        part_r_offset = curr_part['top_row'] - min_row
                        part_c_offset = curr_part['left_index'] - min_col
                        part_datamap = curr_part['data_map']
                        for r in range(part_h):
                            for c in range(part_w):
                                if part_datamap[r][c] is not None:
                                    composite_datamap_list[part_r_offset + r][part_c_offset + c] = part_datamap[r][c]
                    composite_datamap = tuple(tuple(row) for row in composite_datamap_list)

                    # --- NEW: Use the visual fingerprint as the signature ---
                    _, signature, _ = self._create_normalized_fingerprint(composite_datamap)
                    
                    # Generate a simpler log if the moved object is the confirmed agent.
                    if signature == self.world_model.get('player_signature'):
                        log_messages.append(f"🧠 [AGENT] moved by vector {vector}.")
                    else:
                        log_messages.append(f"🧠 COMPOSITE MOVE: Object [{comp_h}x{comp_w}] with fingerprint {signature} moved by vector {vector}.")
                
                true_moves.append({'type': 'composite', 'signature': signature, 'dimensions': (comp_h, comp_w), 'parts': cluster, 'vector': vector})
                for pair in cluster: processed_pairs.append(pair)

        # Log individual moves and collect their structured info.
        for curr, last in move_matched_pairs:
            if (curr, last) not in processed_pairs:
                signature = curr.get('fingerprint')
                # Generate a simpler log if the moved object is the confirmed agent.
                if signature == self.world_model.get('player_signature'):
                    log_messages.append(f"🧠 [AGENT] moved from ({last['top_row']}, {last['left_index']}) to ({curr['top_row']}, {curr['left_index']}).")
                else:
                    log_messages.append(f"🧠 MOVE: Object [{curr['height']}x{curr['width']}] moved from ({last['top_row']}, {last['left_index']}) to ({curr['top_row']}, {curr['left_index']}).")

                vector = (curr['top_row'] - last['top_row'], curr['left_index'] - last['left_index'])
                true_moves.append({'type': 'individual', 'signature': signature, 'curr_obj': curr, 'last_obj': last, 'vector': vector})

        # --- Concept Learning from Movement (Agent and Floor) ---
        moved_agent_obj = None
        if not true_moves:
            agent_part_fingerprints = set()
            if moved_agent_obj and moved_agent_obj.get('parts'):
                agent_part_fingerprints = {part['fingerprint'] for part in moved_agent_obj['parts'] if 'fingerprint' in part}
            
            return log_messages, moved_agent_obj, agent_part_fingerprints

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

                            # --- Map Cleanup Logic ---
                            if self.tile_size:
                                reclassified_count = 0
                                grid_data = latest_grid[0]
                                for tile_coords, cell_type in list(self.tile_map.items()):
                                    if cell_type != CellType.FLOOR:
                                        tile_row = tile_coords[0] * self.tile_size
                                        tile_col = tile_coords[1] * self.tile_size
                                        if 0 <= tile_row < len(grid_data) and 0 <= tile_col < len(grid_data[0]):
                                            tile_color = grid_data[tile_row][tile_col]
                                            if tile_color == floor_candidate:
                                                self.tile_map[tile_coords] = CellType.FLOOR
                                                reclassified_count += 1
                                if reclassified_count > 0:
                                    log_messages.append(f"🧹 Map Cleanup: Reclassified {reclassified_count} tile(s) as newly confirmed floor.")

                            break # Stop after confirming.

        # --- Action Effect Learning ---
        # Learn how actions affect the agent.
        if self.world_model['player_signature'] is not None and action:
            for move in true_moves:
                if move['signature'] == self.world_model['player_signature']:
                    # --- THE AGENT MOVED ---
                    effect_vector = move['vector']

                    # 1. Get the current-frame object descriptions for all agent parts.
                    agent_parts = []
                    if move['type'] == 'composite':
                        agent_parts = [p[0] for p in move['parts']]
                    elif move['type'] == 'individual' and 'curr_obj' in move:
                        agent_parts = [move['curr_obj']]
                    
                    if not agent_parts: continue # Safety check

                    # 2. Combine the parts into a single descriptor for self.last_known_player_obj
                    min_row = min(p['top_row'] for p in agent_parts)
                    max_row = max(p['top_row'] + p['height'] for p in agent_parts)
                    min_col = min(p['left_index'] for p in agent_parts)
                    max_col = max(p['left_index'] + p['width'] for p in agent_parts)

                    # Create a descriptor that includes the parts for future filtering.
                    moved_agent_obj = {
                        'height': max_row - min_row, 
                        'width': max_col - min_col,
                        'top_row': min_row, 
                        'left_index': min_col,
                        'parts': agent_parts  # Store the individual component objects
                    }

                    # 2. Update and test action effect hypotheses
                    # Initialize hypothesis dict for this action if it doesn't exist.
                    if action not in self.action_effect_hypothesis:
                        self.action_effect_hypothesis[action] = {}

                    hypo_dict = self.action_effect_hypothesis[action]
                    hypo_dict[effect_vector] = hypo_dict.get(effect_vector, 0) + 1
                    confidence = hypo_dict[effect_vector]

                    log_messages.append(f"🕵️‍♀️ Action Hypothesis: {action.name} -> move by {effect_vector} (Confidence: {confidence}).")

                    # If confidence is high, check if this updates our world model.
                    if confidence >= self.CONCEPT_CONFIDENCE_THRESHOLD:
                        current_known_vector = self.world_model['action_map'].get(action, {}).get('move_vector')

                        # Update if the action is new or the vector has changed.
                        if effect_vector != current_known_vector:
                            self.world_model['action_map'][action] = {'move_vector': effect_vector}
                            log_messages.append(f"✅ UPDATED Action Effect: {action.name} now moves agent by vector {effect_vector}.")
                            
                            # Re-calculate tile size and generalize the new magnitude.
                            new_magnitude = abs(effect_vector[0]) + abs(effect_vector[1])
                            if new_magnitude > 1: # A move of 1 is ambiguous
                                if self.tile_size != new_magnitude:
                                    self.tile_size = new_magnitude
                                    print(f"📏 Tile Size Re-evaluated: Now {self.tile_size}px based on new movement.")
                                    log_messages.append("🗺️ Tile size changed. Wiping all level-specific map and interaction data.")
                                    self.tile_map.clear()
                                    self.reachable_floor_area.clear()
                                    self.interaction_hypotheses.clear()
                                    self.consumed_tiles_this_life.clear()
                                    self.exploration_plan.clear()
                                    self.exploration_target = None
                                    self.observing_interaction_for_tile = None
                                    self.exploration_phase = ExplorationPhase.INACTIVE

                                # --- Generalize this new magnitude to all other move actions ---
                                log_messages.append(f" -> Generalizing new magnitude of {new_magnitude}px to all other move actions.")
                                for other_action, effect in self.world_model['action_map'].items():
                                    if other_action == action: continue # Skip the action we just updated
                                    
                                    if 'move_vector' in effect:
                                        old_vec = effect['move_vector']
                                        old_mag = abs(old_vec[0]) + abs(old_vec[1])
                                        if old_mag == 0: continue # Skip zero vectors
                                        
                                        # Integer division works here for axis-aligned unit vectors
                                        unit_vec = (old_vec[0] // old_mag, old_vec[1] // old_mag)
                                        
                                        # Create the new vector using the new magnitude
                                        new_vec = (unit_vec[0] * new_magnitude, unit_vec[1] * new_magnitude)
                                        
                                        if old_vec != new_vec:
                                            self.world_model['action_map'][other_action]['move_vector'] = new_vec
                                            log_messages.append(f"    -> Updated {other_action.name} move vector to {new_vec}.")
                            
                            # This hypothesis is now confirmed, clear it to start fresh for the next change.
                            if action in self.action_effect_hypothesis:
                                del self.action_effect_hypothesis[action]

                    break # Agent's move found, no need to check other moves.

            # --- NEW: Enhanced Fuzzy Matching with Full Body Reconstruction ---
            if not moved_agent_obj and self.last_known_player_obj:
                log_messages.append("⚠️ Agent signature not found. Re-acquiring by proximity and direction...")
                last_pos = (self.last_known_player_obj['top_row'], self.last_known_player_obj['left_index'])
                expected_vector = self.world_model.get('action_map', {}).get(self.last_action, {}).get('move_vector')

                best_candidate_move = None
                min_distance = float('inf')
                
                # Find the single best "anchor" part of the agent.
                for move in true_moves:
                    if expected_vector:
                        move_vector = move['vector']
                        dot_product = (expected_vector[0] * move_vector[0]) + (expected_vector[1] * move_vector[1])
                        if dot_product <= 0: continue

                    last_obj_pos = None
                    if move['type'] == 'individual':
                        last_obj_pos = (move['last_obj']['top_row'], move['last_obj']['left_index'])
                    elif move['type'] == 'composite':
                        min_row = min(p[1]['top_row'] for p in move['parts'])
                        min_col = min(p[1]['left_index'] for p in move['parts'])
                        last_obj_pos = (min_row, min_col)
                    
                    if last_obj_pos:
                        distance = math.sqrt((last_pos[0] - last_obj_pos[0])**2 + (last_pos[1] - last_obj_pos[1])**2)
                        if distance < min_distance:
                            min_distance = distance
                            best_candidate_move = move
                
                threshold = self.tile_size * 2.5 if self.tile_size else 24
                if best_candidate_move and min_distance < threshold:
                    log_messages.append(f"✅ Agent Re-acquired: Found anchor part with signature {best_candidate_move['signature']} by proximity (distance: {min_distance:.2f}).")
                    
                    # --- Full Body Reconstruction ---
                    # Now that we have an anchor, find all adjacent, untracked objects to rebuild the full agent.
                    anchor_part = best_candidate_move['curr_obj']
                    reconstructed_parts = [anchor_part]
                    
                    # Create a copy of unmatched objects to search through.
                    search_pool = [obj for obj in current_objects if obj is not anchor_part]
                    if anchor_part in search_pool:
                        search_pool.remove(anchor_part)

                    cluster_q = [anchor_part]
                    while cluster_q:
                        current_part = cluster_q.pop(0)
                        for other_part in list(search_pool):
                            if self._are_objects_adjacent(current_part, other_part):
                                reconstructed_parts.append(other_part)
                                search_pool.remove(other_part)
                                cluster_q.append(other_part)
                    
                    log_messages.append(f"✅ Reconstructed agent with {len(reconstructed_parts)} parts.")

                    # Build the final agent object from all the reconstructed parts.
                    min_row = min(p['top_row'] for p in reconstructed_parts)
                    max_row = max(p['top_row'] + p['height'] for p in reconstructed_parts)
                    min_col = min(p['left_index'] for p in reconstructed_parts)
                    max_col = max(p['left_index'] + p['width'] for p in reconstructed_parts)
                    moved_agent_obj = {
                        'height': max_row - min_row, 'width': max_col - min_col,
                        'top_row': min_row, 'left_index': min_col, 'parts': reconstructed_parts
                    }

        agent_part_fingerprints = set()
        if moved_agent_obj and moved_agent_obj.get('parts'):
            agent_part_fingerprints = {part['fingerprint'] for part in moved_agent_obj['parts'] if 'fingerprint' in part}
        
        return log_messages, moved_agent_obj, agent_part_fingerprints
    
    def _learn_from_interaction_failure(self, action: GameAction, last_grid: list):
        """Analyzes why a known action failed by checking the intended destination area."""
        # --- 1. Check if we have enough information to analyze the failure ---
        player_sig = self.world_model.get('player_signature')
        action_effect = self.world_model['action_map'].get(action)
        
        if not (player_sig and self.last_known_player_obj and action_effect and 'move_vector' in action_effect):
            return

        # --- 2. Calculate the intended destination bounding box ---
        move_vector = action_effect['move_vector']
        last_obj = self.last_known_player_obj
        row_change, col_change = move_vector

        final_top_row = last_obj['top_row'] + row_change
        final_left_index = last_obj['left_index'] + col_change
        final_bottom_row = final_top_row + last_obj['height']
        final_right_index = final_left_index + last_obj['width']
        
        # --- 3. Investigate the destination area for obstacles ---
        grid_height = len(last_grid[0])
        grid_width = len(last_grid[0][0]) if grid_height > 0 else 0
        blocking_colors = Counter()
        floor_color = self.world_model.get('floor_color')
        
        for r in range(final_top_row, final_bottom_row):
            for p_idx in range(final_left_index, final_right_index):
                if 0 <= r < grid_height and 0 <= p_idx < grid_width:
                    color = last_grid[0][r][p_idx]

                    # --- EFFICIENT CHECK: Is this a pre-confirmed wall? ---
                    if color in self.world_model['wall_colors']:
                        print(f"🧱 [WALL] Collision with known wall (Color: {color}) detected.")
                        return # Exit immediately, our job is done.

                    if color != floor_color:
                        blocking_colors[color] += 1
                else:
                    if -1 in self.world_model['wall_colors']:
                        print("🧱 [WALL] Collision with known wall (Out of Bounds) detected.")
                        return
                    blocking_colors[-1] += 1
                        
        if not blocking_colors:
            return

        # --- 4. Process every color found in the collision area as a potential wall ---
        for wall_candidate_color in blocking_colors.keys():
            self.wall_hypothesis[wall_candidate_color] = self.wall_hypothesis.get(wall_candidate_color, 0) + 1
            confidence = self.wall_hypothesis[wall_candidate_color]
            wall_name = "Out of Bounds" if wall_candidate_color == -1 else f"Color {wall_candidate_color}"
            print(f"🧱 Wall Hypothesis: {wall_name} blocked movement (Confidence: {confidence}).")

            # --- 5. Confirm Hypothesis if Threshold is Met ---
            if confidence >= self.CONCEPT_CONFIDENCE_THRESHOLD:
                # Add the color to the set of confirmed walls.
                self.world_model['wall_colors'].add(wall_candidate_color)
                print(f"✅ [WALL] Confirmed: {wall_name} is a wall.")

                # --- Map Cleanup Logic ---
                # This logic now runs for each newly confirmed wall color.
                if self.tile_size:
                    reclassified_count = 0
                    for tile_coords, cell_type in list(self.tile_map.items()):
                        # We only need to check tiles that are not already walls or floors.
                        if cell_type not in [CellType.WALL, CellType.FLOOR]:
                            # Use a full-tile check to see if it's a uniform wall.
                            tile_row_start, tile_col_start = tile_coords[0] * self.tile_size, tile_coords[1] * self.tile_size
                            tile_pixels = [
                                last_grid[0][r][c]
                                for r in range(tile_row_start, tile_row_start + self.tile_size)
                                for c in range(tile_col_start, tile_col_start + self.tile_size)
                                if 0 <= r < grid_height and 0 <= c < grid_width
                            ]
                            
                            # If the tile is made up of only this newly confirmed wall color, reclassify it.
                            if tile_pixels and all(p == wall_candidate_color for p in tile_pixels):
                                self.tile_map[tile_coords] = CellType.WALL
                                reclassified_count += 1
                    
                    if reclassified_count > 0:
                        print(f"🧹 Map Cleanup: Reclassified {reclassified_count} tile(s) as newly confirmed wall ({wall_name}).")
                
                # Once confirmed, we can remove it from the hypothesis tracker.
                if wall_candidate_color in self.wall_hypothesis:
                    del self.wall_hypothesis[wall_candidate_color]
    
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
                        # Define the full row as a rectangle and add it to the ignore list.
                        grid_width = len(self.previous_frame[0][0]) if self.previous_frame and self.previous_frame[0] else 0
                        self.ignored_areas.append({
                            'top_row': row_idx, 'left_index': 0,
                            'height': 1, 'width': grid_width
                        })
                        print(f"✅ Confirmed resource indicator at row {row_idx}! It will now be ignored for state uniqueness checks.")
                        
                        # --- NEW: Re-check active patterns to filter out the new indicator ---
                        if self.active_patterns:
                            print(f"🕵️‍♀️ Re-checking {len(self.active_patterns)} active pattern(s) against the new resource indicator...")
                            
                            valid_patterns = []
                            removed_count = 0
                            for pattern in self.active_patterns:
                                dk = pattern['dynamic_key']
                                # Check if the dynamic key's vertical span overlaps with the indicator row
                                obj_top = dk['top_row']
                                obj_bottom = obj_top + dk['height']
                                
                                if obj_top <= row_idx < obj_bottom:
                                    # This pattern's dynamic key is on the indicator row. It's invalid.
                                    removed_count += 1
                                else:
                                    valid_patterns.append(pattern)
                            
                            if removed_count > 0:
                                print(f"-> Removed {removed_count} pattern(s) that were incorrectly tracking the resource indicator.")
                                self.active_patterns = valid_patterns
                        
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

    def _extract_object_at_tile(self, tile_coords: tuple, grid_data: list, visited_tiles: set) -> dict | None:
        """
        Performs a flood-fill starting from a tile to find, describe, and fingerprint a complete static object.
        This can handle objects that span multiple tiles.
        """
        if tile_coords in visited_tiles:
            return None

        # Get known background colors to define the object's boundaries
        floor_color = self.world_model.get('floor_color')
        wall_colors = self.world_model.get('wall_colors', set())
        background_colors = {floor_color} | wall_colors
        
        # Flood-fill to find all tiles belonging to this object
        q = [tile_coords]
        object_tiles = set(q)
        visited_tiles.add(tile_coords)

        while q:
            current_tile = q.pop(0)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current_tile[0] + dr, current_tile[1] + dc)
                if self.tile_map.get(neighbor) in [CellType.POTENTIALLY_INTERACTABLE, CellType.CONFIRMED_INTERACTABLE] and neighbor not in visited_tiles:
                    visited_tiles.add(neighbor)
                    object_tiles.add(neighbor)
                    q.append(neighbor)
        
        # Now, find all the actual pixels for the object
        object_pixels = set()



        for r_tile, c_tile in object_tiles:
            for r_pixel in range(r_tile * self.tile_size, (r_tile + 1) * self.tile_size):
                for c_pixel in range(c_tile * self.tile_size, (c_tile + 1) * self.tile_size):
                    if 0 <= r_pixel < len(grid_data) and 0 <= c_pixel < len(grid_data[0]):
                        if grid_data[r_pixel][c_pixel] not in background_colors:
                            object_pixels.add((r_pixel, c_pixel))

        if not object_pixels:
            return None

        # Use the same logic from _find_and_describe_objects to create the description
        min_row = min(r for r, _ in object_pixels)
        max_row = max(r for r, _ in object_pixels)
        min_idx = min(p_idx for _, p_idx in object_pixels)
        max_idx = max(p_idx for _, p_idx in object_pixels)
        height, width = max_row - min_row + 1, max_idx - min_idx + 1

        data_map = tuple(tuple(grid_data[r][p] for p in range(min_idx, max_idx + 1)) for r in range(min_row, max_row + 1))
        base_shape, fingerprint, norm_map = self._create_normalized_fingerprint(data_map)

        # Return a complete object description
        return {
            'height': height, 'width': width, 'top_row': min_row,
            'left_index': min_idx, 'color': grid_data[min_row][min_idx], # Sample color
            'data_map': data_map, 'fingerprint': fingerprint, 'base_shape': base_shape    
        }

    def _analyze_and_log_interaction_effect(self, structured_changes: list, effect_type: str, latest_grid: list, static_objects_before_action: list, current_agent_fingerprints: set):
        """Analyzes pixel changes from an interaction and logs them as a hypothesis."""
        if self.observing_interaction_for_tile is None:
            return
        
        object_signature = f"tile_pos_{self.observing_interaction_for_tile}"

        # 1. Define ignore zones to filter out irrelevant changes (player movement, UI).
        ignore_coords = set()
        
        # Add the player's last known position to the ignore zone.
        if self.last_known_player_obj:
            player_box = self.last_known_player_obj
            for r in range(player_box['top_row'], player_box['top_row'] + player_box['height']):
                for c in range(player_box['left_index'], player_box['left_index'] + player_box['width']):
                    ignore_coords.add((r,c))

        # Add the resource indicator row, if known.
        if self.confirmed_resource_indicator:
            indicator_row = self.confirmed_resource_indicator.get('row_index')
            # Add the entire row to the ignore zone.
            grid_width = len(latest_grid[0][0]) if latest_grid and latest_grid[0] else 0
            for c in range(grid_width):
                ignore_coords.add((indicator_row, c))

        # 2. Filter the raw pixel changes using the ignore zones.
        interaction_effects_pixels = []
        for change in structured_changes:
            filtered_pixel_changes = []
            for px in change['changes']:
                if (change['row_index'], px['index']) not in ignore_coords:
                    filtered_pixel_changes.append(px)
            
            if filtered_pixel_changes:
                interaction_effects_pixels.append({'row_index': change['row_index'], 'changes': filtered_pixel_changes})

        if not interaction_effects_pixels:
            print(f"-> No observable '{effect_type}' pixel changes found (excluding agent movement and UI).")
            return

        # 3. Convert the filtered pixel changes into whole OBJECTS.
        effect_objects = self._find_and_describe_objects(interaction_effects_pixels, latest_grid, is_interaction_analysis=True)

        # 4. Filter out any objects that are known parts of the agent.
        if current_agent_fingerprints:
            agent_part_fingerprints = current_agent_fingerprints
            
            if agent_part_fingerprints:
                original_count = len(effect_objects)
                # Keep only the objects whose fingerprint is NOT in the set of agent part fingerprints.
                effect_objects = [obj for obj in effect_objects if obj.get('fingerprint') not in agent_part_fingerprints]
                filtered_count = original_count - len(effect_objects)
                if filtered_count > 0:
                    print(f"🕵️‍♂️ Interaction analysis is ignoring {filtered_count} object(s) matching agent parts.")

        # 5. Log the results and synthesize rules.
        print(f"-> Found {len(effect_objects)} object(s) as a result of the '{effect_type}':")
        for i, obj in enumerate(effect_objects):
            pos = (obj['top_row'], obj['left_index'])
            size = (obj['height'], obj['width'])
            tile_pos = (pos[0] // self.tile_size, pos[1] // self.tile_size) if self.tile_size else pos
            print(f"  - Object {i+1}: A {size[0]}x{size[1]} object at pixel {pos} (tile {tile_pos}).")

        if object_signature not in self.interaction_hypotheses:
            self.interaction_hypotheses[object_signature] = {'immediate_effect': [], 'aftermath_effect': [], 'confidence': 0}

        self.interaction_hypotheses[object_signature][effect_type] = effect_objects
        print(f"💡 EFFECT LEARNED: The '{effect_type}' of interacting with '{object_signature}' has been recorded.")
    
    def _analyze_consumable_aftermath(self, latest_grid: list):
        """Checks if the interacted-with object was consumed (i.e., turned to floor)."""
        interacted_tile = self.observing_interaction_for_tile
        if not interacted_tile or not self.tile_size or not latest_grid:
            return

        floor_color = self.world_model.get('floor_color')
        if floor_color is None:
            print("-> Aftermath check skipped: Floor color is not yet known.")
            return

        # Sample the top-left pixel of the tile to determine its current state.
        tile_r, tile_c = interacted_tile
        tile_pixel_r, tile_pixel_c = tile_r * self.tile_size, tile_c * self.tile_size
        current_tile_color = latest_grid[0][tile_pixel_r][tile_pixel_c]

        object_signature = f"tile_pos_{interacted_tile}"

        if current_tile_color == floor_color:
            print(f"✅ Aftermath: Object at tile {interacted_tile} was consumed (turned to floor).")
            self.tile_map[interacted_tile] = CellType.FLOOR
            if object_signature in self.interaction_hypotheses:
                self.interaction_hypotheses[object_signature]['is_consumable'] = True
            self.consumed_tiles_this_life.add(interacted_tile)
        else:
            print(f"🔄 Aftermath: Object at tile {interacted_tile} is persistent (did not turn to floor).")
            if object_signature in self.interaction_hypotheses:
                self.interaction_hypotheses[object_signature]['is_consumable'] = False

    def _find_and_update_patterns(self, dynamic_objects: list, current_grid: list):
        """
        Scans for patterns by matching dynamic objects to static ones and updates the
        agent's list of currently active patterns.
        """
        if not dynamic_objects or not current_grid:
            return

        print(f"🔬 Checking for patterns based on {len(dynamic_objects)} new object(s)...")
        grid_data = current_grid[0]

        # Get fingerprints of patterns already known to be active to avoid adding duplicates.
        known_pattern_fingerprints = {p['fingerprint'] for p in self.active_patterns}

        for dynamic_key_obj in dynamic_objects:
            # If this object appeared on the tile the agent just left, it's a static
            # object being revealed, not a true dynamic key. Ignore it for pattern matching.
            if self.just_vacated_tile and self.tile_size:
                obj_tile = (dynamic_key_obj['top_row'] // self.tile_size, dynamic_key_obj['left_index'] // self.tile_size)
                if obj_tile == self.just_vacated_tile:
                    print(f"🕵️‍♀️ Ignoring pattern check for object at {obj_tile} (revealed by agent movement).")
                    continue

            # --- NEW: Explicitly ignore objects on the resource indicator row ---
            if self.confirmed_resource_indicator:
                indicator_row = self.confirmed_resource_indicator['row_index']
                # Check if the object's vertical span overlaps with the indicator row
                obj_top = dynamic_key_obj['top_row']
                obj_bottom = obj_top + dynamic_key_obj['height']
                if obj_top <= indicator_row < obj_bottom:
                    continue # Skip this object, it's part of the UI.

            dk_fingerprint = dynamic_key_obj.get('fingerprint')
            dk_color = dynamic_key_obj.get('color')
            if dk_fingerprint is None or dk_color is None: continue

            static_candidates = self._find_static_candidates_by_color(dk_color, grid_data)
            if not static_candidates: continue

            for static_obj in static_candidates:
                if dk_fingerprint == static_obj.get('fingerprint'):
                    dk_size = (dynamic_key_obj['height'], dynamic_key_obj['width'])
                    dk_pos = (dynamic_key_obj['top_row'], dynamic_key_obj['left_index'])
                    sk_size = (static_obj['height'], static_obj['width'])
                    sk_pos = (static_obj['top_row'], static_obj['left_index'])

                    if dk_pos == sk_pos and dk_size == sk_size: continue

                    # Create a unique ID for this specific pattern instance to avoid duplicates.
                    pattern_fingerprint = (dk_fingerprint, static_obj.get('fingerprint'), sk_pos)

                    if pattern_fingerprint not in known_pattern_fingerprints:
                        dk_tile = (dk_pos[0] // self.tile_size, dk_pos[1] // self.tile_size) if self.tile_size else dk_pos
                        sk_tile = (sk_pos[0] // self.tile_size, sk_pos[1] // self.tile_size) if self.tile_size else sk_pos
                        print(f"✅ NEW PATTERN DETECTED: Dynamic key at tile {dk_tile} matches static key at tile {sk_tile}!")

                        new_pattern = {
                            'dynamic_key': dynamic_key_obj,
                            'static_key': static_obj,
                            'fingerprint': pattern_fingerprint
                        }
                        self.active_patterns.append(new_pattern)
                        known_pattern_fingerprints.add(pattern_fingerprint)

    def _print_debug_map(self):
        """Prints a human-readable version of the agent's tile_map to the console."""
        if not self.tile_map:
            print("🗺️ Debug Map: No map data to print.")
            return
        
        # The "playable area" includes reachable tiles and their immediate neighbors.
        display_area = set(self.reachable_floor_area)
        for r_tile, c_tile in self.reachable_floor_area:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r_tile + dr, c_tile + dc)
                if neighbor in self.tile_map:
                    display_area.add(neighbor)

        if not display_area:
            print("🗺️ Debug Map: No playable area found to print.")
            return

        print("\n--- Agent's Debug Map ---")
        player_tile = None
        if self.last_known_player_obj and self.tile_size:
            player_tile = (self.last_known_player_obj['top_row'] // self.tile_size, 
                           self.last_known_player_obj['left_index'] // self.tile_size)

        target_tile = None
        if self.exploration_target and self.tile_size:
            target_tile = (self.exploration_target[0] // self.tile_size, 
                           self.exploration_target[1] // self.tile_size)

        min_r = min(r for r, c in display_area)
        max_r = max(r for r, c in display_area)
        min_c = min(c for r, c in display_area)
        max_c = max(c for r, c in display_area)

        for r in range(min_r, max_r + 1):
            row_str = ""
            for c in range(min_c, max_c + 1):
                current_tile = (r,c)
                if current_tile not in display_area:
                    row_str += "   " # Print empty space for non-playable area
                    continue
                if (r, c) == player_tile:
                    row_str += " P "
                elif (r, c) == target_tile:
                    row_str += " T "
                else:
                    cell = self.tile_map.get((r, c), CellType.UNKNOWN)
                    if cell == CellType.FLOOR:
                        row_str += " . "
                    elif cell == CellType.WALL:
                        row_str += " # "
                    elif cell == CellType.POTENTIALLY_INTERACTABLE:
                        row_str += " ? "
                    elif cell == CellType.CONFIRMED_INTERACTABLE:
                        row_str += " ! "
                    elif cell == CellType.RESOURCE:
                        row_str += " R "
                    else: # UNKNOWN
                        row_str += "   "
            print(row_str)
        print("--- Key: P=Player, T=Target, .=Floor, #=Wall, ?=Potential, !=Confirmed ---\n")

    def _review_and_summarize_interactions(self):
        """Reviews all interactable tiles and summarizes their learned properties."""
        print("\n--- 🧠 Interaction Knowledge Summary 🧠 ---")
        
        interactable_tiles = [
            pos for pos, cell_type in self.tile_map.items() 
            if cell_type in [CellType.POTENTIALLY_INTERACTABLE, CellType.CONFIRMED_INTERACTABLE, CellType.RESOURCE]
            and pos in self.reachable_floor_area
        ]

        if not interactable_tiles:
            print("No interactable tiles were identified on the map.")
            self.has_summarized_interactions = True
            return

        for tile_pos in sorted(interactable_tiles):
            signature = f"tile_pos_{tile_pos}"
            hypothesis = self.interaction_hypotheses.get(signature)

            print(f"Tile {tile_pos}:")

            # --- NEW: Special handling for the goal tile ---
            if tile_pos == self.final_tile_of_level:
                print(f"  - Function: Causes new level when interacted with.")
                print(f"  - Type: Unknown (Level Ended Before Aftermath Observed).")
                continue # Skip the normal summary for this tile.

            if not hypothesis:
                print(f"  - Function: Untested.")
                print(f"  - Type: Unknown.")
                continue

            # 1. Report the learned function/effect
            effect_desc = "No effect observed."
            if hypothesis.get('provides_resource'):
                effect_desc = "Resource (fills resource bar)."
            else:
                immediate_effect_objects = hypothesis.get('immediate_effect', [])
                if immediate_effect_objects:
                    # Describe the first object from the effect for a concise summary.
                    obj = immediate_effect_objects[0]
                    size = (obj['height'], obj['width'])
                    pos = (obj['top_row'], obj['left_index'])
                    tile_pos_effect = (pos[0] // self.tile_size, pos[1] // self.tile_size) if self.tile_size else pos
                    effect_desc = f"Causes a {size[0]}x{size[1]} object to appear/change at tile {tile_pos_effect}."
                    if len(immediate_effect_objects) > 1:
                        effect_desc += f" (+{len(immediate_effect_objects) - 1} other effects)."
            
            print(f"  - Function: {effect_desc}")

            # 2. Report the Type (persistence)
            type_desc = "Unknown (aftermath not observed)."
            is_consumable = hypothesis.get('is_consumable')
            if is_consumable is True:
                type_desc = "Consumable (disappears after use)."
            elif is_consumable is False:
                type_desc = "Persistent (remains after use)."
            
            print(f"  - Type: {type_desc}")
        
        print("-----------------------------------------\n")
        self.has_summarized_interactions = True

    def _summarize_level_goal(self):
        """Analyzes and stores the state of the game at the moment a level is won."""
        print("\n--- 🎯 Post-Level Goal Hypothesis ---")

        goal_hypothesis = {
            'goal_tile_interaction': None,
            'active_patterns_at_win': None
        }

        # 1. Capture the final, winning interaction.
        goal_tile = None
        if self.last_known_player_obj and self.tile_size:
            goal_tile = (self.last_known_player_obj['top_row'] // self.tile_size, self.last_known_player_obj['left_index'] // self.tile_size)
            signature = f"tile_pos_{goal_tile}"
            interaction_summary = self.interaction_hypotheses.get(signature)
            goal_hypothesis['goal_tile_interaction'] = interaction_summary

            print(f"Final Interaction: Stepped on tile {goal_tile}.")
            if not interaction_summary:
                print("  -> Note: No specific interaction effect was recorded for this tile.")

        # 2. Capture any active patterns on the grid.
        if self.active_patterns:
            goal_hypothesis['active_patterns_at_win'] = copy.deepcopy(self.active_patterns)
            print(f"Active Pattern at Win-Time:")
            for i, pattern in enumerate(self.active_patterns):
                dk = pattern['dynamic_key']
                sk = pattern['static_key']
                dk_tile = (dk['top_row'] // self.tile_size, dk['left_index'] // self.tile_size)
                sk_tile = (sk['top_row'] // self.tile_size, sk['left_index'] // self.tile_size)
                print(f"  - Pattern {i+1}: Dynamic object at {dk_tile} matched Static object at {sk_tile}.")
                # Future: Could print more detailed characteristics here.
        else:
            print("No active patterns were present at the end of the level.")

        # 3. Store the hypothesis for future learning.
        self.level_goal_hypotheses.append(goal_hypothesis)
        print(f"Hypothesis stored. Total goal hypotheses: {len(self.level_goal_hypotheses)}")
        print("-------------------------------------\n")

    def _analyze_and_update_moves(self, latest_frame: FrameData):
        """Analyzes the resource bar to determine max and current moves."""
        print(f"Debug Moves Analysis: IndicatorConfirmed={bool(self.confirmed_resource_indicator)}, FullStateSaved={self.resource_bar_full_state is not None}, EmptyStateSaved={self.resource_bar_empty_state is not None}")
        # We need all three components to do anything.
        if not self.confirmed_resource_indicator or self.resource_bar_full_state is None or self.resource_bar_empty_state is None:
            return

        # --- One-time analysis to learn about the bar ---
        if self.max_moves == 0:
            full_bar = self.resource_bar_full_state
            empty_bar = self.resource_bar_empty_state

            # Find the pixels that actually change, and what they change to/from.
            # This makes the logic robust against bars with decorative, non-functional pixels.
            for i in range(len(full_bar)):
                if i < len(empty_bar) and full_bar[i] != empty_bar[i]:
                    # This is a functional pixel of the resource bar.
                    self.resource_bar_indices.append(i)
                    # We only need to set the colors once.
                    if self.resource_pixel_color is None:
                        self.resource_pixel_color = full_bar[i]
                        self.resource_empty_color = empty_bar[i]

            if self.resource_bar_indices:
                self.max_moves = len(self.resource_bar_indices)
                print(f"✅ [RESOURCE] Moves logic initialized. Max Moves: {self.max_moves}. Resource Color: {self.resource_pixel_color}, Empty Color: {self.resource_empty_color}.")
            else:
                # This could happen if the full/empty states are identical for some reason.
                print("⚠️ [RESOURCE] Could not determine move count. Full and empty resource bars are identical.")
                return # Exit if we failed to initialize.

        # --- Per-turn update of the current move count ---
        if not latest_frame.frame or not latest_frame.frame[0]:
            return # Cannot update if the frame is empty.

        indicator_row_index = self.confirmed_resource_indicator['row_index']
        
        # Safety check for frame dimensions
        if indicator_row_index >= len(latest_frame.frame[0]):
            return
            
        current_bar = latest_frame.frame[0][indicator_row_index]
        
        current_count = 0
        for i in self.resource_bar_indices:
            if i < len(current_bar) and current_bar[i] == self.resource_pixel_color:
                current_count += 1
        
        self.current_moves = current_count        