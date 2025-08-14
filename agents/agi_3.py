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

class CellType(Enum):
    """Represents the classification of a single grid cell."""
    UNKNOWN = 0
    FLOOR = 1
    WALL = 2
    POTENTIALLY_INTERACTABLE = 3 # For identified but not fully understood objects
    PLAYER = 4
    CONFIRMED_INTERACTABLE = 5

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
        self.RESOURCE_CONFIDENCE_THRESHOLD = 3 # Actions in a row to confirm
        self.level_knowledge_is_learned = False
        self.wait_action = GameAction.ACTION6 # Use a secondary action for waiting

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

        # --- Interaction Learning ---
        self.observing_interaction_for_tile = None # Stores the coords of the tile being observed
        self.interaction_observation_phase = None # Can be 'IMMEDIATE' or 'AFTERMATH'
        self.interaction_hypotheses = {} # signature -> {'immediate_effect': [], 'aftermath_effect': [], 'confidence': 0}

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

    def _reset_for_new_level(self):
        """Resets all level-specific knowledge for a new level, preserving core learned concepts."""
        # 1. Reset transient state and all object position data.
        self.previous_frame = None
        self.last_action = None
        self.last_known_objects = []
        self.last_known_player_obj = None # Make sure agent has to re-find itself on the new map.

        # 2. Set the agent state to use its existing knowledge.
        # Since the world model is preserved, we can skip discovery and move
        # directly to intelligent action/exploration.
        print("🧠 Knowledge preserved. Skipping discovery and entering action state.")
        self.agent_state = AgentState.RANDOM_ACTION
        
        # 3. Reset all map and exploration data. This "forgets the layout".
        print("🧹 Wiping all level layout and exploration data.")
        self.tile_map = {}
        self.exploration_phase = ExplorationPhase.INACTIVE
        self.exploration_target = None
        self.exploration_plan = []
        self.reachable_floor_area = set()
        self.interaction_hypotheses.clear() # Interaction effects are tied to the specific layout.

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """This is the main decision-making method for the AGI."""
        # --- 1. Store initial level state if not already set ---
        if self.level_start_frame is None:
            # Only store the initial frame if it actually contains data.
            if latest_frame.frame:
                print("--- New Level Detected. Storing initial valid frame and score. ---")
                self.level_start_frame = copy.deepcopy(latest_frame.frame)
                self.level_start_score = latest_frame.score
            else:
                # If the frame is blank, print a message but do nothing else.
                # This allows the normal action-selection logic below to run,
                # preventing the VALIDATION_ERROR. We'll try to store the frame
                # again on the next turn.
                print("--- Ignoring blank starting frame. Waiting for a valid one... ---")
        
        # --- NEW, SIMPLIFIED NEW-LEVEL DETECTION ---
        # If the score has increased at any point, assume it's a new level and reset.
        if latest_frame.score > self.level_start_score:
            print(f"--- New Level Detected (Score Increased)! Resetting layout knowledge. ---")
            self.level_start_frame = copy.deepcopy(latest_frame.frame)
            self.level_start_score = latest_frame.score
            self._reset_for_new_level()
            return self.wait_action
        
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
                                        print(f"  - Object {i+1}: A {size[0]}x{size[1]} object at position {pos}.")

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
        novel_changes_found, known_changes_found, change_descriptions, structured_changes = self.perceive(latest_frame)

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

        # --- Check for massive changes indicating a transition ---

        # --- Check for massive changes indicating a transition ---
        if novel_changes_found:
            is_dimension_change = "Frame dimensions changed" in change_descriptions
            if len(change_descriptions) > self.MASSIVE_CHANGE_THRESHOLD or is_dimension_change:
                print(f"💥 Massive change detected ({len(change_descriptions)} changes). Waiting for stability...")
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

            moved_agent_this_turn = None
            # 2. Track objects from the last frame to the current one
            if self.last_known_objects: # Can only track if we have a "before" state
                tracking_logs, moved_agent_this_turn = self._track_objects(current_objects, self.last_known_objects, latest_frame.frame, object_logic_changes, self.last_action)
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

            # 4. Update the player object's known position
            # If the tracker saw the agent move, use that new position.
            if moved_agent_this_turn:
                self.last_known_player_obj = moved_agent_this_turn
            # 4. Update the player object's known position
            # If the tracker saw the agent move, use that new position.
            if moved_agent_this_turn:
                self.last_known_player_obj = moved_agent_this_turn
            # If no agent moved this turn (e.g., after a reset), we don't know its
            # location. We must wait for movement to find it again.
            else:
                self.last_known_player_obj = None

        # --- Intelligent Exploration Logic ---
        # Check if the agent has learned enough to begin exploring intelligently.
        can_explore = (self.world_model.get('player_signature') and
               self.world_model.get('floor_color') and
               self.world_model.get('action_map') and
               self.last_known_player_obj and
               self.tile_size) 

        if can_explore and self.exploration_phase == ExplorationPhase.INACTIVE:
            print("🤖 World model is sufficiently complete. Activating exploration phase.")
            self.exploration_phase = ExplorationPhase.BUILDING_MAP

        # --- Handle Exploration Phases ---
        if self.exploration_phase != ExplorationPhase.INACTIVE:
            # If a plan is being executed, prioritize it.
            if self.exploration_phase == ExplorationPhase.EXECUTING_PLAN:
                if self.exploration_plan:
                    action = self.exploration_plan.pop(0)
                    print(f"🗺️ Executing plan: {action.name}. {len(self.exploration_plan)} steps remaining.")
                    self.last_action = action
                    self.last_grid_tuple = grid_tuple
                    return action
                else:
                    # This block handles the completion of a plan.
                    if self.interaction_observation_phase == 'AFTERMATH':
                        # This means the "step away" plan just finished.
                        print("-> Stepped away. Observing aftermath...")
                        self._analyze_and_log_interaction_effect(structured_changes, 'aftermath_effect')
                        
                        # End the full observation cycle and return to normal exploration.
                        self.observing_interaction_for_tile = None
                        self.interaction_observation_phase = None
                        self.exploration_phase = ExplorationPhase.BUILDING_MAP
                    
                    else:
                        # This means a normal exploration plan to an interactable has finished.
                        print("✅ Plan complete. Beginning interaction observation.")
                        if self.exploration_target and self.tile_size:
                            target_tile = (self.exploration_target[0] // self.tile_size, self.exploration_target[1] // self.tile_size)

                            if self.tile_map.get(target_tile) == CellType.POTENTIALLY_INTERACTABLE:
                                self.tile_map[target_tile] = CellType.CONFIRMED_INTERACTABLE
                                print(f"✅ Target at {target_tile} confirmed as interactable.")

                            # Set the tile we're observing.
                            self.observing_interaction_for_tile = target_tile
                            
                            # --- Observation 1: Analyze immediate effects ---
                            # The 'structured_changes' from the top of the function are from landing on the tile.
                            self._analyze_and_log_interaction_effect(structured_changes, 'immediate_effect')

                            # --- Create a new micro-plan to step away for Observation 2 ---
                            move_away_plan = self._plan_step_away(target_tile)
                            if move_away_plan:
                                print(f"-> Planning to step away ({move_away_plan[0].name}) to observe aftermath.")
                                self.exploration_plan = move_away_plan
                                self.interaction_observation_phase = 'AFTERMATH'
                                self.exploration_phase = ExplorationPhase.EXECUTING_PLAN # Trigger execution of the new plan
                            else:
                                # If we can't step away, the observation cycle ends here.
                                print("-> Cannot find path to step away. Ending interaction observation.")
                                self.observing_interaction_for_tile = None
                                self.interaction_observation_phase = None
                                self.exploration_phase = ExplorationPhase.BUILDING_MAP
                        else:
                            # If there was no target, just go back to mapping.
                            self.exploration_phase = ExplorationPhase.BUILDING_MAP
            # Build or rebuild the map if needed.
            if self.exploration_phase == ExplorationPhase.BUILDING_MAP:
                print("🗺️ Building/updating the level map...")
                self._build_level_map(latest_frame.frame)
                self.exploration_phase = ExplorationPhase.SEEKING_TARGET

            # Find a new target and create a plan.
            if self.exploration_phase == ExplorationPhase.SEEKING_TARGET:
                print("🗺️ Seeking a new exploration target...")
                target_found = self._find_target_and_plan()
                if target_found:
                    print(f"🎯 New target acquired at {self.exploration_target}. Plan created with {len(self.exploration_plan)} steps.")
                    self.exploration_phase = ExplorationPhase.EXECUTING_PLAN
                    # Execute the first step of the new plan immediately.
                    if self.exploration_plan:
                        action = self.exploration_plan.pop(0)
                        print(f"🗺️ Executing plan: {action.name}. {len(self.exploration_plan)} steps remaining.")
                        self.last_action = action
                        self.last_grid_tuple = grid_tuple
                        return action
                else:
                    print("🧐 No more unknown objects to explore. Reverting to random discovery.")
                    self.exploration_phase = ExplorationPhase.INACTIVE
                    # Fall through to the default random action state

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
        Creates a refined tile-based map. First, it maps the grid by color,
        then finds the player's reachable floor area, and finally re-classifies
        all tiles based on their relationship to that reachable area.
        """
        if not self.tile_size or not grid:
            return
        
        # Preserve tiles that have already been confirmed as interactable before rebuilding.
        confirmed_interactables = {pos for pos, cell_type in self.tile_map.items() if cell_type == CellType.CONFIRMED_INTERACTABLE}

        # --- Pass 1: Initial color-based classification ---
        # We create a temporary map to help the floor-finding algorithm.
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
                sample_color = grid_data[r][c]
                tile_coords = (r // self.tile_size, c // self.tile_size)

                if sample_color in wall_colors:
                    temp_tile_map[tile_coords] = CellType.WALL
                elif sample_color == floor_color:
                    temp_tile_map[tile_coords] = CellType.FLOOR
                else:
                    # Tentatively classify as potentially interactable
                    temp_tile_map[tile_coords] = CellType.POTENTIALLY_INTERACTABLE
        
        # Use the temporary map to find the reachable floor area
        self.tile_map = temp_tile_map
        self.reachable_floor_area = self._find_reachable_floor_tiles()

        # --- Pass 2: Refine the map based on reachability ---
        if not self.reachable_floor_area:
            print("🗺️ No reachable area found. Map will not be refined.")
            # We leave the color-based map as is if nothing is reachable.

        refined_tile_map = {}
        all_tile_coords = list(self.tile_map.keys())

        for tile_coords in all_tile_coords:
            # Rule 0: Preserve confirmed interactables above all else.
            if tile_coords in confirmed_interactables:
                refined_tile_map[tile_coords] = CellType.CONFIRMED_INTERACTABLE
                continue

            # Rule 1: Any tile within the reachable area is FLOOR.
            if tile_coords in self.reachable_floor_area:
                refined_tile_map[tile_coords] = CellType.FLOOR
                continue

            # Rule 2: For other tiles, check if they are adjacent to the floor.
            is_adjacent_to_floor = False
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_tile = (tile_coords[0] + dr, tile_coords[1] + dc)
                if neighbor_tile in self.reachable_floor_area:
                    is_adjacent_to_floor = True
                    break
            
            if is_adjacent_to_floor:
                # It's a WALL if its color matches known wall colors.
                if temp_tile_map.get(tile_coords) == CellType.WALL:
                    refined_tile_map[tile_coords] = CellType.WALL
                # Otherwise, check the color before assuming it's interactable.
                else:
                    # This handles cases where a floor tile wasn't reached by the
                    # initial flood fill, correctly classifying it as FLOOR.
                    sample_color = grid_data[tile_coords[0] * self.tile_size][tile_coords[1] * self.tile_size]
                    if sample_color == floor_color:
                        refined_tile_map[tile_coords] = CellType.FLOOR
                    else:
                        refined_tile_map[tile_coords] = CellType.POTENTIALLY_INTERACTABLE
            else:
                # Rule 3: If it's not floor and not adjacent, it's UNKNOWN.
                refined_tile_map[tile_coords] = CellType.UNKNOWN

         # --- Pass 3: Final Verification Safety Net ---
        # Check all interactable candidates and correct them if their color matches a known type.
        grid_data = grid[0]
        floor_color = self.world_model.get('floor_color')
        wall_colors = self.world_model.get('wall_colors', set())
        
        # We iterate over a copy of the items since we are modifying the dictionary.
        for tile_coords, cell_type in list(refined_tile_map.items()):
            if cell_type in [CellType.POTENTIALLY_INTERACTABLE, CellType.CONFIRMED_INTERACTABLE]:
                # Sample the tile's color from the grid's raw data.
                sample_color = grid_data[tile_coords[0] * self.tile_size][tile_coords[1] * self.tile_size]
                
                # If the color is a known type, correct the classification.
                if floor_color and sample_color == floor_color:
                    refined_tile_map[tile_coords] = CellType.FLOOR
                elif wall_colors and sample_color in wall_colors:
                    refined_tile_map[tile_coords] = CellType.WALL

        # Update the agent's main tile map with the refined version.
        self.tile_map = refined_tile_map
        counts = Counter(self.tile_map.values())
        print(f"🗺️ Refined Map ({len(self.tile_map)} tiles): {counts[CellType.FLOOR]} floor, {counts[CellType.WALL]} wall, {counts.get(CellType.POTENTIALLY_INTERACTABLE, 0)} potentially interactable, {counts.get(CellType.CONFIRMED_INTERACTABLE, 0)} interactable.")
    
    def _find_target_and_plan(self) -> bool:
        """
        Finds the best interactable tile by pathfinding to all available targets
        and picking the one with the shortest path.
        """

        self._print_debug_map()
        
        self.exploration_target = None
        self.exploration_plan = []
        if not self.last_known_player_obj or not self.tile_size:
            return False

        player_pixel_pos = (self.last_known_player_obj['top_row'], self.last_known_player_obj['left_index'])
        player_tile_pos = (player_pixel_pos[0] // self.tile_size, player_pixel_pos[1] // self.tile_size)

        # 1. Find all potential targets on the map.
        potential_targets = [pos for pos, type in self.tile_map.items() if type == CellType.POTENTIALLY_INTERACTABLE]
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
            return False

        # 3. From the list of reachable targets, select the one with the shortest path.
        best_target = min(reachable_targets, key=lambda t: len(t['path']))
        
        # 4. Set the exploration plan based on the best target found.
        self.exploration_target = (best_target['pos'][0] * self.tile_size, best_target['pos'][1] * self.tile_size)
        self.exploration_plan = best_target['path']
        print(f"🎯 New target acquired at {best_target['pos']}. Plan created with {len(self.exploration_plan)} steps.")
        
        return True

    def _find_path_to_target(self, start_tile: tuple, target_tile: tuple) -> list:
        """
        Finds the shortest sequence of actions to reach a target using the learned action model.
        """
        # 1. Build a fresh map from TILE vectors to actions at the start of every call.
        # This ensures we always use the latest learned movement abilities.
        tile_vector_to_action = {}
        if self.tile_size and self.world_model.get('action_map'):
            for action, effect in self.world_model['action_map'].items():
                if 'move_vector' in effect:
                    px_vec = effect['move_vector']
                    # Convert the pixel vector to a tile vector
                    tile_vec = (px_vec[0] // self.tile_size, px_vec[1] // self.tile_size)
                    # Only store actions that result in actual tile movement
                    if tile_vec != (0, 0):
                        tile_vector_to_action[tile_vec] = action

        if not tile_vector_to_action:
            # This will only be printed if the agent genuinely has no learned move actions.
            return []

        # 2. Perform a Breadth-First Search (BFS) to find the shortest path of actions.
        q = [(start_tile, [])]  # Queue stores (current_tile, path_of_actions_so_far)
        visited = {start_tile}

        while q:
            current_tile, path = q.pop(0)

            if current_tile == target_tile:
                return path  # Path of actions found!

            # 3. Explore neighbors using the learned actions.
            for tile_vec, action in tile_vector_to_action.items():
                neighbor_tile = (current_tile[0] + tile_vec[0], current_tile[1] + tile_vec[1])

                tile_type = self.tile_map.get(neighbor_tile)
                can_move_to = tile_type in [CellType.FLOOR, CellType.POTENTIALLY_INTERACTABLE]

                if can_move_to and neighbor_tile not in visited:
                    visited.add(neighbor_tile)
                    # Add the action that gets to the neighbor to the path
                    q.append((neighbor_tile, path + [action]))

        return []
    
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
        if start_tile_type not in [CellType.FLOOR, CellType.POTENTIALLY_INTERACTABLE]:
            print(f"⚠️ Player starting tile {start_tile} is not on a known FLOOR or INTERACTABLE. Aborting flood fill.")
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
                if self.tile_map.get(neighbor_tile) == CellType.FLOOR and neighbor_tile not in visited:
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

    def _find_and_describe_objects(self, structured_changes: list, latest_frame: list) -> list[dict]:
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

                        # --- Map Cleanup Logic ---
                        if self.tile_size:
                            reclassified_count = 0
                            grid_data = latest_frame[0]
                            for tile_coords, cell_type in list(self.tile_map.items()):
                                if cell_type != CellType.FLOOR:
                                    tile_row = tile_coords[0] * self.tile_size
                                    tile_col = tile_coords[1] * self.tile_size
                                    if 0 <= tile_row < len(grid_data) and 0 <= tile_col < len(grid_data[0]):
                                        tile_color = grid_data[tile_row][tile_col]
                                        if tile_color == obj_color:
                                            self.tile_map[tile_coords] = CellType.FLOOR
                                            reclassified_count += 1
                            if reclassified_count > 0:
                                print(f"🧹 Map Cleanup: Reclassified {reclassified_count} tile(s) as newly confirmed floor.")

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

            original_color = from_color_map.get(sample_point) # Get original color from our map

            final_objects.append({
                'height': height, 'width': width, 'top_row': min_row,
                'left_index': min_idx, 'color': obj_color,
                'original_color': original_color,
                'data_map': data_map,
                'background_color': background_color
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
        moved_agent_obj = None
        if not true_moves:
            return log_messages, moved_agent_obj

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

                    # 1. Combine the agent's fragments into a single bounding box
                    agent_parts = [p[0] for p in move['parts']] if move['type'] == 'composite' else [move['last_obj']]
                    
                    min_row = min(p['top_row'] for p in agent_parts)
                    max_row = max(p['top_row'] + p['height'] for p in agent_parts)
                    min_col = min(p['left_index'] for p in agent_parts)
                    max_col = max(p['left_index'] + p['width'] for p in agent_parts)

                    moved_agent_obj = {
                        'height': max_row - min_row, 'width': max_col - min_col,
                        'top_row': min_row, 'left_index': min_col
                    }

                    # 2. Learn the action effect (if not already known)
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
                            if self.tile_size is None:
                                # The tile size is the magnitude of the move vector.
                                # We take the absolute value and check both components of the vector.
                                size_from_vector = abs(effect_vector[0]) + abs(effect_vector[1])
                                if size_from_vector > 1: # A move of 1 is ambiguous
                                    self.tile_size = size_from_vector
                                    print(f"📏 Tile Size Discovered: {self.tile_size}px based on agent movement.")
                            del self.action_effect_hypothesis[action] # Clean up memory

                    break # Agent's move found, no need to check other moves.

        return log_messages, moved_agent_obj
    
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

        # --- 4. If no known walls were hit, proceed with new wall discovery ---
        wall_candidate_color = blocking_colors.most_common(1)[0][0]

        self.wall_hypothesis[wall_candidate_color] = self.wall_hypothesis.get(wall_candidate_color, 0) + 1
        confidence = self.wall_hypothesis[wall_candidate_color]
        wall_name = "Out of Bounds" if wall_candidate_color == -1 else f"Color {wall_candidate_color}"
        print(f"🧱 Wall Hypothesis: {wall_name} blocked movement (Confidence: {confidence}).")

        # --- 5. Confirm Hypothesis if Threshold is Met ---
        if confidence >= self.CONCEPT_CONFIDENCE_THRESHOLD:
            self.world_model['wall_colors'].add(wall_candidate_color)
            print(f"✅ [WALL] Confirmed: {wall_name} is a wall.")

            # --- New Map Cleanup Logic ---
            if self.tile_size:
                reclassified_count = 0
                # Iterate over a copy of the items because the dictionary size may change.
                for tile_coords, cell_type in list(self.tile_map.items()):
                    if cell_type in [CellType.POTENTIALLY_INTERACTABLE, CellType.CONFIRMED_INTERACTABLE]:
                        # Get the top-left pixel of the tile to sample its color from the last grid state.
                        tile_row = tile_coords[0] * self.tile_size
                        tile_col = tile_coords[1] * self.tile_size
                        
                        # Ensure coordinates are within the grid bounds before checking.
                        if 0 <= tile_row < len(last_grid[0]) and 0 <= tile_col < len(last_grid[0][0]):
                            tile_color = last_grid[0][tile_row][tile_col]
                            if tile_color == wall_candidate_color:
                                self.tile_map[tile_coords] = CellType.WALL
                                reclassified_count += 1
                
                if reclassified_count > 0:
                    print(f"🧹 Map Cleanup: Reclassified {reclassified_count} interactable tile(s) as newly confirmed walls.")

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

    def _plan_step_away(self, from_tile: tuple) -> list:
        """Finds an adjacent floor tile and returns a one-step plan to move there."""
        if not self.world_model.get('action_map'):
            return []

        # Invert the action map for easy lookup: tile_vector -> action
        tile_vector_to_action = {}
        if self.tile_size:
            for action, effect in self.world_model['action_map'].items():
                if 'move_vector' in effect:
                    px_vec = effect['move_vector']
                    tile_vec = (px_vec[0] // self.tile_size, px_vec[1] // self.tile_size)
                    if tile_vec != (0, 0):
                        tile_vector_to_action[tile_vec] = action
        
        # Find a valid move action that leads to a floor tile.
        for tile_vec, action in tile_vector_to_action.items():
            neighbor_tile = (from_tile[0] + tile_vec[0], from_tile[1] + tile_vec[1])
            if self.tile_map.get(neighbor_tile) == CellType.FLOOR:
                return [action] # Return a plan with just this one action.

        return [] # Return empty list if no escape route is found.

    def _analyze_and_log_interaction_effect(self, structured_changes: list, effect_type: str):
        """Analyzes pixel changes from an interaction and logs them as a hypothesis."""
        if self.observing_interaction_for_tile is None:
            return

        # 1. Get a simple signature for the object that was interacted with.
        observed_tile = self.observing_interaction_for_tile
        object_signature = None
        if self.tile_size and self.previous_frame:
            tile_top_row = observed_tile[0] * self.tile_size
            tile_left_index = observed_tile[1] * self.tile_size
            
            # Use the color of the tile (before any potential change) as a simple signature.
            sample_color = self.previous_frame[0][tile_top_row][tile_left_index]
            object_signature = f"tile_color_{sample_color}"
        
        if not object_signature:
            return

        # 2. Filter out changes caused by the player's own movement.
        interaction_effects = []
        player_coords = set()
        if self.last_known_player_obj:
            player_box = self.last_known_player_obj
            for r in range(player_box['top_row'], player_box['top_row'] + player_box['height']):
                for c in range(player_box['left_index'], player_box['left_index'] + player_box['width']):
                    player_coords.add((r,c))

        for change in structured_changes:
            change_is_on_player = False
            for px_change in change['changes']:
                if (change['row_index'], px_change['index']) in player_coords:
                    change_is_on_player = True
                    break
            if not change_is_on_player:
                interaction_effects.append(change)

        if not interaction_effects:
            print(f"-> No observable '{effect_type}' pixel changes found (excluding player movement).")
            return

        print(f"-> Found {len(interaction_effects)} raw pixel changes for '{effect_type}'.")

        # 3. Log the raw pixel changes as a hypothesis.
        if object_signature not in self.interaction_hypotheses:
            self.interaction_hypotheses[object_signature] = {'immediate_effect': [], 'aftermath_effect': [], 'confidence': 0}
        
        # Store the raw change data.
        self.interaction_hypotheses[object_signature][effect_type] = interaction_effects
        print(f"📖 Hypothesis logged for '{object_signature}': {effect_type} has been recorded.")
    
    
    def _print_debug_map(self):
        """Prints a human-readable version of the agent's tile_map to the console."""
        if not self.tile_map:
            print("🗺️ Debug Map: No map data to print.")
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

        min_r = min(r for r, c in self.tile_map.keys())
        max_r = max(r for r, c in self.tile_map.keys())
        min_c = min(c for r, c in self.tile_map.keys())
        max_c = max(c for r, c in self.tile_map.keys())

        for r in range(min_r, max_r + 1):
            row_str = ""
            for c in range(min_c, max_c + 1):
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
                    else: # UNKNOWN
                        row_str += "   "
            print(row_str)
        print("--- Key: P=Player, T=Target, .=Floor, #=Wall, ?=Potential, !=Confirmed ---\n")

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