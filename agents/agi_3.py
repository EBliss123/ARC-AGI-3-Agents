# ARC-AGI-3 Main Script
import random
import copy
from enum import Enum
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
        self.actions_to_try = self.primary_actions.copy()
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
            # Reset for the next full run, starting with primary actions
            self.actions_to_try = self.primary_actions.copy()
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

        if self.level_knowledge_is_learned:
            self.agent_state = AgentState.RANDOM_ACTION
        else:
            self.agent_state = AgentState.DISCOVERY
            self.discovery_runs = 0
            self.actions_to_try = self.primary_actions.copy()
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
            for r, c in self.ignore_for_state_hash:
                # Ensure the coordinate is within the current grid's bounds before masking.
                if r < len(grid_for_hashing) and c < len(grid_for_hashing[0]):
                    grid_for_hashing[r][c] = [-1] # Use a constant, non-game value.

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

        current_height, current_width = len(current_frame), len(current_frame[0])
        prev_height, prev_width = len(self.previous_frame), len(self.previous_frame[0])
        if current_height != prev_height or current_width != prev_width:
            print("--- Frame dimensions changed! Analyzing... ---")
            self.previous_frame = copy.deepcopy(current_frame)
            # A dimension change is always considered a novel event.
            return True, False, ["Frame dimensions changed"], []

        for r in range(current_height):
            for c in range(current_width):
                old_pixel_data = self.previous_frame[r][c]
                new_pixel_data = current_frame[r][c]

                if old_pixel_data != new_pixel_data:
                    # First, create structured data for ALL changes.
                    pixel_level_changes = []
                    if len(old_pixel_data) == len(new_pixel_data):
                        for i in range(len(old_pixel_data)):
                            if old_pixel_data[i] != new_pixel_data[i]:
                                pixel_level_changes.append({'index': i, 'from': old_pixel_data[i], 'to': new_pixel_data[i]})
                    if pixel_level_changes:
                        all_structured_changes.append({'coord': (r, c), 'changes': pixel_level_changes})

                    # Second, check if it's a known indicator change or a novel one.
                    if self.confirmed_resource_indicator and (r, c) == self.confirmed_resource_indicator['coord']:
                        known_changes_found = True
                    else:
                        # For novel changes, also create human-readable descriptions for logging.
                        novel_changes_found = True
                        novel_change_descriptions.append(f"  - Changes at coordinate ({r},{c}):")
                        if len(old_pixel_data) == len(new_pixel_data):
                             for change in pixel_level_changes:
                                 novel_change_descriptions.append(f"    - Index {change['index']}: From {change['from']} to {change['to']}")
                        else:
                            novel_change_descriptions.append(f"    - Data lists changed length.")

        self.previous_frame = copy.deepcopy(current_frame)
        return novel_changes_found, known_changes_found, novel_change_descriptions, all_structured_changes

    def _update_resource_indicator_tracking(self, structured_changes: list, action: GameAction):
        """Analyzes changes to find a resource indicator, which depletes on any action."""
        if self.confirmed_resource_indicator or not action:
            return

        changed_coords = {change['coord'] for change in structured_changes}

        # 1. Prune candidates that were expected to change but didn't.
        # Any existing candidate that did NOT change on this turn is inconsistent.
        keys_to_remove = []
        for coord in self.resource_indicator_candidates:
            if coord not in changed_coords:
                print(f"📉 Candidate at {coord} was inconsistent (did not change), removing.")
                keys_to_remove.append(coord)
        
        for key in keys_to_remove:
            del self.resource_indicator_candidates[key]

        # 2. Check all changes for potential indicator patterns
        for change in structured_changes:
            coord = change['coord']
            # This pattern requires a single, distinct index change within the coordinate
            if len(change['changes']) != 1:
                continue
            
            detail = change['changes'][0]
            current_index, old_val, new_val = detail['index'], detail['from'], detail['to']

            if not isinstance(new_val, (int, float)) or not isinstance(old_val, (int, float)):
                continue

            value_direction = 'inc' if new_val > old_val else 'dec'

            if coord in self.resource_indicator_candidates:
                # --- Update Existing Candidate ---
                candidate = self.resource_indicator_candidates[coord]
                
                # Check for consistency in value change direction (action no longer matters)
                if candidate['value_direction'] == value_direction:
                    index_direction = 'inc' if current_index > candidate['last_index'] else 'dec'

                    # On the 2nd hit, establish the index direction
                    if candidate.get('index_direction') is None:
                        candidate['index_direction'] = index_direction
                        candidate['confidence'] += 1
                    # On subsequent hits, ensure the index direction is also consistent
                    elif candidate['index_direction'] == index_direction:
                        candidate['confidence'] += 1
                    else:
                        # Index direction changed, this is not a consistent indicator
                        del self.resource_indicator_candidates[coord]
                        continue
                    
                    candidate['last_index'] = current_index
                    print(f"📈 Resource candidate {coord} confidence is now {candidate['confidence']}.")

                    if candidate['confidence'] >= self.RESOURCE_CONFIDENCE_THRESHOLD:
                        self.confirmed_resource_indicator = {'coord': coord, **candidate}
                        self.ignore_for_state_hash.add(coord)
                        print(f"✅ Confirmed resource indicator at coordinate {coord}! It will now be ignored for state uniqueness checks.")
                        self.resource_indicator_candidates.clear()
                        return
                else:
                    # Value direction was inconsistent, remove
                    del self.resource_indicator_candidates[coord]
            else:
                # --- Add New Candidate ---
                self.resource_indicator_candidates[coord] = {
                    'confidence': 1,
                    'last_index': current_index,
                    'value_direction': value_direction,
                    'index_direction': None # Will be determined on the next consistent change
                }
                print(f"🤔 New resource candidate found at coordinate {coord}.")

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