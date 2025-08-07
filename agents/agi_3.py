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

        # --- State Management ---
        self.agent_state = AgentState.DISCOVERY
        self.discovery_runs = 0
        self.last_action = None
        self.action_effects = {} # Will store actions and all their resulting changes
        self.action_failures = {}
        self.ineffective_actions = [] # Tracks actions that had no effect since the last success
        self.level_start_frame = None
        self.level_start_score = 0
        self.state_action_map = {} # Maps (grid_state, action) -> resulting_grid_state

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
        """Resets the agent's state for a new life or attempt."""
        self.previous_frame = None
        self.last_action = None

        if self.level_knowledge_is_learned:
            print("--- New Life: Knowledge retained. Resuming with random actions. ---")
            self.agent_state = AgentState.RANDOM_ACTION
        else:
            print("--- New Attempt: Resetting Agent State to DISCOVERY ---")
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
        
        # Convert the current grid to a hashable format for memory storage.
        grid_tuple = tuple(tuple(tuple(p) for p in row) for row in latest_frame.frame)

        # Check if this grid state is new before adding it to memory.
        if grid_tuple not in self.visited_grids:
            print(f"🔎 New grid state discovered! Total unique states seen: {len(self.visited_grids) + 1}")
            self.visited_grids.add(grid_tuple)
        
        # --- 1. Check for Win/Loss State First (Highest Priority) ---
        if latest_frame.state is GameState.WIN:
            print("🏆 Level Solved! Awaiting next level. 🏆")
            # We don't reset here, just wait for the new level to load.
            return self.last_action if self.last_action else GameAction.ACTION1
        
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # If the whole game is new/over, reset everything.
            if latest_frame.state == GameState.NOT_PLAYED:
                self.level_knowledge_is_learned = False
            self._reset_for_new_attempt()
            return GameAction.RESET

        # --- 2. Perception & Consequence of Last Action ---
        changes_found, change_descriptions = self.perceive(latest_frame)

        # --- Special Handling for Dimension Changes (New Level or Lost Life) ---
        if changes_found and change_descriptions == ["Frame dimensions changed"]:
            new_grid = latest_frame.frame
            new_score = latest_frame.score

            # A new level is detected if the grid changes AND the score increases[cite: 42].
            is_new_level = (new_grid != self.level_start_frame and new_score > self.level_start_score)

            if is_new_level:
                print(f"--- New Level Detected! Score increased to {new_score}. ---")
                # Update the baseline for the new level
                self.level_start_frame = copy.deepcopy(new_grid)
                self.level_start_score = new_score
                # Attribute success to the last action and describe it as a level advance.
                change_descriptions = [f"Advanced to a new level with score {new_score}."]
            else:
                # If the state matches the start of the level, it's a lost life[cite: 41].
                print("--- Lost a Life (Frame reset to level start). Resetting attempt. ---")
                self._reset_for_new_attempt()
                return GameAction.RESET

        # Process the result of the last action.
        if self.last_action:
            # NEW: Record the state transition for our map
            if self.previous_frame:
                previous_grid_tuple = tuple(tuple(tuple(p) for p in row) for row in self.previous_frame)
                if previous_grid_tuple not in self.state_action_map:
                    self.state_action_map[previous_grid_tuple] = {}
                if self.last_action not in self.state_action_map[previous_grid_tuple]:
                    self.state_action_map[previous_grid_tuple][self.last_action] = grid_tuple
                    print(f"🗺️ Mapped transition: From state {hash(previous_grid_tuple)} via {self.last_action.name} to state {hash(grid_tuple)}.")
            
            if changes_found:
                # --- Action SUCCEEDED (caused a change) ---
                was_cleared = len(self.ineffective_actions) > 0
                self.ineffective_actions.clear()
                clear_message = " Clearing ineffective actions list." if was_cleared else ""

                if self.agent_state == AgentState.DISCOVERY:
                    self.action_effects[self.last_action] = change_descriptions
                    self.discovered_in_current_run = True
                    print(f"Action {self.last_action.name} caused {len(change_descriptions)} changes. Storing success.{clear_message}")
                else: # RANDOM_ACTION state
                    print(f"Known action {self.last_action.name} succeeded, causing {len(change_descriptions)} changes.{clear_message}")

                for description in change_descriptions[:10]:
                    print(description)
                if len(change_descriptions) > 10:
                    print("  - ...and more.")

            else:
                # --- Action FAILED (caused no change) ---
                if self.last_action not in self.ineffective_actions:
                    self.ineffective_actions.append(self.last_action)

                if self.agent_state == AgentState.RANDOM_ACTION:
                    print(f"Known action {self.last_action.name} had no effect. Storing failure context. Ineffective actions: {[a.name for a in self.ineffective_actions]}")
                    context = copy.deepcopy(self.previous_frame)
                    if self.last_action not in self.action_failures:
                        self.action_failures[self.last_action] = []
                    self.action_failures[self.last_action].append(context)

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

        # If discovery is over, use curiosity-driven exploration
        action = GameAction.ACTION1 # Default failsafe action
        if self.action_effects:
            known_actions = list(self.action_effects.keys())
            actions_taken_from_here = self.state_action_map.get(grid_tuple, {})
            
            # Prioritize actions not yet taken from this specific grid
            untried_actions = [a for a in known_actions if a not in actions_taken_from_here]
            candidate_actions = [a for a in untried_actions if a not in self.ineffective_actions]

            if candidate_actions:
                print(f"🧭 Prioritizing untried actions: {[a.name for a in candidate_actions]}")
                action = random.choice(candidate_actions)
            else:
                # Fallback to choosing from any known, effective action
                print("🔂 No novel actions to try from this state. Falling back to known effective actions.")
                effective_actions = [a for a in known_actions if a not in self.ineffective_actions]
                
                # If all known actions become ineffective, reset the list to escape getting stuck
                if not effective_actions and known_actions:
                    print("--- All known actions are ineffective. Resetting list and trying again. ---")
                    self.ineffective_actions.clear()
                    effective_actions = known_actions
                
                if effective_actions:
                    action = random.choice(effective_actions)
        else:
            # Fallback if no actions are known to have effects yet
            all_possible_actions = self.primary_actions + self.secondary_actions
            if all_possible_actions:
                action = random.choice(all_possible_actions)

        self.last_action = action
        return action

    # --- Methods from your original plan ---

    def perceive(self, latest_frame: FrameData) -> tuple[bool, list[str]]:
        """Compares the current frame with the previous one to detect changes.
        Returns a tuple: (bool_if_changes_found, list_of_change_descriptions)."""
        current_frame = latest_frame.frame
        changes_found = False
        change_descriptions = []

        if not current_frame:
            return False, []

        if self.previous_frame is None:
            self.previous_frame = copy.deepcopy(current_frame)
            return False, []

        # Check for dimension changes first
        current_height, current_width = len(current_frame), len(current_frame[0])
        prev_height, prev_width = len(self.previous_frame), len(self.previous_frame[0])
        if current_height != prev_height or current_width != prev_width:
            print("--- Frame dimensions changed! Analyzing... ---")
            self.previous_frame = copy.deepcopy(current_frame)
            return True, ["Frame dimensions changed"]

        # Find and describe pixel changes with index-level detail
        for r in range(current_height):
            for c in range(current_width):

                old_pixel_data = self.previous_frame[r][c]
                new_pixel_data = current_frame[r][c]

                if old_pixel_data != new_pixel_data:
                    changes_found = True
                    change_descriptions.append(f"  - Changes at coordinate ({r},{c}):")

                    # Log every single difference inside the pixel's data list
                    if len(old_pixel_data) == len(new_pixel_data):
                        for i in range(len(old_pixel_data)):
                            if old_pixel_data[i] != new_pixel_data[i]:
                                old_val = old_pixel_data[i]
                                new_val = new_pixel_data[i]
                                change_descriptions.append(f"    - Index {i}: From {old_val} to {new_val}")
                    else:
                        change_descriptions.append(f"    - Data lists changed length.")

        # Save a deep copy for the next comparison
        self.previous_frame = copy.deepcopy(current_frame)

        return changes_found, change_descriptions

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