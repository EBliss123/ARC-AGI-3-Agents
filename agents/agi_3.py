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

        # --- State Management ---
        self.agent_state = AgentState.DISCOVERY
        self.discovery_runs = 0
        self.last_action = None
        self.action_effects = {} # Will store actions and all their resulting changes
        self.action_failures = {}

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

        print(f"Custom AGI initialized for game: {self.game_id}")

    def _end_discovery_run(self):
        """Helper method to finalize a discovery run and set up for the next."""
        self.discovery_runs += 1
        print(f"--- Discovery Run {self.discovery_runs} Complete ---")

        if self.discovery_runs >= 3:
            print("--- All discovery runs complete. Switching to RANDOM_ACTION state. ---")
            self.agent_state = AgentState.RANDOM_ACTION
        else:
            # Reset for the next full run, starting with primary actions
            self.actions_to_try = self.primary_actions.copy()
            self.discovery_sub_phase = 'PRIMARY'
            self.discovered_in_current_run = False

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        # The agent stops this attempt if it wins the level.
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """This is the main decision-making method for the AGI."""
        # --- 1. Perception ---
        # First, see what happened as a result of the last action.
        changes_found, change_descriptions = self.perceive(latest_frame)

        # --- 2. Process Last Action's Result ---
        # Next, process that outcome based on the agent's state.
        if self.last_action:
            if self.agent_state == AgentState.DISCOVERY and changes_found:
                # SUCCESS during DISCOVERY: A new rule is found.
                self.action_effects[self.last_action] = change_descriptions
                self.discovered_in_current_run = True
                
                print(f"Action {self.last_action.name} caused {len(change_descriptions)} changes. Storing success.")
                for description in change_descriptions[:10]:
                    print(description)
                if len(change_descriptions) > 10:
                    print("  - ...and more.")

            elif self.agent_state == AgentState.RANDOM_ACTION:
                if changes_found:
                    # SUCCESS during RANDOM_ACTION: A known action worked as expected.
                    print(f"Known action {self.last_action.name} succeeded, causing {len(change_descriptions)} changes.")
                    for description in change_descriptions[:10]:
                        print(description)
                    if len(change_descriptions) > 10:
                        print("  - ...and more.")
                else:
                    # FAILURE during RANDOM_ACTION: A known action failed. Log the context.
                    print(f"Known action {self.last_action.name} had no effect. Storing failure context.")
                    context = copy.deepcopy(self.previous_frame)
                    if self.last_action not in self.action_failures:
                        self.action_failures[self.last_action] = []
                    self.action_failures[self.last_action].append(context)

        # --- 3. Handle Game State Resets ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # (This block remains unchanged)
            print("--- New Attempt: Resetting Agent State to DISCOVERY ---")
            self.agent_state = AgentState.DISCOVERY
            self.discovery_runs = 0
            self.actions_to_try = self.primary_actions.copy()
            self.last_action = None
            self.discovery_sub_phase = 'PRIMARY'
            self.discovered_in_current_run = False
            return GameAction.RESET

        # --- 4. Choose a New Action to Take ---
        # Finally, decide what to do in this new turn.
        if self.agent_state == AgentState.DISCOVERY:
            # (This block remains unchanged)
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

        # If discovery is over, we are in the RANDOM_ACTION state.
        if self.action_effects:
            action = random.choice(list(self.action_effects.keys()))
        else:
            all_possible_actions = self.primary_actions + self.secondary_actions
            if all_possible_actions:
                action = random.choice(all_possible_actions)
            else:
                action = GameAction.NOOP
        
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
            print("--- Frame dimensions changed! ---")
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