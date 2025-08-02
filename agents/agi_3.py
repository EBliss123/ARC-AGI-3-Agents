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
        self.agent_state = AgentState.DISCOVERY
        self.discovery_runs = 0
        self.last_action = None
        self.discovered_actions = set()
        self.all_actions = [a for a in GameAction if a is not GameAction.RESET]
        self.actions_to_try = self.all_actions.copy()


        print(f"Custom AGI initialized for game: {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        # The agent stops this attempt if it wins the level.
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """This is the main decision-making method for the AGI."""
        
        # --- AGI LOGIC PIPELINE ---
        
        # 1. Perception: Check for changes from the last action. 
        changes_found, change_descriptions = self.perceive(latest_frame)

        # If the last action caused a change, add it to our set of useful actions.
        if changes_found and self.last_action:
            print(f"Action {self.last_action.name} caused changes. Storing it.")
            for description in change_descriptions:
                print(description)
            self.discovered_actions.add(self.last_action)

        # Handle game state resets.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            print("--- New Attempt: Resetting Agent State to DISCOVERY ---")
            self.agent_state = AgentState.DISCOVERY
            self.discovery_runs = 0
            self.actions_to_try = self.all_actions.copy()
            self.last_action = None
            return GameAction.RESET

        # --- State-Based Action Selection ---
        
        # 2. Action Discovery: Systematically try all actions for 3 runs. 
        if self.agent_state == AgentState.DISCOVERY:
            if not self.actions_to_try:
                self.discovery_runs += 1
                print(f"--- Discovery Run {self.discovery_runs} Complete ---")
                if self.discovery_runs >= 3:
                    print("--- All discovery runs complete. Switching to RANDOM_ACTION state. ---")
                    self.agent_state = AgentState.RANDOM_ACTION
                else:
                    # Start the next run
                    self.actions_to_try = self.all_actions.copy()

            if self.agent_state == AgentState.DISCOVERY:
                print(f"Discovery Run {self.discovery_runs + 1}: Trying actions.")
                action = self.actions_to_try.pop(0)
                self.last_action = action
                return action

        # 3. Curiosity-Driven Action: Use discovered actions.
        # This part runs after discovery is complete.
        if self.discovered_actions:
            # [cite_start]Prioritize actions that led to new grid states. [cite: 44]
            print(f"Choosing from {len(self.discovered_actions)} discovered actions.")
            action = random.choice(list(self.discovered_actions))
        else:
            # Fallback if no actions caused any change after 3 runs.
            print("No discovered actions. Choosing from all possible actions.")
            action = random.choice(self.all_actions)
        
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
                # We can stop collecting descriptions if we have enough, to keep logs clean
                if len(change_descriptions) > 10:
                    break

                old_pixel_data = self.previous_frame[r][c]
                new_pixel_data = current_frame[r][c]

                if old_pixel_data != new_pixel_data:
                    changes_found = True
                    change_descriptions.append(f"  - Changes at coordinate ({r},{c}):")

                    # Log every single difference inside the pixel's data list
                    if len(old_pixel_data) == len(new_pixel_data):
                        for i in range(len(old_pixel_data)):
                            if old_pixel_data[i] != new_pixel_data[i]:
                                if len(change_descriptions) > 10:
                                    change_descriptions.append("    - ...and more.")
                                    break
                                old_val = old_pixel_data[i]
                                new_val = new_pixel_data[i]
                                change_descriptions.append(f"    - Index {i}: From {old_val} to {new_val}")
                    else:
                        change_descriptions.append(f"    - Data lists changed length.")
            if len(change_descriptions) > 10:
                break

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