import numpy as np
import logging
import random
from collections import defaultdict

# Imports from the game framework's files
from .agent import Agent
from .structs import FrameData, GameAction, GameState, ComplexAction

class GameObserver:
    """Analyzes the game grid to identify objects and changes."""
    def analyze_grid(self, grid):
        """Scans the grid to find all objects and their properties."""
        objects = defaultdict(list)
        for r_idx, row in enumerate(grid):
            for c_idx, cell in enumerate(row):
                color_key = tuple(cell)
                objects[color_key].append((r_idx, c_idx))
        return dict(objects)

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
        self.player_identified = False
        self.last_player_pos = None

    def _analyze_grid_changes(self, grid, latest_frame):
        """Compares the last grid with the current one to find all changes."""
        changes = []

        def get_center(coords):
            if not coords: return None
            sum_x = sum(x for y, x in coords)
            sum_y = sum(y for y, x in coords)
            count = len(coords)
            return (int(round(sum_y / count)), int(round(sum_x / count)))

        last_objects = self.observer.analyze_grid(self.last_grid)
        current_objects = self.observer.analyze_grid(latest_frame.frame)

        last_obj_set = {(color, frozenset(coords)) for color, coords in last_objects.items()}
        current_obj_set = {(color, frozenset(coords)) for color, coords in current_objects.items()}

        disappeared = list(last_obj_set - current_obj_set)
        appeared = list(current_obj_set - last_obj_set)

        # Use a copy of the lists to safely remove items as we pair them
        unmatched_appeared = appeared[:]

        # Look for objects that moved
        for d_color, d_coords in disappeared:
            best_match = None
            for a_color, a_coords in unmatched_appeared:
                d_center = get_center(d_coords)
                a_center = get_center(a_coords)
                distance = abs(d_center[0] - a_center[0]) + abs(d_center[1] - a_center[1])

                if 0.5 < distance < 1.5:
                    changes.append({'type': 'MOVE', 'from_color': d_color, 'to_color': a_color, 'distance': distance})
                    best_match = (a_color, a_coords)
                    break # Found a clear move, stop looking for this disappeared object

            if best_match:
                unmatched_appeared.remove(best_match)

        # Any remaining items likely changed in place (color, shape, etc.)
        for a_color, a_coords in unmatched_appeared:
            changes.append({'type': 'OTHER_CHANGE', 'new_obj': (a_color, a_coords)})

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

        # On Turn 1, analyze the TRUE initial grid state (after RESET)
        if self.action_counter == 1:
            self.initial_objects = self.observer.analyze_grid(grid)
            logging.info(f"--- Turn 1: Initial analysis complete. Found {len(self.initial_objects)} colored object groups. ---")

        # Analyze the result of the LAST turn
        if not np.array_equal(grid, self.last_grid):
            logging.info(f"--- Change detected! Action {self.last_action_taken} was effective. ---")
            if self.last_action_taken not in self.discovered_actions:
                self.discovered_actions.append(self.last_action_taken)

            # Analyze all changes and log them in the RuleEngine
            changes = self._analyze_grid_changes(grid, latest_frame)
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
                background_color = (0, 0, 0)
                if self.initial_objects:
                    for color, coords_list in self.initial_objects.items():
                        if color == background_color:
                            continue
                        sum_x = sum(x for y, x in coords_list)
                        sum_y = sum(y for y, x in coords_list)
                        count = len(coords_list)
                        center_x = int(round(sum_x / count))
                        center_y = int(round(sum_y / count))
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
            reasoning += ". Exploring the map."
            # Filter out RESET (action 0) and CLICK (action 6) from the list of actions to use for exploring.
            exploring_actions = [a for a in self.discovered_actions if a not in [0, 6]]

            if exploring_actions:
                action_num = random.choice(exploring_actions)
            else:
                logging.warning("No effective movement actions found to explore with.")
                action_num = 0


        # --- Create and return the final action object ---
        action_obj = GameAction.from_id(action_num)
        if action_data:
            action_obj.set_data(action_data)
        action_obj.reasoning = reasoning

        self.last_grid = np.copy(grid)
        self.last_action_taken = action_num

        return action_obj