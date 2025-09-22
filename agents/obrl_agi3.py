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
        self.last_success_contexts = {}
        self.last_relationships = {}
        self.last_score = 0
        self.is_new_level = True
        self.removed_objects_memory = {}
        self.object_id_counter = 0
        self.weights = {}
        self.action_counts = {}
        self.last_state_key = None
        self.learning_rate = 0.1  # Alpha
        self.discount_factor = 0.9 # Gamma
        self.is_waiting_for_stability = False
        self.visited_states = set()
        self.seen_object_states = set()
        self.recent_effect_patterns = deque(maxlen=20)
        self.seen_configurations = set()
        self.total_unique_changes = 0
        self.total_moves = 0
        self.failed_action_blacklist = set()
        self.turns_without_discovery = 0

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        This method is called by the game to get the next action.
        """
        # If the game is over or hasn't started, the correct action is to reset.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.actions_printed = False  # Reset the print flag for the new game.
            self.last_score = 0
            self.is_new_level = True
            return GameAction.RESET
        
        current_summary = self._perceive_objects(latest_frame)
        current_relationships = self._analyze_relationships(current_summary)

        # If this is the first scan (last summary is empty), print the full summary.
        if not self.last_object_summary or self.is_new_level:
            print("--- Initial Frame Summary ---")
            self.is_new_level = False # We've handled the "new level" state, so turn the flag off.
            
            # Assign initial persistent IDs to all objects discovered on the first frame.
            self.object_id_counter = 0
            for obj in current_summary:
                self.object_id_counter += 1
                obj['id'] = f'obj_{self.object_id_counter}'
            
            if not current_summary:
                print("No objects found.")
            self._print_full_summary(current_summary)

            if current_relationships:
                print("\n--- Relationship Analysis ---")
                output_lines = []
                def format_id_list_str(id_set):
                    id_list = sorted(list(id_set))
                    if len(id_list) < 2:
                        return f"Object {id_list[0]}"
                    if len(id_list) == 2:
                        return f"Objects {id_list[0]} and {id_list[1]}"
                    return "Objects " + ", ".join(map(str, id_list[:-1])) + f", and {id_list[-1]}"

                for rel_type, groups in sorted(current_relationships.items()):
                    for value, ids in sorted(groups.items()):
                        value_str = value
                        if rel_type == 'Size':
                            value_str = f"{value[0]}x{value[1]}"
                        output_lines.append(f"- {rel_type} Group ({value_str}): {format_id_list_str(ids)}")
                
                for line in output_lines:
                    print(line)

        # On subsequent turns, analyze the outcome of the previous action.
        else:
            prev_summary = self.last_object_summary
            changes, current_summary = self._log_changes(prev_summary, current_summary)

            # --- LEVEL CHANGE DETECTION & HANDLING ---
            current_score = latest_frame.score
            if current_score > self.last_score:
                print(f"\n--- LEVEL CHANGE DETECTED (Score increased from {self.last_score} to {current_score}) ---")
                
                # Print summary of the old level's last frame (which is self.last_object_summary)
                print("\n--- Final Frame Summary (Old Level) ---")
                self._print_full_summary(self.last_object_summary)

                # Reset agent's learning and memory for the new level
                print("Resetting agent's memory for new level.")
                
                self.rule_hypotheses = {}
                self.last_success_contexts = {}
                self.last_relationships = {}
                self.last_action_context = None
                self.is_new_level = True
                self.removed_objects_memory = {}
                self.object_id_counter = 0
                
                # --- Reset core RL knowledge ---
                self.weights = {}
                self.action_counts = {}

                # --- Reset performance trackers and blacklists ---
                self.total_unique_changes = 0
                self.total_moves = 0
                self.failed_action_blacklist.clear()
                self.seen_configurations.clear()

            # Update the score tracker for the next turn.
            self.last_score = current_score

            self._log_relationship_changes(self.last_relationships, current_relationships)

            # --- Prepare for Learning & Rule Analysis ---
            current_state_key = self._get_state_key(current_summary)

            if self.last_action_context:
                # --- Analyze the outcome of the previous action ---
                prev_action_name, prev_coords = self.last_action_context
                learning_key = self._get_learning_key(prev_action_name, prev_coords)
                just_finished_waiting = False
                is_failure_case = False
                novel_state_count = 0
                
                # Determine the outcomes first
                if changes:
                    if self.failed_action_blacklist:
                        print("A successful action was found, clearing the failure blacklist.")
                        self.failed_action_blacklist.clear()
                    
                    # --- Create a specific, per-object learning key for each change ---
                    per_object_keys = []
                    prev_summary_map = {obj['id']: obj for obj in prev_summary}

                    for change_str in changes:
                        if "Object id_" in change_str:
                            id_num_str = change_str.split('id_')[1].split()[0]
                            obj_id = f"obj_{id_num_str}"
                            
                            if obj_id in prev_summary_map:
                                target_in_prev_state = prev_summary_map[obj_id]
                                prev_action_name, _ = self.last_action_context
                                base_action_key = self._get_learning_key(prev_action_name, None)
                                
                                # Create the object's complete state description (properties + position)
                                object_state = (self._get_stable_id(target_in_prev_state), target_in_prev_state['position'])
                                
                                # The new key is (ACTION, OBJECT_ID, OBJECT_STATE)
                                contextual_key = (base_action_key, target_in_prev_state['id'], object_state)
                                per_object_keys.append(contextual_key)
                    
                    # Success Path: Calculate how many unique states were created
                    affected_object_ids = set()
                    for change_str in changes:
                        if "Object id_" in change_str:
                            id_num_str = change_str.split('id_')[1].split()[0]
                            affected_object_ids.add(f"obj_{id_num_str}")

                    unique_log_messages = []
                    for obj in current_summary:
                        if obj['id'] in affected_object_ids:
                            state_fingerprint = (obj['id'], self._get_stable_id(obj), obj['position'])
                            if state_fingerprint not in self.seen_configurations:
                                self.seen_configurations.add(state_fingerprint)
                                obj_id_str = obj['id'].replace('obj_', 'id_')
                                size_str = f"{obj['size'][0]}x{obj['size'][1]}"
                                log_msg = (
                                    f"- Unique State: Object {obj_id_str} recorded in new state "
                                    f"(Color: {obj['color']}, Size: {size_str}, Pos: {obj['position']}, "
                                    f"Fingerprint: {obj['fingerprint']})."
                                )
                                unique_log_messages.append(log_msg)
                    novel_state_count = len(unique_log_messages)
                
                else:
                    # No changes occurred. Check if this is a failure, but NOT if we just finished waiting.
                    if not just_finished_waiting:
                        prev_action_name, prev_coords = self.last_action_context
                        learning_key = self._get_learning_key(prev_action_name, prev_coords)
                        if learning_key in self.last_success_contexts or self.action_counts.get((self.last_state_key, learning_key), 0) > 0:
                            is_failure_case = True

                # --- Now, with all outcomes known, learn from the last action ---
                self._learn_from_outcome(latest_frame, changes, current_summary, novel_state_count, is_failure_case, learning_key)

                # --- Handle all Logging and Rule Analysis *after* learning ---
                if changes:
                    print("--- Change Log ---")
                    for change in changes:
                        print(change)
                    
                    if unique_log_messages:
                        print("\n--- Unique Change Log ---")
                        for msg in sorted(unique_log_messages):
                            print(msg)

                    
                    # Report each change under its own per-object contextual key
                    for i, key in enumerate(per_object_keys):
                        # Pass only the single relevant change string
                        self._analyze_and_report(key, [changes[i]])
                    self.last_success_contexts[learning_key] = prev_summary
                
                    # --- Tag rules that led to non-unique outcomes ---
                    if changes and not is_failure_case:
                        # First, get the set of IDs for objects that had a unique change
                        unique_object_ids = set()
                        for msg in unique_log_messages:
                            if "Object id_" in msg:
                                id_num_str = msg.split('id_')[1].split()[0]
                                unique_object_ids.add(f"obj_{id_num_str}")

                        # Now, tag the rules for any changed object that was NOT unique
                        for key in per_object_keys:
                            # key = (base_action_key, object_id, object_state)
                            obj_id_from_key = key[1]
                            if obj_id_from_key not in unique_object_ids:
                                if key in self.rule_hypotheses:
                                    self.rule_hypotheses[key]['is_boring'] = True

                elif is_failure_case:
                    # Blacklist the action, regardless of its history.
                    self.failed_action_blacklist.add(learning_key)
                    print(f"Action {learning_key} has been blacklisted until a success occurs.")
                    
                    # Only perform failure analysis if we have a past success to compare against.
                    if learning_key in self.last_success_contexts:
                        print(f"\n--- Failure Detected for Action {learning_key} ---")
                        last_success_summary = self.last_success_contexts[learning_key]
                        self._analyze_failures(learning_key, last_success_summary, prev_summary)
                    else:
                        # This is for a repeated, unproductive action that has never been successful.
                        print(f"Action {learning_key} is unproductive, but has no prior success record to analyze.")
                        
            elif changes:
                # Fallback for when changes happen without a known previous action
                print("--- Change Log ---")
                for change in changes:
                    print(change)

            if self.is_waiting_for_stability:
                if changes:
                    print("Animation in progress, observing...")
                    # Update our memory to check against the next frame
                    self.last_object_summary = current_summary
                    self.last_relationships = current_relationships
                    # Return a default, benign action and end the turn
                    return latest_frame.available_actions[0] if latest_frame.available_actions else GameAction.RESET
                else:
                    # The world is now stable, so we can proceed with a new action.
                    print("Stability reached. Resuming control.")
                    self.is_waiting_for_stability = False
                    just_finished_waiting = True

        # Update the memory for the next turn.
        self.last_object_summary = current_summary
        self.last_relationships = current_relationships

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

        # --- Proactive "Boring" Move Scan ---
        boring_predictions = []
        for obj in current_summary:
            for action_template in game_specific_actions:
                # Create the specific 3-part key for this potential object/action pair
                coords_for_key = {'x': obj['position'][1], 'y': obj['position'][0]} if 'CLICK' in action_template.name or action_template.name == 'ACTION6' else None
                base_action_key = self._get_learning_key(action_template.name, coords_for_key)
                object_state = (self._get_stable_id(obj), obj['position'])
                action_key = (base_action_key, obj['id'], object_state)

                hypothesis = self.rule_hypotheses.get(action_key)
                if hypothesis and hypothesis.get('is_boring', False):
                    obj_id_str = obj['id'].replace('obj_', 'id_')
                    boring_predictions.append(f"- Object {obj_id_str}: Action {base_action_key} is predicted to be boring.")

        if boring_predictions:
            print("\n--- Boring Move Predictions ---")
            # Use set() to remove any duplicate predictions before printing
            for prediction in sorted(list(set(boring_predictions))):
                print(prediction)

        # --- RL: Score and select the best action ---
        current_state_key = self._get_state_key(current_summary)
        best_move = None
        best_score = -float('inf')

        if not possible_moves:
             # If there are no possible moves, we might need a fallback.
             # For now, let's try to use a generic click or another default.
             possible_moves.append({'type': random.choice([a for a in GameAction if a is not GameAction.RESET]), 'object': None})

        all_scores_debug = []
        for move in possible_moves:
            action_template = move['type']
            target_object = move['object']
            coords_for_context = None

            if target_object:
                pos = target_object['position']
                coords_for_context = {'x': pos[1], 'y': pos[0]}

            action_key = self._get_learning_key(action_template.name, coords_for_context)

            # --- Check if the action is currently blacklisted ---
            if action_key in self.failed_action_blacklist:
                continue # Skip this action

            features = self._extract_features(current_summary, move)
            
            # Calculate Q-value as the dot product of features and weights
            q_value = 0.0
            for feature, value in features.items():
                q_value += self.weights.get(feature, 0.0) * value
            
            action_count = self.action_counts.get((current_state_key, action_key), 0)
            if action_count == 0:
                exploration_bonus = 25.0 # Large bonus to encourage trying a new action
            else:
                exploration_bonus = 1.0 / (1.0 + action_count) # Smaller bonus for less-used actions

            # --- Boring Penalty ---
            # Penalize actions that are known to lead to non-unique states.
            boring_penalty = 0.0
            if target_object:
                # Case 1: This is a CLICK action on a specific object.
                hypothesis = self.rule_hypotheses.get(action_key)
                if hypothesis and hypothesis.get('is_boring', False):
                    boring_penalty = -20.0
                    obj_id = target_object['id'].replace('obj_', 'id_')
                    print(f"Heuristic: Predicting a 'boring' outcome for {action_key[0]} on object {obj_id}. Applying penalty.")
            else:
                # Case 2: This is a GLOBAL action. Scan all objects for potential boring outcomes.
                action_name = action_template.name
                for obj in current_summary:
                    # Construct the specific key for this global action + this object
                    obj_action_key = (action_name, obj['id'], (self._get_stable_id(obj), obj['position']))
                    hypothesis = self.rule_hypotheses.get(obj_action_key)
                    if hypothesis and hypothesis.get('is_boring', False):
                        # Add a penalty for each predicted boring outcome
                        boring_penalty -= 20.0
                        obj_id = obj['id'].replace('obj_', 'id_')
                        print(f"Heuristic: Predicting a 'boring' outcome for {action_name} on object {obj_id}. Applying penalty.")

            score = q_value + exploration_bonus + boring_penalty

            if score > best_score:
                best_score = score
                best_move = move

            # Create a user-friendly name for the debug log
            if target_object:
                obj_id = target_object['id'].replace('obj_', 'id_')
                debug_name = f"CLICK on {obj_id}"
            else:
                debug_name = action_template.name
            all_scores_debug.append(f"{debug_name} (Score: {score:.2f})")
        
        # --- Fallback if all available actions were blacklisted ---
        if best_move is None and possible_moves:
            print("Warning: All available actions were blacklisted. Clearing blacklist to break deadlock.")
            self.failed_action_blacklist.clear()
            best_move = possible_moves[0] # Force select the first available action

        # --- Prepare the chosen action to be returned ---
        chosen_template = best_move['type']
        chosen_object = best_move['object']
        coords_for_context = None

        debug_scores_str = "{ " + ", ".join(sorted(all_scores_debug)) + " }"

        if chosen_object:
            pos = chosen_object['position']
            click_y, click_x = pos[0], pos[1]
            coords_for_context = {'x': click_x, 'y': click_y}
            obj_id = chosen_object['id'].replace('obj_', 'id_')
            print(f"RL Agent chose CLICK on object {obj_id} (Score: {best_score:.2f}) {debug_scores_str}")
            chosen_template.set_data(coords_for_context)
        else:
            print(f"RL Agent chose action: {chosen_template.name} (Score: {best_score:.2f}) {debug_scores_str}")
        
        action_to_return = chosen_template
        self.last_state_key = current_state_key # Remember the state for the next learning cycle

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
                    pos_parts = parts[1].replace('.', '').split(' to ')
                    start_pos, end_pos = ast.literal_eval(pos_parts[0]), ast.literal_eval(pos_parts[1])
                    event.update({
                        'vector': (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                    })
                    events.append(event)

                elif change_type == 'RECOLORED':
                    from_color_str = details.split(' from ')[1].split(' to ')[0]
                    to_color_str = details.split(' to ')[1].replace('.', '')
                    event.update({
                        'from_color': int(from_color_str),
                        'to_color': int(to_color_str)
                    })
                    events.append(event)
                
                elif change_type == 'SHAPE_CHANGED':
                    fp_part = details.split('fingerprint: ')[1]
                    from_fp_str, to_fp_str = fp_part.replace(').','').split(' -> ')
                    event.update({
                        'from_fingerprint': int(from_fp_str),
                        'to_fingerprint': int(to_fp_str)
                    })
                    events.append(event)
                
                elif change_type in ['GROWTH', 'SHRINK', 'TRANSFORM']:
                    start_pos_str = details.split(') ')[0] + ')'
                    start_pos = ast.literal_eval(start_pos_str.split(' at ')[1])
                    end_pos_str = details.split('now at ')[1].replace('.', '')
                    end_pos = ast.literal_eval(end_pos_str)
                    event.update({'start_position': start_pos, 'end_position': end_pos})
                    
                    if '(from ' in details: # Growth/Shrink have size details
                        from_size_str = details.split('(from ')[1].split(' to ')[0]
                        to_size_str = details.split(' to ')[1].split(')')[0]
                        pixel_diff = int(details.split(' by ')[1].split(' pixels')[0])
                        event.update({
                            'from_size': ast.literal_eval(from_size_str.replace('x', ',')),
                            'to_size': ast.literal_eval(to_size_str.replace('x', ',')),
                            'pixel_delta': pixel_diff if change_type == 'GROWTH' else -pixel_diff
                        })
                    events.append(event)

                elif change_type in ['NEW', 'REMOVED']:
                    id_str = details.split(') ')[0] + ')'
                    id_tuple = ast.literal_eval(id_str.split('ID ')[1])
                    pos_str = '(' + details.split('(')[-1].replace('.', '')
                    position = ast.literal_eval(pos_str)
                    event.update({
                        'position': position, 'fingerprint': id_tuple[0],
                        'color': id_tuple[1], 'size': id_tuple[2], 'pixels': id_tuple[3]
                    })
                    events.append(event)

            except (ValueError, IndexError, SyntaxError):
                continue
        return events

    def _analyze_and_report(self, action_key: str, changes: list[str]):
        """Compares new events with a stored hypothesis to find consistent, refined rules."""
        
        def find_best_match(event_to_match, candidate_list):
            """Finds the best matching event from a list based on event type."""
            event_type = event_to_match['type']

            # Special logic to track objects that transform and move between turns
            if event_type in ['GROWTH', 'SHRINK', 'TRANSFORM']:
                # Match the END position of the old event with the START position of a new one
                if 'end_position' in event_to_match:
                    for candidate in candidate_list:
                        if event_to_match['end_position'] == candidate.get('start_position'):
                            candidate_list.remove(candidate)
                            return candidate
            
            # --- Existing logic for matching events by stable properties ---
            match_key_map = {
                'MOVED': 'fingerprint',
                'SHAPE_CHANGED': 'position',
                'RECOLORED': 'position',
                'NEW': 'position',
                'REMOVED': 'position',
            }
            match_key = match_key_map.get(event_type)

            if match_key and match_key in event_to_match:
                for candidate in candidate_list:
                    if event_to_match[match_key] == candidate.get(match_key):
                        candidate_list.remove(candidate)
                        return candidate

            # Fallback: if no direct match and only one candidate of this type, it must be the one.
            if len(candidate_list) == 1 and candidate_list[0]['type'] == event_type:
                return candidate_list.pop(0)

            return None

        def refine_rule(old_rule, new_event):
            """Intersects the properties of a rule and a new event, keeping only commonalities."""
            refined = {}
            for key, value in old_rule.items():
                if key in new_event and new_event[key] == value:
                    refined[key] = value
            refined['type'] = old_rule['type'] # Always preserve the event type
            return refined

        new_events = self._parse_change_logs_to_events(changes)
        if not new_events: return

        hypothesis = self.rule_hypotheses.get(action_key)

        if not hypothesis:
            # First observation: create the hypothesis with initial confidence.
            self.rule_hypotheses[action_key] = {
                'rules': new_events,
                'attempts': 1,
                'confirmations': 1,
                'confidence': 1.0,
                'is_boring': False,
            }
            print(f"\n--- Initial Case File for {action_key} (Confidence: 100%) ---")
            for event in new_events:
                details = {k:v for k,v in event.items() if k != 'type'}
                print(f"- Observed {event['type']} with details: {details}")
            return
        
        hypothesis['attempts'] += 1

        # --- Stage 2 & 3: Match events and refine the hypothesis ---
        refined_rules = []
        # Group events by type to make matching manageable
        new_events_by_type = {}
        for event in new_events:
            new_events_by_type.setdefault(event['type'], []).append(event)

        # For each rule in our old hypothesis, try to find its twin in the new events
        for old_rule in hypothesis['rules']:
            rule_type = old_rule['type']
            candidates = new_events_by_type.get(rule_type, [])
            
            match = find_best_match(old_rule, candidates)
            
            if match:
                # If we found a match, refine the rule by finding the intersection of properties
                refined_rule = refine_rule(old_rule, match)
                refined_rules.append(refined_rule)

        if refined_rules:
            hypothesis['confirmations'] += 1
        
        hypothesis['rules'] = refined_rules
        hypothesis['confidence'] = hypothesis['confirmations'] / hypothesis['attempts']
        
        confidence_percent = hypothesis['confidence'] * 100
        print(f"\n--- Refined Hypothesis for {action_key} (Confidence: {confidence_percent:.0f}%) ---")
        if not refined_rules:
            print("No consistent rules could be confirmed from the new observation.")
        else:
            for rule in refined_rules:
                details = {k:v for k,v in rule.items() if k != 'type'}
                print(f"- Confirmed Rule: A {rule['type']} event occurs with consistent properties: {details}")

    def _analyze_failures(self, action_key: str, last_success_summary: list[dict], current_failure_summary: list[dict]):
        """Compares a failure state to the last known success state to find differences."""
        print("Comparing current state to last successful state to find potential failure preconditions...")
        
        # We can reuse our powerful _log_changes function for this comparison
        differences, _ = self._log_changes(last_success_summary, current_failure_summary, assign_new_ids=False)
        
        if not differences:
            print("No obvious differences found between this failure state and the last success state.")
        else:
            # Create a lookup map of new objects by their position for easy access
            new_objects_by_pos = {obj['position']: obj for obj in current_failure_summary}

            print("--- Failure Precondition Log (Differences from last success) ---")
            for diff in differences:
                print(diff)  # Print the original change log

                if 'REMOVED:' in diff:
                    continue  # No new object to describe

                # --- Try to parse the final position of the object from the log string ---
                final_pos = None
                try:
                    # Case 1: Fuzzy match with "now at"
                    if ' now at ' in diff:
                        pos_str = diff.split(' now at ')[1].replace('.', '')
                        final_pos = ast.literal_eval(pos_str)
                    # Case 2: A move event with "to"
                    elif ' moved from ' in diff and ' to ' in diff:
                        pos_str = diff.split(' to ')[1].replace('.', '')
                        final_pos = ast.literal_eval(pos_str)
                    # Case 3: In-place changes or new objects with "at"
                    elif ' at ' in diff:
                        details_part = diff.split(' at ')[1]
                        start = details_part.find('(')
                        end = details_part.find(')')
                        if start != -1 and end != -1:
                            pos_str = details_part[start : end + 1]
                            final_pos = ast.literal_eval(pos_str)
                except (ValueError, IndexError, SyntaxError):
                    # If parsing fails, we can't find the object, so just skip.
                    continue
                
                # --- Find and print the object's full description ---
                if final_pos and final_pos in new_objects_by_pos:
                    obj = new_objects_by_pos[final_pos]
                    size_str = f"{obj['size'][0]}x{obj['size'][1]}"
                    desc = (
                        f"  - Object is now {size_str} at {obj['position']} with color {obj['color']}, "
                        f"{obj['pixels']} pixels, and fingerprint {obj['fingerprint']}."
                    )
                    print(desc)

    def _analyze_relationships(self, object_summary: list[dict]) -> dict:
        """Analyzes object relationships and returns a structured dictionary of groups."""
        if not object_summary or len(object_summary) < 2:
            return {}

        # Use capitalized keys for cleaner log titles
        rel_data = {
            'Color': {},
            'Shape': {},
            'Size': {},
            'Pixel': {}
        }
        for obj in object_summary:
            obj_id = int(obj['id'].replace('obj_', ''))
            
            rel_data['Color'].setdefault(obj['color'], set()).add(obj_id)
            rel_data['Shape'].setdefault(obj['fingerprint'], set()).add(obj_id)
            rel_data['Size'].setdefault(obj['size'], set()).add(obj_id)
            rel_data['Pixel'].setdefault(obj['pixels'], set()).add(obj_id)

        # Filter out groups with only one member, as they don't represent a relationship
        final_rels = {}
        for rel_type, groups in rel_data.items():
            final_rels[rel_type] = {
                value: ids for value, ids in groups.items() if len(ids) > 1
            }
        
        # Clean up empty relationship types
        final_rels = {k: v for k, v in final_rels.items() if v}
        return final_rels
    
    def _log_relationship_changes(self, old_rels: dict, new_rels: dict):
        """Compares two relationship dictionaries and logs which objects joined, left, or replaced others in groups."""
        output_lines = []
        
        def format_ids(id_set):
            """Formats a set of object IDs into a human-readable string."""
            if not id_set: return ""
            id_list = sorted(list(id_set))
            if len(id_list) == 1:
                return f"object {id_list[0]}" # Lowercase 'o' for use in a sentence
            return f"objects " + ", ".join(map(str, id_list))

        all_rel_types = sorted(list(set(old_rels.keys()) | set(new_rels.keys())))

        for rel_type in all_rel_types:
            old_groups = old_rels.get(rel_type, {})
            new_groups = new_rels.get(rel_type, {})
            all_values = set(old_groups.keys()) | set(new_groups.keys())

            for value in all_values:
                old_ids = old_groups.get(value, set())
                new_ids = new_groups.get(value, set())

                if old_ids == new_ids:
                    continue

                joined_ids = new_ids - old_ids
                left_ids = old_ids - new_ids

                value_str = value
                if rel_type == 'Size':
                    value_str = f"{value[0]}x{value[1]}"

                # Only log changes for meaningful groups (size > 1)
                if len(new_ids) > 1 or len(old_ids) > 1:
                    if joined_ids and left_ids:
                        output_lines.append(f"- {rel_type} Group ({value_str}): {format_ids(joined_ids).capitalize()} replaced {format_ids(left_ids)}.")
                    elif joined_ids:
                        output_lines.append(f"- {rel_type} Group ({value_str}): {format_ids(joined_ids).capitalize()} joined.")
                    elif left_ids:
                        output_lines.append(f"- {rel_type} Group ({value_str}): {format_ids(left_ids).capitalize()} left.")

        if output_lines:
            print("\n--- Relationship Change Log ---")
            for line in sorted(output_lines):
                print(line)
            print()

    def _print_full_summary(self, summary: list[dict]):
        """Prints a formatted summary of all objects in a list."""
        if not summary:
            print("No objects found.")
            return

        for obj in summary:
            obj_id = obj['id'].replace('obj_', 'id_')
            size_str = f"{obj['size'][0]}x{obj['size'][1]}"
            print(
                f"- Object {obj_id}: Found a {size_str} object of color {obj['color']} "
                f"at position {obj['position']} with {obj['pixels']} pixels "
                f"and shape fingerprint {obj['fingerprint']}."
            )

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
    
    def _log_changes(self, old_summary: list[dict], new_summary: list[dict], assign_new_ids=True) -> tuple[list[str], list[dict]]:
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

                    # Propagate the persistent ID from the old object to its new version.
                    new_obj['id'] = old_obj['id']

                    # If anything changed, it's an event. If not, it's stable.
                    # In either case, we've explained this pair of objects.
                    if color_changed and shape_changed:
                        changes.append(
                            f"- TRANSFORM: Object {old_obj['id'].replace('obj_', 'id_')} changed shape "
                            f"(fingerprint: {old_obj['fingerprint']} -> {new_obj['fingerprint']}) "
                            f"and color (from {old_obj['color']} to {new_obj['color']})."
                        )
                    elif color_changed:
                        size_str = f"{old_obj['size'][0]}x{old_obj['size'][1]}"
                        changes.append(
                            f"- RECOLORED: Object {old_obj['id'].replace('obj_', 'id_')} "
                            f"changed color from {old_obj['color']} to {new_obj['color']}."
                        )
                    elif shape_changed:
                        changes.append(
                            f"- SHAPE_CHANGED: Object {old_obj['id'].replace('obj_', 'id_')} changed shape "
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
                # Propagate the persistent ID.
                new_inst['id'] = old_inst['id']
                # A move is only a move if the position is different.
                if old_inst['position'] != new_inst['position']:
                    changes.append(f"- MOVED: Object {old_inst['id'].replace('obj_', 'id_')} moved from {old_inst['position']} to {new_inst['position']}.")
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
                    # Propagate the persistent ID.
                    new_obj['id'] = old_obj['id']
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
                    else:  # TRANSFORM
                        # Figure out what exactly changed to make the log more descriptive
                        changed_parts = []
                        if old_obj['fingerprint'] != new_obj['fingerprint']:
                            changed_parts.append("shape")
                        if old_obj['color'] != new_obj['color']:
                            changed_parts.append(f"color from {old_obj['color']} to {new_obj['color']}")
                        
                        size_changed = old_obj['size'] != new_obj['size']
                        size_details = f" (size {old_size_str} -> {new_size_str})" if size_changed else ""

                        if not changed_parts:
                            # Handles rare cases, describe it as transformed with size info if relevant
                            details = f"transformed{size_details}"
                        else:
                            details = f"changed { ' and '.join(changed_parts) }{size_details}"

                    changes.append(f"- {event_type}: Object {old_obj['id'].replace('obj_', 'id_')} at {old_obj['position']} {details}, now at {new_obj['position']}.")

                    matched_old.add(id(old_obj))
                    matched_new.add(id(new_obj))
            
            # Remove the newly matched objects from the unexplained lists
            old_unexplained = [obj for obj in old_unexplained if id(obj) not in matched_old]
            new_unexplained = [obj for obj in new_unexplained if id(obj) not in matched_new]

        # --- Final Pass: Log remaining as REMOVED, NEW, or REAPPEARED ---
        for obj in old_unexplained:
            stable_id = self._get_stable_id(obj)
            changes.append(f"- REMOVED: Object {obj['id'].replace('obj_', 'id_')} (ID {stable_id}) at {obj['position']} has disappeared.")
            # Update memory with the persistent ID, storing a list for each stable ID type.
            if stable_id not in self.removed_objects_memory:
                self.removed_objects_memory[stable_id] = deque()
            self.removed_objects_memory[stable_id].append(obj['id'])

        for obj in new_unexplained:
            stable_id = self._get_stable_id(obj)

            if assign_new_ids:
                # Live mode: Check for reappearance or assign a new persistent ID.
                if stable_id in self.removed_objects_memory:
                    persistent_id = self.removed_objects_memory[stable_id].popleft()
                    if not self.removed_objects_memory[stable_id]:
                        del self.removed_objects_memory[stable_id]
                    obj['id'] = persistent_id
                    changes.append(f"- REAPPEARED: Object {persistent_id.replace('obj_', 'id_')} (ID {stable_id}) has reappeared at {obj['position']}.")
                else:
                    self.object_id_counter += 1
                    new_id = f'obj_{self.object_id_counter}'
                    obj['id'] = new_id
                    changes.append(f"- NEW: Object {new_id.replace('obj_', 'id_')} (ID {stable_id}) has appeared at {obj['position']}.")
            else:
                # Analysis mode: Do not assign a new ID. Log with the object's existing ID.
                changes.append(f"- NEW: Object {obj['id'].replace('obj_', 'id_')} (ID {stable_id}) has appeared at {obj['position']}.")

        return sorted(changes), new_summary
    
    def _get_state_key(self, object_summary: list[dict]) -> str:
        """Creates a stable, hashable key representing the current state."""
        if not object_summary:
            return "empty"
        # Create a tuple of stable object IDs, sorted to ensure consistency
        state_tuple = tuple(sorted(self._get_stable_id(obj) for obj in object_summary))
        return str(hash(state_tuple))
    
    def _learn_from_outcome(self, latest_frame: FrameData, changes: list[str], new_summary: list[dict], novel_state_count: int, is_failure: bool, learning_key: str):
        """Calculates reward and updates model weights based on the last action's outcome."""
        if not self.last_state_key or not self.last_action_context:
            return

        # 1. Calculate reward (same as before)
        reward = 0
        # --- Normalized reward for unique state discovery ---
        self.total_moves += 1
        self.total_unique_changes += novel_state_count

        # Calculate the running average of unique changes per move.
        average_unique_change = self.total_unique_changes / self.total_moves if self.total_moves > 0 else 0

        # Reward actions that perform better than average, penalize those that do worse.
        performance_vs_average = novel_state_count - average_unique_change
        reward += performance_vs_average * 15 # Multiplier makes this a strong signal

        # --- Specific penalty for unexpected failures ---
        if is_failure:
            reward -= 15 # Heavy penalty for an action that should have worked but didn't.

        # --- Escalating penalty for being in an unproductive loop ---
        if novel_state_count > 0:
            # A discovery was made, so the drought is over.
            self.turns_without_discovery = 0
        else:
            # No discovery, the drought continues and the penalty increases.
            self.turns_without_discovery += 10
            reward -= self.turns_without_discovery

        if latest_frame.score > self.last_score:
            reward += 100
        if changes:
            # Reward is now proportional to the number of changes caused.
            # An action that causes more things to happen is considered more valuable.
            reward += len(changes)
        else:
            # The penalty for an action that does nothing remains.
            reward -= 5

        # --- Object-Level Novelty Reward ---
        novel_object_count = 0
        if new_summary:
            for obj in new_summary:
                # We use the stable ID to represent a unique object state
                object_stable_id = self._get_stable_id(obj)
                if object_stable_id not in self.seen_object_states:
                    novel_object_count += 1
                    # Add the new object state to our long-term memory
                    self.seen_object_states.add(object_stable_id)
        
        if novel_object_count > 0:
            # The reward is proportional to how many new object types were created.
            # We can tune the multiplier (e.g., 3) as needed.
            reward += novel_object_count * 3
        elif not changes:
             # If no new objects were created AND no changes happened, it was a wasted turn.
             # This overlaps with the inaction penalty but reinforces it.
             reward -= 2
        
        # Check if the action was a "discovery" (i.e., created new object states).
        was_discovery = novel_object_count > 0
        if not was_discovery:
            # If it wasn't a discovery, penalize it based on how many times we've tried it.
            # The count is for the PREVIOUS state-action pair, before the update.
            repetition_count = self.action_counts.get((self.last_state_key, learning_key), 0)
            # The penalty increases with each repetition of the unproductive action.
            reward -= repetition_count

        # --- Effect Pattern Novelty Reward ---
        if changes:
            # Create a "fingerprint" of the outcome based on the types of changes.
            change_types = sorted([log.split(':')[0].replace('- ', '') for log in changes])
            effect_pattern_key = tuple(change_types)
            
            # Check how many times this exact pattern has occurred recently.
            pattern_count_in_history = self.recent_effect_patterns.count(effect_pattern_key)
            
            if pattern_count_in_history == 0:
                # This is a brand new type of outcome we haven't seen recently.
                reward += 20 # Large bonus for discovering a new mechanism.
            else:
                # We've seen this pattern before, so it's less interesting.
                # Apply a penalty that increases the more we repeat the pattern.
                reward -= pattern_count_in_history * 5 # Increased penalty for unoriginal outcomes

            # Add the new pattern to our recent history.
            self.recent_effect_patterns.append(effect_pattern_key)

        # 2. Estimate the best possible Q-value from the new state
        max_q_for_next_state = 0
        
        # We need to construct the possible moves for the *new* state to evaluate them
        next_possible_moves = []
        click_action_template = next((action for action in latest_frame.available_actions if action.name == 'ACTION6'), None)
        for action in latest_frame.available_actions:
            if action.name != 'ACTION6':
                next_possible_moves.append({'type': action, 'object': None})
        if click_action_template and new_summary:
            for obj in new_summary:
                next_possible_moves.append({'type': click_action_template, 'object': obj})
        
        if next_possible_moves:
            q_values_for_next_state = []
            for move in next_possible_moves:
                features = self._extract_features(new_summary, move)
                next_q = sum(self.weights.get(f, 0.0) * v for f, v in features.items())
                q_values_for_next_state.append(next_q)
            max_q_for_next_state = max(q_values_for_next_state)

        # 3. Calculate the prediction error (Temporal Difference)
        prev_action_name, prev_coords = self.last_action_context
        prev_move_mock = {'type': GameAction[prev_action_name], 'object': self._find_object_by_coords(prev_coords)}
        
        # We need the summary from the *previous* state to get the old features
        prev_summary = self.last_object_summary 
        prev_features = self._extract_features(prev_summary, prev_move_mock)
        
        old_q_prediction = sum(self.weights.get(f, 0.0) * v for f, v in prev_features.items())
        
        temporal_difference = reward + (self.discount_factor * max_q_for_next_state) - old_q_prediction

        # 4. Update the weights
        for feature, value in prev_features.items():
            # The update rule: adjust weight in proportion to the error and the feature's value
            self.weights[feature] = self.weights.get(feature, 0.0) + (self.learning_rate * temporal_difference * value)
        
        # Update action count for exploration bonus
        self.action_counts[(self.last_state_key, learning_key)] = self.action_counts.get((self.last_state_key, learning_key), 0) + 1

    def _find_object_by_coords(self, coords: dict | None) -> dict | None:
        """Helper to find an object in the last summary based on click coordinates."""
        if not coords:
            return None
        for obj in self.last_object_summary:
            # This is a simplification; a more robust check would see if the click is within the object's bounds.
            if obj['position'] == (coords.get('y'), coords.get('x')):
                return obj
        return None

    def _extract_features(self, summary: list[dict], move: dict) -> dict:
        """
        Extracts a dictionary of features for a given state and action (move).
        The model will learn a weight for each feature.
        """
        features = {}
        action_template = move['type']
        target_object = move['object']

        # --- Bias Feature (always on) ---
        features['bias'] = 1.0

        # --- Action Type Features ---
        for action_type in GameAction:
            # Is this action the one we are considering?
            features[f'action_is_{action_type.name}'] = 1.0 if action_template.name == action_type.name else 0.0

        # --- Target Object & Relationship Features (if it's a click) ---
        if target_object:
            obj_id = target_object['id']
            
            # --- Object-Specific Features ONLY ---
            # By including the object's unique ID in the feature name and removing all
            # global or relational features, we achieve perfect isolation.
            features[f'target_color_for_{obj_id}'] = target_object['color'] / 15.0
            features[f'target_pixels_for_{obj_id}'] = target_object['pixels'] / 4096.0
            features[f'target_size_w_for_{obj_id}'] = target_object['size'][1] / 64.0
            features[f'target_size_h_for_{obj_id}'] = target_object['size'][0] / 64.0
            
            # --- We intentionally DO NOT use generic or relational features for clicks ---
            
        else: # This is a non-click, global action
            # --- Action Type Features ---
            for action_type in GameAction:
                features[f'action_is_{action_type.name}'] = 1.0 if action_template.name == action_type.name else 0.0

            # --- Global State Features ---
            features['total_objects'] = len(summary) / 50.0 # Max objects
            if summary:
                unique_colors = len(set(obj['color'] for obj in summary))
                features['unique_colors'] = unique_colors / 15.0 # Max colors
        
        # --- Curiosity & Knowledge Features ---
        coords_for_key = {'x': target_object['position'][1], 'y': target_object['position'][0]} if target_object else None
        base_action_key = self._get_learning_key(action_template.name, coords_for_key)
        contextual_id_part = None
        contextual_state_part = None
        if target_object:
            # The context includes both the specific ID and the object's current state.
            contextual_id_part = target_object['id']
            object_state = (self._get_stable_id(target_object), target_object['position'])
            contextual_state_part = object_state
        action_key = (base_action_key, contextual_id_part, contextual_state_part)
        
        hypothesis = self.rule_hypotheses.get(action_key)
        if hypothesis:
            features['rule_confidence'] = hypothesis['confidence']
            features['is_novel_action'] = 0.0 # We have a rule for it, so it's not novel
        else:
            features['rule_confidence'] = 0.0 # No rule exists, confidence is zero
            features['is_novel_action'] = 1.0 # This action has never been tried and resulted in a change

        return features