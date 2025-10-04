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
        self.success_contexts = {}
        self.failure_contexts = {}
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
        self.action_history = {}
        self.is_new_level = True
        self.final_summary_before_level_change = None
        self.last_adjacencies = {}
        self.failure_patterns = {}
        self.last_match_groups = {}
        self.level_state_history = []
        self.win_condition_hypotheses = []
        self.level_milestones = []
        self.seen_event_types_in_level = set()
        self.last_alignments = {}

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        This method is called by the game to get the next action.
        """
        changes = []
        # If the game is over or hasn't started, the correct action is to reset.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.actions_printed = False  # Reset the print flag for the new game.
            self.is_new_level = True
            self.object_id_counter = 0
            return GameAction.RESET
        
        current_summary = self._perceive_objects(latest_frame)
        current_relationships, current_adjacencies, current_match_groups = self._analyze_relationships(current_summary)
        current_alignments = self._analyze_alignments(current_summary)

        # If this is the first scan (last summary is empty), print the full summary.
        if not self.last_object_summary or self.is_new_level:
            self.level_state_history = []
            self.is_new_level = False
            self.level_milestones = []
            self.seen_event_types_in_level = set()
            self.last_alignments = {}

            id_map = {} # To store {old_id: new_id} mappings
            new_obj_to_old_id_map = {} # Initialize our map to handle the first frame case
            
            # If we have a saved frame, perform correlation before final ID assignment.
            if self.final_summary_before_level_change is not None:
                print("--- Correlating Objects Across Levels ---")
                
                # This temporary map will link a new object to its old ID.
                new_obj_to_old_id_map = {}

                # --- Two-Pass Matching to find correspondences ---
                unmatched_old = list(self.final_summary_before_level_change)
                unmatched_new = list(current_summary)
                
                # Pass 1: Strict Matching
                old_strict_map = {(self._get_stable_id(obj), obj['position']): obj for obj in unmatched_old}
                pass1_newly_matched = []
                for new_obj in unmatched_new:
                    strict_key = (self._get_stable_id(new_obj), new_obj['position'])
                    if strict_key in old_strict_map:
                        old_obj = old_strict_map[strict_key]
                        new_obj_to_old_id_map[id(new_obj)] = old_obj['id']
                        pass1_newly_matched.append(new_obj)
                        unmatched_old.remove(old_obj)
                unmatched_new = [obj for obj in unmatched_new if obj not in pass1_newly_matched]

                # Pass 2: Flexible Matching (Moves)
                if unmatched_old and unmatched_new:
                    old_flexible_map = {}
                    for obj in unmatched_old:
                        old_flexible_map.setdefault(self._get_stable_id(obj), deque()).append(obj)
                    
                    pass2_newly_matched = []
                    newly_matched_old_in_pass2 = []
                    for new_obj in unmatched_new:
                        stable_id = self._get_stable_id(new_obj)
                        if stable_id in old_flexible_map and old_flexible_map[stable_id]:
                            old_obj = old_flexible_map[stable_id].popleft()
                            
                            new_obj_to_old_id_map[id(new_obj)] = old_obj['id']
                            info = f"moved from {old_obj['position']} to {new_obj['position']}"
                            new_obj['cross_level_change_info'] = info
                            
                            pass2_newly_matched.append(new_obj)
                            newly_matched_old_in_pass2.append(old_obj)

                    if pass2_newly_matched:
                        unmatched_new = [obj for obj in unmatched_new if obj not in pass2_newly_matched]
                        old_ids_to_remove = {id(obj) for obj in newly_matched_old_in_pass2}
                        unmatched_old = [obj for obj in unmatched_old if id(obj) not in old_ids_to_remove]

                # Pass 3: Transformation Matching (e.g., color change at same position)
                if unmatched_old and unmatched_new:
                    old_pos_map = {obj['position']: obj for obj in unmatched_old}
                    pass3_newly_matched = []
                    
                    for new_obj in unmatched_new:
                        if new_obj['position'] in old_pos_map:
                            old_obj = old_pos_map[new_obj['position']]
                            
                            # Check for a color change with the same shape/size
                            if (old_obj['fingerprint'] == new_obj['fingerprint'] and
                                old_obj['size'] == new_obj['size'] and
                                old_obj['color'] != new_obj['color']):
                                
                                # This is a match. Store old ID and the change info.
                                change_info = f"recolored from color {old_obj['color']} to {new_obj['color']}"
                                new_obj['cross_level_change_info'] = change_info
                                new_obj_to_old_id_map[id(new_obj)] = old_obj['id']
                                
                                pass3_newly_matched.append(new_obj)
                                unmatched_old.remove(old_obj)
                                del old_pos_map[new_obj['position']] # Prevent re-matching
                    
                    unmatched_new = [obj for obj in unmatched_new if obj not in pass3_newly_matched]

                # Pass 4: Resize/Reshape Matching (allows moves)
                if unmatched_old and unmatched_new:
                    potential_pairs = []
                    for old_obj in unmatched_old:
                        for new_obj in unmatched_new:
                            # Match condition: must have same color, but different shape/size.
                            if (old_obj['color'] == new_obj['color'] and
                                (old_obj['fingerprint'] != new_obj['fingerprint'] or old_obj['pixels'] != new_obj['pixels'])):
                                
                                pos1, pos2 = old_obj['position'], new_obj['position']
                                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                                potential_pairs.append({'old': old_obj, 'new': new_obj, 'dist': distance})
                    
                    # Sort by distance to greedily match the closest pairs first
                    potential_pairs.sort(key=lambda p: p['dist'])
                    
                    pass4_matched_old = set()
                    pass4_matched_new = set()
                    for pair in potential_pairs:
                        old_obj, new_obj = pair['old'], pair['new']
                        # If either object in this pair has already been matched, skip.
                        if id(old_obj) in pass4_matched_old or id(new_obj) in pass4_matched_new:
                            continue
                        
                        # This is our best match for these two objects.
                        new_obj_to_old_id_map[id(new_obj)] = old_obj['id']
                        
                        # --- Generate a descriptive string ---
                        old_size_str = f"{old_obj['size'][0]}x{old_obj['size'][1]}"
                        new_size_str = f"{new_obj['size'][0]}x{new_obj['size'][1]}"
                        info_parts = []
                        
                        if new_obj['pixels'] > old_obj['pixels']:
                            info_parts.append(f"grew from {old_size_str} to {new_size_str}")
                        elif new_obj['pixels'] < old_obj['pixels']:
                            info_parts.append(f"shrank from {old_size_str} to {new_size_str}")
                        elif old_obj['fingerprint'] != new_obj['fingerprint']:
                            info_parts.append("changed shape")
                        
                        if old_obj['position'] != new_obj['position']:
                            info_parts.append(f"moved to {new_obj['position']}")
                            
                        new_obj['cross_level_change_info'] = ", ".join(info_parts)
                        
                        # Add to matched sets to prevent re-matching
                        pass4_matched_old.add(id(old_obj))
                        pass4_matched_new.add(id(new_obj))
                        
                    # Clean up the master lists
                    if pass4_matched_old:
                        unmatched_old = [obj for obj in unmatched_old if id(obj) not in pass4_matched_old]
                        unmatched_new = [obj for obj in unmatched_new if id(obj) not in pass4_matched_new]

                self.final_summary_before_level_change = None

            # --- Final Re-Numbering and Memory Migration ---
            print("--- Finalizing Frame with Sequential IDs ---")
            self.object_id_counter = 0
            
            # Sort by position for a clean, predictable order.
            sorted_summary = sorted(current_summary, key=lambda o: (o['position'][0], o['position'][1]))
            
            for obj in sorted_summary:
                self.object_id_counter += 1
                new_id = f'obj_{self.object_id_counter}'
                
                # If this object was matched, record the mapping from its old ID to its new one.
                if id(obj) in new_obj_to_old_id_map:
                    old_id = new_obj_to_old_id_map[id(obj)]
                    id_map[old_id] = new_id
                
                obj['id'] = new_id # Assign the new, clean ID.
            
            current_summary = sorted_summary

            if id_map:
                self._remap_memory(id_map)

            print("--- Initial Frame Summary ---")
            if not current_summary:
                print("No objects found.")
            
            # Create an inverse map for logging purposes.
            new_to_old_id_map = {v: k for k, v in id_map.items()}
            self._print_full_summary(current_summary, new_to_old_id_map)

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

                if current_adjacencies:
                    print("\n--- Initial Adjacency Analysis (Top, Right, Bottom, Left) ---")
                    for obj_id, contacts in sorted(current_adjacencies.items(), key=lambda item: int(item[0].split('_')[1])):
                        # Format the contact list into the desired string
                        contact_ids = [c.replace('obj_', '') if 'obj_' in c else c for c in contacts]
                        contact_tuple_str = ", ".join(contact_ids)
                        clean_obj_id = obj_id.replace('obj_', 'id_')
                        print(f"- Object {clean_obj_id} ({contact_tuple_str})")

                if current_alignments:
                    print("\n--- Initial Alignment Analysis ---")
                    def format_align_ids(id_set):
                        id_list = sorted(list(id_set))
                        if len(id_list) < 2: return f"Object {id_list[0]}"
                        if len(id_list) == 2: return f"Objects {id_list[0]} and {id_list[1]}"
                        return "Objects " + ", ".join(map(str, id_list[:-1])) + f", and {id_list[-1]}"
                    
                    for align_type, groups in sorted(current_alignments.items()):
                        for coord, ids in sorted(groups.items()):
                            print(f"- '{align_type}' Alignment at {coord}: {format_align_ids(ids)}")

                if current_match_groups:
                    print("\n--- Object Match Type Analysis ---")
                    print_order = ['Exact', 'Color', 'Fingerprint', 'Size', 'Pixels']
                    group_counter = 1
                    for match_type in print_order:
                        if match_type in current_match_groups:
                            groups_dict = current_match_groups[match_type]
                            label = f"Exact Matches" if match_type == "Exact" else f"Matches (Except {match_type})"
                            print(f"- {label}:")
                            
                            for props, group in groups_dict.items():
                                sorted_ids = sorted(group, key=int)
                                id_list_str = ", ".join([f"id_{id_num}" for id_num in sorted_ids])
                                
                                # Add property details to the group title
                                props_str_parts = []
                                if match_type == 'Exact':
                                    props_str_parts.append(f"Color:{props[0]}")
                                    props_str_parts.append(f"Fingerprint:{props[1]}")
                                    props_str_parts.append(f"Size:{props[2][0]}x{props[2][1]}")
                                    props_str_parts.append(f"Pixels:{props[3]}")
                                elif match_type == 'Color':
                                    props_str_parts.append(f"Fingerprint:{props[0]}")
                                    props_str_parts.append(f"Size:{props[1][0]}x{props[1][1]}")
                                    props_str_parts.append(f"Pixels:{props[2]}")
                                elif match_type == 'Fingerprint':
                                    props_str_parts.append(f"Color:{props[0]}")
                                    props_str_parts.append(f"Size:{props[1][0]}x{props[1][1]}")
                                    props_str_parts.append(f"Pixels:{props[2]}")
                                elif match_type == 'Size':
                                    props_str_parts.append(f"Color:{props[0]}")
                                    props_str_parts.append(f"Fingerprint:{props[1]}")
                                    props_str_parts.append(f"Pixels:{props[2]}")
                                elif match_type == 'Pixels':
                                    props_str_parts.append(f"Color:{props[0]}")
                                    props_str_parts.append(f"Fingerprint:{props[1]}")
                                    props_str_parts.append(f"Size:{props[2][0]}x{props[2][1]}")
                                
                                props_str = f" ({', '.join(props_str_parts)})" if props_str_parts else ""
                                print(f"  - Group {group_counter}{props_str}: {id_list_str}")
                                group_counter += 1

        # On subsequent turns, analyze the outcome of the previous action.
        else:
            changes = []
            prev_summary = self.last_object_summary
            changes, current_summary = self._log_changes(prev_summary, current_summary)

            # --- LEVEL CHANGE DETECTION & HANDLING ---
            current_score = latest_frame.score
            if current_score > self.last_score:
                print(f"\n--- LEVEL CHANGE DETECTED (Score increased from {self.last_score} to {current_score}) ---")
                
                # Analyze what was unique about the winning state compared to the rest of the level
                if self.level_state_history:
                    winning_context = self.level_state_history[-1] # The last state before the score change
                    historical_contexts = self.level_state_history[:-1]
                    self._analyze_win_condition(self.level_state_history, self.level_milestones)
                
                # Print the full summary of the final frame of the level that was just won.
                print("\n--- Final Frame Summary (Old Level) ---")
                self._print_full_summary(self.last_object_summary)

                print("\nSaving final frame for cross-level object matching.")
                # Save the final summary of the completed level for comparison.
                self.final_summary_before_level_change = self.last_object_summary
                # Signal that the next frame is the start of a new level, but preserve all memory.
                self.is_new_level = True

            # Update the score tracker for the next turn.
            self.last_score = current_score

            self._log_relationship_changes(self.last_relationships, current_relationships)
            self._log_adjacency_changes(self.last_adjacencies, current_adjacencies)
            self._log_alignment_changes(self.last_alignments, current_alignments)
            self._log_match_type_changes(self.last_match_groups, current_match_groups)

            # --- Prepare for Learning & Rule Analysis ---
            current_state_key = self._get_state_key(current_summary)

            if self.last_action_context:
                # --- Analyze the outcome of the previous action ---
                prev_action_name, prev_target_id = self.last_action_context
                learning_key = self._get_learning_key(prev_action_name, prev_target_id)
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
                                prev_action_name, prev_target_id = self.last_action_context
                                base_action_key = self._get_learning_key(prev_action_name, prev_target_id)
                                
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

                    # --- Milestone Detection ---
                    # A milestone is any turn with a unique change, OR the discovery of a new game mechanic.
                    is_milestone = False
                    if novel_state_count > 0:
                        is_milestone = True

                    # Check for new mechanic discovery
                    current_event_types = {log.split(':')[0].replace('- ', '') for log in changes}
                    newly_discovered_types = current_event_types - self.seen_event_types_in_level
                    if newly_discovered_types:
                        is_milestone = True
                        self.seen_event_types_in_level.update(newly_discovered_types)
                        print(f"Milestone Discovery: New event types observed: {sorted(list(newly_discovered_types))}")
                    
                    if is_milestone:
                        # The current state will be appended to the history at the end of this method.
                        # Its index will be the current length of the history.
                        milestone_index = len(self.level_state_history)
                        self.level_milestones.append(milestone_index)
                        print(f"Logging Milestone at state index {milestone_index}.")


                else:
                    # No changes occurred. Check if this is a failure, but NOT if we just finished waiting.
                    if not just_finished_waiting:
                        # We already have the correct learning_key from the top of the block.
                        if learning_key in self.success_contexts or self.action_counts.get((self.last_state_key, learning_key), 0) > 0:
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
                    success_context = {
                        'summary': prev_summary,
                        'rels': self.last_relationships,
                        'adj': self.last_adjacencies
                    }
                    self.success_contexts.setdefault(learning_key, []).append(success_context)
                
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
                    
                    print(f"\n--- Failure Detected for Action {learning_key} ---")
                    # Store the context of this failure before analyzing
                    failure_context = {
                        'summary': prev_summary,
                        'rels': self.last_relationships,
                        'adj': self.last_adjacencies
                    }
                    self.failure_contexts.setdefault(learning_key, []).append(failure_context)

                    # Get all history for this action
                    successes = self.success_contexts.get(learning_key, [])
                    failures = self.failure_contexts.get(learning_key, [])

                    # Perform the new, advanced analysis
                    self._analyze_failures(learning_key, successes, failures, failure_context)
                        
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
        self.last_adjacencies = current_adjacencies
        self.last_relationships = current_relationships
        self.last_alignments = current_alignments
        self.last_match_groups = current_match_groups

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

            # --- Failure Prediction Penalty ---
            failure_penalty = 0.0
            known_failure_pattern = self.failure_patterns.get(action_key)
            if known_failure_pattern:
                is_match = True # Assume the current state matches the pattern until proven otherwise.
                
                # Check Adjacency Patterns
                pattern_adj = known_failure_pattern.get('adj', {})
                for obj_id, pattern_contacts in pattern_adj.items():
                    current_contacts = tuple(current_adjacencies.get(obj_id, ['na']*4))
                    # Compare the current state against the pattern, ignoring wildcards ('x')
                    for i in range(4):
                        if pattern_contacts[i] != 'x' and pattern_contacts[i] != current_contacts[i]:
                            is_match = False
                            break
                    if not is_match:
                        break
                
                # (Future enhancement: Could add checks for relationship patterns here as well)

                if is_match:
                    failure_penalty = -50.0 # Apply a heavy penalty for a predicted failure
                    print(f"Heuristic: Current state matches a known failure pattern for {action_key}. Applying penalty.")

            score += failure_penalty

            goal_bonus = self._calculate_goal_bonus(move)
            score += goal_bonus

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
        target_id_for_context = chosen_object['id'] if chosen_object else None
        self.last_action_context = (action_to_return.name, target_id_for_context)
        
        # Store the current state for the level history before ending the turn
        current_context = {
            'summary': current_summary,
            'rels': current_relationships,
            'adj': current_adjacencies,
            'match': current_match_groups,
            'align': current_alignments,
            'events': changes 
        }
        if self.last_action_context:
             # This is a simplification; a full implementation would pass the `changes` variable.
             # For now, this structure will help us in the next step.
             current_context['last_action_changes'] = []
        
        self.level_state_history.append(current_context)

        return action_to_return

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """
        This method is called by the game to see if the agent thinks it is done.
        """
        return False

    def _find_single_property_change_match(self, new_obj):
        """Finds a removed object that matches the new object in all but one property."""
        new_stable_id = self._get_stable_id(new_obj)
        
        for old_stable_id in self.removed_objects_memory.keys():
            diffs = {}
            # old_stable_id = (fingerprint, color, size, pixels)
            if new_stable_id[0] != old_stable_id[0]:
                diffs['shape'] = (old_stable_id[0], new_stable_id[0])
            if new_stable_id[1] != old_stable_id[1]:
                diffs['color'] = (old_stable_id[1], new_stable_id[1])
            if new_stable_id[2] != old_stable_id[2]:
                diffs['size'] = (old_stable_id[2], new_stable_id[2])
            if new_stable_id[3] != old_stable_id[3]:
                diffs['pixels'] = (old_stable_id[3], new_stable_id[3])

            if len(diffs) == 1:
                # We found a match with exactly one property change.
                return old_stable_id, diffs
                
        return None, None
    
    def _remap_memory(self, id_map: dict[str, str]):
        """Updates all learning structures to use new IDs after a re-numbering."""
        print(f"Remapping memory for {len(id_map)} objects...")

        def remap_key(key):
            if isinstance(key, str) and '_' in key:
                parts = key.split('_')
                action_part, old_id_part = parts[0], '_'.join(parts[1:])
                if old_id_part in id_map:
                    return f"{action_part}_{id_map[old_id_part]}"
            return key

        # Remap weights, action_counts, and action_history
        for memory_dict in [self.weights, self.action_counts, self.action_history]:
            keys_to_remap = [k for k in memory_dict.keys() if any(old_id in str(k) for old_id in id_map)]
            for old_key in keys_to_remap:
                new_key = remap_key(old_key) if isinstance(old_key, str) else \
                          (remap_key(old_key[0]), remap_key(old_key[1]))
                if new_key != old_key:
                    memory_dict[new_key] = memory_dict.pop(old_key)

        # Remap rule_hypotheses, which have a more complex key structure
        keys_to_remap = [k for k in self.rule_hypotheses.keys()]
        for old_key in keys_to_remap:
            # old_key = (base_action_key, object_id, object_state)
            base_action, old_id, obj_state = old_key
            if old_id in id_map:
                new_id = id_map[old_id]
                new_key = (base_action, new_id, obj_state)
                self.rule_hypotheses[new_key] = self.rule_hypotheses.pop(old_key)
    
    def _get_learning_key(self, action_name: str, target_id: str | None) -> str:
        """Generates a unique key for learning, specific to an object ID for CLICK actions."""
        if action_name == 'ACTION6' and target_id:
            return f"{action_name}_{target_id}"
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

    def _find_common_context(self, contexts: list[dict]) -> dict:
        """
        Finds the common context across a list of attempts, using wildcard 'x' for
        inconsistent adjacency properties.
        """
        if not contexts:
            return {'adj': {}, 'rels': {}}

        # --- Adjacency Analysis with Wildcards ---
        # Start with the first case as the baseline pattern. Convert to lists for mutability.
        master_adj = {obj_id: list(contacts) for obj_id, contacts in contexts[0].get('adj', {}).items()}
        
        # Iteratively refine the master pattern against all other contexts
        for i in range(1, len(contexts)):
            next_adj = contexts[i].get('adj', {})
            
            # Use list(master_adj.keys()) to iterate safely while potentially deleting keys
            for obj_id in list(master_adj.keys()):
                master_pattern = master_adj[obj_id]
                next_contacts = next_adj.get(obj_id)
                
                # If the object doesn't exist in the next context, the pattern is broken.
                if not next_contacts:
                    del master_adj[obj_id]
                    continue
                
                # Compare each direction (top, right, bottom, left)
                for i in range(4):
                    # If a direction is already a wildcard, it stays a wildcard.
                    if master_pattern[i] == 'x':
                        continue
                    # If the contacts for this direction differ, it becomes a wildcard.
                    if master_pattern[i] != next_contacts[i]:
                        master_pattern[i] = 'x'
                
                # If the whole pattern has become wildcards, it's not useful information.
                if all(d == 'x' for d in master_pattern):
                    del master_adj[obj_id]

        # Convert lists back to tuples for the final result
        common_adj = {obj_id: tuple(pattern) for obj_id, pattern in master_adj.items()}

        # --- Relationship Analysis (Strict Intersection - logic is unchanged) ---
        common_rels = copy.deepcopy(contexts[0].get('rels', {}))
        for i in range(1, len(contexts)):
            next_rels = contexts[i].get('rels', {})
            temp_common_rels = {}
            common_rel_types = set(common_rels.keys()) & set(next_rels.keys())
            for rel_type in common_rel_types:
                groups1, groups2 = common_rels[rel_type], next_rels[rel_type]
                common_values = set(groups1.keys()) & set(groups2.keys())
                for value in common_values:
                    if groups1[value] == groups2[value]:
                        temp_common_rels.setdefault(rel_type, {})[value] = groups1[value]
            common_rels = temp_common_rels
        
        return {'adj': common_adj, 'rels': common_rels}

    def _analyze_failures(self, action_key: str, all_success_contexts: list[dict], all_failure_contexts: list[dict], current_failure_context: dict):
        """
        Analyzes failures by finding conditions that are consistent across all failures
        AND have never been observed in any past success.
        """
        
        if not all_success_contexts or not all_failure_contexts:
            print("\n--- Failure Analysis ---")
            print("Cannot perform differential analysis: insufficient history of successes or failures.")
            return

        print("\n--- Failure Analysis: Consistent Differentiating Conditions ---")
        
        # 1. Find the common context for all successes and all failures.
        common_success_context = self._find_common_context(all_success_contexts)
        common_failure_context = self._find_common_context(all_failure_contexts)
        
        # 2. Build a comprehensive set of ALL states ever observed in ANY success case.
        # This will be used to veto any differences that aren't exclusive to failures.
        observed_in_any_success_adj = set()
        observed_in_any_success_rels = set()
        for context in all_success_contexts:
            for obj_id, contacts in context.get('adj', {}).items():
                observed_in_any_success_adj.add((obj_id, tuple(contacts)))
            for rel_type, groups in context.get('rels', {}).items():
                for value, ids in groups.items():
                    observed_in_any_success_rels.add((rel_type, value, frozenset(ids)))

        diffs_found = False

        # 3. Compare the master contexts, applying the veto check.
        # Adjacency Differences
        adj_s = common_success_context['adj']
        adj_f = common_failure_context['adj']
        all_adj_ids = set(adj_s.keys()) | set(adj_f.keys())

        for obj_id in sorted(list(all_adj_ids), key=lambda x: int(x.split('_')[1])):
            contacts_s = tuple(adj_s.get(obj_id, ['na']*4))
            contacts_f = tuple(adj_f.get(obj_id, ['na']*4))

            if contacts_s != contacts_f:
                # VETO CHECK: Is this failure condition truly novel, or did it appear in at least one success?
                if (obj_id, contacts_f) not in observed_in_any_success_adj:
                    diffs_found = True
                    failure_pattern = []
                    for i in range(4):
                        if contacts_s[i] == contacts_f[i]:
                            failure_pattern.append('x')
                        else:
                            contact = contacts_f[i]
                            failure_pattern.append(contact.replace('obj_', '') if 'obj_' in contact else contact)
                    
                    pattern_str = f"({', '.join(failure_pattern)})"
                    clean_id = obj_id.replace('obj_', 'id_')
                    print(f"- Adjacency Difference for {clean_id}: Failures consistently exhibit pattern {pattern_str}, which has never occurred in a success.")

        # Relationship Differences
        rels_s = common_success_context['rels']
        rels_f = common_failure_context['rels']
        all_rel_types = set(rels_s.keys()) | set(rels_f.keys())

        for rel_type in all_rel_types:
            groups_s, groups_f = rels_s.get(rel_type, {}), rels_f.get(rel_type, {})
            all_values = set(groups_s.keys()) | set(groups_f.keys())
            for value in all_values:
                ids_s, ids_f = groups_s.get(value, set()), groups_f.get(value, set())
                if ids_s != ids_f:
                    # VETO CHECK:
                    if (rel_type, value, frozenset(ids_f)) not in observed_in_any_success_rels:
                        diffs_found = True
                        value_str = f"{value[0]}x{value[1]}" if rel_type == 'Size' else value
                        print(f"- {rel_type} Group ({value_str}) Difference: Failures consistently have members {sorted(list(ids_f))}, which has never occurred in a success.")
        
        if diffs_found:
            # This is a key insight. Store the common failure context that was successfully
            # differentiated from all successes. This becomes our new theory for why this action fails.
            self.failure_patterns[action_key] = common_failure_context

        if not diffs_found:
            print("No conditions found that are both consistent across all failures and unique to them.")

    def _analyze_win_condition(self, level_history: list[dict], milestone_indices: list[int]):
        """
        Analyzes a completed level by breaking it into chapters based on milestones
        and generating competing hypotheses about the winning "recipe".
        """
        print("\n--- Win Condition Analysis (V2) ---")
        if len(level_history) < 2:
            print("Insufficient history for analysis.")
            return

        # The final state is always the last one in the history before the win.
        winning_state_index = len(level_history) - 1
        
        # Combine the start, milestones, and end into a list of key moments.
        # Ensure the start (index 0) and end are always included and the list is sorted with no duplicates.
        key_indices = sorted(list(set([0] + milestone_indices + [winning_state_index])))
        
        print(f"Analyzing level based on {len(key_indices)} key moments at indices: {key_indices}")

        # --- Step 1: Analyze the "chapters" between key moments ---
        level_chapters = []
        print("\n--- Chapter Analysis ---")
        for i in range(len(key_indices) - 1):
            start_index = key_indices[i]
            end_index = key_indices[i+1]

            start_context = level_history[start_index]
            end_context = level_history[end_index]
            
            delta = self._get_context_delta(start_context, end_context)
            
            # Only store and report on chapters that had a significant change.
            if delta:
                chapter_info = {
                    'start': start_index,
                    'end': end_index,
                    'delta': delta
                }
                level_chapters.append(chapter_info)
                # For logging, create a summary of the delta
            delta_summary = []
            if 'rels' in delta:
                for rel_type, changes in delta['rels'].items():
                    for change in changes:
                        delta_summary.append(f"new '{rel_type}' group for value '{change['value']}'")
            if 'adjs' in delta:
                for adj_change in delta['adjs']:
                    delta_summary.append(f"new adjacency for id_{adj_change['obj_id'].split('_')[1]}")
            if 'aligns' in delta:
                for align_type, changes in delta['aligns'].items():
                    for change in changes:
                        delta_summary.append(f"new '{align_type}' alignment at {change['coord']}")
            if 'objs' in delta:
                if 'added' in delta['objs']:
                    delta_summary.append(f"{len(delta['objs']['added'])} new object(s) appeared")
                if 'removed' in delta['objs']:
                    delta_summary.append(f"{len(delta['objs']['removed'])} object(s) were removed")
            
            if delta_summary:
                print(f"- Chapter ({start_index} -> {end_index}): Delta includes {', '.join(delta_summary)}.")

        if not level_chapters:
            print("No significant deltas found between key moments.")

        if not self.win_condition_hypotheses:
            # --- Step 2: Formulate Competing Hypotheses (The "Recipes") ---
            new_hypotheses = []
            hyp_counter = 0

            # Recipe 1: The "Final Step" - based on the very last significant chapter.
            if level_chapters:
                last_chapter = level_chapters[-1]
                pattern_list = self._format_delta_as_pattern_list(last_chapter['delta'])
                if pattern_list:
                    hyp_counter += 1
                    new_hypotheses.append({
                        'id': f'hyp_{hyp_counter}', 'type': 'static_pattern', 'confidence': 1.0,
                        'description': f"Final Step (from turn {last_chapter['start']})",
                        'conditions': pattern_list
                    })

            # Recipe 2: "Causal Chain" hypotheses - connecting key milestones to the final step.
            if len(level_chapters) > 1:
                last_chapter = level_chapters[-1]
                final_step_patterns = self._format_delta_as_pattern_list(last_chapter['delta'])
                
                # Create a sequential hypothesis for each milestone, treating it as a prerequisite to the final step.
                # We iterate up to the second-to-last chapter, as the last one is handled separately.
                for chapter in level_chapters[:-1]:
                    milestone_patterns = self._format_delta_as_pattern_list(chapter['delta'])
                    
                    if milestone_patterns and final_step_patterns:
                        hyp_counter += 1
                        new_hypotheses.append({
                            'id': f'hyp_{hyp_counter}',
                            'type': 'sequential_pattern',
                            'confidence': 1.0,
                            'description': f"Event at turn {chapter['start']} followed by Final Step",
                            'conditions': [
                                {'description': f"Step from turn {chapter['start']}", 'patterns': milestone_patterns},
                                {'description': f"Step from turn {last_chapter['start']}", 'patterns': final_step_patterns}
                            ]
                        })
            
            # Recipe 3: The "Full Recipe" - a sequential pattern of all chapter deltas.
            if len(level_chapters) > 1:
                full_recipe_steps = []
                for chapter in level_chapters:
                    step_patterns = self._format_delta_as_pattern_list(chapter['delta'])
                    if step_patterns:
                        full_recipe_steps.append({
                            'description': f"Step from turn {chapter['start']}",
                            'patterns': step_patterns
                        })
                if full_recipe_steps:
                    hyp_counter += 1
                    new_hypotheses.append({
                        'id': f'hyp_{hyp_counter}', 'type': 'sequential_pattern', 'confidence': 1.0,
                        'description': 'Full sequence of events',
                        'conditions': full_recipe_steps
                    })

            # Recipe 4: The "Start-to-Finish" Recipe - a simple delta of the whole level.
            start_context = level_history[0]
            end_context = level_history[winning_state_index]
            overall_delta = self._get_context_delta(start_context, end_context)
            pattern_list = self._format_delta_as_pattern_list(overall_delta)
            if pattern_list:
                hyp_counter += 1
                new_hypotheses.append({
                    'id': f'hyp_{hyp_counter}', 'type': 'static_pattern', 'confidence': 1.0,
                    'description': 'Overall level goal',
                    'conditions': pattern_list
                })

             # --- Step 3: Remove duplicate hypotheses and store them ---
            unique_hypotheses = []
            seen_conditions = set()
            for hyp in new_hypotheses:
                # Create a hashable representation of the conditions to check for duplicates
                conditions_str = str(sorted(hyp['conditions'], key=lambda x: str(x)))
                if conditions_str not in seen_conditions:
                    unique_hypotheses.append(hyp)
                    seen_conditions.add(conditions_str)
            
            self.win_condition_hypotheses = unique_hypotheses

        else:
            # --- PRUNING MODE (After Level 2+) ---
            print(f"\n--- Pruning {len(self.win_condition_hypotheses)} Hypotheses Based on New Evidence ---")
            surviving_hypotheses = []
            
            for hyp in self.win_condition_hypotheses:
                is_consistent = self._check_hypothesis_against_level(hyp, level_chapters)
                
                if is_consistent:
                    # Reinforce confidence for consistent theories
                    hyp['confidence'] = min(1.0, hyp['confidence'] + 0.25)
                    surviving_hypotheses.append(hyp)
                    print(f"- {hyp['id']} ({hyp['description']}) was CONSISTENT. Confidence -> {hyp['confidence']:.0%}")
                else:
                    # Penalize inconsistent theories (Confidence Decay)
                    hyp['confidence'] *= 0.5
                    if hyp['confidence'] >= 0.25: # Confidence threshold
                        surviving_hypotheses.append(hyp)
                        print(f"- {hyp['id']} ({hyp['description']}) was INCONSISTENT. Confidence -> {hyp['confidence']:.0%}")
                    else:
                        print(f"- {hyp['id']} ({hyp['description']}) was DELETED. Confidence fell below threshold.")
            
            self.win_condition_hypotheses = surviving_hypotheses
        
        # --- Final Report ---
        if self.win_condition_hypotheses:
            print(f"\n--- Current Top Hypotheses ({len(self.win_condition_hypotheses)} total) ---")
            # Sort by confidence to see the best theories first
            sorted_hyps = sorted(self.win_condition_hypotheses, key=lambda x: x['confidence'], reverse=True)
            for hyp in sorted_hyps[:5]: # Log top 5
                
                if hyp['type'] == 'static_pattern':
                    condition_summary = []
                    for cond in hyp['conditions']:
                        cond_type = cond.get('type')
                        if cond_type == 'group':
                            condition_summary.append(f"form a '{cond.get('property')}' group")
                        elif cond_type == 'adjacency':
                            condition_summary.append("create an adjacency")
                        elif cond_type == 'alignment':
                            condition_summary.append(f"create a '{cond.get('align_type')}' alignment")
                        elif cond_type == 'object_appeared':
                            condition_summary.append("cause a new object to appear")
                        elif cond_type == 'event':
                            condition_summary.append(f"cause a '{cond.get('event_type')}' event")
                    
                    # Sort the summary for consistent output and remove duplicates
                    condition_summary = sorted(list(set(condition_summary)))
                    print(f"- {hyp['id']} ({hyp['description']}) [Conf: {hyp['confidence']:.0%}]: Goal is to {', '.join(condition_summary)}.")
                
                elif hyp['type'] == 'sequential_pattern':
                    print(f"- {hyp['id']} ({hyp['description']}) [Conf: {hyp['confidence']:.0%}]: Requires completing {len(hyp['conditions'])} steps.")

    def _get_context_delta(self, start_context: dict, end_context: dict) -> dict:
        """Finds significant changes (deltas) between a start and end context."""
        delta = {}
        
        # --- 1. Relationship Deltas ---
        new_rels = {}
        start_rels = start_context.get('rels', {})
        end_rels = end_context.get('rels', {})
        for rel_type, end_groups in end_rels.items():
            start_groups = start_rels.get(rel_type, {})
            for value, end_ids in end_groups.items():
                start_ids = start_groups.get(value, set())
                if end_ids > start_ids: # Group appeared or grew
                    new_rels.setdefault(rel_type, []).append({'value': value, 'members': end_ids, 'change': 'grew'})
        if new_rels:
            delta['rels'] = new_rels

        # --- 2. Adjacency Deltas ---
        new_adjs = []
        start_adj = start_context.get('adj', {})
        end_adj = end_context.get('adj', {})
        for obj_id, end_contacts in end_adj.items():
            start_contacts = tuple(start_adj.get(obj_id, ('na', 'na', 'na', 'na')))
            if tuple(end_contacts) != start_contacts:
                new_adjs.append({'obj_id': obj_id, 'contacts': tuple(end_contacts)})
        if new_adjs:
            delta['adjs'] = new_adjs

        # --- Alignment Deltas ---
        new_aligns = {}
        start_aligns = start_context.get('align', {})
        end_aligns = end_context.get('align', {})
        for align_type, end_groups in end_aligns.items():
            start_groups = start_aligns.get(align_type, {})
            for coord, end_ids in end_groups.items():
                start_ids = start_groups.get(coord, set())
                if end_ids > start_ids: # An alignment group appeared or grew
                    new_aligns.setdefault(align_type, []).append({'coord': coord, 'members': end_ids})
        if new_aligns:
            delta['aligns'] = new_aligns

        # --- 3. Individual Object Deltas (New/Removed) ---
        changed_objs = {}
        start_summary = start_context.get('summary', [])
        end_summary = end_context.get('summary', [])
        start_ids = {obj['id'] for obj in start_summary}
        end_ids = {obj['id'] for obj in end_summary}
        
        added_ids = end_ids - start_ids
        removed_ids = start_ids - end_ids
        
        if added_ids:
            changed_objs['added'] = [obj for obj in end_summary if obj['id'] in added_ids]
        if removed_ids:
            changed_objs['removed'] = [obj for obj in start_summary if obj['id'] in removed_ids]
        if changed_objs:
            delta['objs'] = changed_objs

        # --- 4. Direct Event Deltas ---
        # We only care about the events that happened at the end of the chapter
        end_events = end_context.get('events', [])
        if end_events:
            delta['events'] = end_events
            
        return delta
    
    def _format_delta_as_pattern_list(self, delta: dict) -> list[dict]:
        """Converts a delta dictionary into a list of GENERALIZED pattern dictionaries."""
        patterns = []
        # Generalize relationship patterns by type, not value
        for rel_type in delta.get('rels', {}):
            patterns.append({'type': 'group', 'property': rel_type})
        
        # Generalize adjacency patterns
        if 'adjs' in delta:
            patterns.append({'type': 'adjacency'})

        # Generalize alignment patterns by type, not coordinate
        for align_type in delta.get('aligns', {}):
            patterns.append({'type': 'alignment', 'align_type': align_type})

        # Generalize object appearance
        if delta.get('objs', {}).get('added'):
            patterns.append({'type': 'object_appeared'})
        
        # Generalize events by their type (MOVED, GROWTH, etc.)
        for event_str in delta.get('events', []):
            event_type = event_str.split(':')[0].replace('- ', '')
            patterns.append({'type': 'event', 'event_type': event_type})

        # Remove duplicates by converting to a set of tuples and back to dicts
        unique_patterns = {tuple(p.items()) for p in patterns}
        return [dict(t) for t in unique_patterns]

    def _calculate_goal_bonus(self, move: dict) -> float:
        """
        Calculates a score bonus based on how well a potential move helps achieve
        any of the current win condition hypotheses. This acts as the "Scoreboard" system.
        """
        if not self.win_condition_hypotheses or not self.rule_hypotheses:
            return 0.0

        total_bonus = 0.0
        action_template = move['type']
        target_object = move['object']

        # Determine the base action key for looking up rules.
        coords_for_key = {'x': target_object['position'][1], 'y': target_object['position'][0]} if target_object else None
        base_action_key = self._get_learning_key(action_template.name, coords_for_key)
        
        # --- Step 1: Predict the Action's Outcome ---
        # Find all known effects for this specific action from our rule hypotheses.
        known_effects = set()
        for rule_key, hypothesis in self.rule_hypotheses.items():
            rule_base_key = rule_key[0]
            # We only consider rules with high confidence to make reliable predictions
            if rule_base_key == base_action_key and hypothesis.get('confidence', 0.0) > 0.75:
                # The rule_key is a tuple: (base_action_key, object_id, object_state)
                # We need to ensure the rule applies to the object we're targeting.
                if target_object and rule_key[1] == target_object['id']:
                    for event in hypothesis.get('rules', []):
                        # Generalize the effect into a searchable "creates_property_value" format
                        if event.get('type') == 'RECOLORED': known_effects.add(f"creates_Color_{event.get('to_color')}")
                        elif event.get('type') == 'SHAPE_CHANGED': known_effects.add(f"creates_Shape_{event.get('to_fingerprint')}")
                        elif 'to_size' in event: known_effects.add(f"creates_Size_{event.get('to_size')}")

        if not known_effects:
            return 0.0

        # --- Step 2: Check Predictions Against Each Hypothesis ---
        for hyp in self.win_condition_hypotheses:
            required_patterns = []
            if hyp['type'] == 'static_pattern':
                required_patterns = hyp['conditions']
            elif hyp['type'] == 'sequential_pattern' and hyp['conditions']:
                # For a sequence, focus on achieving the first step.
                required_patterns = hyp['conditions'][0].get('patterns', [])

            for pattern in required_patterns:
                if pattern.get('type') == 'group':
                    prop = pattern.get('property')
                    val = pattern.get('value')
                    required_effect = f"creates_{prop}_{val}"
                    
                    if required_effect in known_effects:
                        # --- Step 3: Calculate a Confidence-Weighted Bonus ---
                        bonus = 25.0 * hyp['confidence']
                        total_bonus += bonus
                        print(f"Goal-Seeking Bonus: Action on {target_object['id']} helps '{hyp['description']}' (Confidence: {hyp['confidence']:.0%}). Adding {bonus:.2f} bonus.")
        
        return total_bonus
    
    def _check_hypothesis_against_level(self, hypothesis: dict, level_chapters: list[dict]) -> bool:
        """Checks if a given GENERALIZED hypothesis is consistent with the events of a completed level."""
        hyp_type = hypothesis['type']
        hyp_conditions = hypothesis['conditions']

        if not hyp_conditions:
            return False

        # Get a flat list of all unique abstract patterns that occurred in the entire level
        level_patterns_observed = set()
        for chapter in level_chapters:
            chapter_patterns = self._format_delta_as_pattern_list(chapter['delta'])
            for p in chapter_patterns:
                level_patterns_observed.add(tuple(p.items()))
        
        # Convert set of tuples back to list of dicts for easy lookup
        level_patterns_observed_list = [dict(t) for t in level_patterns_observed]

        if hyp_type == 'static_pattern':
            # Check if all required abstract conditions were observed at least once in the level.
            return all(cond in level_patterns_observed_list for cond in hyp_conditions)

        elif hyp_type == 'sequential_pattern':
            # For a sequence, check if the steps' abstract patterns appeared in the correct order.
            chapter_idx = 0
            for step in hyp_conditions:
                # The patterns required for this specific step in the recipe
                step_required_patterns = step['patterns']
                found_step = False
                # Search for a chapter in the level that satisfies this step's requirements
                while chapter_idx < len(level_chapters):
                    # Get the abstract patterns for the current chapter
                    chapter_patterns_in_step = self._format_delta_as_pattern_list(level_chapters[chapter_idx]['delta'])
                    chapter_idx += 1
                    # Check if this chapter satisfies all conditions for the current step in the recipe
                    if all(cond in chapter_patterns_in_step for cond in step_required_patterns):
                        found_step = True
                        break # Found this step, move on to the next one in the recipe
                
                if not found_step:
                    return False # A required step was not found in order.
            
            return True # All steps were found in the correct order.
        
        return False

    def _analyze_relationships(self, object_summary: list[dict]) -> tuple[dict, dict, dict]:
        """Analyzes object relationships and returns a structured dictionary of groups."""
        if not object_summary or len(object_summary) < 2:
            return {}, {}, {}

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

        # --- Pixel-Perfect Adjacency Analysis ---
        temp_adj = {}
        # Create a quick lookup map of pixel coordinates to object IDs
        pixel_map = {}
        for obj in object_summary:
            for pixel in obj['pixel_coords']:
                pixel_map[pixel] = obj['id']

        # Pass 1: Collect all contacts for every object using the pixel map
        for obj_a in object_summary:
            a_id = obj_a['id']
            for r, c in obj_a['pixel_coords']:
                # Check neighbors (top, right, bottom, left)
                neighbors = {
                    'top': (r - 1, c),
                    'bottom': (r + 1, c),
                    'left': (r, c - 1),
                    'right': (r, c + 1)
                }
                for direction, coord in neighbors.items():
                    if coord in pixel_map:
                        b_id = pixel_map[coord]
                        if a_id != b_id:
                            # Reverse the direction for obj_a's perspective
                            # If B is on top of A's pixel, then B is a 'top' contact for A.
                            temp_adj.setdefault(a_id, {}).setdefault(direction, set()).add(b_id)

        # Pass 2: Simplify contacts into the final format (top, right, bottom, left)
        adjacency_map = {}
        for obj_id, contacts in temp_adj.items():
            result = ['na'] * 4 # [top, right, bottom, left]
            
            top_contacts = contacts.get('top', set())
            if len(top_contacts) == 1: result[0] = top_contacts.pop()

            right_contacts = contacts.get('right', set())
            if len(right_contacts) == 1: result[1] = right_contacts.pop()

            bottom_contacts = contacts.get('bottom', set())
            if len(bottom_contacts) == 1: result[2] = bottom_contacts.pop()
            
            left_contacts = contacts.get('left', set())
            if len(left_contacts) == 1: result[3] = left_contacts.pop()

            # Only add to the map if there's at least one unique contact
            if any(res != 'na' for res in result):
                adjacency_map[obj_id] = result
        
        # Clean up empty relationship types
        final_rels = {k: v for k, v in final_rels.items() if v}
        
        # --- Match Type Analysis ---
        match_groups = {}
        processed_ids = set()

        # Pass 1: Find Exact Matches
        exact_match_key = lambda o: (o['color'], o['fingerprint'], o['size'], o['pixels'])
        temp_groups = {}
        for obj in object_summary:
            temp_groups.setdefault(exact_match_key(obj), []).append(int(obj['id'].replace('obj_', '')))
        
        exact_groups_dict = {key: group for key, group in temp_groups.items() if len(group) > 1}
        if exact_groups_dict:
            match_groups['Exact'] = exact_groups_dict
            for group in exact_groups_dict.values():
                processed_ids.update(group)

        # Subsequent Passes: Find partial matches, excluding objects already in a more specific match group.
        partial_match_definitions = {
            "Color":       lambda o: (o['fingerprint'], o['size'], o['pixels']),
            "Fingerprint": lambda o: (o['color'], o['size'], o['pixels']),
            "Size":        lambda o: (o['color'], o['fingerprint'], o['pixels']),
            "Pixels":      lambda o: (o['color'], o['fingerprint'], o['size']),
        }

        for match_type, key_func in partial_match_definitions.items():
            temp_groups = {}
            # Only consider objects not yet processed
            unprocessed_objects = [o for o in object_summary if int(o['id'].replace('obj_', '')) not in processed_ids]
            
            for obj in unprocessed_objects:
                temp_groups.setdefault(key_func(obj), []).append(int(obj['id'].replace('obj_', '')))
            
            partial_groups_dict = {key: group for key, group in temp_groups.items() if len(group) > 1}
            if partial_groups_dict:
                match_groups[match_type] = partial_groups_dict
                for group in partial_groups_dict.values():
                    processed_ids.update(group)
        
        return final_rels, adjacency_map, match_groups
    
    def _analyze_alignments(self, object_summary: list[dict]) -> dict:
        """Analyzes and groups objects based on horizontal and vertical alignments."""
        if len(object_summary) < 2:
            return {}

        # Pre-calculate alignment coordinates for each object
        for obj in object_summary:
            y, x = obj['position']
            h, w = obj['size']
            obj['top_y'] = y
            obj['bottom_y'] = y + h - 1
            obj['left_x'] = x
            obj['right_x'] = x + w - 1
            obj['center_y'] = y + h // 2
            obj['center_x'] = x + w // 2

        alignment_types = ['top_y', 'bottom_y', 'center_y', 'left_x', 'right_x', 'center_x']
        alignment_groups = {align_type: {} for align_type in alignment_types}

        for align_type in alignment_types:
            for obj in object_summary:
                coord = obj[align_type]
                obj_id = int(obj['id'].replace('obj_', ''))
                alignment_groups[align_type].setdefault(coord, set()).add(obj_id)

        # Filter out groups with only one member
        final_alignments = {}
        for align_type, groups in alignment_groups.items():
            filtered_groups = {coord: ids for coord, ids in groups.items() if len(ids) > 1}
            if filtered_groups:
                final_alignments[align_type] = filtered_groups
        
        return final_alignments

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

    def _log_adjacency_changes(self, old_adj: dict, new_adj: dict):
        """Compares two simplified adjacency maps and logs the differences."""
        if old_adj == new_adj:
            return

        output_lines = []
        all_ids = set(old_adj.keys()) | set(new_adj.keys())

        for obj_id in sorted(list(all_ids), key=lambda x: int(x.split('_')[1])):
            old_contacts = old_adj.get(obj_id, ['na'] * 4)
            new_contacts = new_adj.get(obj_id, ['na'] * 4)

            if old_contacts != new_contacts:
                clean_id = obj_id.replace('obj_', 'id_')
                
                old_ids = [c.replace('obj_', '') if 'obj_' in c else c for c in old_contacts]
                new_ids = [c.replace('obj_', '') if 'obj_' in c else c for c in new_contacts]
                
                old_str = f"({', '.join(old_ids)})"
                new_str = f"({', '.join(new_ids)})"
                
                output_lines.append(f"- Adjacency Change for Object {clean_id}: contacts changed from {old_str} to {new_str}.")
        
        if output_lines:
            print("\n--- Adjacency Change Log ---")
            for line in output_lines:
                print(line)
            print()

    def _log_alignment_changes(self, old_aligns: dict, new_aligns: dict):
        """Compares two alignment dictionaries and logs the differences."""
        if old_aligns == new_aligns:
            return

        output_lines = []
        all_align_types = sorted(list(set(old_aligns.keys()) | set(new_aligns.keys())))

        def format_ids(id_set):
            id_list = sorted(list(id_set))
            if len(id_list) == 1: return f"object {id_list[0]}"
            return f"objects " + ", ".join(map(str, id_list))

        for align_type in all_align_types:
            old_groups = old_aligns.get(align_type, {})
            new_groups = new_aligns.get(align_type, {})
            all_coords = set(old_groups.keys()) | set(new_groups.keys())

            for coord in all_coords:
                old_ids = old_groups.get(coord, set())
                new_ids = new_groups.get(coord, set())

                if old_ids == new_ids:
                    continue

                joined = new_ids - old_ids
                left = old_ids - new_ids

                if joined and left:
                    output_lines.append(f"- '{align_type}' Alignment at {coord}: {format_ids(joined).capitalize()} replaced {format_ids(left)}.")
                elif joined:
                    output_lines.append(f"- '{align_type}' Alignment at {coord}: {format_ids(joined).capitalize()} joined.")
                elif left:
                    output_lines.append(f"- '{align_type}' Alignment at {coord}: {format_ids(left).capitalize()} left.")

        if output_lines:
            print("\n--- Alignment Change Log ---")
            for line in sorted(output_lines):
                print(line)
            print()

    def _log_match_type_changes(self, old_matches: dict, new_matches: dict):
        """Compares two match group dictionaries and logs group-level changes."""
        if old_matches == new_matches:
            return

        output_lines = []
        all_match_types = sorted(list(set(old_matches.keys()) | set(new_matches.keys())))

        def format_ids(id_set):
            id_list = sorted(list(id_set))
            if len(id_list) == 1: return f"object id_{id_list[0]}"
            return f"objects " + ", ".join([f"id_{i}" for i in id_list])

        def format_props(match_type, props):
            """Builds a descriptive string of the shared properties that define a group."""
            parts = []
            if match_type == 'Exact':
                # props = (color, fingerprint, size, pixels)
                parts.append(f"Color:{props[0]}")
                parts.append(f"Fingerprint:{props[1]}")
                parts.append(f"Size:{props[2][0]}x{props[2][1]}")
                parts.append(f"Pixels:{props[3]}")
            elif match_type == 'Color': # This means "Except Color"
                # props = (fingerprint, size, pixels)
                parts.append(f"Fingerprint:{props[0]}")
                parts.append(f"Size:{props[1][0]}x{props[1][1]}")
                parts.append(f"Pixels:{props[2]}")
            elif match_type == 'Fingerprint':
                # props = (color, size, pixels)
                parts.append(f"Color:{props[0]}")
                parts.append(f"Size:{props[1][0]}x{props[1][1]}")
                parts.append(f"Pixels:{props[2]}")
            elif match_type == 'Size':
                # props = (color, fingerprint, pixels)
                parts.append(f"Color:{props[0]}")
                parts.append(f"Fingerprint:{props[1]}")
                parts.append(f"Pixels:{props[2]}")
            elif match_type == 'Pixels':
                # props = (color, fingerprint, size)
                parts.append(f"Color:{props[0]}")
                parts.append(f"Fingerprint:{props[1]}")
                parts.append(f"Size:{props[2][0]}x{props[2][1]}")
            
            return f" ({', '.join(parts)})" if parts else ""

        for match_type in all_match_types:
            old_groups = old_matches.get(match_type, {})
            new_groups = new_matches.get(match_type, {})
            all_props = set(old_groups.keys()) | set(new_groups.keys())
            
            label = f"Exact Match" if match_type == "Exact" else f"Except {match_type} Match"

            for props in all_props:
                old_ids = set(old_groups.get(props, []))
                new_ids = set(new_groups.get(props, []))

                if old_ids == new_ids: continue

                joined = new_ids - old_ids
                left = old_ids - new_ids
                props_str = format_props(match_type, props)

                if joined and left:
                    output_lines.append(f"- {label} Group {props_str}: {format_ids(joined).capitalize()} replaced {format_ids(left)}.")
                elif joined:
                    output_lines.append(f"- {label} Group {props_str}: {format_ids(joined).capitalize()} joined.")
                elif left:
                    output_lines.append(f"- {label} Group {props_str}: {format_ids(left).capitalize()} left.")

        if output_lines:
            print("\n--- Match Type Change Log ---")
            for line in sorted(output_lines):
                print(line)
            print()

    def _print_full_summary(self, summary: list[dict], new_to_old_map: dict = None):
        """Prints a formatted summary of all objects, noting previous IDs if available."""
        if not summary:
            print("No objects found.")
            return

        for obj in summary:
            obj_id = obj['id'].replace('obj_', 'id_')
            size_str = f"{obj['size'][0]}x{obj['size'][1]}"
            
            # --- Build the descriptive formerly string ---
            formerly_parts = []
            if new_to_old_map and obj['id'] in new_to_old_map:
                old_id = new_to_old_map[obj['id']].replace('obj_', 'id_')
                formerly_parts.append(f"formerly {old_id}")
            
            if 'cross_level_change_info' in obj:
                formerly_parts.append(obj['cross_level_change_info'])
            
            formerly_str = ""
            if formerly_parts:
                formerly_str = f" ({', '.join(formerly_parts)})"

            print(
                f"- Object {obj_id}{formerly_str}: Found a {size_str} object of color {obj['color']} "
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
                    'fingerprint': shape_fingerprint,
                    'pixel_coords': frozenset(object_pixels),
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
        matched_old_in_pass1 = {id(o) for o, n in matches_to_remove}
        matched_new_in_pass1 = {id(n) for o, n in matches_to_remove}
        old_unexplained = [obj for obj in old_unexplained if id(obj) not in matched_old_in_pass1]
        new_unexplained = [obj for obj in new_unexplained if id(obj) not in matched_new_in_pass1]

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
        
        matched_old_in_pass2 = {id(o) for o, n in moves_to_remove}
        matched_new_in_pass2 = {id(n) for o, n in moves_to_remove}
        old_unexplained = [obj for obj in old_unexplained if id(obj) not in matched_old_in_pass2]
        new_unexplained = [obj for obj in new_unexplained if id(obj) not in matched_new_in_pass2]

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

        # --- Final Pass: Log remaining as REMOVED, and then handle NEW, REAPPEARED, or TRANSFORMED ---
        used_persistent_ids = set() # Safeguard to prevent using the same revived ID twice in one turn.

        for obj in old_unexplained:
            stable_id = self._get_stable_id(obj)
            changes.append(f"- REMOVED: Object {obj['id'].replace('obj_', 'id_')} (ID {stable_id}) at {obj['position']} has disappeared.")
            self.removed_objects_memory.setdefault(stable_id, deque()).append(obj['id'])

        if assign_new_ids:
            unmatched_new = []
            # Step 1: Check for perfect matches (REAPPEARED) first.
            for obj in new_unexplained:
                stable_id = self._get_stable_id(obj)
                if stable_id in self.removed_objects_memory:
                    id_deque = self.removed_objects_memory[stable_id]
                    found_id = None
                    while id_deque:
                        candidate_id = id_deque.popleft()
                        if candidate_id not in used_persistent_ids:
                            found_id = candidate_id
                            break
                    
                    if found_id:
                        obj['id'] = found_id
                        used_persistent_ids.add(found_id)
                        changes.append(f"- REAPPEARED: Object {found_id.replace('obj_', 'id_')} (ID {stable_id}) reappeared at {obj['position']}.")
                    else:
                        unmatched_new.append(obj)

                    if not id_deque:
                        del self.removed_objects_memory[stable_id]
                else:
                    unmatched_new.append(obj)
            
            # Step 2: For the remainder, check for fuzzy matches (TRANSFORMED).
            still_unmatched = []
            for obj in unmatched_new:
                old_match_id, changed_properties = self._find_single_property_change_match(obj)
                if old_match_id and old_match_id in self.removed_objects_memory:
                    id_deque = self.removed_objects_memory[old_match_id]
                    found_id = None
                    while id_deque:
                        candidate_id = id_deque.popleft()
                        if candidate_id not in used_persistent_ids:
                            found_id = candidate_id
                            break
                    
                    if found_id:
                        obj['id'] = found_id
                        used_persistent_ids.add(found_id)
                        prop_name, (old_val, new_val) = list(changed_properties.items())[0]
                        changes.append(f"- REAPPEARED & TRANSFORMED: Object {found_id.replace('obj_', 'id_')} reappeared, changing {prop_name} from {old_val} to {new_val}, now at {obj['position']}.")
                    else:
                        still_unmatched.append(obj)

                    if not id_deque:
                        del self.removed_objects_memory[old_match_id]
                else:
                    still_unmatched.append(obj)

            # Step 3: Anything left is truly NEW.
            for obj in still_unmatched:
                self.object_id_counter += 1
                new_id = f'obj_{self.object_id_counter}'
                obj['id'] = new_id
                stable_id = self._get_stable_id(obj)
                changes.append(f"- NEW: Object {new_id.replace('obj_', 'id_')} (ID {stable_id}) appeared at {obj['position']}.")

        # --- Final Counter Synchronization ---
        # After all IDs are assigned, ensure the global counter tracks the
        # highest ID ever used to prevent any future collisions.
        max_current_id = 0
        for obj in new_summary:
            try:
                id_num = int(obj['id'].replace('obj_', ''))
                if id_num > max_current_id:
                    max_current_id = id_num
            except (ValueError, AttributeError):
                continue
        
        # The counter should only ever increase to match the highest ID seen.
        if self.object_id_counter < max_current_id:
            self.object_id_counter = max_current_id

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
            # Apply a small base penalty for doing nothing.
            reward -= 5
            # Check the action's lifetime history.
            history = self.action_history.get(learning_key)
            if history and history['successes'] == 0:
                # This action has a history of ONLY ever failing. Punish it exponentially.
                failed_attempts = history['attempts']
                exponential_penalty = (10 ** failed_attempts) - 1
                reward -= exponential_penalty

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
        prev_action_name, prev_target_id = self.last_action_context
        # Find the object in the previous summary by its persistent ID
        prev_object = next((obj for obj in self.last_object_summary if obj['id'] == prev_target_id), None)
        prev_move_mock = {'type': GameAction[prev_action_name], 'object': prev_object}
        
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

        # Update the lifetime action history
        history = self.action_history.setdefault(learning_key, {'attempts': 0, 'successes': 0})
        history['attempts'] += 1
        if changes:
            history['successes'] += 1

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
        action_name = self._get_learning_key(action_template.name, target_object['id'] if target_object else None)

        # --- Action-Specific Bias Feature ---
        features[f'bias_for_{action_name}'] = 1.0

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

        # --- Curiosity & Knowledge Features ---
        target_id_for_key = target_object['id'] if target_object else None
        base_action_key = self._get_learning_key(action_template.name, target_id_for_key)
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
            features[f'rule_confidence_for_{action_name}'] = hypothesis['confidence']
            features[f'is_novel_action_for_{action_name}'] = 0.0
        else:
            features[f'rule_confidence_for_{action_name}'] = 0.0
            features[f'is_novel_action_for_{action_name}'] = 1.0
        return features