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
        self.current_level_id_map = {}
        self.last_diag_adjacencies = {}
        self.last_diag_alignments = {}
        self.transition_history = []
        self.novelty_ratio_history = []
        self.current_state_id = None
        self.click_failure_counts = {}
        self.object_blacklist = set()
        self.state_counter = 0
        self.action_to_state_map = {}
        self.novel_state_details = {}
        self.boring_state_details = []
        self.actions_from_state = {}
        self.state_transition_map = {}

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

        # If this is the first scan (last summary is empty), print the full summary.
        if not self.last_object_summary or self.is_new_level:
            self.click_failure_counts = {}
            self.object_blacklist = set()
            self.level_state_history = []
            self.is_new_level = False
            self.level_milestones = []
            self.seen_event_types_in_level = set()
            self.last_alignments = {}
            self.boring_state_details = []
            self.state_transition_map = {}

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
            self.current_level_id_map = id_map
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

            current_relationships, current_adjacencies, current_diag_adjacencies, current_match_groups = self._analyze_relationships(current_summary)
            current_alignments = self._analyze_alignments(current_summary)
            current_diag_alignments = self._analyze_diagonal_alignments(current_summary)
            current_conjunctions = self._analyze_conjunctions(current_relationships, current_alignments)

            print("--- Initial Frame Summary ---")
            if not current_summary:
                print("No objects found.")
            
            # Create an inverse map for logging purposes.
            new_to_old_id_map = {v: k for k, v in id_map.items()}
            self._print_full_summary(current_summary, new_to_old_id_map)

            if current_relationships:
                print("\n--- Relationship Analysis ---")

                def format_align_ids(id_set):
                        id_list = sorted(list(id_set))
                        if len(id_list) < 2: return f"Object {id_list[0]}"
                        if len(id_list) == 2: return f"Objects {id_list[0]} and {id_list[1]}"
                        return "Objects " + ", ".join(map(str, id_list[:-1])) + f", and {id_list[-1]}"
                
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

                if current_diag_alignments:
                    print("\n--- Initial Diagonal Alignment Analysis ---")
                    for align_type, groups in sorted(current_diag_alignments.items()):
                        for line_idx, ids in enumerate(groups):
                            print(f"- '{align_type}' Alignment (Line {line_idx + 1}): {format_align_ids(ids)}")

                if current_diag_adjacencies:
                    print("\n--- Initial Diagonal Adjacency (TR, BR, BL, TL) ---")
                    for obj_id, contacts in sorted(current_diag_adjacencies.items(), key=lambda item: int(item[0].split('_')[1])):
                        contact_ids = [c.replace('obj_', '') if 'obj_' in c else c for c in contacts]
                        contact_tuple_str = ", ".join(contact_ids)
                        clean_obj_id = obj_id.replace('obj_', 'id_')
                        print(f"- Object {clean_obj_id} ({contact_tuple_str})")

                if current_alignments:
                    print("\n--- Initial Alignment Analysis ---")   
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

            current_relationships, current_adjacencies, current_diag_adjacencies, current_match_groups = self._analyze_relationships(current_summary)
            current_alignments = self._analyze_alignments(current_summary)
            current_diag_alignments = self._analyze_diagonal_alignments(current_summary)
            current_conjunctions = self._analyze_conjunctions(current_relationships, current_alignments)

            # --- LEVEL CHANGE DETECTION & HANDLING ---
            current_score = latest_frame.score
            if current_score > self.last_score:
                print(f"\n--- LEVEL CHANGE DETECTED (Score increased from {self.last_score} to {current_score}) ---")
                
                # Construct the current (winning) context from the live variables
                winning_context = {
                    'summary': current_summary,
                    'rels': current_relationships,
                    'adj': current_adjacencies,
                    'match': current_match_groups,
                    'align': current_alignments,
                    'events': changes
                }
                
                # Create a temporary history list that includes the final winning state for analysis
                history_for_analysis = self.level_state_history + [winning_context]
                self._analyze_win_condition(history_for_analysis, self.level_milestones, self.current_level_id_map)

                self.novelty_ratio_history = []
                self.state_counter = 0
                self.action_to_state_map = {}
                self.novel_state_details = {}
                self.actions_from_state = {}

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
            self._log_diag_adjacency_changes(self.last_diag_adjacencies, current_diag_adjacencies)
            self._log_match_type_changes(self.last_match_groups, current_match_groups)
            self._log_alignment_changes(self.last_diag_alignments, current_diag_alignments, is_diagonal=True)

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
                    
                    # --- Handle successful click on a blacklisted object ---
                    if prev_action_name == 'ACTION6' and prev_target_id:
                        if prev_target_id in self.object_blacklist:
                            print(f"Object {prev_target_id.replace('obj_', 'id_')} caused a change. Removing from blacklist.")
                            self.object_blacklist.remove(prev_target_id)
                            # Reset the failure count upon success
                            self.click_failure_counts[prev_target_id] = 0

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
                        print(f"Logging Milestone:")

                else:
                    # --- Handle click that caused no change ---
                    if prev_action_name == 'ACTION6' and prev_target_id:
                        failure_count = self.click_failure_counts.get(prev_target_id, 0) + 1
                        self.click_failure_counts[prev_target_id] = failure_count
                        self.object_blacklist.add(prev_target_id)
                        
                        failure_str = f"{failure_count} times" if failure_count > 1 else "for the first time"
                        print(f"Clicking object {prev_target_id.replace('obj_', 'id_')} caused no change ({failure_str}). Blacklisting.")

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
                        'adj': self.last_adjacencies,
                        'diag_adj': self.last_diag_adjacencies,
                        'diag_align': self.last_diag_alignments
                    }
                    self.success_contexts.setdefault(learning_key, []).append(success_context)

                elif is_failure_case:
                    # Blacklist the action, regardless of its history.
                    self.failed_action_blacklist.add(learning_key)
                    print(f"Action {learning_key} has been blacklisted until a success occurs.")
                    
                    print(f"\n--- Failure Detected for Action {learning_key} ---")
                    # Store the context of this failure before analyzing
                    failure_context = {
                        'summary': prev_summary,
                        'rels': self.last_relationships,
                        'adj': self.last_adjacencies,
                        'diag_adj': self.last_diag_adjacencies,
                        'diag_align': self.last_diag_alignments
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
        self.last_diag_adjacencies = current_diag_adjacencies
        self.last_relationships = current_relationships
        self.last_alignments = current_alignments
        self.last_diag_alignments = current_diag_alignments
        self.last_match_groups = current_match_groups

        # This is the REAL list of actions for this specific game on this turn.
        game_specific_actions = sorted(latest_frame.available_actions, key=lambda a: a.name)

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

        # --- Prepare for Learning & Rule Analysis ---
        current_state_key = self._get_state_key(current_summary)

        # --- Pre-calculate Unmet Goals for the current state ---
        unmet_abstract_goals = set()
        if self.win_condition_hypotheses:
            required_abstract_patterns = {
                (rule['abstract_pattern']['pattern_type'], rule['abstract_pattern']['sub_type'])
                for rule in self.win_condition_hypotheses
            }
            current_context = {
                'summary': current_summary, 'rels': current_relationships, 'adj': current_adjacencies,
                'align': current_alignments, 'match': current_match_groups, 'events': changes
            }
            patterns_already_true = self._extract_patterns_from_context(current_context)
            true_abstract_patterns = {(p['pattern_type'], p['sub_type']) for p in patterns_already_true}
            unmet_abstract_goals = required_abstract_patterns - true_abstract_patterns

        # --- RL: Score and select the best action ---
        best_moves = []
        best_score = -float('inf')

        if not possible_moves:
            print("Warning: No moves were generated. Falling back to the first available game action.")
            # The game_specific_actions list is already sorted, so this is deterministic.
            if game_specific_actions:
                possible_moves.append({'type': game_specific_actions[0], 'object': None})
            else:
                # Absolute last resort if the environment provides no actions
                possible_moves.append({'type': GameAction.ACTION1, 'object': None})

        all_scores_debug = []
        for move in possible_moves:
            action_template = move['type']
            target_object = move['object']
            target_id = target_object['id'] if target_object else None
            action_key = self._get_learning_key(action_template.name, target_id)

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

            score = q_value + exploration_bonus

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
            
            # --- Object Blacklist Penalty ---
            if target_object and target_object['id'] in self.object_blacklist:
                failure_count = self.click_failure_counts.get(target_object['id'], 1)
                object_blacklist_penalty = -5000 - (((failure_count - 1) * 100) ** 2)
                failure_penalty += object_blacklist_penalty
                obj_id_str = target_object['id'].replace('obj_', 'id_')
                print(f"Heuristic: Object {obj_id_str} is blacklisted (failed {failure_count} time(s)). Applying penalty of {object_blacklist_penalty}.")

            score += failure_penalty

            # --- Penalty for Repeating Actions in a Known State ---
            if self.current_state_id is not None:
                previously_tried_actions = self.actions_from_state.get(self.current_state_id, set())
                if action_key in previously_tried_actions:
                    score -= 100.0 # Heavy penalty for repeating an action from a known state.
                    print(f"Heuristic: Action {action_key} has already been tried from State {self.current_state_id}. Applying penalty.")

            # --- Open-Ended State Bonus (V2) ---
            # Bonus for actions that lead to states with high exploration potential.
            open_ended_bonus = 0.0
            predicted_state_id = self.action_to_state_map.get(action_key)
            if predicted_state_id and predicted_state_id != 'boring':
                predicted_state_details = self.novel_state_details.get(predicted_state_id)
                if predicted_state_details:
                    # Calculate the total number of actions that were available in that state.
                    state_summary = predicted_state_details.get('summary', [])
                    state_actions = predicted_state_details.get('available_actions', [])
                    
                    num_objects = len(state_summary)
                    num_non_clicks = len([a for a in state_actions if a.name != 'ACTION6'])
                    click_available = any(a.name == 'ACTION6' for a in state_actions)
                    num_clicks = num_objects if click_available else 0
                    total_actions_in_next_state = num_clicks + num_non_clicks
                    
                    # Find how many actions have already been tried from that state.
                    actions_tried_from_next_state = self.actions_from_state.get(predicted_state_id, set())
                    num_tried = len(actions_tried_from_next_state)

                    # The bonus is proportional to the number of UNTRIED actions.
                    num_untried = total_actions_in_next_state - num_tried
                    if num_untried > 0:
                        open_ended_bonus = num_untried * 10.0
                    
                    if open_ended_bonus > 0:
                        print(f"Heuristic: Action {action_key} leads to State {predicted_state_id}, which has {num_untried} untried actions. Adding bonus of {open_ended_bonus:.2f}.")
            
            score += open_ended_bonus

            # Pass the pre-calculated unmet goals to the bonus function
            goal_bonus = self._calculate_goal_bonus(move, current_summary, unmet_abstract_goals)
            score += goal_bonus

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

            # Create a user-friendly name for the debug log
            if target_object:
                obj_id = target_object['id'].replace('obj_', 'id_')
                debug_name = f"CLICK on {obj_id}"
            else:
                debug_name = action_template.name
            all_scores_debug.append(f"{debug_name} (Score: {score:.2f})")
        
        # --- Fallback if all available actions were blacklisted ---
        best_move = None
        if best_moves:
            if len(best_moves) > 1:
                # Deterministic Tie-Breaking:
                # Sort by a stable identifier. Clicks are sorted by their target object ID.
                # Non-clicks are already in a deterministic order, but we can sort by name for safety.
                print(f"Tie detected between {len(best_moves)} actions. Applying deterministic sort.")
                best_moves.sort(key=lambda m: (m['object']['id'] if m['object'] else 'non_click', m['type'].name))
            best_move = best_moves[0]
        
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
            'conj': current_conjunctions,
            'events': changes 
        }
        if self.last_action_context:
             # This is a simplification; a full implementation would pass the `changes` variable.
             # For now, this structure will help us in the next step.
             current_context['last_action_changes'] = []
        
        self.level_state_history.append(current_context)

        # --- Record Action Taken From Current State ---
        if self.current_state_id is not None and self.last_action_context:
            action_name, target_id = self.last_action_context
            action_key = self._get_learning_key(action_name, target_id)
            
            # Add the action to the set of actions taken from this state
            known_actions = self.actions_from_state.setdefault(self.current_state_id, set())
            known_actions.add(action_key)
            
            # Log the full set of known actions for this state
            actions_list_str = ", ".join(sorted(list(known_actions)))
            print(f"Memory Update: From State {self.current_state_id}, known actions are: [{actions_list_str}].")

            # --- Calculate and Log State Exploration Percentage ---
            num_objects = len(current_summary)
            num_non_clicks = len([a for a in game_specific_actions if a.name != 'ACTION6'])

            # Only count objects as potential actions if a CLICK action is actually available.
            click_action_is_available = any(a.name == 'ACTION6' for a in game_specific_actions)
            num_clickable_actions = num_objects if click_action_is_available else 0
            
            total_available_actions = num_clickable_actions + num_non_clicks

            if total_available_actions > 0:
                num_actions_taken = len(known_actions)
                exploration_percentage = (num_actions_taken / total_available_actions) * 100
                print(f"Memory Update: State {self.current_state_id} is {exploration_percentage:.1f}% explored ({num_actions_taken}/{total_available_actions} actions taken).")

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
    
    def _parse_changes_for_state_memory(self, changes: list[str]) -> list[dict]:
        """Parses change logs into a structured list for state memory."""
        structured_outcomes = []
        for log in changes:
            try:
                # --- Common parsing for all change types ---
                change_type_raw, details = log.replace('- ', '', 1).split(': ', 1)
                
                if 'id_' not in details:
                    continue

                id_details = details.split('id_')[1]
                obj_id_num = id_details.split()[0].rstrip('):.')
                obj_id = f"id_{obj_id_num}"
                
                outcome = {'object_id': obj_id}

                # --- Type-specific parsing ---
                if 'MOVED' in change_type_raw:
                    outcome['type'] = 'move'
                    end_pos_str = details.split(' to ')[1].replace('.', '')
                    outcome['end_state'] = {'position': ast.literal_eval(end_pos_str)}
                    structured_outcomes.append(outcome)
                
                elif 'RECOLORED' in change_type_raw:
                    outcome['type'] = 'recolor'
                    end_color_str = details.split(' to ')[1].replace('.', '')
                    outcome['end_state'] = {'color': int(end_color_str)}
                    structured_outcomes.append(outcome)

                elif 'SHAPE_CHANGED' in change_type_raw:
                    outcome['type'] = 'shape_change'
                    if ' -> ' in details:
                        end_fp_str = details.split(' -> ')[1].replace(').', '')
                    else:
                        end_fp_str = details.split('fingerprint: ')[1].replace(').','')
                    outcome['end_state'] = {'fingerprint': int(end_fp_str)}
                    structured_outcomes.append(outcome)

                elif 'REAPPEARED' in change_type_raw:
                    outcome['type'] = 'reappear'
                    end_pos_str = details.split(' at ')[1].replace('.', '')
                    outcome['end_state'] = {'position': ast.literal_eval(end_pos_str)}
                    structured_outcomes.append(outcome)

                elif 'TRANSFORM' in change_type_raw:
                    outcome['type'] = 'transform'
                    end_state = {}
                    if 'fingerprint:' in details and ' -> ' in details:
                        end_fp_str = details.split('fingerprint: ')[1].split(' -> ')[1].split(')')[0]
                        end_state['fingerprint'] = int(end_fp_str)
                    if 'color (from ' in details and ' to ' in details:
                        end_color_str = details.split(' to ')[1].split(')')[0]
                        end_state['color'] = int(end_color_str)
                    outcome['end_state'] = end_state
                    structured_outcomes.append(outcome)

                elif 'GROWTH' in change_type_raw or 'SHRINK' in change_type_raw:
                    outcome['type'] = 'growth' if 'GROWTH' in change_type_raw else 'shrink'
                    end_pos_str = details.split('now at ')[1].replace('.', '')
                    outcome['end_state'] = {'position': ast.literal_eval(end_pos_str)}
                    structured_outcomes.append(outcome)

            except (IndexError, ValueError, SyntaxError):
                continue
                
        return structured_outcomes

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

    def _analyze_win_condition(self, level_history: list[dict], milestone_indices: list[int], id_map: dict = None):
        """
        Analyzes the winning state to create and refine "Role-Based" hypotheses.
        A rule consists of an abstract pattern, roles for objects, and group properties.
        """
        print("\n--- Win Condition Analysis (V5 - Role-Based) ---")
        if not level_history: return
        if id_map is None: id_map = {}

        winning_context = level_history[-1]
        
        if not self.win_condition_hypotheses:
            # --- CREATION MODE (After Level 1) ---
            print("No Master Recipe found. Creating initial Role-Based rules...")
            
            # 1. Use the Uniqueness Filter to find the most significant patterns.
            win_patterns = self._extract_patterns_from_context(winning_context)
            historical_contexts = level_history[:-1]
            seen_patterns_set = set()
            for context in historical_contexts:
                historical_patterns = self._extract_patterns_from_context(context)
                for p in historical_patterns:
                    seen_patterns_set.add(str(p)) # Use string for hashable representation
            
            unique_win_patterns = [p for p in win_patterns if str(p) not in seen_patterns_set]
            print(f"Found {len(win_patterns)} patterns in win state; {len(unique_win_patterns)} are unique.")

            # 2. For each unique pattern, create a detailed "Role-Based" rule.
            new_rules = []
            summary_map = {int(obj['id'].replace('obj_', '')): obj for obj in winning_context.get('summary', [])}

            for i, pattern in enumerate(unique_win_patterns):
                obj_ids = pattern['object_ids']
                
                # 3. Create the roles based on the objects in this first instance.
                roles = {}
                all_props_in_pattern = []
                for role_idx, obj_id in enumerate(sorted(list(obj_ids))):
                    obj = summary_map.get(obj_id)
                    if not obj: continue
                    
                    # The initial "learned properties" for a role are just the object's full properties.
                    initial_props = frozenset({
                        'color': obj['color'], 'shape': obj['fingerprint'], 'size': obj['size']
                    }.items())
                    
                    roles[f'role_{role_idx}'] = {
                        'learned_properties': initial_props,
                        'history': [obj_id] # Start the history for this role
                    }
                    all_props_in_pattern.append(dict(initial_props))

                # 4. Find the initial "group properties" - what's common to all roles.
                group_common_props = {}
                if all_props_in_pattern:
                    first_props = all_props_in_pattern[0]
                    for key in first_props:
                        if all(p.get(key) == first_props[key] for p in all_props_in_pattern):
                            group_common_props[key] = first_props[key]

                new_rules.append({
                    'id': f'rule_{i+1}',
                    'abstract_pattern': {'pattern_type': pattern['pattern_type'], 'sub_type': pattern['sub_type']},
                    'roles': roles,
                    'group_properties': frozenset(group_common_props.items()),
                    'confidence': 1.0
                })
            
            self.win_condition_hypotheses = new_rules

        else:
            # --- REFINING MODE (After Level 2+) ---
            master_recipe_rules = self.win_condition_hypotheses
            print(f"\n--- Refining {len(master_recipe_rules)} Master Rules Based on New Evidence ---")

            # Get the set of abstract patterns that were unique to this new win.
            new_win_patterns = self._extract_patterns_from_context(winning_context)
            historical_contexts = level_history[:-1]
            seen_patterns_set = {str(p) for c in historical_contexts for p in self._extract_patterns_from_context(c)}
            unique_new_patterns = [p for p in new_win_patterns if str(p) not in seen_patterns_set]
            
            # Create a simple set of abstract types for easy lookup, e.g., {('alignment', 'bottom_y'), ...}
            unique_new_abstract_patterns = {
                (p['pattern_type'], p['sub_type']) for p in unique_new_patterns
            }

            surviving_rules = []
            for rule in master_recipe_rules:
                # Check if the rule's abstract pattern was found in the new level's unique events
                rule_abstract = (rule['abstract_pattern']['pattern_type'], rule['abstract_pattern']['sub_type'])
                
                if rule_abstract in unique_new_abstract_patterns:
                    # The rule is consistent. Reinforce its confidence.
                    rule['confidence'] = min(1.0, rule['confidence'] + 0.1)
                    surviving_rules.append(rule)
                    print(f"- Rule '{rule['id']}' ({rule['abstract_pattern']['sub_type']}) was REINFORCED. Confidence -> {rule['confidence']:.0%}")
                else:
                    # The rule is inconsistent with this win. Delete it immediately.
                    print(f"- Rule '{rule['id']}' ({rule['abstract_pattern']['sub_type']}) was INCONSISTENT and has been DELETED.")
            
            self.win_condition_hypotheses = surviving_rules
            
        # --- Final Report ---
        if self.win_condition_hypotheses:
            print(f"\n--- Current Master Recipe ({len(self.win_condition_hypotheses)} rules) ---")
            sorted_rules = sorted(self.win_condition_hypotheses, key=lambda x: x['confidence'], reverse=True)

            for rule in sorted_rules[:10]: # Log top 10 rules
                pattern = rule['abstract_pattern']
                desc = f"A '{pattern['sub_type']}' pattern"
                
                group_props_str = ", ".join([f"{k}:{v}" for k, v in rule['group_properties']])
                if not group_props_str: group_props_str = "None"

                print(f"\n- {rule['id']} [Conf: {rule['confidence']:.0%}]: Requires {desc}")
                print(f"  - Group Property Consistency: [{group_props_str}]")
                print(f"  - Role Analysis:")
                
                for role_id, role_data in sorted(rule['roles'].items()):
                    props_str = ", ".join([f"{k}:{v}" for k, v in role_data['learned_properties']])
                    history_str = ", ".join(map(str, role_data['history']))
                    print(f"    - {role_id}: Consistent properties [{props_str}]. Seen as objects [{history_str}]")

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

    def _calculate_goal_bonus(self, move: dict, current_summary: list[dict], unmet_goals: set) -> float:
        """
        Performs a "one-step lookahead" simulation. It predicts the next state based on
        the given move, analyzes that hypothetical state, and awards a bonus if it
        is predicted to satisfy an unmet goal from the Master Recipe.
        """
        if not unmet_goals or not self.rule_hypotheses:
            return 0.0

        action_template = move['type']
        target_object = move['object']
        
        # --- Step 1: Find all high-confidence rules that apply to this specific action ---
        applicable_rules = []
        for rule_key, hypothesis in self.rule_hypotheses.items():
            rule_base_key, rule_obj_id, rule_obj_state = rule_key

            if hypothesis.get('confidence', 0.0) < 0.9: # Use a high threshold for planning
                continue

            is_match = False
            if target_object: # This is a targeted click action
                # The rule must match the action, the specific object clicked, and that object's current state.
                base_action_key = self._get_learning_key(action_template.name, target_object['id'])
                current_object_state = (self._get_stable_id(target_object), target_object['position'])
                if rule_base_key == base_action_key and rule_obj_id == target_object['id'] and rule_obj_state == current_object_state:
                    is_match = True
            else: # This is a global action
                # The rule must match the action name. The `rule_obj_id` tells us which object will be affected.
                base_action_key = self._get_learning_key(action_template.name, None)
                if rule_base_key == base_action_key:
                    is_match = True
            
            if is_match:
                # Store the key along with the rule for easier access later
                hypothesis['key'] = rule_key
                applicable_rules.append(hypothesis)

        if not applicable_rules:
            return 0.0 # No reliable predictions for this action exist.

        # --- Step 2: Build a Hypothetical Future State ---
        future_summary = copy.deepcopy(current_summary)
        future_summary_map = {obj['id']: obj for obj in future_summary}

        for rule in applicable_rules:
            # The rule_key contains the ID of the object that will be affected.
            affected_obj_id = rule['key'][1] 
            if affected_obj_id not in future_summary_map:
                continue
            
            obj_to_change = future_summary_map[affected_obj_id]
            
            for event in rule.get('rules', []):
                # Apply predictable state changes. For now, this handles MOVED and RECOLORED.
                if event.get('type') == 'MOVED' and 'vector' in event:
                    vy, vx = event['vector']
                    old_pos = obj_to_change['position']
                    obj_to_change['position'] = (old_pos[0] + vy, old_pos[1] + vx)
                
                elif event.get('type') == 'RECOLORED' and 'to_color' in event:
                    obj_to_change['color'] = event['to_color']
        
        # --- Step 3: Analyze the Hypothetical Future State ---
        future_rels, future_adj, _, __ = self._analyze_relationships(future_summary)
        future_aligns = self._analyze_alignments(future_summary)
        future_context = {
            'summary': future_summary, 'rels': future_rels, 'adj': future_adj,
            'align': future_aligns, 'match': {}, 'events': []
        }
        predicted_patterns = self._extract_patterns_from_context(future_context)
        predicted_abstract_patterns = {(p['pattern_type'], p['sub_type']) for p in predicted_patterns}

        # --- Step 4: Award Bonus if the Future State Achieves an Unmet Goal ---
        helpful_predictions = predicted_abstract_patterns & unmet_goals
        
        if helpful_predictions:
            bonus = 75.0 * len(helpful_predictions) # Increased bonus for high-confidence planning
            summary = sorted([f"{p[1]}" for p in helpful_predictions])
            obj_id_str = target_object['id'].replace('obj_', 'id_') if target_object else 'GLOBAL'
            action_name = self._get_learning_key(action_template.name, target_object['id'] if target_object else None)
            
            print(f"Goal-Seeking Bonus: Action {action_name} on {obj_id_str} is predicted to achieve unmet goals: {summary}. Adding {bonus:.2f} bonus.")
            return bonus
        
        return 0.0
    
    def _extract_patterns_from_context(self, context: dict) -> list[dict]:
        """
        Scans a single context and returns a list of multi-layered patterns,
        each including abstract, property, and identity information.
        """
        patterns = []
        summary = context.get('summary', [])
        if not summary: return []
        
        summary_map = {int(obj['id'].replace('obj_', '')): obj for obj in summary}

        def get_common_properties(obj_ids: set) -> dict:
            """Finds the shared properties among a set of object IDs."""
            if not obj_ids: return {}
            
            first_obj = summary_map.get(list(obj_ids)[0])
            if not first_obj: return {}

            common = {
                'color': first_obj['color'],
                'shape': first_obj['fingerprint'],
                'size': first_obj['size']
            }
            
            for obj_id in list(obj_ids)[1:]:
                obj = summary_map.get(obj_id)
                if not obj: return {}
                if obj['color'] != common.get('color'): common.pop('color', None)
                if obj['fingerprint'] != common.get('shape'): common.pop('shape', None)
                if obj['size'] != common.get('size'): common.pop('size', None)
            
            return {k: v for k, v in common.items() if v is not None}

        # --- Process Relationships, Alignments, and Conjunctions ---
        perception_modules = {
            'rels': context.get('rels', {}),
            'aligns': context.get('align', {}),
            'conjs': context.get('conj', {})
        }

        for module_name, module_data in perception_modules.items():
            for group_type, groups in module_data.items():
                for value, obj_ids_int in groups.items():
                    common_props = get_common_properties(obj_ids_int)
                    patterns.append({
                        'pattern_type': module_name,
                        'sub_type': group_type,
                        'object_ids': frozenset(obj_ids_int),
                        'common_properties': frozenset(common_props.items())
                    })
        
        # --- Process Adjacencies ---
        for obj_id_str, contacts in context.get('adj', {}).items():
            # Only generate a pattern if there is at least one contact to analyze
            if not any('obj_' in c for c in contacts):
                continue

            obj_id_int = int(obj_id_str.replace('obj_', ''))
            
            # Create an abstract fingerprint of the neighborhood (e.g., ('na', 'contact', 'na', 'na')).
            # This captures the arrangement without binding to specific object IDs in the subtype.
            abstract_config = tuple(['contact' if 'obj_' in c else 'na' for c in contacts])
            
            contact_ids_int = {int(c.replace('obj_', '')) for c in contacts if 'obj_' in c}
            all_ids = {obj_id_int} | contact_ids_int
            
            patterns.append({
                'pattern_type': 'adjacency',
                'sub_type': abstract_config,
                'object_ids': frozenset(all_ids),
                'common_properties': frozenset(get_common_properties(all_ids).items())
            })

        # --- Process Final Turn Events ---
        for event_str in context.get('events', []):
            event_type = event_str.split(':')[0].replace('- ', '')
            if "Object id_" in event_str:
                obj_id_int = int(event_str.split('id_')[1].split(' ')[0])
                obj_props = get_common_properties({obj_id_int})
                patterns.append({
                    'pattern_type': 'event',
                    'sub_type': event_type,
                    'object_ids': frozenset({obj_id_int}),
                    'common_properties': frozenset(obj_props.items())
                })

        # Return a unique list of patterns
        unique_patterns = {str(p) for p in patterns}
        return [eval(s) for s in unique_patterns]

    def _analyze_relationships(self, object_summary: list[dict]) -> tuple[dict, dict, dict, dict]:
        """Analyzes object relationships and returns a structured dictionary of groups."""
        if not object_summary or len(object_summary) < 2:
            return {}, {}, {}, {}

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
        temp_diag_adj = {}
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

                diagonal_neighbors = {
                    'top_right': (r - 1, c + 1),
                    'bottom_right': (r + 1, c + 1),
                    'bottom_left': (r + 1, c - 1),
                    'top_left': (r - 1, c - 1),
                }
                for direction, coord in diagonal_neighbors.items():
                    if coord in pixel_map:
                        b_id = pixel_map[coord]
                        if a_id != b_id:
                            temp_diag_adj.setdefault(a_id, {}).setdefault(direction, set()).add(b_id)

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

        # Pass 3: Simplify diagonal contacts into their own map
        diag_adjacency_map = {}
        for obj_id, contacts in temp_diag_adj.items():
            # Order: TR, BR, BL, TL
            result = ['na'] * 4 
            
            dir_map = {
                'top_right': 0, 'bottom_right': 1, 'bottom_left': 2, 'top_left': 3
            }
            
            for direction, index in dir_map.items():
                contact_set = contacts.get(direction, set())
                if len(contact_set) == 1:
                    result[index] = contact_set.pop()

            # Only add to the map if there's at least one unique contact
            if any(res != 'na' for res in result):
                diag_adjacency_map[obj_id] = result
        
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
        
        return final_rels, adjacency_map, diag_adjacency_map, match_groups
    
    def _analyze_conjunctions(self, relationships: dict, alignments: dict) -> dict:
        """
        Finds all partial-profile matches between objects. This discovers conjunctions
        of any complexity (2-part, 3-part, etc.) in a single, efficient pass.
        """
        # --- Step 1: Create a "Profile" for Each Object ---
        # A profile is a set of all group signatures an object belongs to.
        object_group_map = {}
        
        # Consolidate all group-based perception modules
        all_modules = {
            **relationships,
            **alignments
        }

        for group_type, groups in all_modules.items():
            for value, obj_ids in groups.items():
                # The group signature is a tuple, e.g., ('Color', 3) or ('bottom_y', 42)
                group_signature = (group_type, value)
                for obj_id in obj_ids:
                    object_group_map.setdefault(obj_id, set()).add(group_signature)
        
        if len(object_group_map) < 2:
            return {}

        # --- Step 2: Compare Every Pair to Find "Common Ground" ---
        # This dictionary will store the results, mapping a common profile to a set of objects.
        # e.g., {frozenset({('Color',3), ('bottom_y',42)}): {1, 5, 7}}
        temp_conjunctions = {}
        obj_ids = sorted(list(object_group_map.keys()))

        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                obj_A_id = obj_ids[i]
                obj_B_id = obj_ids[j]

                profile_A = object_group_map[obj_A_id]
                profile_B = object_group_map[obj_B_id]
                
                # Find the intersection of their profiles
                common_groups = profile_A & profile_B
                
                # We only care about conjunctions of 2 or more properties.
                if len(common_groups) > 1:
                    # The frozenset of common groups is our unique key for this conjunction.
                    conjunction_key = frozenset(common_groups)
                    temp_conjunctions.setdefault(conjunction_key, set()).update({obj_A_id, obj_B_id})

        # --- Step 3: Format the Output for the Rest of the System ---
        final_conjunctions = {}
        for common_profile, obj_ids in temp_conjunctions.items():
            if len(obj_ids) > 1:
                # Create a human-readable name for the conjunction type, e.g., "Color_and_bottom_y"
                type_names = sorted([item[0] for item in common_profile])
                conj_type_name = "_and_".join(type_names)
                
                # Sort the profile by the type name first to ensure a consistent order
                sorted_profile = sorted(list(common_profile))
                value_tuple = tuple([item[1] for item in sorted_profile])

                final_conjunctions.setdefault(conj_type_name, {})[value_tuple] = obj_ids
        
        return final_conjunctions

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
    
    def _analyze_diagonal_alignments(self, object_summary: list[dict]) -> dict:
        """Finds groups of objects aligned on a perfect diagonal (slope of 1 or -1)."""
        if len(object_summary) < 2:
            return {}

        # Pre-calculate center points for all objects
        for obj in object_summary:
            y, x = obj['position']
            h, w = obj['size']
            obj['center_y'] = y + h // 2
            obj['center_x'] = x + w // 2

        coord_map = {(obj['center_y'], obj['center_x']): int(obj['id'].replace('obj_', '')) for obj in object_summary}
        processed_ids = set()
        final_alignments = {
            'top_left_to_bottom_right': [],
            'top_right_to_bottom_left': [],
        }

        # Iterate through every unique pair of objects
        for i in range(len(object_summary)):
            obj_a = object_summary[i]
            id_a = int(obj_a['id'].replace('obj_', ''))
            if id_a in processed_ids:
                continue

            for j in range(i + 1, len(object_summary)):
                obj_b = object_summary[j]
                id_b = int(obj_b['id'].replace('obj_', ''))
                if id_b in processed_ids:
                    continue

                # Calculate slope between the two centers
                y1, x1 = obj_a['center_y'], obj_a['center_x']
                y2, x2 = obj_b['center_y'], obj_b['center_x']
                
                if x2 - x1 == 0: continue # Avoid division by zero for vertical lines
                slope = (y2 - y1) / (x2 - x1)

                if abs(slope) == 1.0:
                    # Found a valid diagonal pair, now "walk" the line to find all members
                    line_members = {id_a, id_b}
                    
                    # Determine the direction (step vector) of the line
                    step_y = 1 if y2 > y1 else -1
                    step_x = 1 if x2 > x1 else -1

                    # Walk forward from the "second" object (obj_b)
                    next_y, next_x = y2 + step_y, x2 + step_x
                    while (next_y, next_x) in coord_map:
                        found_id = coord_map[(next_y, next_x)]
                        line_members.add(found_id)
                        next_y, next_x = next_y + step_y, next_x + step_x

                    # Walk backward from the "first" object (obj_a)
                    prev_y, prev_x = y1 - step_y, x1 - step_x
                    while (prev_y, prev_x) in coord_map:
                        found_id = coord_map[(prev_y, prev_x)]
                        line_members.add(found_id)
                        prev_y, prev_x = prev_y - step_y, prev_x - step_x
                    
                    if len(line_members) > 1:
                        align_type = 'top_left_to_bottom_right' if slope == 1.0 else 'top_right_to_bottom_left'
                        final_alignments[align_type].append(frozenset(line_members))
                        processed_ids.update(line_members)
        
        # Clean up empty alignment types and remove duplicate lines
        final_results = {}
        for align_type, groups in final_alignments.items():
            if groups:
                # Using a set of frozensets ensures all lines are unique
                unique_groups = sorted([sorted(list(g)) for g in set(groups)])
                final_results[align_type] = [set(g) for g in unique_groups]
                
        return final_results

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

    def _log_diag_adjacency_changes(self, old_adj: dict, new_adj: dict):
        """Compares two diagonal adjacency maps and logs the differences."""
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
                
                output_lines.append(f"- Diagonal Adjacency for Object {clean_id}: contacts changed from {old_str} to {new_str}.")
        
        if output_lines:
            print("\n--- Diagonal Adjacency Change Log ---")
            for line in output_lines:
                print(line)
            print()

    def _log_alignment_changes(self, old_aligns: dict, new_aligns: dict, is_diagonal: bool = False):
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
            old_groups = old_aligns.get(align_type, [] if is_diagonal else {})
            new_groups = new_aligns.get(align_type, [] if is_diagonal else {})

            if is_diagonal:
                # Handle list-based format for diagonal alignments
                old_sets = {frozenset(g) for g in old_groups}
                new_sets = {frozenset(g) for g in new_groups}
                
                joined_groups = new_sets - old_sets
                left_groups = old_sets - new_sets
                log_prefix = f"'{align_type}' Diagonal Alignment"

                for group in joined_groups:
                    output_lines.append(f"- {log_prefix}: {format_ids(group).capitalize()} appeared.")
                for group in left_groups:
                    output_lines.append(f"- {log_prefix}: {format_ids(group).capitalize()} disappeared.")

            else:
                # Handle dictionary-based format for cardinal alignments
                all_coords = set(old_groups.keys()) | set(new_groups.keys())
                for coord in all_coords:
                    old_ids = old_groups.get(coord, set())
                    new_ids = new_groups.get(coord, set())

                    if old_ids == new_ids: continue

                    joined = new_ids - old_ids
                    left = old_ids - new_ids
                    log_prefix = f"'{align_type}' Alignment at {coord}"

                    if joined and left:
                        output_lines.append(f"- {log_prefix}: {format_ids(joined).capitalize()} replaced {format_ids(left)}.")
                    elif joined:
                        output_lines.append(f"- {log_prefix}: {format_ids(joined).capitalize()} joined.")
                    elif left:
                        output_lines.append(f"- {log_prefix}: {format_ids(left).capitalize()} left.")

        if output_lines:
            title = "Diagonal Alignment Change Log" if is_diagonal else "Alignment Change Log"
            print(f"\n--- {title} ---")
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

        return sorted(objects, key=lambda o: (o['position'][0], o['position'][1]))
    
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
        for stable_id in sorted(list(movable_ids)):
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

        from_state_id = self.current_state_id

        # 1. Calculate reward
        reward = 0
        # --- Normalized reward for unique state discovery ---
        total_changes = len(changes)
        current_novelty_ratio = (novel_state_count / total_changes) if total_changes > 0 else 0.0

        # Calculate the running average novelty ratio from history.
        average_novelty_ratio = sum(self.novelty_ratio_history) / len(self.novelty_ratio_history) if self.novelty_ratio_history else 0.0
        
        # The performance score is how much better (or worse) this turn's ratio is than the average.
        # This will be a value between -1.0 and 1.0.
        performance_score = current_novelty_ratio - average_novelty_ratio
        
        # --- State Assignment based on Performance ---
        
        # First, parse the raw change logs into a structured format for memory.
        structured_outcomes = self._parse_changes_for_state_memory(changes)

        if performance_score >= 0:
            self.state_counter += 1
            self.action_to_state_map[learning_key] = self.state_counter
            print(f"Outcome was novel. Assigning new state: State {self.state_counter}")
            self.current_state_id = self.state_counter
            
            # Record the structured details and full context of this novel state
            self.novel_state_details[self.state_counter] = {
                'transitions': structured_outcomes,
                'summary': new_summary,
                'available_actions': latest_frame.available_actions
            }
            print(f"  - Stored snapshot for State {self.state_counter} (summary, actions, transitions).")

        else:
            self.action_to_state_map[learning_key] = 'boring'
            print("Outcome was boring. No new state assigned.")

            # Step 1: Record the structured details of this boring outcome to the log.
            if structured_outcomes:
                self.boring_state_details.append({'transitions': structured_outcomes})
                print(f"  - Logged boring outcome with {len(structured_outcomes)} structured transitions.")

            # Step 2: Now, use this outcome to find the most similar novel state in memory.
            if not structured_outcomes or not self.novel_state_details:
                if structured_outcomes: # Only print if there was something to compare
                    print("  - No novel state history exists to compare against.")
            else:
                # Helper function to make nested dictionaries and lists hashable for comparison.
                def make_hashable(obj):
                    if isinstance(obj, dict):
                        return frozenset((k, make_hashable(v)) for k, v in sorted(obj.items()))
                    if isinstance(obj, list):
                        return tuple(make_hashable(v) for v in obj)
                    return obj

                boring_transitions_set = {make_hashable(t) for t in structured_outcomes}
                
                best_match_state_id = None
                max_match_score = 0 # Start at 0, as we only care about positive matches

                for state_id, details in self.novel_state_details.items():
                    novel_transitions = details.get('transitions', [])
                    if not novel_transitions:
                        continue

                    novel_transitions_set = {make_hashable(t) for t in novel_transitions}
                    
                    # The score is the number of common transitions (set intersection).
                    match_score = len(boring_transitions_set.intersection(novel_transitions_set))

                    if match_score > max_match_score:
                        max_match_score = match_score
                        best_match_state_id = state_id
                
                if best_match_state_id is not None:
                    print(f"  - Match Found: This outcome is most similar to novel State {best_match_state_id} (Score: {max_match_score}).")
                    self.current_state_id = best_match_state_id
                    self.action_to_state_map[learning_key] = best_match_state_id
                    print(f"  - Agent context is now considered to be State {self.current_state_id}.")
                    
                    # Announce the actions that have been tried from this state before
                    known_actions = self.actions_from_state.get(self.current_state_id, set())
                    if known_actions:
                        actions_list_str = ", ".join(sorted(list(known_actions)))
                        print(f"  - Previously tried actions from this state: [{actions_list_str}].")
                    else:
                        print("  - No actions have been taken from this state before.")
                else:
                    print("  - No matching novel state found in memory.")

        to_state_id = self.current_state_id # Capture state after the action.

        # --- Record and report all state transitions ---
        if from_state_id is not None and to_state_id is not None:
            # Always update the map with the latest observation for this action
            self.state_transition_map.setdefault(from_state_id, {})[learning_key] = to_state_id

            # Get the full dictionary of transitions from the origin state
            all_known_transitions = self.state_transition_map[from_state_id]
            
            # Format the list for printing
            transitions_str_parts = []
            for action, dest_state in sorted(all_known_transitions.items()):
                 transitions_str_parts.append(f"{action} -> State {dest_state}")
            
            full_transitions_str = ", ".join(transitions_str_parts)
            print(f"Memory Update: From State {from_state_id}, known transitions are: [{full_transitions_str}].")
        
        # Since the score is now a ratio, the reward multiplier needs to be larger to have an impact.
        reward += performance_score * 50

        print(f"Novelty Analysis: Current ratio is {current_novelty_ratio:.2%} ({novel_state_count}/{total_changes} unique). Average is {average_novelty_ratio:.2%}. Performance score: {performance_score:.2f}.")

        # Now, update the history for the next turn's calculation.
        if total_changes > 0:
            self.novelty_ratio_history.append(current_novelty_ratio)

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

        # --- Effect Pattern Novelty Reward (for discovering new game mechanics) ---
        if changes:
            change_types = sorted([log.split(':')[0].replace('- ', '') for log in changes])
            effect_pattern_key = tuple(change_types)
            pattern_count_in_history = self.recent_effect_patterns.count(effect_pattern_key)
            
            if pattern_count_in_history == 0:
                reward += 20 # Bonus for a new type of outcome.
            else:
                reward -= pattern_count_in_history * 5 # Penalty for repeating old outcomes.
            
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