from .agent import Agent, FrameData
from .structs import GameAction, GameState
from collections import deque
import copy
import ast
import statistics

class ObmlAgi3Agent(Agent):
    """
    Placeholder Agent for OBML AGI 3. Replace with actual implementation.
    """
    def __init__(self, params: dict = None, **kwargs):
        super().__init__(**kwargs)

        self.object_id_counter = 0
        self.removed_objects_memory = {}
        self.last_object_summary = []
        self.last_relationships = {}
        self.last_adjacencies = {}
        self.last_diag_adjacencies = {}
        self.last_alignments = {}
        self.last_diag_alignments = {}
        self.last_match_groups = {}
        self.is_new_level = True
        self.final_summary_before_level_change = None
        self.current_level_id_map = {}
        self.last_action_context = None  # Stores the action we just took
        self.success_contexts = {}
        self.failure_contexts = {}
        self.failure_patterns = {}
        self.rule_hypotheses = {}
        self.seen_outcomes = set()
        self.level_state_history = []
        self.win_condition_hypotheses = []
        self.actions_printed = False
        self.last_score = 0

        # --- Debug Channels ---
        # Set these to True or False to control the debug output.
        self.debug_channels = {
            'PERCEPTION': False,      # Object finding, relationships, new level setup
            'CHANGES': True,         # All "Change Log" prints
            'STATE_GRAPH': True,     # State understanding
            'HYPOTHESIS': True,      # "Initial Hypotheses", "Refined Hypothesis"
            'FAILURE': False,         # "Failure Analysis", "Failure Detected"
            'WIN_CONDITION': True,   # "LEVEL CHANGE DETECTED", "Win Condition Analysis"
            'ACTION_SCORE': True,    # All scoring prints
            'CONTEXT_DETAILS': False # Keep or remove large prints
        }

    def _reset_agent_memory(self):
        """
        Resets all agent learning and memory to a clean state.
        This is called at the start of a new game.
        """
        self.object_id_counter = 0
        self.removed_objects_memory = {}
        self.last_object_summary = []
        self.last_relationships = {}
        self.last_adjacencies = {}
        self.last_diag_adjacencies = {}
        self.last_alignments = {}
        self.last_diag_alignments = {}
        self.last_match_groups = {}
        self.is_new_level = True
        self.final_summary_before_level_change = None
        self.current_level_id_map = {}
        self.last_action_context = None
        self.success_contexts = {}
        self.failure_contexts = {}
        self.failure_patterns = {}
        self.rule_hypotheses = {}
        self.seen_outcomes = set()
        self.level_state_history = []
        self.win_condition_hypotheses = []
        self.actions_printed = False
        self.last_score = 0

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Analyzes the current frame, compares it to the previous frame,
        and logs all perceived changes.
        """
        # If the game is over or hasn't started, the correct action is to reset.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.object_id_counter = 0
            self.is_new_level = True # Reset for the next game
            self.actions_printed = False # Reset the print flag
            return GameAction.RESET
        
        # --- 1. Perceive Raw Objects ---
        # --- Print available actions once ---
        if latest_frame.available_actions and not self.actions_printed:
            action_names = [action.name for action in latest_frame.available_actions]
            if self.debug_channels['PERCEPTION']:
                print(f"\n--- Discovered game-specific actions: {action_names} ---")
            self.actions_printed = True
        
        # This just finds the blobs of pixels, without persistent IDs
        current_summary = self._perceive_objects(latest_frame)
        changes = []

        # --- 2. Compare Current State to Previous State ---
        if not self.last_object_summary or self.is_new_level:
            # --- FIRST FRAME LOGIC ---
            current_score = latest_frame.score
            
            # --- Perform a full "brain wipe" ---
            self._reset_agent_memory()
            
            if self.debug_channels['PERCEPTION'] and current_score > self.last_score:
                print(f"\n--- Level Cleared (Score: {current_score}): Resetting history. ---")
            
            # Now, set the state for the new level
            self.is_new_level = False
            self.last_score = current_score

            id_map = {}
            new_obj_to_old_id_map = {}
            
            # (Skipping cross-level correlation for now, as it requires
            # self.final_summary_before_level_change, which is not yet set)

            # --- Final Re-Numbering (simple version for first frame) ---
            if self.debug_channels['PERCEPTION']: print("--- Finalizing Frame with Sequential IDs ---")
            self.object_id_counter = 0
            sorted_summary = sorted(current_summary, key=lambda o: (o['position'][0], o['position'][1]))
            for obj in sorted_summary:
                self.object_id_counter += 1
                new_id = f'obj_{self.object_id_counter}'
                obj['id'] = new_id
            current_summary = sorted_summary

            # --- Analyze the fully ID'd summary ---
            (current_relationships, current_adjacencies, current_diag_adjacencies, 
             current_match_groups, current_alignments, current_diag_alignments, 
             current_conjunctions) = self._analyze_relationships(current_summary)

            # --- Print Initial Summary (if debug is on) ---
            if self.debug_channels['PERCEPTION']:
                print("--- Initial Frame Summary ---")
                self._print_full_summary(current_summary)
                
                print("\n--- Initial Relationship Analysis ---")
                def format_id_list_str(id_set):
                    id_list = sorted(list(id_set))
                    if len(id_list) < 2: return f"Object {id_list[0]}"
                    if len(id_list) == 2: return f"Objects {id_list[0]} and {id_list[1]}"
                    return "Objects " + ", ".join(map(str, id_list[:-1])) + f", and {id_list[-1]}"

                for rel_type, groups in sorted(current_relationships.items()):
                    for value, ids in sorted(groups.items()):
                        value_str = f"{value[0]}x{value[1]}" if rel_type == 'Size' else value
                        print(f"- {rel_type} Group ({value_str}): {format_id_list_str(ids)}")
                
                if current_adjacencies:
                    print("\n--- Initial Adjacency Analysis (T,R,B,L) ---")
                    for obj_id, contacts in sorted(current_adjacencies.items(), key=lambda item: int(item[0].split('_')[1])):
                        contact_ids = [c.replace('obj_', '') if 'obj_' in c else c for c in contacts]
                        print(f"- Object {obj_id.replace('obj_', 'id_')} ({', '.join(contact_ids)})")

                if current_diag_adjacencies:
                    print("\n--- Initial Diagonal Adjacency (TR,BR,BL,TL) ---")
                    for obj_id, contacts in sorted(current_diag_adjacencies.items(), key=lambda item: int(item[0].split('_')[1])):
                        contact_ids = [c.replace('obj_', '') if 'obj_' in c else c for c in contacts]
                        print(f"- Object {obj_id.replace('obj_', 'id_')} ({', '.join(contact_ids)})")

                if current_alignments:
                    print("\n--- Initial Alignment Analysis ---")   
                    for align_type, groups in sorted(current_alignments.items()):
                        for coord, ids in sorted(groups.items()):
                            print(f"- '{align_type}' Alignment at {coord}: {format_id_list_str(ids)}")

                if current_diag_alignments:
                    print("\n--- Initial Diagonal Alignment Analysis ---")
                    for align_type, groups in sorted(current_diag_alignments.items()):
                        for line_idx, ids in enumerate(groups):
                            print(f"- '{align_type}' Alignment (Line {line_idx + 1}): {format_id_list_str(ids)}")

                if current_match_groups:
                    print("\n--- Object Match Type Analysis ---")
                    for match_type, groups_dict in current_match_groups.items():
                        label = f"Exact Matches" if match_type == "Exact" else f"Matches (Except {match_type})"
                        print(f"- {label}:")
                        for props, group in groups_dict.items():
                            id_list_str = ", ".join([i.replace('obj_', 'id_') for i in sorted(group, key=lambda x: int(x.split('_')[1]))])
                            print(f"  - Group {props}: {id_list_str}")

        else:
            # --- SUBSEQUENT FRAME LOGIC ---
            prev_summary = self.last_object_summary
            
            # This function compares old and new summaries, assigns persistent IDs,
            # and returns a list of change strings.
            changes, current_summary = self._log_changes(prev_summary, current_summary)

            # Now, analyze the new summary *after* persistent IDs are assigned
            (current_relationships, current_adjacencies, current_diag_adjacencies, 
             current_match_groups, current_alignments, current_diag_alignments, 
             current_conjunctions) = self._analyze_relationships(current_summary)

            # --- Check for score changes (win detection) ---
            current_score = latest_frame.score
            if current_score > self.last_score:
                if self.debug_channels['WIN_CONDITION']:
                    print(f"\n--- LEVEL CHANGE DETECTED (Score increased from {self.last_score} to {current_score}) ---")
                # (Win analysis logic will go here)
                self.is_new_level = True # Signal for the *next* frame
            self.last_score = current_score

            # --- Log all changes found ---
            if changes and self.debug_channels['CHANGES']:
                print("--- Change Log ---")
                for change in changes:
                    print(change)

            self._log_relationship_changes(self.last_relationships, current_relationships)
            self._log_adjacency_changes(self.last_adjacencies, current_adjacencies)
            self._log_diag_adjacency_changes(self.last_diag_adjacencies, current_diag_adjacencies)
            self._log_match_type_changes(self.last_match_groups, current_match_groups)
            self._log_alignment_changes(self.last_alignments, current_alignments, is_diagonal=False)
            self._log_alignment_changes(self.last_diag_alignments, current_diag_alignments, is_diagonal=True)

            # --- Analyze the outcome of the previous action ---
            if self.last_action_context:
                learning_key = self.last_action_context # This is now 'ACTION4' or 'ACTION6_obj_2'
        
                # This is the full context of the *previous* state, when the action was taken.
                prev_context = {
                    'summary': prev_summary,
                    'rels': self.last_relationships,
                    'adj': self.last_adjacencies,
                    'diag_adj': self.last_diag_adjacencies,
                    'align': self.last_alignments,
                    'diag_align': self.last_diag_alignments,
                    'match': self.last_match_groups
                }

                if changes:
                    # --- SUCCESS: Learn rules from the outcome (Unified Logic) ---
                    prev_summary_map = {obj['id']: obj for obj in prev_summary}
                    
                    # Find all objects that *changed*
                    affected_ids = set()
                    for change_str in changes:
                            if "Object id_" in change_str:
                                try:
                                    id_num_str = change_str.split('id_')[1].split()[0]
                                    affected_ids.add(f"obj_{id_num_str}")
                                except IndexError:
                                    continue
                    
                    # 1. Learn rules for all AFFECTED objects
                    for obj_id in affected_ids:
                        if obj_id in prev_summary_map:
                            # The key is (action_key, affected_object_id)
                            # e.g., ('ACTION6_obj_2', 'obj_5') or ('ACTION4', 'obj_5')
                            hypothesis_key = (learning_key, obj_id) 
                            
                            self.success_contexts.setdefault(hypothesis_key, []).append(prev_context) # Log success
                            
                            obj_id_str = obj_id.replace('obj_', 'id_')
                            object_specific_changes = [c for c in changes if f"Object {obj_id_str}" in c]

                            if object_specific_changes:
                                self._analyze_and_report(hypothesis_key, object_specific_changes, prev_context)
                    
                    # 2. Learn "no change" rules for all UNAFFECTED objects
                    all_prev_ids = set(prev_summary_map.keys())
                    unaffected_ids = all_prev_ids - affected_ids
                    
                    for obj_id in unaffected_ids:
                        if obj_id in prev_summary_map: # Check if object still exists
                            hypothesis_key = (learning_key, obj_id) # e.g., ('ACTION6_obj_2', 'obj_1') or ('ACTION4', 'obj_1')
                            
                            # This is now treated as a "failure" (no change)
                            self.failure_contexts.setdefault(hypothesis_key, []).append(prev_context)
                            
                            # Analyze this specific object's failure history
                            successes = self.success_contexts.get(hypothesis_key, [])
                            failures = self.failure_contexts.get(hypothesis_key, [])
                            self._analyze_failures(hypothesis_key, successes, failures, prev_context)
                
                else:
                    # --- FAILURE (No changes occurred) ---
                    if self.debug_channels['FAILURE']:
                        print(f"\n--- Global Failure Detected for Action {learning_key} (No Changes) ---")
                    
                    # We must log a "no change" failure for EVERY object on the screen.
                    for obj in prev_summary:
                        obj_id = obj['id']
                        # The key is (action_key, affected_object_id)
                        hypothesis_key = (learning_key, obj_id) # e.g., ('ACTION6_obj_2', 'obj_1') or ('ACTION4', 'obj_1')
                        
                        self.failure_contexts.setdefault(hypothesis_key, []).append(prev_context)
                        
                        # Analyze this specific object's failure history
                        successes = self.success_contexts.get(hypothesis_key, [])
                        failures = self.failure_contexts.get(hypothesis_key, [])
                        self._analyze_failures(hypothesis_key, successes, failures, prev_context)
                        self._analyze_and_report(hypothesis_key, [], prev_context)
                    
        # --- 3. Update Memory For Next Turn ---
        # This runs every frame, saving the state we just analyzed
        self.last_object_summary = current_summary
        self.last_adjacencies = current_adjacencies
        self.last_diag_adjacencies = current_diag_adjacencies
        self.last_relationships = current_relationships
        self.last_alignments = current_alignments
        self.last_diag_alignments = current_diag_alignments
        self.last_match_groups = current_match_groups

        # --- 4. Choose an Action (Discovery Profiler Logic) ---
        action_to_return = None
        chosen_object = None
        chosen_object_id = None
        
        # Get the full context of the *current* state for prediction
        current_full_context = {
            'summary': current_summary,
            'rels': current_relationships,
            'adj': current_adjacencies,
            'diag_adj': current_diag_adjacencies,
            'align': current_alignments,
            'diag_align': current_diag_alignments,
            'match': current_match_groups
        }

        # Build a list of all possible moves
        all_possible_moves = []
        click_action_template = None
        for action in latest_frame.available_actions:
            if action.name == 'ACTION6':
                click_action_template = action
            else:
                all_possible_moves.append({'template': action, 'object': None})
        if click_action_template and current_summary:
            for obj in current_summary:
                all_possible_moves.append({'template': click_action_template, 'object': obj})
        
        # --- Deterministic Profiling ---
        move_profiles = []
        
        for move in all_possible_moves:
            action_template = move['template']
            target_obj = move['object']
            target_id = target_obj['id'] if target_obj else None
            
            # This is the "base" key for this action, e.g. "ACTION4" or "ACTION6_obj_5"
            base_action_key_str = self._get_learning_key(action_template.name, target_id)
            
            profile = {'unknowns': 0, 'discoveries': 0, 'boring': 0, 'failures': 0}
            predicted_fingerprints_for_this_move = set()
            all_predicted_events_for_move = []

            # --- Unified Profiling Logic ---
            # We profile the effect of this action (Global or Targeted)
            # on ALL objects on the screen.
            for obj in current_summary:
                obj_id = obj['id']
                
                # The hypothesis key is (action_key, affected_object_id)
                # e.g., ('ACTION4', 'obj_1') or ('ACTION6_obj_2', 'obj_1')
                hypothesis_key = (base_action_key_str, obj_id)

                # 1. Check for a predicted per-object FAILURE
                if hypothesis_key in self.failure_patterns:
                    failure_rule = self.failure_patterns[hypothesis_key]
                    if self._context_matches_pattern(current_full_context, failure_rule):
                        profile['failures'] += 1
                        continue # This object will fail, check the next object

                # 2. If no failure, predict the per-object SUCCESS outcome
                predicted_event_list = self._predict_outcome(hypothesis_key, current_full_context)
                
                # --- Convert event list to a hashable fingerprint ---
                predicted_outcome_fingerprint = None
                if predicted_event_list is not None:
                    hashable_events = [tuple(sorted(e.items())) for e in predicted_event_list]
                    predicted_outcome_fingerprint = tuple(sorted(hashable_events))
                    all_predicted_events_for_move.extend(predicted_event_list)
                # --- End conversion ---
                
                # 3. Tally the results
                if predicted_event_list is None:
                    profile['unknowns'] += 1
                elif predicted_outcome_fingerprint == ():
                    profile['failures'] += 1 # Predicted "no change"
                elif predicted_outcome_fingerprint in self.seen_outcomes:
                    profile['boring'] += 1 # Predicted "repetitive change"
                else:
                    profile['discoveries'] += 1 # Predicted "new, novel change"
                    predicted_fingerprints_for_this_move.add(predicted_outcome_fingerprint)

            move_profiles.append((move, profile, predicted_fingerprints_for_this_move, all_predicted_events_for_move))

        # --- NEW: Debug print of all move profiles ---
        if self.debug_channels['ACTION_SCORE']:
            print("\n--- Full Profile List (Before Sort) ---")
            if not move_profiles:
                print("  (No moves to profile)")
            
            # Sort for display purposes only
            sorted_for_print = sorted(move_profiles, key=lambda x: (
                x[1]['unknowns'], 
                x[1]['discoveries'], 
                -x[1]['failures'], 
                x[1]['boring']
            ), reverse=True)

            for i, (move, profile, _, _) in enumerate(sorted_for_print):
                action_name = move['template'].name
                target_name = f" on {move['object']['id']}" if move['object'] else ""
                
                prefix = "  -> " if i == 0 else "     " # Highlight the winner
                
                print(f"{prefix}{action_name}{target_name} -> "
                        f"U:{profile['unknowns']} D:{profile['discoveries']} "
                        f"B:{profile['boring']} F:{profile['failures']}")

        # --- Deterministic Priority-Based Sorting ---
        if move_profiles:
            # Sort by:
            # 1. Most Unknowns (desc)
            # 2. Most Discoveries (desc)
            # 3. Fewest Failures (asc)
            # 4. Most Boring (desc) - (to break ties, prefer action over inaction)
            move_profiles.sort(key=lambda x: (
                x[1]['unknowns'], 
                x[1]['discoveries'], 
                -x[1]['failures'], 
                x[1]['boring']
            ), reverse=True)
            
            # --- NEW: 1-Step Lookahead Tie-Breaker ---
            top_profile_score = (
                move_profiles[0][1]['unknowns'], 
                move_profiles[0][1]['discoveries'], 
                -move_profiles[0][1]['failures']
            )
            
            # Find all moves tied with the best score
            tied_moves = []
            for move_tuple in move_profiles:
                # move_tuple is (move, profile, fingerprints, events)
                move, profile, fingerprints, events = move_tuple
                current_score = (profile['unknowns'], profile['discoveries'], -profile['failures'])
                
                if current_score == top_profile_score:
                    # Only check "boring" moves (U=0, D=0) for lookahead
                    if profile['unknowns'] == 0 and profile['discoveries'] == 0:
                        tied_moves.append(move_tuple)
                    elif not tied_moves: 
                        # This is the first (and best) non-boring move.
                        # No lookahead needed, just pick this one.
                        tied_moves.append(move_tuple)
                        break
                else:
                    # We are past the tied scores
                    break
            
            chosen_move_tuple = None
            if len(tied_moves) > 1:
                # --- Run Lookahead ---
                if self.debug_channels['ACTION_SCORE']: 
                    print(f"\n--- Running 1-Step Lookahead for {len(tied_moves)} Boring Moves ---")
                
                lookahead_scores = []
                for move_tuple in tied_moves:
                    move, profile, fingerprints, events = move_tuple
                    
                    # 1. Simulate the state
                    hypothetical_summary = self._get_hypothetical_summary(current_summary, events)
                    
                    # 2. Profile the *future* state
                    future_profile = self._get_hypothetical_profile(hypothetical_summary, latest_frame.available_actions)
                    
                    if self.debug_channels['ACTION_SCORE']:
                        target_name = f" on {move['object']['id']}" if move['object'] else ""
                        print(f"  - Move {move['template'].name}{target_name} -> Future Profile: "
                              f"U:{future_profile['unknowns']} D:{future_profile['discoveries']} "
                              f"B:{future_profile['boring']} F:{future_profile['failures']}")
                    
                    lookahead_scores.append((move_tuple, future_profile))
                
                # Sort by best *future* profile
                lookahead_scores.sort(key=lambda x: (
                    x[1]['unknowns'], 
                    x[1]['discoveries'], 
                    -x[1]['failures']
                ), reverse=True)
                
                # The best move is the winner of this sort
                chosen_move_tuple = lookahead_scores[0][0]
            
            else:
                # No tie, or only one move, just pick the first one
                chosen_move_tuple = tied_moves[0]
            # --- End Lookahead ---

            # Choose the best move
            chosen_move, best_profile, new_fingerprints_to_add, _best_events = chosen_move_tuple
            
            action_to_return = chosen_move['template']
            chosen_object = chosen_move['object']
            
            # Add any new discoveries to our "seen" list
            self.seen_outcomes.update(new_fingerprints_to_add)

            # --- Logging ---
            action_name = action_to_return.name
            target_name = ""
            if chosen_object:
                chosen_object_id = chosen_object['id']
                pos = chosen_object['position']
                action_to_return.set_data({'x': pos[1], 'y': pos[0]})
                target_name = f" on {chosen_object_id.replace('obj_', 'id_')}"
            
            if self.debug_channels['ACTION_SCORE']:
                print(f"\n--- Discovery Profiler ---")
                print(f"Chose: {action_name}{target_name}")
                print(f"Profile: U:{best_profile['unknowns']} D:{best_profile['discoveries']} B:{best_profile['boring']} F:{best_profile['failures']}")
        
        else:
            action_to_return = GameAction.RESET # Fallback

        # --- Store action for next turn's analysis ---
        learning_key_for_storage = self._get_learning_key(action_to_return.name, chosen_object_id if chosen_object else None)
        self.last_action_context = learning_key_for_storage
        
        # --- Store state for level history ---
        current_context = {
            'summary': current_summary,
            'rels': current_relationships,
            'adj': current_adjacencies,
            'diag_adj': current_diag_adjacencies,
            'align': current_alignments,
            'diag_align': current_diag_alignments,
            'match': current_match_groups,
            'events': changes 
        }
        self.level_state_history.append(current_context)

        return action_to_return

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return False
    
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
    
    def _get_learning_key(self, action_name: str, target_id: str | None) -> str:
        """Generates a unique key for learning, specific to an object ID for CLICK actions."""
        if action_name == 'ACTION6' and target_id:
            return f"{action_name}_{target_id}"
        return action_name

    def _find_common_context(self, contexts: list[dict]) -> dict:
        """
        Finds the common context across a list of attempts.
        This new version checks all perception modules.
        """
        if not contexts:
            return {}

        # Start with the first context as the baseline
        common_context = copy.deepcopy(contexts[0])
        
        # Remove summary, we only care about the analysis
        common_context.pop('summary', None) 

        # Iteratively find the intersection
        for i in range(1, len(contexts)):
            next_context = contexts[i]
            intersection = {} # The intersection of common_context and next_context
            
            # 1. Adjacency and Diagonal Adjacency
            for key in ['adj', 'diag_adj']:
                adj_A = common_context.get(key, {})
                adj_B = next_context.get(key, {})
                common_ids = set(adj_A.keys()) & set(adj_B.keys())
                for obj_id in common_ids:
                    if adj_A[obj_id] == adj_B[obj_id]:
                        intersection.setdefault(key, {})[obj_id] = adj_A[obj_id]

            # 2. Relationship, Alignment, and Match Diffs (dict-of-dicts)
            for key in ['rels', 'align', 'match']:
                rels_A = common_context.get(key, {})
                rels_B = next_context.get(key, {})
                common_rel_types = set(rels_A.keys()) & set(rels_B.keys())
                for rel_type in common_rel_types:
                    groups_A = rels_A[rel_type]
                    groups_B = rels_B[rel_type]
                    common_values = set(groups_A.keys()) & set(groups_B.keys())
                    for value in common_values:
                        if groups_A[value] == groups_B[value]:
                            intersection.setdefault(key, {}).setdefault(rel_type, {})[value] = groups_A[value]

            # 3. Diagonal Alignment Diffs (dict-of-lists)
            key = 'diag_align'
            rels_A = common_context.get(key, {})
            rels_B = next_context.get(key, {})
            common_rel_types = set(rels_A.keys()) & set(rels_B.keys())
            for rel_type in common_rel_types:
                groups_A = rels_A[rel_type]
                groups_B = rels_B[rel_type]
                frozensets_A = {frozenset(s) for s in groups_A}
                frozensets_B = {frozenset(s) for s in groups_B}
                common_lines_frozensets = frozensets_A & frozensets_B
                if common_lines_frozensets:
                    intersection.setdefault(key, {})[rel_type] = [set(fs) for fs in common_lines_frozensets]
            
            # The new common_context is this intersection
            common_context = intersection

        # We don't need wildcards ('x') for this method,
        # because we're intersecting full contexts, not patterns.
        return common_context

    def _analyze_failures(self, action_key: str, all_success_contexts: list[dict], all_failure_contexts: list[dict], current_failure_context: dict):
        """
        Analyzes failures by finding conditions that are consistent across all failures
        AND have never been observed in any past success.
        
        This new version learns multiple rules:
        1. A "default" rule if no successes exist.
        2. A "common" rule (what all failures share).
        3. A "differentiating" rule (what makes failures different from successes).
        """
        if not all_failure_contexts:
            return

        # --- Rule 1: No Successes Ever ---
        if not all_success_contexts:
            # This action has ONLY ever failed. It should fail in ALL contexts.
            # An empty rule {} is a "default" rule that always matches.
            default_failure_rule = {} 
            
            if self.debug_channels['FAILURE']: 
                print(f"\n--- Failure Analysis for {action_key}: (No successes on record) ---")
                print(f"  Learning rule: Action *always* fails. (Default failure rule stored)")
            
            # Store this as the *only* rule in a new list.
            self.failure_patterns[action_key] = [default_failure_rule]
            return
        
        if self.debug_channels['FAILURE']: 
            print(f"\n--- Failure Analysis for {action_key}: Consistent Differentiating Conditions ---")
        
        # --- Step 1: Find common contexts ---
        common_success_context = self._find_common_context(all_success_contexts)
        common_failure_context = self._find_common_context(all_failure_contexts)
        
        # --- Rule 2: The "Common Failure" Rule ---
        # This is the rule for "what do all failures have in common?"
        # We'll store this rule.
        
        # --- Step 2: Build a veto list of all states ever seen in *any* success ---
        observed_in_any_success = {
            'adj': set(), 'diag_adj': set(), 'rels': set(),
            'align': set(), 'diag_align': set(), 'match': set()
        }
        for context in all_success_contexts:
            for obj_id, contacts in context.get('adj', {}).items():
                observed_in_any_success['adj'].add((obj_id, tuple(contacts)))
            for obj_id, contacts in context.get('diag_adj', {}).items():
                observed_in_any_success['diag_adj'].add((obj_id, tuple(contacts)))
            for rel_type, groups in context.get('rels', {}).items():
                for value, ids in groups.items():
                    observed_in_any_success['rels'].add((rel_type, value, frozenset(ids)))
            for rel_type, groups in context.get('align', {}).items():
                for value, ids in groups.items():
                    observed_in_any_success['align'].add((rel_type, value, frozenset(ids)))
            for rel_type, groups in context.get('match', {}).items():
                for value, ids in groups.items():
                    observed_in_any_success['match'].add((rel_type, value, frozenset(ids)))
            for rel_type, groups in context.get('diag_align', {}).items():
                for line_set in groups:
                    observed_in_any_success['diag_align'].add((rel_type, frozenset(line_set)))

        # --- Rule 3: The "Differentiating" Rule ---
        # The rule is "what is in the failure context that is different from the success context"
        differentiating_rule = self._find_context_difference(common_failure_context, common_success_context)
        
        # --- Step 4: Validate and Prune the Differentiating Rule ---
        pruned_differentiating_rule = {}
        diffs_found = False

        # Helper for logging
        def log_diff(module_name, identifier, success_val, failure_val):
            nonlocal diffs_found
            diffs_found = True
            if self.debug_channels['FAILURE']:
                if self.debug_channels['CONTEXT_DETAILS']:
                    print(f"- {module_name} Difference for {identifier}:")
                    print(f"  - In Successes: {success_val}")
                    print(f"  - In Failures:  {failure_val} (This state was never seen in a success)")
                else:
                    print(f"- {module_name} Difference for {identifier}: Found a differentiating rule.")

        # Check Adj and Diag_Adj
        for key in ['adj', 'diag_adj']:
            adj_f = differentiating_rule.get(key, {})
            for obj_id, contacts_f in adj_f.items():
                if (obj_id, contacts_f) not in observed_in_any_success[key]:
                    pruned_differentiating_rule.setdefault(key, {})[obj_id] = contacts_f
                    log_diff(key.upper(), obj_id, common_success_context.get(key, {}).get(obj_id, "na"), contacts_f)

        # Check Rels, Align, and Match
        for key in ['rels', 'align', 'match']:
            rels_f = differentiating_rule.get(key, {})
            for rel_type, groups_f in rels_f.items():
                for value, ids_f in groups_f.items():
                    if (rel_type, value, frozenset(ids_f)) not in observed_in_any_success[key]:
                        pruned_differentiating_rule.setdefault(key, {}).setdefault(rel_type, {})[value] = ids_f
                        success_val = common_success_context.get(key, {}).get(rel_type, {}).get(value, "na")
                        log_diff(f"{key.upper()} Group", f"({rel_type}, {value})", success_val, ids_f)
                            
        # Check Diag_Align
        key = 'diag_align'
        rels_f = differentiating_rule.get(key, {})
        for rel_type, lines_f_list in rels_f.items():
            lines_f = {frozenset(s) for s in lines_f_list}
            for line_f in lines_f:
                if (rel_type, line_f) not in observed_in_any_success[key]:
                    pruned_differentiating_rule.setdefault(key, {}).setdefault(rel_type, []).append(line_f)
                    log_diff(f"{key.upper()} Line", f"({rel_type})", "Not present", line_f)
        
        # --- Step 5: Store All Learned Rules ---
        # We will store the rules in a list.
        # Clear any old rules and add the new ones.
        self.failure_patterns[action_key] = []
        
        # Add the common failure rule (what all failures share)
        if common_failure_context:
            self.failure_patterns[action_key].append(common_failure_context)
            if self.debug_channels['FAILURE']: print(f"  Learning rule (Common): Stored common context of all failures.")

        # Add the pruned differentiating rule (what's unique to failures)
        if diffs_found and pruned_differentiating_rule:
            if pruned_differentiating_rule != common_failure_context:
                self.failure_patterns[action_key].append(pruned_differentiating_rule)
                if self.debug_channels['FAILURE']: print(f"  Learning rule (Diff): Stored differentiating context.")
        elif not diffs_found:
            if self.debug_channels['FAILURE']: print(f"  (No unique differentiating conditions found for this key)")

        if not self.failure_patterns[action_key]:
            if self.debug_channels['FAILURE']: print(f"  (No failure rules could be learned)")
            del self.failure_patterns[action_key]

    def _parse_change_logs_to_events(self, changes: list[str]) -> list[dict]:
        """Parses a list of human-readable change logs into a list of structured event dictionaries."""
        events = []
        for log_str in changes:
            try:
                change_type, details = log_str.replace('- ', '', 1).split(': ', 1)
                event = {'type': change_type}
                
                # --- NEW: Extract object ID ---
                obj_id_str = ""
                if 'Object id_' in details:
                    if change_type in ['NEW', 'REMOVED']:
                         # Format: "Object id_X (ID ...)"
                        obj_id_str = details.split(' ')[1].replace('id_', '')
                    else:
                        # Format: "Object id_X moved..."
                        obj_id_str = details.split(' ')[1].replace('id_', '')
                
                if obj_id_str.isdigit():
                    event['id'] = f"obj_{obj_id_str}"
                # --- End NEW ---

                if change_type == 'MOVED':
                    parts = details.split(' moved from ')
                    pos_parts = parts[1].replace('.', '').split(' to ')
                    start_pos, end_pos = ast.literal_eval(pos_parts[0]), ast.literal_eval(pos_parts[1])
                    event.update({'vector': (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])})
                    events.append(event)
                elif change_type == 'RECOLORED':
                    from_color_str = details.split(' from ')[1].split(' to ')[0]
                    to_color_str = details.split(' to ')[1].replace('.', '')
                    event.update({'from_color': int(from_color_str), 'to_color': int(to_color_str)})
                    events.append(event)
                elif change_type == 'SHAPE_CHANGED':
                    fp_part = details.split('fingerprint: ')[1]
                    from_fp_str, to_fp_str = fp_part.replace(').','').split(' -> ')
                    event.update({'from_fingerprint': int(from_fp_str), 'to_fingerprint': int(to_fp_str)})
                    events.append(event)
                elif change_type in ['GROWTH', 'SHRINK', 'TRANSFORM']:
                    start_pos_str = details.split(') ')[0] + ')'
                    start_pos = ast.literal_eval(start_pos_str.split(' at ')[1])
                    end_pos_str = details.split('now at ')[1].replace('.', '')
                    end_pos = ast.literal_eval(end_pos_str)
                    event.update({'start_position': start_pos, 'end_position': end_pos})
                    if '(from ' in details:
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
            except (ValueError, IndexError, SyntaxError, AttributeError):
                continue
        return events

    def _analyze_and_report(self, action_key: tuple, changes: list[str], full_context: dict):
        """
        Learns from a successful action by storing the full context
        and re-analyzing rule ambiguities.
        """
        new_events = self._parse_change_logs_to_events(changes)

        hashable_events = []
        if new_events: # Only loop if there are events
            for event in new_events:
                stable_event_tuple = tuple(sorted(event.items()))
                hashable_events.append(stable_event_tuple)
        
        # An empty tuple () is the fingerprint for "no change"
        outcome_fingerprint = tuple(sorted(hashable_events))

        # 1. Get or create the hypothesis
        hypothesis = self.rule_hypotheses.setdefault(action_key, {
            'outcomes': {},          # Stores the raw contexts for each outcome
            'differentiated_rules': {} # Stores the *learned rule* (common context) for each outcome
        })

        # 2. Add this event's context to the history for this outcome
        outcome_data = hypothesis['outcomes'].setdefault(outcome_fingerprint, {
            'contexts': [],
            'confirmations': 0,
            'rules': new_events # Store the parsed outcome for prediction
        })
        outcome_data['contexts'].append(full_context)
        outcome_data['confirmations'] += 1
        
        # 3. Check for ambiguity
        if len(hypothesis['outcomes']) == 1:
            # --- Case 1: No Ambiguity ---
            # This is the first outcome, or a confirmation of the only outcome.
            # The "rule" is just the common context of all observations.
            # This is the only outcome ever seen. It MUST be the default rule.
            hypothesis['differentiated_rules'][outcome_fingerprint] = {} # An empty rule means "default"
            common_context = {} # For the logging print below
            
            if self.debug_channels['HYPOTHESIS']:
                print(f"\n--- Learned Default Rule for {action_key} (First Outcome) ---")
                print(f"  Confirmations: {outcome_data['confirmations']}.")
                # --- MODIFIED PRINT ---
                if self.debug_channels['CONTEXT_DETAILS']:
                    print(f"  Rule (Common Context): {common_context}")
                else:
                    print(f"  Rule (Common Context) refined.")
        
        else:
            # --- Case 2: Ambiguity Detected ---
            # We have multiple possible outcomes for this action.
            # We must re-calculate all rules by finding the *differences*
            # between all outcomes.
            if self.debug_channels['HYPOTHESIS']:
                print(f"\n--- Ambiguity Detected for {action_key}. Re-analyzing all outcomes. ---")
            self._analyze_ambiguity(action_key, hypothesis)

    def _find_context_difference(self, context_A: dict, context_B: dict) -> dict:
        """
        Compares two common_contexts (A and B) and returns a new context pattern
        containing all elements that are in A and are different in B.
        """
        diff_pattern = {}

        for key in ['adj', 'diag_adj']:
            adj_A = context_A.get(key, {})
            adj_B = context_B.get(key, {})
            all_adj_ids = set(adj_A.keys()) | set(adj_B.keys())

            for obj_id in all_adj_ids:
                contacts_A = tuple(adj_A.get(obj_id, ['na']*4))
                contacts_B = tuple(adj_B.get(obj_id, ['na']*4))
                
                if contacts_A != contacts_B:
                    if obj_id in adj_A:
                        diff_pattern.setdefault(key, {})[obj_id] = contacts_A

        for key in ['rels', 'align', 'match']:
            rels_A = context_A.get(key, {})
            rels_B = context_B.get(key, {})
            all_rel_types = set(rels_A.keys()) | set(rels_B.keys())

            for rel_type in all_rel_types:
                groups_A = rels_A.get(rel_type, {})
                groups_B = rels_B.get(rel_type, {})
                all_values = set(groups_A.keys()) | set(groups_B.keys())
                
                for value in all_values:
                    ids_A = groups_A.get(value, set())
                    ids_B = groups_B.get(value, set())
                    
                    if ids_A != ids_B:
                        if value in groups_A:
                            diff_pattern.setdefault(key, {}).setdefault(rel_type, {})[value] = ids_A

        key = 'diag_align'
        rels_A = context_A.get(key, {})
        rels_B = context_B.get(key, {})
        all_rel_types = set(rels_A.keys()) | set(rels_B.keys())

        for rel_type in all_rel_types:
            groups_A = rels_A.get(rel_type, [])
            groups_B = rels_B.get(rel_type, [])
            
            frozensets_A = {frozenset(s) for s in groups_A}
            frozensets_B = {frozenset(s) for s in groups_B}

            if frozensets_A != frozensets_B:
                if rel_type in rels_A:
                    diff_pattern.setdefault(key, {})[rel_type] = groups_A

        return diff_pattern
    
    def _intersect_contexts(self, context_A: dict, context_B: dict) -> dict:
        """
        Finds the intersection of two context patterns (A and B).
        A context pattern is a (potentially partial) context.
        """
        if not context_A: return context_B
        if not context_B: return context_A
        
        intersection = {}

        for key in ['adj', 'diag_adj']:
            adj_A = context_A.get(key, {})
            adj_B = context_B.get(key, {})
            
            common_ids = set(adj_A.keys()) & set(adj_B.keys())
            for obj_id in common_ids:
                if adj_A[obj_id] == adj_B[obj_id]:
                    intersection.setdefault(key, {})[obj_id] = adj_A[obj_id]

        for key in ['rels', 'align', 'match']:
            rels_A = context_A.get(key, {})
            rels_B = context_B.get(key, {})
            
            common_rel_types = set(rels_A.keys()) & set(rels_B.keys())
            for rel_type in common_rel_types:
                groups_A = rels_A[rel_type]
                groups_B = rels_B[rel_type]
                
                common_values = set(groups_A.keys()) & set(groups_B.keys())
                for value in common_values:
                    if groups_A[value] == groups_B[value]:
                        intersection.setdefault(key, {}).setdefault(rel_type, {})[value] = groups_A[value]

        key = 'diag_align'
        rels_A = context_A.get(key, {})
        rels_B = context_B.get(key, {})
        common_rel_types = set(rels_A.keys()) & set(rels_B.keys())

        for rel_type in common_rel_types:
            groups_A = rels_A[rel_type]
            groups_B = rels_B[rel_type]
            
            frozensets_A = {frozenset(s) for s in groups_A}
            frozensets_B = {frozenset(s) for s in groups_B}
            
            common_lines_frozensets = frozensets_A & frozensets_B
            
            if common_lines_frozensets:
                intersection.setdefault(key, {})[rel_type] = [set(fs) for fs in common_lines_frozensets]

        return intersection

    def _analyze_ambiguity(self, action_key: tuple, hypothesis: dict):
        """
        Analyzes an ambiguous hypothesis (one with multiple outcomes) by comparing
        every outcome to every *other* outcome to find a unique, differentiating context.
        """
        outcome_keys = list(hypothesis['outcomes'].keys())
        if len(outcome_keys) < 2:
            return

        if self.debug_channels['FAILURE']:
            print(f"\n--- Ambiguity Analysis for {action_key} ---")
            print(f"  Hypothesis has {len(outcome_keys)} outcomes. Running full comparison.")

        try:
            # --- Step 1: Get the common context for every outcome ---
            all_common_contexts = {}
            for key in outcome_keys:
                contexts = hypothesis['outcomes'][key].get('contexts', [])
                if not contexts:
                    if self.debug_channels['FAILURE']: print(f"  Analysis failed: Contexts not stored for outcome {key}.")
                    return
                all_common_contexts[key] = self._find_common_context(contexts)

            # --- Step 2: Find all differentiated rules ---
            differentiated_rules = {}
            all_outcome_keys = list(hypothesis['outcomes'].keys())
            
            no_change_key = () # The "no change" fingerprint
            
            if no_change_key not in all_common_contexts:
                # --- Legacy/Fallback Logic ---
                # This should not happen with the new failure logic, but as a fallback:
                # We don't have a "no change" baseline, so just find any diffs.
                if self.debug_channels['FAILURE']: print("  (Fallback: No 'no change' key found in ambiguity.)")
                for i in range(len(all_outcome_keys)):
                    target_key = all_outcome_keys[i]
                    common_target = all_common_contexts[target_key]
                    intersection_of_diffs = None
                    for j in range(len(all_outcome_keys)):
                        if i == j: continue
                        compare_key = all_outcome_keys[j]
                        common_compare = all_common_contexts[compare_key]
                        current_diff = self._find_context_difference(common_target, common_compare)
                        if intersection_of_diffs is None:
                            intersection_of_diffs = current_diff
                        else:
                            intersection_of_diffs = self._intersect_contexts(intersection_of_diffs, current_diff)
                    differentiated_rules[target_key] = intersection_of_diffs
            
            else:
                # --- Standard Logic: Use "no change" as the baseline ---
                if self.debug_channels['FAILURE']: print("  (Using 'no change' as baseline for ambiguity.)")
                baseline_context = all_common_contexts[no_change_key]
                
                # 1. The "no change" rule is always the default
                differentiated_rules[no_change_key] = {}
                
                # 2. Find specific rules for all *other* outcomes
                for target_key in all_outcome_keys:
                    if target_key == no_change_key:
                        continue
                        
                    target_context = all_common_contexts[target_key]
                    
                    # Find what is different in this rule *compared to the failure baseline*
                    diff_rule = self._find_context_difference(target_context, baseline_context)
                    
                    if diff_rule:
                        # This is a valid, specific rule
                        differentiated_rules[target_key] = diff_rule
                    else:
                        # This rule is a "lie". It happened in the *same context* as a failure.
                        # The failure () takes precedence. Do NOT add a rule for this outcome.
                        if self.debug_channels['FAILURE']:
                            print(f"  (Pruning invalid rule for {target_key}: context matches failure.)")

            # --- NEW Step 3: Analyze and Print the rules ---
            if self.debug_channels['HYPOTHESIS']:
                # Find all the *other* positive rules
                all_positive_rules = [rule for rule in differentiated_rules.values() if rule]
                
                for i in range(len(all_outcome_keys)):
                    target_key = all_outcome_keys[i]
                    rule = differentiated_rules[target_key]
                    
                    if rule:
                        # This is a POSITIVE rule
                        print(f"  - Found POSITIVE Rule for Outcome {i+1} (Result: {target_key}).")
                        if self.debug_channels['CONTEXT_DETAILS']:
                            print(f"    Rule: {rule}")
                    else:
                        # This is a NEGATIVE (default) rule
                        print(f"  - Learned NEGATIVE Rule for Outcome {i+1} (Result: {target_key}).")
                        if self.debug_channels['CONTEXT_DETAILS']:
                            if not all_positive_rules:
                                print(f"    (This is the 'default' outcome; no other positive rules exist yet)")
                            else:
                                # Build the "NOT" string from all other positive rules
                                not_rules_str = " AND NOT ".join([str(r) for r in all_positive_rules])
                                print(f"    (Occurs when NOT ({not_rules_str}))")
            
            hypothesis['differentiated_rules'] = differentiated_rules

        except Exception as e:
            if self.debug_channels['FAILURE']:
                print(f"  Ambiguity analysis failed with error: {e}")

    def _context_matches_pattern(self, current_context: dict, pattern_data: dict | list) -> bool:
        """
        Checks if the current_context (live state) matches a stored pattern or ANY pattern in a list.
        """
        if isinstance(pattern_data, list):
            if not pattern_data:
                return False # No rules in the list to match
            
            # Check if ANY rule in the list matches
            for pattern in pattern_data:
                if self._context_matches_pattern_single(current_context, pattern):
                    return True # Found a match
            return False # No rules in the list matched
        
        # Original behavior: pattern_data is a single dict
        return self._context_matches_pattern_single(current_context, pattern_data)

    def _context_matches_pattern_single(self, current_context: dict, pattern: dict) -> bool:
        """
        Checks if the current_context (live state) matches a stored pattern.
        The pattern can be partial (e.g., only checking for one adjacency).
        """
        if not pattern:
            # An empty pattern is a "default" rule and should only match
            # if no other specific patterns do. We'll handle this in the
            # prediction function, not here. For a direct check, empty is True.
            return True 

        try:
            # Check Adjacency Patterns
            for key in ['adj', 'diag_adj']:
                pattern_adj = pattern.get(key, {})
                current_adj = current_context.get(key, {})
                for obj_id, pattern_contacts in pattern_adj.items():
                    current_contacts = tuple(current_adj.get(obj_id, ['na']*4))
                    for i in range(4):
                        if pattern_contacts[i] != 'x' and pattern_contacts[i] != current_contacts[i]:
                            return False # Mismatch
            
            # Check Relationship, Alignment, and Match Patterns
            for key in ['rels', 'align', 'match']:
                pattern_rels = pattern.get(key, {})
                current_rels = current_context.get(key, {})
                for rel_type, pattern_groups in pattern_rels.items():
                    current_groups = current_rels.get(rel_type, {})
                    for value, pattern_ids in pattern_groups.items():
                        current_ids = current_groups.get(value, set())
                        if pattern_ids != current_ids:
                            return False # Mismatch
            
            # Check Diagonal Alignment Patterns
            key = 'diag_align'
            pattern_rels = pattern.get(key, {})
            current_rels = current_context.get(key, {})
            for rel_type, pattern_groups in pattern_rels.items():
                current_lines_frozensets = {frozenset(s) for s in current_rels.get(rel_type, [])}
                pattern_lines_frozensets = {frozenset(s) for s in pattern_groups}
                if not pattern_lines_frozensets.issubset(current_lines_frozensets):
                    return False # Mismatch
        
        except Exception:
            return False # Failed to parse, so it's not a match

        return True # All pattern checks passed

    def _predict_outcome(self, hypothesis_key: tuple, current_context: dict) -> list[dict] | None:
        """
        Predicts the outcome (as a list of event dicts) of an action given the current context.
        Returns:
        - A list of event dicts if a rule matches.
        - An empty list [] if the rule predicts "no change".
        - None if the action is completely unknown.
        """
        hypothesis = self.rule_hypotheses.get(hypothesis_key)
        
        if not hypothesis:
            return None # This action is an "Unknown"

        differentiated_rules = hypothesis.get('differentiated_rules', {})
        if not differentiated_rules:
            return None

        positive_rules = []
        default_rule_outcome_key = None # This will be the fingerprint

        for outcome_fingerprint, rule_pattern in differentiated_rules.items():
            if rule_pattern:
                positive_rules.append((outcome_fingerprint, rule_pattern))
            else:
                default_rule_outcome_key = outcome_fingerprint
        
        positive_rules.sort(key=lambda x: str(x[1])) 
        
        for outcome_fingerprint, rule_pattern in positive_rules:
            if self._context_matches_pattern(current_context, rule_pattern):
                # Found a specific rule. Return its event list.
                return hypothesis['outcomes'][outcome_fingerprint]['rules']
        
        if default_rule_outcome_key is not None:
            # No specific rule matched, so the default rule applies.
            return hypothesis['outcomes'][default_rule_outcome_key]['rules']

        # No rules matched the context. Treat as unknown.
        return None

    def _get_hypothetical_summary(self, current_summary: list[dict], predicted_events_list: list[dict]) -> list[dict]:
        """
        Takes a summary and a list of predicted events and builds a
        hypothetical future state summary.
        """
        hypothetical_summary = copy.deepcopy(current_summary)
        obj_map = {obj['id']: obj for obj in hypothetical_summary}
        ids_to_remove = set()
        new_objects = []
        
        # We must process removals first
        for event in predicted_events_list:
            if event.get('type') == 'REMOVED':
                obj_id = event.get('id')
                if obj_id:
                    ids_to_remove.add(obj_id)

        # Apply all other changes
        for event in predicted_events_list:
            obj_id = event.get('id')
            event_type = event.get('type')
            
            if event_type == 'NEW':
                # Create a new object. We can't know the 'pixel_coords',
                # but this is enough for relationship/alignment analysis.
                new_obj = {
                    'id': obj_id,
                    'color': event['color'],
                    'pixels': event['pixels'],
                    'position': event['position'],
                    'size': event['size'],
                    'fingerprint': event['fingerprint'],
                    'pixel_coords': frozenset(), # Adjacency analysis will fail, but others will work
                }
                new_objects.append(new_obj)
                continue
            
            # For all other events, find the target object
            if not obj_id or obj_id not in obj_map or obj_id in ids_to_remove:
                continue
                
            target_obj = obj_map[obj_id]

            if event_type == 'MOVED':
                dr, dc = event['vector']
                r, c = target_obj['position']
                target_obj['position'] = (r + dr, c + dc)
            elif event_type == 'RECOLORED':
                target_obj['color'] = event['to_color']
            elif event_type == 'SHAPE_CHANGED':
                target_obj['fingerprint'] = event['to_fingerprint']
            elif event_type in ['GROWTH', 'SHRINK', 'TRANSFORM']:
                # These events have absolute 'to' states
                if 'end_position' in event:
                    target_obj['position'] = event['end_position']
                if 'to_size' in event:
                    target_obj['size'] = event['to_size']
                if 'pixel_delta' in event:
                    # Note: 'pixels' is an int, not a list
                    target_obj['pixels'] = target_obj.get('pixels', 0) + event['pixel_delta']
                if 'to_fingerprint' in event:
                     target_obj['fingerprint'] = event['to_fingerprint']
        
        # Build the final list
        final_summary = [obj for obj_id, obj in obj_map.items() if obj_id not in ids_to_remove]
        final_summary.extend(new_objects)
        return final_summary
    
    def _get_hypothetical_profile(self, hypothetical_summary: list[dict], available_actions: list[GameAction]) -> dict:
        """
        Analyzes a hypothetical future state and returns the
        profile of the *best* action that could be taken from it.
        """
        # --- 1. Analyze the hypothetical state ---
        (current_relationships, current_adjacencies, current_diag_adjacencies, 
         current_match_groups, current_alignments, current_diag_alignments, 
         current_conjunctions) = self._analyze_relationships(hypothetical_summary)

        current_full_context = {
            'summary': hypothetical_summary,
            'rels': current_relationships,
            'adj': current_adjacencies,
            'diag_adj': current_diag_adjacencies,
            'align': current_alignments,
            'diag_align': current_diag_alignments,
            'match': current_match_groups
        }

        # --- 2. Build all possible moves ---
        all_possible_moves = []
        click_action_template = None
        for action in available_actions:
            if action.name == 'ACTION6':
                click_action_template = action
            else:
                all_possible_moves.append({'template': action, 'object': None})
        if click_action_template and hypothetical_summary:
            for obj in hypothetical_summary:
                all_possible_moves.append({'template': click_action_template, 'object': obj})
        
        # --- 3. Run the Discovery Profiler ---
        move_profiles = []
        default_profile = {'unknowns': 0, 'discoveries': 0, 'boring': 0, 'failures': 0}
        
        if not all_possible_moves:
            return default_profile

        for move in all_possible_moves:
            action_template = move['template']
            target_obj = move['object']
            target_id = target_obj['id'] if target_obj else None
            
            base_action_key_str = self._get_learning_key(action_template.name, target_id)
            
            profile = {'unknowns': 0, 'discoveries': 0, 'boring': 0, 'failures': 0}
            predicted_fingerprints_for_this_move = set()
            all_predicted_events_for_move = []

            if target_id:
                # --- Case 1: This is a TARGETED action (e.g., ACTION6_obj_1) ---
                
                # 1. Check for a predicted FAILURE for this specific action
                if base_action_key_str in self.failure_patterns:
                    failure_rule = self.failure_patterns[base_action_key_str]
                    if self._context_matches_pattern(current_full_context, failure_rule):
                        profile['failures'] = 1
                    
                # 2. If no failure was triggered, predict the SUCCESS outcome
                if profile['failures'] == 0:
                    hypothesis_key = (base_action_key_str.split('_')[0], target_id)
                    predicted_event_list = self._predict_outcome(hypothesis_key, current_full_context)
                    
                    # --- Convert event list to a hashable fingerprint ---
                    predicted_outcome_fingerprint = None
                    if predicted_event_list is not None:
                        hashable_events = [tuple(sorted(e.items())) for e in predicted_event_list]
                        predicted_outcome_fingerprint = tuple(sorted(hashable_events))
                        all_predicted_events_for_move.extend(predicted_event_list)
                    # --- End conversion ---
                    
                    # 3. Tally the result
                    if predicted_event_list is None:
                        profile['unknowns'] = 1
                    elif predicted_outcome_fingerprint == ():
                        profile['failures'] += 1 # Predicted "no change"
                    elif predicted_outcome_fingerprint in self.seen_outcomes:
                        profile['boring'] = 1 # Predicted "repetitive change"
                    else:
                        profile['discoveries'] = 1 # Predicted "new, novel change"
                        predicted_fingerprints_for_this_move.add(predicted_outcome_fingerprint)
            else:
                for obj in hypothetical_summary:
                    obj_id = obj['id']
                    hypothesis_key = (base_action_key_str.split('_')[0], obj_id)

                    if hypothesis_key in self.failure_patterns:
                        failure_rule = self.failure_patterns[hypothesis_key]
                        if self._context_matches_pattern(current_full_context, failure_rule):
                            profile['failures'] += 1
                            continue

                    # 2. If no failure, predict the per-object SUCCESS outcome
                    predicted_event_list = self._predict_outcome(hypothesis_key, current_full_context)
                    
                    # --- Convert event list to a hashable fingerprint ---
                    predicted_outcome_fingerprint = None
                    if predicted_event_list is not None:
                        hashable_events = [tuple(sorted(e.items())) for e in predicted_event_list]
                        predicted_outcome_fingerprint = tuple(sorted(hashable_events))
                        all_predicted_events_for_move.extend(predicted_event_list)
                    # --- End conversion ---
                    
                    # 3. Tally the results
                    if predicted_event_list is None:
                        profile['unknowns'] += 1
                    elif predicted_outcome_fingerprint == ():
                        profile['failures'] += 1 # Predicted "no change"
                    elif predicted_outcome_fingerprint in self.seen_outcomes:
                        profile['boring'] += 1 # Predicted "repetitive change"
                    else:
                        profile['discoveries'] += 1 # Predicted "new, novel change"
                        predicted_fingerprints_for_this_move.add(predicted_outcome_fingerprint)

            move_profiles.append((move, profile, predicted_fingerprints_for_this_move, all_predicted_events_for_move))

        # --- 4. Sort and return the best profile ---
        move_profiles.sort(key=lambda x: (
            x[1]['unknowns'], 
            x[1]['discoveries'], 
            -x[1]['failures'], 
            x[1]['boring']
        ), reverse=True)
            
        _chosen_move, best_profile, _new_fingerprints, _best_events = move_profiles[0]
        return best_profile

    def _analyze_relationships(self, object_summary: list[dict]) -> tuple:
        """
        Analyzes all object relationships, adjacencies, alignments, and
        conjunctions from a given object summary.
        
        Returns a tuple containing all 7 analysis dictionaries:
        (relationships, adjacencies, diag_adjacencies, match_groups,
         alignments, diag_alignments, conjunctions)
        """
        
        # --- 0. Handle Empty/Trivial Cases ---
        if not object_summary:
            return {}, {}, {}, {}, {}, {}, {}
        
        if len(object_summary) < 2:
            # Can't have relationships with only one object
            return {}, {}, {}, {}, {}, {}, {}
            
        # --- 1. Basic Relationships, Adjacencies, & Match Groups ---
        
        relationships = {}
        adjacency_map = {}
        diag_adjacency_map = {}
        match_groups = {}

        # Use capitalized keys for cleaner log titles
        rel_data = {
            'Color': {}, 'Shape': {}, 'Size': {}, 'Pixel': {}
        }
        for obj in object_summary:
            obj_id = obj['id']
            rel_data['Color'].setdefault(obj['color'], set()).add(obj_id)
            rel_data['Shape'].setdefault(obj['fingerprint'], set()).add(obj_id)
            rel_data['Size'].setdefault(obj['size'], set()).add(obj_id)
            rel_data['Pixel'].setdefault(obj['pixels'], set()).add(obj_id)

        # Filter out groups with only one member
        for rel_type, groups in rel_data.items():
            relationships[rel_type] = {
                value: ids for value, ids in groups.items() if len(ids) > 1
            }
        relationships = {k: v for k, v in relationships.items() if v} # Clean empty types

        # --- Pixel-Perfect Adjacency Analysis ---
        temp_adj = {}
        temp_diag_adj = {}
        pixel_map = {}
        for obj in object_summary:
            for pixel in obj['pixel_coords']:
                pixel_map[pixel] = obj['id']

        for obj_a in object_summary:
            a_id = obj_a['id']
            for r, c in obj_a['pixel_coords']:
                neighbors = {'top': (r - 1, c), 'bottom': (r + 1, c), 'left': (r, c - 1), 'right': (r, c + 1)}
                for direction, coord in neighbors.items():
                    if coord in pixel_map:
                        b_id = pixel_map[coord]
                        if a_id != b_id:
                            temp_adj.setdefault(a_id, {}).setdefault(direction, set()).add(b_id)

                diagonal_neighbors = {'top_right': (r - 1, c + 1), 'bottom_right': (r + 1, c + 1), 'bottom_left': (r + 1, c - 1), 'top_left': (r - 1, c - 1)}
                for direction, coord in diagonal_neighbors.items():
                    if coord in pixel_map:
                        b_id = pixel_map[coord]
                        if a_id != b_id:
                            temp_diag_adj.setdefault(a_id, {}).setdefault(direction, set()).add(b_id)

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
            if any(res != 'na' for res in result):
                adjacency_map[obj_id] = result

        for obj_id, contacts in temp_diag_adj.items():
            result = ['na'] * 4 # Order: TR, BR, BL, TL
            dir_map = {'top_right': 0, 'bottom_right': 1, 'bottom_left': 2, 'top_left': 3}
            for direction, index in dir_map.items():
                contact_set = contacts.get(direction, set())
                if len(contact_set) == 1:
                    result[index] = contact_set.pop()
            if any(res != 'na' for res in result):
                diag_adjacency_map[obj_id] = result
        
        # --- Match Type Analysis ---
        processed_ids = set()
        exact_match_key = lambda o: (o['color'], o['fingerprint'], o['size'], o['pixels'])
        temp_groups = {}
        for obj in object_summary:
            temp_groups.setdefault(exact_match_key(obj), []).append(obj['id'])
        exact_groups_dict = {key: group for key, group in temp_groups.items() if len(group) > 1}
        if exact_groups_dict:
            match_groups['Exact'] = exact_groups_dict
            for group in exact_groups_dict.values():
                processed_ids.update(group)

        partial_match_definitions = [
            ("Color",       lambda o: (o['fingerprint'], o['size'], o['pixels'])),
            ("Fingerprint", lambda o: (o['color'], o['size'], o['pixels'])),
            ("Size",        lambda o: (o['color'], o['fingerprint'], o['pixels'])),
            ("Pixels",      lambda o: (o['color'], o['fingerprint'], o['size'])),
        ]
        for match_type, key_func in partial_match_definitions:
            temp_groups = {}
            unprocessed_objects = [o for o in object_summary if o['id'] not in processed_ids]
            for obj in unprocessed_objects:
                temp_groups.setdefault(key_func(obj), []).append(obj['id'])
            partial_groups_dict = {key: group for key, group in temp_groups.items() if len(group) > 1}
            if partial_groups_dict:
                match_groups[match_type] = partial_groups_dict
                for group in partial_groups_dict.values():
                    processed_ids.update(group)
        
        # --- 2. Cardinal Alignments ---
        
        alignments = {}
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
                obj_id = obj['id']
                alignment_groups[align_type].setdefault(coord, set()).add(obj_id)
        
        for align_type, groups in alignment_groups.items():
            filtered_groups = {coord: ids for coord, ids in groups.items() if len(ids) > 1}
            if filtered_groups:
                alignments[align_type] = filtered_groups

        # --- 3. Diagonal Alignments ---
        
        diag_alignments = {}
        coord_map = {(obj['center_y'], obj['center_x']): obj['id'] for obj in object_summary}
        processed_ids = set()
        final_alignments = {'top_left_to_bottom_right': [], 'top_right_to_bottom_left': []}

        for i in range(len(object_summary)):
            obj_a = object_summary[i]
            id_a = obj_a['id']
            if id_a in processed_ids: continue
            for j in range(i + 1, len(object_summary)):
                obj_b = object_summary[j]
                id_b = obj_b['id']
                if id_b in processed_ids: continue
                y1, x1 = obj_a['center_y'], obj_a['center_x']
                y2, x2 = obj_b['center_y'], obj_b['center_x']
                if x2 - x1 == 0: continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) == 1.0:
                    line_members = {id_a, id_b}
                    step_y = 1 if y2 > y1 else -1
                    step_x = 1 if x2 > x1 else -1
                    next_y, next_x = y2 + step_y, x2 + step_x
                    while (next_y, next_x) in coord_map:
                        found_id = coord_map[(next_y, next_x)]
                        line_members.add(found_id)
                        next_y, next_x = next_y + step_y, next_x + step_x
                    prev_y, prev_x = y1 - step_y, x1 - step_x
                    while (prev_y, prev_x) in coord_map:
                        found_id = coord_map[(prev_y, prev_x)]
                        line_members.add(found_id)
                        prev_y, prev_x = prev_y - step_y, prev_x - step_x
                    if len(line_members) > 1:
                        align_type = 'top_left_to_bottom_right' if slope == 1.0 else 'top_right_to_bottom_left'
                        final_alignments[align_type].append(frozenset(line_members))
                        processed_ids.update(line_members)
        
        for align_type, groups in final_alignments.items():
            if groups:
                unique_groups = sorted([sorted(list(g)) for g in set(groups)])
                diag_alignments[align_type] = [set(g) for g in unique_groups]
        
        # --- 4. Conjunctions ---
        
        conjunctions = {}
        object_group_map = {}
        # We can only find conjunctions if we found relationships or alignments
        all_modules = {**relationships, **alignments}
        
        if all_modules:
            for group_type, groups in all_modules.items():
                for value, obj_ids in groups.items():
                    group_signature = (group_type, value)
                    for obj_id in obj_ids:
                        object_group_map.setdefault(obj_id, set()).add(group_signature)
            
            if len(object_group_map) >= 2:
                temp_conjunctions = {}
                obj_ids = sorted(list(object_group_map.keys()))
                for i in range(len(obj_ids)):
                    for j in range(i + 1, len(obj_ids)):
                        obj_A_id, obj_B_id = obj_ids[i], obj_ids[j]
                        profile_A = object_group_map.get(obj_A_id, set())
                        profile_B = object_group_map.get(obj_B_id, set())
                        common_groups = profile_A & profile_B
                        if len(common_groups) > 1:
                            conjunction_key = frozenset(common_groups)
                            temp_conjunctions.setdefault(conjunction_key, set()).update({obj_A_id, obj_B_id})

            for common_profile, obj_ids in temp_conjunctions.items():
                if len(obj_ids) > 1:
                    type_names = sorted([item[0] for item in common_profile])
                    conj_type_name = "_and_".join(type_names)
                    sorted_profile = sorted(list(common_profile))
                    value_tuple = tuple([item[1] for item in sorted_profile])
                    conjunctions.setdefault(conj_type_name, {})[value_tuple] = obj_ids
        
        # --- 5. Return All Results ---
        return relationships, adjacency_map, diag_adjacency_map, match_groups, alignments, diag_alignments, conjunctions
    
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
                        changed_parts = []
                        if old_obj['fingerprint'] != new_obj['fingerprint']:
                            changed_parts.append("shape")
                        if old_obj['color'] != new_obj['color']:
                            changed_parts.append(f"color from {old_obj['color']} to {new_obj['color']}")
                        
                        size_changed = old_obj['size'] != new_obj['size']
                        size_details = f" (size {old_size_str} -> {new_size_str})" if size_changed else ""

                        if not changed_parts:
                            details = f"transformed{size_details}"
                        else:
                            details = f"changed { ' and '.join(changed_parts) }{size_details}"

                    changes.append(f"- {event_type}: Object {old_obj['id'].replace('obj_', 'id_')} at {old_obj['position']} {details}, now at {new_obj['position']}.")

                    matched_old.add(id(old_obj))
                    matched_new.add(id(new_obj))
            
            old_unexplained = [obj for obj in old_unexplained if id(obj) not in matched_old]
            new_unexplained = [obj for obj in new_unexplained if id(obj) not in matched_new]

        # --- Final Pass: Log remaining as REMOVED, and then handle NEW, REAPPEARED, or TRANSFORMED ---
        used_persistent_ids = set() 

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
        max_current_id = 0
        for obj in new_summary:
            try:
                id_num = int(obj['id'].replace('obj_', ''))
                if id_num > max_current_id:
                    max_current_id = id_num
            except (ValueError, AttributeError):
                continue
        
        if self.object_id_counter < max_current_id:
            self.object_id_counter = max_current_id

        return sorted(changes), new_summary
    
    def _find_single_property_change_match(self, new_obj):
        """Finds a removed object that matches the new object in all but one property."""
        new_stable_id = self._get_stable_id(new_obj)
        
        for old_stable_id in sorted(list(self.removed_objects_memory.keys())):
            diffs = {}
            if new_stable_id[0] != old_stable_id[0]:
                diffs['shape'] = (old_stable_id[0], new_stable_id[0])
            if new_stable_id[1] != old_stable_id[1]:
                diffs['color'] = (old_stable_id[1], new_stable_id[1])
            if new_stable_id[2] != old_stable_id[2]:
                diffs['size'] = (old_stable_id[2], new_stable_id[2])
            if new_stable_id[3] != old_stable_id[3]:
                diffs['pixels'] = (old_stable_id[3], new_stable_id[3])

            if len(diffs) == 1:
                return old_stable_id, diffs
                
        return None, None
    
    def _remap_recursive(self, data: any, id_map: dict[str, str]) -> any:
        """
        Recursively traverses a data structure (dict, list, set, etc.) and
        replaces all occurrences of old object IDs with new ones based on id_map.
        """
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                new_key = id_map.get(k, k)
                new_dict[new_key] = self._remap_recursive(v, id_map)
            return new_dict
        elif isinstance(data, list):
            return [self._remap_recursive(item, id_map) for item in data]
        elif isinstance(data, set):
            return {self._remap_recursive(item, id_map) for item in data}
        elif isinstance(data, frozenset):
            return frozenset({self._remap_recursive(item, id_map) for item in data})
        elif isinstance(data, tuple):
            return tuple([self._remap_recursive(item, id_map) for item in data])
        elif isinstance(data, str):
            return id_map.get(data, data)
        else:
            return data

    def _remap_memory(self, id_map: dict[str, str]):
        """Updates all learning structures to use new IDs after a re-numbering."""
        if self.debug_channels['PERCEPTION']: print(f"Remapping memory for {len(id_map)} objects...")
        
        # This function will be expanded later, but for now it's a placeholder
        # as obml_agent doesn't have the same memory structures as obrl_agent.
        pass
    
    def _print_full_summary(self, summary: list[dict], new_to_old_map: dict = None):
        """Prints a formatted summary of all objects, noting previous IDs if available."""
        if not summary:
            if self.debug_channels['PERCEPTION']: print("No objects found.")
            return

        for obj in summary:
            obj_id = obj['id'].replace('obj_', 'id_')
            size_str = f"{obj['size'][0]}x{obj['size'][1]}"
            
            formerly_parts = []
            if new_to_old_map and obj['id'] in new_to_old_map:
                old_id = new_to_old_map[obj['id']].replace('obj_', 'id_')
                formerly_parts.append(f"formerly {old_id}")
            
            if 'cross_level_change_info' in obj:
                formerly_parts.append(obj['cross_level_change_info'])
            
            formerly_str = ""
            if formerly_parts:
                formerly_str = f" ({', '.join(formerly_parts)})"

            if self.debug_channels['PERCEPTION']: print(
                f"- Object {obj_id}{formerly_str}: Found a {size_str} object of color {obj['color']} "
                f"at position {obj['position']} with {obj['pixels']} pixels "
                f"and shape fingerprint {obj['fingerprint']}."
            )

    def _log_relationship_changes(self, old_rels: dict, new_rels: dict):
        """Compares two relationship dictionaries and logs which objects joined, left, or replaced others in groups."""
        output_lines = []
        
        def format_ids(id_set):
            if not id_set: return ""
            id_list = sorted(list(id_set))
            if len(id_list) == 1:
                return f"object {id_list[0]}"
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

                if len(new_ids) > 1 or len(old_ids) > 1:
                    if joined_ids and left_ids:
                        output_lines.append(f"- {rel_type} Group ({value_str}): {format_ids(joined_ids).capitalize()} replaced {format_ids(left_ids)}.")
                    elif joined_ids:
                        output_lines.append(f"- {rel_type} Group ({value_str}): {format_ids(joined_ids).capitalize()} joined.")
                    elif left_ids:
                        output_lines.append(f"- {rel_type} Group ({value_str}): {format_ids(left_ids).capitalize()} left.")

        if output_lines:
            if self.debug_channels['CHANGES']:
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
            if self.debug_channels['CHANGES']:
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
            if self.debug_channels['CHANGES']:
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
            id_list = sorted(list(id_set), key=lambda x: int(x.split('_')[1]))
            if len(id_list) == 1: return f"object {id_list[0].replace('obj_', 'id_')}"
            return f"objects " + ", ".join([i.replace('obj_', 'id_') for i in id_list])

        for align_type in all_align_types:
            old_groups = old_aligns.get(align_type, [] if is_diagonal else {})
            new_groups = new_aligns.get(align_type, [] if is_diagonal else {})

            if is_diagonal:
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
            if self.debug_channels['CHANGES']:
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
            parts = []
            if match_type == 'Exact':
                parts.append(f"Color:{props[0]}")
                parts.append(f"Fingerprint:{props[1]}")
                parts.append(f"Size:{props[2][0]}x{props[2][1]}")
                parts.append(f"Pixels:{props[3]}")
            elif match_type == 'Color': 
                parts.append(f"Fingerprint:{props[0]}")
                parts.append(f"Size:{props[1][0]}x{props[1][1]}")
                parts.append(f"Pixels:{props[2]}")
            elif match_type == 'Fingerprint':
                parts.append(f"Color:{props[0]}")
                parts.append(f"Size:{props[1][0]}x{props[1][1]}")
                parts.append(f"Pixels:{props[2]}")
            elif match_type == 'Size':
                parts.append(f"Color:{props[0]}")
                parts.append(f"Fingerprint:{props[1]}")
                parts.append(f"Pixels:{props[2]}")
            elif match_type == 'Pixels':
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
                left = old_ids - old_ids
                props_str = format_props(match_type, props)

                if joined and left:
                    output_lines.append(f"- {label} Group {props_str}: {format_ids(joined).capitalize()} replaced {format_ids(left)}.")
                elif joined:
                    output_lines.append(f"- {label} Group {props_str}: {format_ids(joined).capitalize()} joined.")
                elif left:
                    output_lines.append(f"- {label} Group {props_str}: {format_ids(left).capitalize()} left.")

        if output_lines:
            if self.debug_channels['CHANGES']:
                print("\n--- Match Type Change Log ---")
                for line in sorted(output_lines):
                    print(line)
                print()