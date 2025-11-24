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
        self.banned_action_keys = set()
        self.permanent_banned_actions = set()
        self.successful_click_actions = set()

        # --- NEW: CLASSIFICATION MEMORY ---
        self.concrete_witness_registry = {}
        self.abstract_witness_registry = {}
        self.phenomenal_witness_registry = {}

        # --- NEW: Session/Trial Tracking ---
        self.global_action_counter = 0  # Tracks the passing of time/trials
        self.last_action_id = 0         # ID of the action that caused the current state

        # --- NEW: TRANSITION MEMORY (Scientific Standard) ---
        # Structure: self.transition_counts[start_state][action_family][end_state] = count
        # Start/End State Signature: (Color, Fingerprint, Size)
        self.transition_counts = {}

        self.global_action_counter = 0
        self.last_action_id = 0
        
        self.concrete_event_counts = {}
        self.phenomenal_event_counts = {}
        
        self.action_consistency_counts = {} 
        self.performed_action_types = set()
        self.productive_action_types = set() # Tracks actions that caused change

        # --- Debug Channels ---
        # Set these to True or False to control the debug output.
        self.debug_channels = {
            'PERCEPTION': False,      # Object finding, relationships, new level setup
            'CHANGES': True,         # All "Change Log" prints
            'STATE_GRAPH': False,     # State understanding
            'HYPOTHESIS': False,      # "Initial Hypotheses", "Refined Hypothesis"
            'FAILURE': False,         # "Failure Analysis", "Failure Detected"
            'WIN_CONDITION': False,   # "LEVEL CHANGE DETECTED", "Win Condition Analysis"
            'ACTION_SCORE': True,    # All scoring prints
            'CONTEXT_DETAILS': False # Keep or remove large prints
        }

    def _reset_agent_memory(self):
        """
        Resets all agent learning and memory to a clean state.
        This is called at the start of a new game.
        """
        # Reset the "brain" (long-term learning)
        self.success_contexts = {}
        self.failure_contexts = {}
        self.failure_patterns = {}
        self.rule_hypotheses = {}
        self.seen_outcomes = set()
        self.win_condition_hypotheses = []
        self.last_score = 0
        self.permanent_banned_actions = set()
        self.successful_click_actions = set()

        # NEW: Reset Transition Memory
        self.transition_counts = {}
        
        # NEW: Registry for Concrete Exclusivity Check
        # Mapping: Concrete Signature -> Set of Action Names seen with it
        self.concrete_witness_registry = {}
        # Mapping: Abstract Signature -> Set of Action Names seen with it
        self.abstract_witness_registry = {}
        # Mapping: Phenomenal Signature -> Set of Action Names seen with it
        self.phenomenal_witness_registry = {}
        # Count how many times a specific event has occurred total (Fix for KeyError)
        self.concrete_event_counts = {}
        self.phenomenal_event_counts = {}
        self.action_consistency_counts = {} 
        self.performed_action_types = set()

        # Also reset the level state
        self._reset_level_state()

    def _reset_level_state(self):
        """Resets only the memory for the current level."""
        self.object_id_counter = 0
        self.removed_objects_memory = {}
        self.last_object_summary = []
        self.last_relationships = {}
        self.last_adjacencies = {}
        self.last_diag_adjacencies = {}
        self.last_alignments = {}
        self.last_diag_alignments = {}
        self.last_match_groups = {}
        self.final_summary_before_level_change = None
        self.current_level_id_map = {}
        self.last_action_context = None
        self.level_state_history = []
        self.banned_action_keys = set()
        self.actions_printed = False # This is per-level, not per-game
        self.performed_action_types = set()

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Analyzes the current frame, compares it to the previous frame,
        and logs all perceived changes.
        """
        # If the game hasn't started, this is a new game. Do a full "brain wipe".
        if latest_frame.state == GameState.NOT_PLAYED:
            self._reset_agent_memory()
            return GameAction.RESET
        
        # If the game is over, this is a "retry" of the same level.
        # Only reset the level state, but KEEP the learned rules.
        if latest_frame.state == GameState.GAME_OVER:
            self._reset_level_state()
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
            # This block runs on the first frame of a new level, OR a retry.
            current_score = latest_frame.score
            
            if self.is_new_level:
                # --- This is a NEW LEVEL (from score increase) ---
                # We must wipe the brain.
                self._reset_agent_memory()
                if self.debug_channels['PERCEPTION']:
                     print(f"\n--- Level Cleared (Score: {current_score}): Wiping brain and resetting history. ---")

            else:
                # --- This is a RETRY (from GAME_OVER) ---
                # We are just re-perceiving the level. DO NOT wipe the brain.
                # The brain was preserved by the GAME_OVER check.
                if self.debug_channels['PERCEPTION']:
                    print(f"\n--- Retrying Level (Score: {current_score}): Re-perceiving level. ---")

            # Now, set the state for this "first frame" (applies to both cases)
            self.is_new_level = False # We have now handled the "new level" state
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
            
            # --- NEW: Check for score increase *before* analysis ---
            current_score = latest_frame.score
            score_increased = current_score > self.last_score
            # --- End NEW ---
            
            # This function compares old and new summaries, assigns persistent IDs,
            # and returns a list of change strings.
            changes, current_summary = self._log_changes(prev_summary, current_summary)

            # Now, analyze the new summary *after* persistent IDs are assigned
            (current_relationships, current_adjacencies, current_diag_adjacencies, 
             current_match_groups, current_alignments, current_diag_alignments, 
             current_conjunctions) = self._analyze_relationships(current_summary)

            # --- Check for score changes (win detection) ---
            if score_increased:
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
            if self.last_action_context and self.last_object_summary:
                learning_key = self.last_action_context 
        
                # This is the full context of the *previous* state
                prev_context = {
                    'summary': prev_summary,
                    'rels': self.last_relationships,
                    'adj': self.last_adjacencies,
                    'diag_adj': self.last_diag_adjacencies,
                    'align': self.last_alignments,
                    'diag_align': self.last_diag_alignments,
                    'match': self.last_match_groups
                }

                # 1. Parse text changes back to structured event dicts
                events = self._parse_change_logs_to_events(changes)

                # 2. Classify Events (Strict Logic + Exception Solving)
                # FIX: Pass 'prev_context' so the solver can compare states
                direct_events, global_events, ambiguous_events = self._classify_event_stream(events, learning_key, prev_context)
                
                # 3. Resolve Ambiguity (The Pipeline)
                promoted_events = self._resolve_ambiguous_events(
                    ambiguous_events, events, current_summary, prev_summary, 
                    current_adjacencies, self.last_adjacencies, learning_key
                )
                
                # Merge promoted events into Direct for learning
                direct_events.extend(promoted_events)
                ambiguous_events = [e for e in ambiguous_events if e not in promoted_events]

                # 4. Map events to specific objects for DIRECT learning
                obj_events_map = {obj['id']: [] for obj in self.last_object_summary}
                for event in direct_events:
                    if 'id' in event and event['id'] in obj_events_map:
                        obj_events_map[event['id']].append(event)

                # 5. Unified Learning Loop (RUN BEFORE PRINTING)
                
                # A. Learn Direct Rules (Action -> Result)
                for obj in self.last_object_summary:
                    obj_id = obj['id']
                    specific_events = obj_events_map[obj_id]
                    hypothesis_key = (learning_key, obj_id)
                    self._analyze_result(hypothesis_key, specific_events, prev_context)
                
                # B. Learn Global Rules (Environment -> Result)
                for event in global_events:
                    abst_sig = self._get_abstract_signature(event)
                    global_key = ('GLOBAL', str(abst_sig)) 
                    self._analyze_result(global_key, [event], prev_context)

                # --- DETAILED PRINTING (Now uses updated brain) ---
                if self.debug_channels['CHANGES']:
                    
                    def _fmt_val(e):
                        if 'vector' in e: return f"Move {e['vector']}"
                        if 'to_color' in e: return f"Recolor to {e['to_color']}"
                        if 'pixel_delta' in e: return f"Size {'+' if e['pixel_delta']>0 else ''}{e['pixel_delta']}"
                        if 'to_fingerprint' in e: return f"Shape {e['to_fingerprint']}"
                        if 'type' in e and e['type'] == 'NEW': return f"Spawn ({e.get('color')}, {e.get('size')})"
                        return e['type']

                    if global_events: 
                        print(f"  -> Filtered {len(global_events)} Global events:")
                        for e in global_events:
                            abst_sig = self._get_abstract_signature(e)
                            global_key = ('GLOBAL', str(abst_sig))
                            rule_str = self._format_rule_description(global_key)
                            
                            print(f"     * {e['type']} on {e.get('id', 'Unknown')}")
                            print(f"       [Explanation] Global Rule: Inevitable '{_fmt_val(e)}'.")
                            print(f"       [Prediction]  {rule_str}")

                    if direct_events: 
                        print(f"  -> Processing {len(direct_events)} Direct events:")
                        for e in direct_events:
                            # Call with single tuple key
                            direct_key = (learning_key, e.get('id'))
                            rule_str = self._format_rule_description(direct_key)
                            
                            # --- NEW: Prepend Condition if found ---
                            prefix = ""
                            if 'condition' in e:
                                prefix = f"[EXCEPTION FOUND] {e['condition']} => "
                            # ---------------------------------------

                            print(f"     * {e['type']} on {e.get('id', 'Unknown')}")
                            print(f"       [Explanation] Direct Causality: Action consistently causes '{_fmt_val(e)}'")
                            print(f"       [Prediction]  {prefix}{rule_str}")

                    if ambiguous_events:
                        print(f"  -> Ignored {len(ambiguous_events)} Ambiguous events:")
                        for wrapper in ambiguous_events:
                            e = wrapper['event']
                            reason = wrapper['reason']
                            fix = wrapper['fix']
                            
                            print(f"     * {e['type']} on {e.get('id', 'Unknown')}")
                            print(f"       [Status] {reason}")
                            print(f"       [Needs]  {fix}")
                            
                # --- Failsafe: Track banned actions ---
                # Note: We count success if there are DIRECT events or score increased.
                # Ambiguous events are NOT considered success yet (conservative).
                action_succeeded = bool(direct_events) or score_increased

                if not action_succeeded and self.last_action_context: # Last action was a total failure
                    self.banned_action_keys.add(self.last_action_context)
                    if self.debug_channels['FAILURE']:
                        print(f"--- Failsafe: Banning action '{self.last_action_context}' due to failure. Total banned: {len(self.banned_action_keys)} ---")
                    
                    # --- MODIFIED: Permanent Ban Logic ---
                    if (self.last_action_context.startswith('ACTION6_') and
                        self.last_action_context not in self.successful_click_actions):
                        self.permanent_banned_actions.add(self.last_action_context)
                        if self.debug_channels['FAILURE']:
                            print(f"--- Failsafe: Permanently banning '{self.last_action_context}' (never succeeded). Total permanent: {len(self.permanent_banned_actions)} ---")
                    # --- End MODIFIED ---
                
                elif action_succeeded: # Last action had *some* success
                    if self.banned_action_keys:
                        if self.debug_channels['FAILURE']:
                            print(f"--- Failsafe: Success detected. Clearing {len(self.banned_action_keys)} banned actions. ---")
                        self.banned_action_keys.clear() # Clear ban
                    
                    # --- NEW: Track Successful Clicks ---
                    if self.last_action_context and self.last_action_context.startswith('ACTION6_'):
                        self.successful_click_actions.add(self.last_action_context)
                    # --- End NEW ---
                # --- End Failsafe ---

                # 2. Map events to specific objects
                # Only map DIRECT events to the learner.
                obj_events_map = {obj['id']: [] for obj in self.last_object_summary}
                
                for event in direct_events:
                    if 'id' in event and event['id'] in obj_events_map:
                        obj_events_map[event['id']].append(event)

                # 3. Unified Learning Loop
                # We process EVERY object from the previous frame.
                for obj in self.last_object_summary:
                    obj_id = obj['id']
                    specific_events = obj_events_map[obj_id]
                    
                    # Key: (ActionName, TargetID)
                    hypothesis_key = (learning_key, obj_id)
                    
                    # Learn (Success OR Failure is handled uniformly inside this function)
                    self._analyze_result(hypothesis_key, specific_events, prev_context)
                    
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
            for obj in current_summary:
                obj_id = obj['id']
                hypothesis_key = (base_action_key_str, obj_id)

                # 1. Strict Prediction
                predicted_events, confidence = self._predict_outcome(hypothesis_key, current_full_context)
                
                # 2. Categorize
                if predicted_events is None:
                    # Unknown: We have NO rule (Success or Failure) that matches this state.
                    profile['unknowns'] += 1
                elif not predicted_events:
                    # Failure: We matched a "No Change" consistency rule.
                    profile['failures'] += 1
                else:
                    # Success: We matched a "Change" consistency rule.
                    hashable_events = [tuple(sorted(e.items())) for e in predicted_events]
                    fp = tuple(sorted(hashable_events))
                    predicted_fingerprints_for_this_move.add(fp)
                    all_predicted_events_for_move.extend(predicted_events)
                    
                    if fp in self.seen_outcomes and confidence >= 2:
                        profile['boring'] += 1
                    else:
                        profile['discoveries'] += 1
            move_profiles.append((move, profile, predicted_fingerprints_for_this_move, all_predicted_events_for_move))

# --- Failsafe: 2-Stage Ban Filtering ---
        
        # --- Stage 1: Temporary "banned_action_keys" (cleared on success) ---
        if self.banned_action_keys and len(move_profiles) > 1:
            initial_count_temp = len(move_profiles)
            filtered_profiles_temp = []
            for move_tuple in move_profiles:
                move, _, _, _ = move_tuple
                move_key = self._get_learning_key(move['template'].name, move['object']['id'] if move['object'] else None)
                if move_key not in self.banned_action_keys:
                    filtered_profiles_temp.append(move_tuple)
            
            if not filtered_profiles_temp:
                # All moves are *temporarily* banned. This is a stalemate.
                # Clear the temporary list and proceed with all original moves.
                if self.debug_channels['FAILURE']:
                    print(f"--- Failsafe: All moves temporarily banned. Clearing {len(self.banned_action_keys)} temp bans. ---")
                self.banned_action_keys.clear()
                # We proceed with the original move_profiles
            
            elif len(filtered_profiles_temp) < initial_count_temp:
                # We filtered some temp bans and have options left.
                move_profiles = filtered_profiles_temp
                if self.debug_channels['FAILURE']:
                    print(f"--- Failsafe: Removed {initial_count_temp - len(filtered_profiles_temp)} temp banned actions. ---")
            
            # else: no temp bans were found, proceed with original move_profiles
        
        # --- Stage 2: Permanent "permanent_banned_actions" (ACTION6 only) ---
        if self.permanent_banned_actions and len(move_profiles) > 1:
            initial_count_perm = len(move_profiles)
            filtered_profiles_perm = []
            for move_tuple in move_profiles:
                move, _, _, _ = move_tuple
                move_key = self._get_learning_key(move['template'].name, move['object']['id'] if move['object'] else None)
                if move_key not in self.permanent_banned_actions:
                    filtered_profiles_perm.append(move_tuple)
            
            if not filtered_profiles_perm:
                # All *remaining* moves are *permanently* banned.
                # This is the "last resort" scenario.
                # We do NOT clear the permanent list. We just ignore it for this turn.
                if self.debug_channels['FAILURE']:
                    print(f"--- Failsafe: All valid moves are permanently banned. Ignoring {len(self.permanent_banned_actions)} permanent bans for this turn. ---")
                # We proceed with the move_profiles list from Stage 1
            
            elif len(filtered_profiles_perm) < initial_count_perm:
                # We successfully filtered permanent bans.
                move_profiles = filtered_profiles_perm
                if self.debug_channels['FAILURE']:
                    print(f"--- Failsafe: Removed {initial_count_perm - len(filtered_profiles_perm)} permanent banned actions. ---")
            
            # else: no permanent bans were found, proceed
        
        # --- End Failsafe ---

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
            
            # --- Updated: Recursive Lookahead Tie-Breaker ---
            
            chosen_move_tuple = None
            
            # If we have multiple best moves, and they are "Boring" (U=0, D=0),
            # we need to check which one sets up a better future.
            top_score_is_boring = (move_profiles[0][1]['unknowns'] == 0 and 
                                   move_profiles[0][1]['discoveries'] == 0 and
                                   move_profiles[0][1]['failures'] == 0)

            if top_score_is_boring:
                # Find all ties for first place
                tied_moves = []
                top_score = (move_profiles[0][1]['unknowns'], move_profiles[0][1]['discoveries'], -move_profiles[0][1]['failures'])
                
                for mp in move_profiles:
                    current_score = (mp[1]['unknowns'], mp[1]['discoveries'], -mp[1]['failures'])
                    if current_score == top_score:
                        tied_moves.append(mp)
                    else:
                        break # Sorted list, so we can stop
                
                if len(tied_moves) > 1:
                    if self.debug_channels['ACTION_SCORE']: 
                        print(f"\n--- Running Recursive Lookahead (Depth 3) for {len(tied_moves)} Boring Moves ---")

                    best_deep_score = (-999, -999, -999, -999)
                    best_move_idx = 0
                    
                    for i, move_tuple in enumerate(tied_moves):
                        move, _, _, _ = move_tuple
                        
                        # Call the recursive function
                        # Returns (Unknowns, Discoveries, -Failures, -Distance)
                        deep_score = self._recursive_lookahead_score(current_summary, move['template'], move['object'], 0)
                        
                        if self.debug_channels['ACTION_SCORE']:
                             target = f" on {move['object']['id']}" if move['object'] else ""
                             print(f"  - Move {move['template'].name}{target} -> Chain Score: {deep_score}")

                        if deep_score > best_deep_score:
                            best_deep_score = deep_score
                            best_move_idx = i
                    
                    chosen_move_tuple = tied_moves[best_move_idx]
                else:
                    chosen_move_tuple = tied_moves[0] # No ties
            else:
                # If the best move is already an Unknown or Discovery, just take it!
                chosen_move_tuple = move_profiles[0]
            
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
        
        # --- NEW: Timestamp the Action ---
        # We increment the counter to define a new "Scientific Trial"
        self.global_action_counter += 1
        self.last_action_id = self.global_action_counter
        # ---------------------------------
        
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
        Finds the GLOBAL intersection of multiple game states.
        FIX: Uses Intersection for groups instead of Strict Equality.
        """
        if not contexts: return {}

        common_rule = copy.deepcopy(contexts[0])
        common_rule.pop('summary', None)
        common_rule.pop('events', None)

        for i in range(1, len(contexts)):
            next_ctx = contexts[i]
            
            # --- 1. Intersect Adjacencies ---
            for key in ['adj', 'diag_adj']:
                if key not in common_rule: continue
                current_map = common_rule[key]
                next_map = next_ctx.get(key, {})
                
                ids_to_remove = []
                for obj_id, current_contacts in current_map.items():
                    if obj_id not in next_map or current_contacts != next_map[obj_id]:
                        ids_to_remove.append(obj_id)
                for obj_id in ids_to_remove:
                    del current_map[obj_id]
                if not current_map: del common_rule[key]

            # --- 2. Intersect Global Relationships (The Fix) ---
            for key in ['rels', 'align', 'match']:
                if key not in common_rule: continue
                current_types = common_rule[key]
                next_types = next_ctx.get(key, {})
                
                types_to_remove = []
                for t_name, curr_groups in current_types.items():
                    if t_name not in next_types:
                        types_to_remove.append(t_name)
                        continue
                    
                    next_groups = next_types[t_name]
                    values_to_remove = []
                    for val, curr_ids in curr_groups.items():
                        if val not in next_groups:
                            values_to_remove.append(val)
                            continue
                        
                        # FIX: Intersect the sets instead of requiring equality
                        # If {7, 8} became {7, 8, 9}, we keep {7, 8}.
                        common_ids = set(curr_ids) & set(next_groups[val])
                        
                        if common_ids:
                            curr_groups[val] = common_ids
                        else:
                            values_to_remove.append(val)

                    for val in values_to_remove:
                        del curr_groups[val]
                    if not curr_groups: types_to_remove.append(t_name)
                for t_name in types_to_remove:
                    del current_types[t_name]
                if not current_types: del common_rule[key]

            # --- 3. Intersect Diagonal Alignments ---
            key = 'diag_align'
            if key in common_rule:
                current_types = common_rule[key]
                next_types = next_ctx.get(key, {})
                types_to_remove = []
                for t_name, curr_lines_list in current_types.items():
                    if t_name not in next_types:
                        types_to_remove.append(t_name)
                        continue
                    
                    curr_lines = {frozenset(l) for l in curr_lines_list}
                    next_lines = {frozenset(l) for l in next_types[t_name]}
                    common_lines = curr_lines & next_lines
                    
                    if common_lines:
                        current_types[t_name] = [set(fs) for fs in common_lines]
                    else:
                        types_to_remove.append(t_name)
                for t_name in types_to_remove:
                    del current_types[t_name]
                if not current_types: del common_rule[key]

        return common_rule

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
    
    def _get_concrete_signature(self, event: dict):
        """
        Returns a tuple representing the exact event identity: (Type, ID, Value).
        Used to check if the exact same thing happened under different circumstances.
        """
        e_type = event['type']
        obj_id = event.get('id')
        
        val = None
        if e_type == 'MOVED':
            val = event.get('vector')
        elif e_type == 'RECOLORED':
            val = event.get('to_color')
        elif e_type in ['GROWTH', 'SHRINK']:
            val = event.get('pixel_delta')
        elif e_type == 'TRANSFORM':
            val = event.get('to_fingerprint')
        elif e_type == 'SHAPE_CHANGED':  # <--- WAS MISSING
            val = event.get('to_fingerprint')
        elif e_type == 'NEW':
            # For NEW, the "ID" is the location/appearance, not the assigned string
            obj_id = event.get('position') 
            val = (event.get('color'), event.get('size'))
        elif e_type == 'REMOVED':
            val = 'removed'

        # Fallback for unknown types to prevent 'None' values from merging distinct events
        if val is None and e_type not in ['REMOVED', 'NEW']:
             val = 'unknown_change'

        return (e_type, obj_id, val)

    def _get_object_state(self, obj_summary: dict) -> tuple:
        """
        Extracts the Intrinsic State signature for an object.
        Used to define the 'Start State' for scientific comparison.
        INCLUDES: Color, Fingerprint (Shape), Size, Pixels (Mass).
        EXCLUDES: Position, IDs, Relationships (these are Context, not Identity).
        """
        return (
            obj_summary['color'], 
            obj_summary['fingerprint'], 
            obj_summary['size'],
            obj_summary['pixels']
        )
    
    def _get_abstract_signature(self, event: dict):
        """
        Returns (Type, Value). Ignores the specific Object ID.
        Used to detect phenomena like 'Something always turns Red' regardless of target.
        """
        e_type = event['type']
        
        val = None
        if e_type == 'MOVED':
            val = event.get('vector')
        elif e_type == 'RECOLORED':
            val = event.get('to_color')
        elif e_type in ['GROWTH', 'SHRINK']:
            val = event.get('pixel_delta')
        elif e_type == 'TRANSFORM':
            val = event.get('to_fingerprint')
        elif e_type == 'SHAPE_CHANGED':
            val = event.get('to_fingerprint')
        elif e_type == 'NEW':
            # For Abstract NEW, we only care about WHAT appeared (Color/Size), not WHERE
            val = (event.get('color'), event.get('size'))
        elif e_type == 'REMOVED':
            val = 'removed'

        if val is None: 
            val = 'unknown'
            
        return (e_type, val)

    def _analyze_result(self, action_key: tuple, events: list[dict], full_context: dict):
        """
        Unified Learner.
        Updates the rule for the current outcome.
        CRITICAL FIX: Prevents "Universal Failure" learning.
        If the intersection of failures results in an empty/generic rule,
        we discard the rule and keep the raw contexts as specific 'Landmines'.
        """
        # 1. Create Fingerprint (Empty tuple = Failure)
        hashable_events = []
        if events:
            for event in events:
                stable_event_tuple = tuple(sorted(event.items()))
                hashable_events.append(stable_event_tuple)
        outcome_fingerprint = tuple(sorted(hashable_events))

        # 2. Get Hypothesis Structure
        hypothesis = self.rule_hypotheses.setdefault(action_key, {})
        
        # 3. Get or Initialize Outcome Data
        if outcome_fingerprint not in hypothesis:
            hypothesis[outcome_fingerprint] = {
                'contexts': [],
                'rule': None,
                'raw_events': events
            }
        
        outcome_data = hypothesis[outcome_fingerprint]
        outcome_data['contexts'].append(full_context)
        
        # 4. Update the Consistency Rule (The Intersection)
        candidate_rule = self._find_common_context(outcome_data['contexts'])

        # --- LANDMINE LOGIC START ---
        # If this is a FAILURE (empty events) and the rule is EMPTY (generic),
        # we reject the rule. We do not want to learn "Everything Fails".
        if not events and not candidate_rule:
             outcome_data['rule'] = None # No rule. Use raw contexts (Landmines).
             if self.debug_channels['HYPOTHESIS']:
                 print(f"  [Failure Analysis] Intersection is generic. Keeping {len(outcome_data['contexts'])} specific landmines instead of a rule.")
        else:
             outcome_data['rule'] = candidate_rule
             if self.debug_channels['HYPOTHESIS']:
                lbl = "FAILURE" if not events else "SUCCESS"
                print(f"  Learned {lbl} Consistency Rule for {action_key} (seen {len(outcome_data['contexts'])} times).")
        # --- LANDMINE LOGIC END ---

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

    def _context_matches_pattern(self, current_context: dict, rule: dict) -> bool:
        """
        Checks if the 'current_context' (the live screen) satisfies 
        EVERY global condition in the 'rule'.
        """
        if not rule:
            # Empty rule = "No specific conditions required" = Matches everything
            return True

        try:
            # 1. Check Global Adjacencies (e.g. Is Obj 7 touching Obj 12?)
            for key in ['adj', 'diag_adj']:
                if key not in rule: continue
                rule_map = rule[key]
                curr_map = current_context.get(key, {})
                
                for obj_id, rule_contacts in rule_map.items():
                    # If the rule says Obj 7 exists, it must exist now
                    if obj_id not in curr_map: return False
                    
                    # If the rule says Obj 7 touches Obj 12, it must touch Obj 12 now
                    if curr_map[obj_id] != rule_contacts: return False

            # 2. Check Global Relationships (e.g. Is Obj 5 the same color as Obj 9?)
            for key in ['rels', 'align', 'match']:
                if key not in rule: continue
                rule_types = rule[key]
                curr_types = current_context.get(key, {})
                
                for t_name, rule_groups in rule_types.items():
                    if t_name not in curr_types: return False
                    curr_groups = curr_types[t_name]
                    
                    for val, rule_ids in rule_groups.items():
                        if val not in curr_groups: return False
                        # The group membership must be identical
                        if set(curr_groups[val]) != set(rule_ids): return False

            # 3. Check Diagonal Alignments
            key = 'diag_align'
            if key in rule:
                rule_types = rule[key]
                curr_types = current_context.get(key, {})
                for t_name, rule_lines in rule_types.items():
                    if t_name not in curr_types: return False
                    
                    curr_lines_fs = {frozenset(l) for l in curr_types[t_name]}
                    for line in rule_lines:
                        if frozenset(line) not in curr_lines_fs: return False

            return True

        except Exception:
            return False

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

    def _predict_outcome(self, hypothesis_key: tuple, current_context: dict) -> tuple[list|None, int]:
        """
        Checks the current context against ALL learned consistency rules.
        
        CRITICAL UPDATE: "Local Landmine" Priority.
        We ALWAYS check the raw failure history for a direct match on the target object.
        If the target object (ID/Color/Shape/Size) is identical to a past failure,
        we predict Failure immediately, overriding any broad intersection rules.
        """
        hypothesis = self.rule_hypotheses.get(hypothesis_key)
        if not hypothesis:
            return None, 0
        
        matches = []
        
        # Extract the target object from the current context for local comparison.
        target_id = hypothesis_key[1]
        current_target_obj = None
        if target_id:
            for obj in current_context['summary']:
                if obj['id'] == target_id:
                    current_target_obj = obj
                    break

        for fingerprint, data in hypothesis.items():
            rule = data['rule']
            
            # --- 1. Local Landmine Check (High Priority) ---
            is_local_landmine = False

            # --- MODIFIED: Only run this check for ACTION6 (clicks) ---
            action_name_str = hypothesis_key[0]
            if action_name_str.startswith('ACTION6') and current_target_obj and not data['raw_events']: # Only check if this is a Failure outcome
                 for past_ctx in data['contexts']:
                    past_target_obj = None
                    
            if current_target_obj and not data['raw_events']: # Only check if this is a Failure outcome
                 for past_ctx in data['contexts']:
                    past_target_obj = None
                    # Optimization: We assume past_ctx structure holds the summary
                    for obj in past_ctx['summary']:
                        if obj['id'] == target_id:
                            past_target_obj = obj
                            break
                    
                    if past_target_obj:
                        # Strict check: If intrinsic properties are identical, it's the same "Dead Object"
                        if (current_target_obj['color'] == past_target_obj['color'] and
                            current_target_obj['fingerprint'] == past_target_obj['fingerprint'] and
                            current_target_obj['size'] == past_target_obj['size']):
                            
                            is_local_landmine = True
                            break
            
            if is_local_landmine:
                matches.append(data)
                continue # We found a match, move to next hypothesis item

            # --- 2. Standard Rule Match ---
            # If it wasn't a direct landmine hit, check the learned rule (if any)
            if rule is not None:
                if self._context_matches_pattern(current_context, rule):
                    matches.append(data)

        if not matches:
            return None, 0
        
        # --- Success Selection ---
        def get_rule_complexity(match_data):
            r = match_data.get('rule', {})
            if not r: return 0
            score = 0
            for k in ['adj', 'diag_adj']: score += len(r.get(k, {}))
            for k in ['rels', 'align', 'match']:
                for t, groups in r.get(k, {}).items(): score += len(groups)
            score += len(r.get('diag_align', {}))
            return score

        matches.sort(key=lambda m: (
            get_rule_complexity(m), 
            len(m['contexts'])
        ), reverse=True)

        best_match = matches[0]
        return best_match['raw_events'], len(best_match['contexts'])

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
                    # FIX: Unpack tuple (events, confidence)
                    predicted_event_list, confidence = self._predict_outcome(hypothesis_key, current_full_context)
                    
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
                    elif predicted_outcome_fingerprint in self.seen_outcomes and confidence >= 2:
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
                    # FIX: Unpack tuple (events, confidence)
                    predicted_event_list, confidence = self._predict_outcome(hypothesis_key, current_full_context)
                    
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
                    elif predicted_outcome_fingerprint in self.seen_outcomes and confidence >= 2:
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

    def _recursive_lookahead_score(self, current_summary: list[dict], action_template, target_obj, depth: int) -> tuple:
        """
        Recursively simulates moves to find the long-term value of an action.
        Returns a score tuple: (Has_Unknowns, Has_Discoveries, -Failures, -Steps_To_Reward)
        """
        MAX_DEPTH = 10  # Don't look deeper than X turns to save CPU/prevent loops
        
        # 1. Build Context for Prediction
        # We need to re-analyze relationships for the simulation to be accurate
        (rels, adj, diag_adj, match, align, diag_align, conj) = self._analyze_relationships(current_summary)
        
        context = {
            'summary': current_summary,
            'rels': rels, 'adj': adj, 'diag_adj': diag_adj,
            'align': align, 'diag_align': diag_align, 'match': match
        }
        
        # 2. Predict Outcome of THIS move
        target_id = target_obj['id'] if target_obj else None
        action_key = self._get_learning_key(action_template.name, target_id)
        
        # We need to predict for ALL objects (Global) or just the Target
        predicted_events = []
        
        if target_id:
            # Targeted Action
            base_key = (action_key.split('_')[0], target_id)
            
            # Check Failure Landmines first (Veto)
            if action_key in self.failure_patterns and self._context_matches_pattern(context, self.failure_patterns[action_key]):
                return (0, 0, -1, 0) # Immediate Failure
            
            events, conf = self._predict_outcome(base_key, context)
            if events is None: return (1, 0, 0, 0) # Found an Unknown! Immediate reward.
            if not events: return (0, 0, -1, 0) # Failure
            predicted_events = events
            
            # Check if this specific outcome is a Discovery
            hashable = tuple(sorted([tuple(sorted(e.items())) for e in events]))
            if hashable not in self.seen_outcomes or conf < 2:
                return (0, 1, 0, 0) # Found a Discovery!
                
        else:
            # Global Action (simulate for all objects)
            # For simplicity in lookahead, we just check if ANY object triggers Unknown/Discovery
            total_events = []
            for obj in current_summary:
                base_key = (action_key, obj['id'])
                events, conf = self._predict_outcome(base_key, context)
                
                if events is None: return (1, 0, 0, 0) # Unknown
                if events:
                    hashable = tuple(sorted([tuple(sorted(e.items())) for e in events]))
                    if hashable not in self.seen_outcomes or conf < 2:
                        return (0, 1, 0, 0) # Discovery
                    total_events.extend(events)
            
            if not total_events: return (0, 0, -1, 0) # All objects failed
            predicted_events = total_events

        # 3. If we are here, the move was "Boring" (Success, but known).
        # We must look deeper.
        
        if depth >= MAX_DEPTH:
            return (0, 0, 0, 0) # Hit bottom, nothing found.

        # 4. Simulate the Future State
        future_summary = self._get_hypothetical_summary(current_summary, predicted_events)
        
        # 5. Find the BEST move from this future state
        # (We reuse the logic from choose_action, simplified)
        best_future_score = (-999, -999, -999, -999)
        
        # Construct next possible moves
        next_moves = []
        if not future_summary: return (0, 0, 0, 0) # Empty board

        click_template = None
        if action_template.name == 'ACTION6': click_template = action_template # Reuse current template if it's ACTION6
        
        # Only test ACTION6 on objects for speed in recursion
        if click_template:
            for obj in future_summary:
                next_moves.append((click_template, obj))
        
        # Evaluate next moves
        for next_action, next_obj in next_moves:
            # RECURSE: Add 1 to depth
            score = self._recursive_lookahead_score(future_summary, next_action, next_obj, depth + 1)
            
            # The score tuple is (U, D, -F, -Steps). 
            # We want to maximize U, D, and minimize F and Steps.
            # Current score format doesn't account for steps, let's adjust it.
            # If score found something (U>0 or D>0), we subtract depth from a "Urgency" score.
            
            found_u, found_d, neg_f, neg_steps = score
            
            # Decay the value slightly for being further away
            current_recursive_score = (found_u, found_d, neg_f, neg_steps - 1)
            
            if current_recursive_score > best_future_score:
                best_future_score = current_recursive_score
                
        return best_future_score

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

        # --- FIX: Allow Singleton Groups (len >= 1) ---
        # We must allow unique properties (e.g. "The only Green object") to be part of the rule.
        for rel_type, groups in rel_data.items():
            relationships[rel_type] = {
                value: ids for value, ids in groups.items() if len(ids) >= 1
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
    
    def _detect_pixel_overlaps(self, active_obj: dict, previous_summary: list[dict]) -> list[str]:
        """
        Checks if 'active_obj' (from current frame) intersects with any object 
        from the PREVIOUS frame (excluding itself).
        
        Used to detect 'Virtual Interactions' where one object moves on top of another,
        causing the bottom object to disappear (occlusion or consumption).
        
        Returns: List of object IDs from the PREVIOUS frame that are being overlapped.
        """
        overlapped_ids = []
        
        # The set of absolute (row, col) coordinates for the active object
        current_pixels = active_obj['pixel_coords']
        
        for prev_obj in previous_summary:
            # 1. Skip the object itself 
            # (We don't care that the Player overlaps the Player's old position)
            if prev_obj['id'] == active_obj['id']:
                continue
            
            # 2. Check for intersection
            # We use isdisjoint() for efficiency. If sets are NOT disjoint, they overlap.
            if not current_pixels.isdisjoint(prev_obj['pixel_coords']):
                overlapped_ids.append(prev_obj['id'])
                
        return overlapped_ids

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
            
    def _get_phenomenal_signature(self, event: dict):
        """
        Returns (Type, ID). Ignores the specific Value/Result.
        Used to detect if a specific object is 'unstable' across different actions.
        """
        e_type = event['type']
        obj_id = event.get('id')
        
        # Normalize unstable appearance events (The "Cycle" detector)
        if e_type in ['TRANSFORM', 'SHAPE_CHANGED', 'RECOLORED']:
            e_type = 'APPEARANCE_CHANGE'
        
        # Normalize size changes
        elif e_type in ['GROWTH', 'SHRINK']:
            e_type = 'SIZE_CHANGE'
            
        # For NEW events, the 'ID' is essentially the spawning phenomenon itself
        if e_type == 'NEW':
             return ('NEW', 'ANY')
            
        return (e_type, obj_id)

    def _classify_event_stream(self, current_events: list[dict], current_action_key: str, current_context: dict) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Classifies events using the 'Survivor System' (Hypothesis Elimination).
        
        Hypotheses:
        1. SPECIFIC GLOBAL: The exact result (ID+Value) happens regardless of action.
        2. DIRECT: The result is consistent AND exclusive to this action.
        3. ABSTRACT GLOBAL: The phenomenon (Value) happens regardless of action.
        
        Priority: Specific Global > Direct > Abstract Global.
        """
        direct_events = []
        global_events = []
        ambiguous_events = [] 
        
        # --- CRITICAL: Treat every target as a unique Action Family ---
        action_family = current_action_key 
        
        target_id_from_action = None
        if 'ACTION6_' in current_action_key:
            target_id_from_action = current_action_key.replace('ACTION6_', '')

        self.performed_action_types.add(action_family)
        if current_events:
            self.productive_action_types.add(action_family)
            
        current_trial_id = self.last_action_id

        transitions_to_analyze = []
        changed_obj_ids = {e['id']: e for e in current_events if 'id' in e}
        
        # A. Build Transitions (Explicit Changes)
        for event in current_events:
            if 'id' not in event: continue
            obj_id = event['id']
            prev_obj = next((o for o in self.last_object_summary if o['id'] == obj_id), None)
            if prev_obj:
                start_state = self._get_object_state(prev_obj)
                end_state_sig = self._get_concrete_signature(event)
                transitions_to_analyze.append({
                    'event': event, 'start': start_state, 'end': end_state_sig
                })

        # B. Implicit Events (Failures/No Change)
        if target_id_from_action and target_id_from_action not in changed_obj_ids:
            prev_obj = next((o for o in self.last_object_summary if o['id'] == target_id_from_action), None)
            if prev_obj:
                start_state = self._get_object_state(prev_obj)
                end_state_sig = ('NO_CHANGE', target_id_from_action, None)
                self._update_transition_memory(start_state, action_family, end_state_sig, current_trial_id)

        # C. Run The Survivor System
        for item in transitions_to_analyze:
            start = item['start']
            end = item['end'] # (Type, ID, Value)
            event = item['event']
            
            self._update_transition_memory(start, action_family, end, current_trial_id)
            
            history = self.transition_counts.get(start, {})
            this_action_history = history.get(action_family, {})
            
            # --- 1. Initialize Hypotheses ---
            is_specific_global = True
            is_abstract_global = True
            is_direct = True
            
            current_specific = end
            current_abstract = (end[0], end[2])

            # --- 2. Internal Consistency Check (Zero Tolerance) ---
            if len(this_action_history) > 1:
                is_specific_global = False
                is_abstract_global = False
                is_direct = False

            # --- 3. External Control Check ---
            control_group_found = False
            
            if is_direct or is_specific_global or is_abstract_global:
                for other_action, outcomes in history.items():
                    if other_action == action_family: continue
                    control_group_found = True
                    
                    for other_end in outcomes:
                        other_specific = other_end
                        other_abstract = (other_end[0], other_end[2])
                        
                        # KILL GLOBAL: If other action produced a DIFFERENT result
                        if other_specific != current_specific:
                            is_specific_global = False
                        if other_abstract != current_abstract:
                            is_abstract_global = False
                            
                        # KILL DIRECT: If other action produced the SAME result (Loss of Exclusivity)
                        if other_specific == current_specific:
                            is_direct = False

            # --- 4. Final Verdict ---
            
            if not control_group_found:
                if not (is_specific_global or is_abstract_global or is_direct):
                     pass # Fall through to Exception Solver
                else:
                    ambiguous_events.append({
                        'event': event, 'reason': "Correlation Only",
                        'fix': "Needs Negative Control (Divergence) to prove Causality."
                    })
                    continue

            # Case A: The Stipulation (All Dead / Exception)
            if not (is_specific_global or is_abstract_global or is_direct):
                
                # Check for Global Pattern
                is_global_pattern = False
                for other_action, outcomes in history.items():
                    if other_action == action_family: continue
                    for other_end in outcomes:
                        if (other_end[0], other_end[2]) == current_abstract:
                            is_global_pattern = True; break
                    if is_global_pattern: break

                # Zoom Out: Try to solve the condition
                condition_str = self._solve_conditional_rule(start, action_family, end, current_context)
                
                if condition_str:
                    event['condition'] = condition_str
                    success_trials = this_action_history.get(end, set())
                    if len(success_trials) >= 2:
                        if is_global_pattern:
                            event['_abstract_global'] = True
                            global_events.append(event)
                        else:
                            direct_events.append(event) 
                    else:
                        ambiguous_events.append({
                            'event': event, 'reason': "Exception Hypothesis (N=1)", 
                            'fix': f"Found potential condition '{condition_str}'. Needs replication."
                        })
                elif is_global_pattern:
                    event['_abstract_global'] = True
                    event['condition'] = "(Time/Cycle Driven)" 
                    global_events.append(event)
                else:
                    reason = "Contradiction Found"
                    fix = "Action produces variable results. Needs Context Refinement."
                    ambiguous_events.append({'event': event, 'reason': reason, 'fix': fix})
                continue

            # Case B: Specific Global (Priority 1)
            if is_specific_global:
                global_events.append(event)
                continue

            # Case C: Direct Survivor (Priority 2)
            if is_direct:
                success_trials = this_action_history.get(end, set())
                if len(success_trials) >= 2:
                    direct_events.append(event)
                else:
                    ambiguous_events.append({
                        'event': event, 'reason': "New Event (N=1)",
                        'fix': "Hypothesis verified but needs sample size (N>=2)."
                    })
                continue
            
            # Case D: Abstract Global (Priority 3)
            if is_abstract_global:
                event['_abstract_global'] = True 
                global_events.append(event)
                continue

            # Case E: Ambiguous Overlap
            ambiguous_events.append({
                'event': event, 'reason': "Ambiguous Overlap",
                'fix': "Data supports conflicting hypotheses."
            })

        return direct_events, global_events, ambiguous_events

    def _update_transition_memory(self, start, action, end, trial_id):
        """
        Helper to update the nested transition dictionary safely.
        STRICT: Only logs the specific action provided. No pooling.
        """
        if start not in self.transition_counts:
            self.transition_counts[start] = {}
        if action not in self.transition_counts[start]:
            self.transition_counts[start][action] = {}
        
        if end not in self.transition_counts[start][action]:
            self.transition_counts[start][action][end] = set()
            
        # Add the unique ID of this action instance.
        # The set automatically handles duplicates (e.g., multiple events in one turn = N=1).
        self.transition_counts[start][action][end].add(trial_id)

    def _resolve_ambiguous_events(self, ambiguous_wrappers: list[dict], current_events: list[dict], 
                                  current_summary: list[dict], prev_summary: list[dict],
                                  current_adj: dict, prev_adj: dict, learning_key: str):
        """
        The Resolution Phase.
        Analyzes Ambiguous events to checks for REACTIVE triggers (State-Dependent).
        """
        promoted_direct_events = []
        
        if not ambiguous_wrappers:
            return promoted_direct_events

        # Identify "Agitators" (Objects that moved this turn)
        mover_ids = {e['id'] for e in current_events if e['type'] == 'MOVED'}
        movers = [obj for obj in current_summary if obj['id'] in mover_ids]

        for wrapper in ambiguous_wrappers:
            event = wrapper['event'] # Extract raw event from the wrapper
            victim_id = event.get('id')
            if not victim_id: continue
            
            # --- TEST 1: REACTIVE CHECK (The "State" Test) ---
            
            # Case A: Overlap (Bulldozer)
            if event['type'] == 'REMOVED':
                for mover in movers:
                    overlapped_ids = self._detect_pixel_overlaps(mover, prev_summary)
                    if victim_id in overlapped_ids:
                        if self.debug_channels['HYPOTHESIS']:
                            print(f"  [Resolution] EXPLAINED REACTIVE: {victim_id} was REMOVED because {mover['id']} overlapped it.")
                        break

            # Case B: Overlap (Self-Move)
            elif victim_id in mover_ids:
                actor = next((obj for obj in current_summary if obj['id'] == victim_id), None)
                if actor:
                    overlapped_ids = self._detect_pixel_overlaps(actor, prev_summary)
                    if overlapped_ids:
                        if self.debug_channels['HYPOTHESIS']:
                            trigger_obj = overlapped_ids[0] 
                            print(f"  [Resolution] EXPLAINED REACTIVE: {victim_id} {event['type']} because it overlapped {trigger_obj}.")

            # Case C: Adjacency (Electric Fence)
            else:
                curr_contacts = set(current_adj.get(victim_id, [])); curr_contacts.discard('na')
                prev_contacts = set(prev_adj.get(victim_id, [])); prev_contacts.discard('na')
                new_contacts = curr_contacts - prev_contacts
                if new_contacts:
                    trigger_obj = list(new_contacts)[0]
                    if self.debug_channels['HYPOTHESIS']:
                        print(f"  [Resolution] EXPLAINED REACTIVE: {victim_id} {event['type']} because it touched {trigger_obj}.")

        return promoted_direct_events
    
    def _format_rule_description(self, hypothesis_key: tuple) -> str:
        """
        Returns a readable rule string using Background Subtraction.
        Explicitly prints Object IDs for every condition to verify bindings.
        """
        hypothesis = self.rule_hypotheses.get(hypothesis_key)
        if not hypothesis: return "(No rule learned yet)"
            
        best_outcome = None
        max_seen = -1
        for fingerprint, data in hypothesis.items():
            if data['raw_events'] and len(data['contexts']) > max_seen:
                max_seen = len(data['contexts'])
                best_outcome = data
        
        if not best_outcome: return "(History: Consistently No Change)"
        rule = best_outcome['rule']
        if not rule: return "IF (Always/Unconditional)"

        # --- Calculate Static Background (Intersection of ALL history) ---
        static_context = None
        if self.level_state_history:
            # Sample start, middle, end
            indices = {0, len(self.level_state_history)-1}
            if len(self.level_state_history) > 5: indices.add(len(self.level_state_history)//2)
            sample_history = [self.level_state_history[i] for i in indices]
            
            static_context = sample_history[0]
            for ctx in sample_history[1:]:
                static_context = self._intersect_contexts(static_context, ctx)

        def is_static(key, sub_key, val):
            if not static_context: return False
            if key not in static_context: return False
            if sub_key not in static_context[key]: return False
            return static_context[key][sub_key] == val

        conditions = []
        
        # Format Adjacencies (Subtracting Static)
        for key in ['adj', 'diag_adj']:
            if key in rule:
                for obj, contacts in rule[key].items():
                    if is_static(key, obj, contacts): continue
                    specifics = [c for c in contacts if c != 'x' and c != 'na']
                    if specifics: conditions.append(f"Adj({obj} to {specifics})")

        # Format Relationships/Alignments (Subtracting Static)
        for key in ['rels', 'align', 'match']:
            if key in rule:
                for type_name, groups in rule[key].items():
                    for val, ids in groups.items():
                        # Check if this exact group exists in background
                        is_group_static = False
                        if static_context and key in static_context:
                            static_types = static_context[key]
                            if type_name in static_types:
                                if val in static_types[type_name] and static_types[type_name][val] == ids:
                                    is_group_static = True
                        if not is_group_static:
                            # --- THE FIX: Print the IDs involved! ---
                            # e.g. turns "Color=3" into "Color=3(['obj_34'])"
                            id_list_str = str(list(ids))
                            conditions.append(f"{type_name}={val}{id_list_str}")
        
        if 'diag_align' in rule:
             for type_name in rule['diag_align']:
                conditions.append(f"DiagAlign({type_name})")

        if not conditions: return "IF (Generic/Background Context)"
        return "IF " + " AND ".join(conditions)

    def _solve_conditional_rule(self, start_state, action_family, current_end_sig, current_context) -> str | None:
        """
        The 'Exception Solver'.
        Compares Current Context vs. Historical Conflict Context.
        Checks Adjacencies, Alignments, and Match Groups for a differentiator.
        Returns a simple condition string, or None if the difference is too complex.
        """
        history = self.transition_counts.get(start_state, {}).get(action_family, {})
        
        conflicting_end = None
        conflicting_trial_ids = set()
        
        for end_sig, trials in history.items():
            if end_sig != current_end_sig:
                conflicting_end = end_sig
                conflicting_trial_ids = trials
                break 
        
        if not conflicting_end or not conflicting_trial_ids: return None

        past_trial_id = max(conflicting_trial_ids)
        history_index = past_trial_id - 1
        
        if history_index < 0 or history_index >= len(self.level_state_history): return None
            
        past_context = self.level_state_history[history_index]
        
        diffs = []
        
        # --- 1. Check Adjacency Differences ---
        curr_adj = current_context.get('adj', {})
        past_adj = past_context.get('adj', {})
        
        all_ids = set(curr_adj.keys()) | set(past_adj.keys())
        for obj_id in all_ids:
            c_contacts = set(curr_adj.get(obj_id, [])); c_contacts.discard('na')
            p_contacts = set(past_adj.get(obj_id, [])); p_contacts.discard('na')
            
            if c_contacts != p_contacts:
                added = c_contacts - p_contacts
                if added: diffs.append(f"Adj({obj_id} to {list(added)})")
                
                removed = p_contacts - c_contacts
                if removed: diffs.append(f"NOT Adj({obj_id} to {list(removed)})")

        # --- 2. Check Alignment Differences (Position Context) ---
        # This solves cases where "The Object in the Center" works but "The Object in the Corner" fails.
        curr_align = current_context.get('align', {})
        past_align = past_context.get('align', {})
        
        # We iterate over alignment types (center_x, top_y, etc.)
        all_types = set(curr_align.keys()) | set(past_align.keys())
        for align_type in all_types:
            c_groups = curr_align.get(align_type, {})
            p_groups = past_align.get(align_type, {})
            
            # Check if the involved objects have different alignment values
            all_coords = set(c_groups.keys()) | set(p_groups.keys())
            for coord in all_coords:
                c_ids = c_groups.get(coord, set())
                p_ids = p_groups.get(coord, set())
                
                if c_ids != p_ids:
                    # Simplify: Just report the property existence for now to avoid clutter
                    # We assume the diff is relevant if it involves the target implicitly
                    # (We can't filter by target ID easily here as start_state doesn't have it, 
                    # but strict diffs usually isolate the active object).
                    if len(diffs) < 3: # Heuristic limit
                        added = c_ids - p_ids
                        if added: diffs.append(f"{align_type}={coord}{list(added)}")

        # --- 3. Check Match Group Differences (Global Context) ---
        # This solves "It only works if a Blue object exists"
        curr_match = current_context.get('match', {})
        past_match = past_context.get('match', {})
        
        for m_type in set(curr_match.keys()) | set(past_match.keys()):
            c_groups = curr_match.get(m_type, {})
            p_groups = past_match.get(m_type, {})
            
            for props, c_ids in c_groups.items():
                p_ids = p_groups.get(props, [])
                if set(c_ids) != set(p_ids):
                     if len(diffs) < 3:
                        diffs.append(f"{m_type}={props}") # e.g. "Color=5 Group Changed"

        # Garbage Filter: If it's too complex, it's probably a hidden variable we can't see yet.
        if len(diffs) > 3:
            return None 

        if diffs: return " AND ".join(diffs)
        return None