from .agent import Agent, FrameData
from .structs import GameAction, GameState
from collections import deque
import copy
import ast
from typing import Optional
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
        self.run_counter = 0
        self.last_score = 0
        self.banned_action_keys = set()
        self.permanent_banned_actions = set()
        self.successful_click_actions = set()
        self.last_grid_size = (64, 64) # Default, updates on perceive

        # --- NEW: CLASSIFICATION MEMORY ---
        self.concrete_witness_registry = {}
        self.abstract_witness_registry = {}
        self.phenomenal_witness_registry = {}

        # --- NEW: Session/Trial Tracking ---
        self.global_action_counter = 0  # Tracks the passing of time/trials
        self.last_action_id = 0         # ID of the action that caused the current state

        # --- NEW: DETERMINISTIC SCIENTIFIC MEMORY ---
        # 1. The Truth Table: Stores raw results of experiments.
        # Structure: self.truth_table[Scientific_State][Action_Key] = {Result_Sig: [Turn_ID_List]}
        self.truth_table = {}

        # 2. Certified Laws: Rules that passed Positive & Negative control tests.
        # Key: (Scientific_State, Action_Key) -> Value: {'type': 'DIRECT'|'GLOBAL', 'result': Result}
        self.certified_laws = {}

        # 3. State Refinements (The Splitter): Tracks how we define "State".
        # Key: Base_Intrinsic_Sig -> Value: List of Context Keys (e.g. ['adj_top', 'match_Color'])
        self.state_refinements = {}

        # 4. Global Invariants: Rules that apply to EVERYTHING (e.g. Gravity).
        self.global_invariants = {}

        self.global_action_counter = 0
        self.last_action_id = 0
        
        self.concrete_event_counts = {}
        self.phenomenal_event_counts = {}
        
        self.action_consistency_counts = {} 
        self.performed_action_types = set()
        self.productive_action_types = set() # Tracks actions that caused change

        # --- NEW: Relational Color Memory ---
        # Stores discovered laws like: ('ACTION6_obj_1', 'obj_5') -> {'Source: Adjacency(0,1)'}
        self.relational_constraints = {}

        # --- Stability Tracking ---
        self.stability_streak = 0
        self.forcing_wait = False
        self.last_stable_pixels = None

        # --- NEW: Global Forecasting Memory ---
        self.global_event_history = {}  # Map: AbstractSig -> List of Turn IDs (When did it happen?)
        self.global_precursors = {}     # Map: AbstractSig -> Precursor Context (What signaled it?)
        self.global_cycles = {}         # Map: AbstractSig -> Period (int) (How often does it happen?)
        # --------------------------------------

        # --- NEW: Proven Law Registry ---
        # Stores rules that have passed the 100% rigor test.
        # Key: (Action_Family, Result_Signature) -> Value: Classification ('DIRECT', 'GLOBAL')
        self.proven_rules = {} 
        # --------------------------------

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

        # --- NEW: Log File Setup ---
        self.log_file_path = "game_debug.log"
        
        # Overwrite the file at the start of each run
        try:
            with open(self.log_file_path, "w") as f:
                f.write("--- NEW RUN STARTED ---\n")
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")

    def _log_to_file(self, message: str):
        """Appends a message to the debug log file."""
        try:
            with open(self.log_file_path, "a") as f:
                f.write(message + "\n")
        except Exception:
            pass # Fail silently to not crash the agent

    def _print_and_log(self, message: str):
        """Prints to console AND appends to the debug log file."""
        print(message) # Show you
        try:
            with open(self.log_file_path, "a") as f:
                f.write(message + "\n") # Show me
        except Exception:
            pass

    def _reset_agent_memory(self):
        """
        Resets all agent learning and memory to a clean state.
        This is called at the start of a new game.
        """

        # --- NEW: Log File Reset ---
        # Ensure the log is wiped clean at the beginning of every run/seed.
        if hasattr(self, 'log_file_path'):
            try:
                with open(self.log_file_path, "w") as f:
                    f.write("--- NEW RUN STARTED ---\n")
            except Exception:
                pass

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
        self.performed_action_types = {}

        self.truth_table = {}
        self.certified_laws = {}
        self.state_refinements = {}
        self.global_invariants = {}
        self.level_state_history = []
        self.global_action_counter = 0
        
        # Reset per-level state
        self._reset_level_state()

    def _reset_level_state(self):
        """Resets only the memory for the current level."""
        self.object_id_counter = 0
        self.performed_action_types = {}
        self.productive_action_types = set()
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

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Analyzes the current frame, checks for stability, and logs perceived changes.
        """
        # --- 1. Game Start / Game Over Logic ---
        if latest_frame.state == GameState.NOT_PLAYED:
            self._reset_agent_memory()
            self.run_counter += 1 # New Run
            return GameAction.RESET
        
        if latest_frame.state == GameState.GAME_OVER:
            # --- NEW: Scientific Death Analysis ---
            # Instead of just resetting, we treat this as a "Terminal Event" for the previous action.
            if self.last_action_context and self.last_object_summary:
                learning_key = self.last_action_context
                
                # Reconstruct the context of the world *before* we died
                prev_context = {
                    'summary': self.last_object_summary,
                    'rels': self.last_relationships,
                    'adj': self.last_adjacencies,
                    'diag_adj': self.last_diag_adjacencies,
                    'align': self.last_alignments,
                    'diag_align': self.last_diag_alignments,
                    'match': self.last_match_groups
                }
                
                # Broadcast the "Death" event to every object in memory.
                # The Survivor System will filter out which object's state actually *correlated* with the death.
                terminal_events = []
                for obj in self.last_object_summary:
                    terminal_events.append({'type': 'TERMINAL', 'outcome': 'LOSS', 'id': obj['id']})
                
                if self.debug_channels['FAILURE']:
                    print(f"\n--- GAME OVER DETECTED ---")
                    print(f"Injecting TERMINAL events for analysis on action {learning_key}...")

                # Run the standard classification pipeline
                direct, global_ev, ambiguous = self._classify_event_stream(terminal_events, learning_key, prev_context)
                
                # Learn from the classified results
                # A. Direct (Action-Specific Death)
                obj_events_map = {obj['id']: [] for obj in self.last_object_summary}
                for event in direct:
                    if 'id' in event and event['id'] in obj_events_map:
                        obj_events_map[event['id']].append(event)
                
                for obj in self.last_object_summary:
                    hypothesis_key = (learning_key, obj['id'])
                    self._analyze_result(hypothesis_key, obj_events_map[obj['id']], prev_context)

                # B. Global (State-Specific Death)
                for event in global_ev:
                    abst_sig = self._get_abstract_signature(event)
                    global_key = ('GLOBAL', str(abst_sig)) 
                    self._analyze_result(global_key, [event], prev_context)
            # --------------------------------------

            self._reset_level_state()
            self.forcing_wait = False
            self.run_counter += 1 # New Run
            return GameAction.RESET

        # --- 2. Animation Guard: If server says busy, we wait ---
        if not latest_frame.available_actions:
            return None

        # --- 3. Stability Check (Robust Streak Logic) ---
        current_pixels = latest_frame.frame[0] if latest_frame.frame else []
        
        # A. Force wait after any action to see result
        if self.forcing_wait:
            self.forcing_wait = False
            self.last_stable_pixels = current_pixels
            self.stability_streak = 0
            return None 
            
        # B. Compare pixels to last known state
        is_visually_identical = (self.last_stable_pixels is not None and 
                                 current_pixels == self.last_stable_pixels)

        if not is_visually_identical:
            if self.debug_channels['CHANGES'] and self.stability_streak > 0:
                print("--- Stability Broken: Screen moved. Resetting streak. ---")
            self.last_stable_pixels = current_pixels
            self.stability_streak = 0
            return None # Wait for stability
        
        else:
            # Screen is stable. Increment streak.
            self.stability_streak += 1
            REQUIRED_STREAK = 1 # Wait for 2 consecutive stable frames (~1.0s)
            
            if self.stability_streak < REQUIRED_STREAK:
                return None # Wait for confirmation
        
        # C. Authorized to ACT. Next time we must wait again.
        self.forcing_wait = True 
        self.last_stable_pixels = current_pixels
        # -----------------------------------------------------
        
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
                self._print_and_log("--- Change Log ---")
                for change in changes:
                    self._print_and_log(change)

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

                # --- NEW: Pre-calculate Physics for Classification ---
                # We run this BEFORE classification so the notes are available 
                # for the classifier and debug logs.
                for e in events:
                    # UPDATED: Allow Mass Changes (Growth/Shrink) to enter Physics
                    if e['type'] in ['SHAPE_CHANGED', 'TRANSFORM', 'GROWTH', 'SHRINK']:
                        # Unpack Tuple
                        result = self._check_interaction_physics(e['id'], events, prev_context, current_summary)
                        if result:
                            note, agitators = result
                            e['_physics_note'] = note
                            e['_physics_agitators'] = agitators
                # -----------------------------------------------------

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
                    
                    # --- NEW: Update Global History ---
                    if abst_sig not in self.global_event_history:
                        self.global_event_history[abst_sig] = []
                    
                    # Avoid duplicates for the same turn
                    if self.last_action_id not in self.global_event_history[abst_sig]:
                        self.global_event_history[abst_sig].append(self.last_action_id)
                    # ----------------------------------

                    global_key = ('GLOBAL', str(abst_sig)) 
                    self._analyze_result(global_key, [event], prev_context)

                # --- NEW: Run Global Forecasting Analysis ---
                self._analyze_global_patterns()

                # --- NEW: Rescue Ambiguous Events via Relational Logic ---
                # UPDATED: Disabled. We do not rescue ambiguous events with hypotheses.
                # We strictly wait for the Truth Table to certify them.
                # ---------------------------------------------------------

                # --- DETAILED PRINTING (Now uses updated brain) ---
                if self.debug_channels['CHANGES']:
                    
                    # Define map for logging lookups
                    prev_obj_map = {o['id']: o for o in prev_context['summary']}
                    
                    def _fmt_val(e):
                        if e.get('is_until'):
                            val = e.get('vector') if 'vector' in e else f"+{e.get('pixel_delta')}"
                            action_verb = "Move" if 'vector' in e else "Grow"
                            return f"{action_verb} {val} (Until {e['until_cond']})"
                        
                        if e.get('is_absolute'):
                            if 'vector' in e: return f"Move to {e['abs_coords']}"
                            if 'to_size' in e: return f"Grow to Size {e['abs_coords']}"
                        
                        suffix = ""
                        if e.get('_is_variable'): suffix = " (Variable Magnitude/Shape)"
                        
                        if 'vector' in e: return f"Move {e['vector']}{suffix}"
                        if 'to_color' in e: return f"Recolor to {e['to_color']}{suffix}"
                        if 'pixel_delta' in e: return f"Size {'+' if e['pixel_delta']>0 else ''}{e['pixel_delta']}{suffix}"
                        if 'to_fingerprint' in e: return f"Shape {e['to_fingerprint']}{suffix}"
                        return e['type']

                    if global_events: 
                        self._print_and_log(f"  -> Filtered {len(global_events)} Global events:")
                        for e in global_events:
                            obj_id = e.get('id')
                            if not obj_id or obj_id not in prev_obj_map: continue
                            
                            prev_obj = prev_obj_map[obj_id]
                            state_sig = self._get_scientific_state(prev_obj, prev_context)
                            
                            # Check for Global Sequence Law
                            law = self.certified_laws.get((state_sig, 'ANY'))
                            
                            self._print_and_log(f"     * [GLOBAL] {e['type']} on {e.get('id', 'Unknown')}")

                            if law and law.get('type') == 'GLOBAL_SEQUENCE':
                                seq_str = " -> ".join(map(str, law['sequence']))
                                self._print_and_log(f"       [Explanation] Global Cycle: Target ID Pattern {seq_str}")
                                self._print_and_log(f"       [Prediction]  Next Target ID: {law.get('next_prediction', 'Unknown')}")
                            else:
                                result_sig = self._get_concrete_signature(e)
                                # FIX: Use 'learning_key' instead of 'action_key'
                                rule_str = self._format_rule_description('GLOBAL', learning_key, state_sig, result_sig)
                                self._print_and_log(f"       [Explanation] Global Rule: Inevitable '{_fmt_val(e)}'.")
                                self._print_and_log(f"       [Prediction]  IF {rule_str}")

                    if direct_events: 
                        self._print_and_log(f"  -> Processing {len(direct_events)} Direct events:")
                        for e in direct_events:
                            obj_id = e.get('id')
                            if not obj_id or obj_id not in prev_obj_map: continue
                            
                            prev_obj = prev_obj_map[obj_id]
                            state_sig = self._get_scientific_state(prev_obj, prev_context)

                            # Check for Direct Sequence Law
                            law = self.certified_laws.get((state_sig, learning_key))
                            
                            self._print_and_log(f"     * [DIRECT] {e['type']} on {e.get('id', 'Unknown')}")
                            if '_physics_note' in e:
                                self._print_and_log(f"       [Physics]     {e['_physics_note']}")
                            
                            if law and law.get('type') == 'DIRECT_SEQUENCE':
                                seq_str = " -> ".join(map(str, law['sequence']))
                                self._print_and_log(f"       [Explanation] Direct Cycle: Value Pattern {seq_str}")
                                
                                # Calculate next value for display (replicating prediction logic)
                                current_val = None
                                r_type = law['result_type']
                                if r_type == 'RECOLORED': current_val = prev_obj['color']
                                elif r_type in ['SHAPE_CHANGED', 'TRANSFORM']: current_val = prev_obj['fingerprint']
                                elif r_type in ['GROWTH', 'SHRINK']: current_val = prev_obj['pixels']
                                
                                next_val = "Unknown"
                                if current_val in law['sequence']:
                                    idx = law['sequence'].index(current_val)
                                    next_val = law['sequence'][(idx + 1) % len(law['sequence'])]
                                
                                self._print_and_log(f"       [Prediction]  Next Value: {next_val}")
                            else:
                                result_sig = self._get_concrete_signature(e)
                                rule_str = self._format_rule_description('DIRECT', learning_key, state_sig, result_sig)
                                self._print_and_log(f"       [Explanation] Direct Causality: Action consistently causes '{_fmt_val(e)}'")
                                self._print_and_log(f"       [Prediction]  IF {rule_str}")

                    if ambiguous_events:
                        self._print_and_log(f"  -> Ignored {len(ambiguous_events)} Ambiguous events:")
                        for wrapper in ambiguous_events:
                            e = wrapper['event']
                            reason = wrapper['reason']
                            fix = wrapper['fix']
                            detail = wrapper.get('detail', '')
                            
                            # 1. Get the Readable Value (e.g., "Move (0, -1)")
                            val_str = _fmt_val(e)

                            self._print_and_log(f"     * [AMBIGUOUS] {e['type']} on {e.get('id', 'Unknown')}")
                            
                            # 2. Print WHAT happened (The "New" observation)
                            self._print_and_log(f"       [Observed] {val_str}")
                            
                            self._print_and_log(f"       [Status]   {reason}")
                            if detail:
                                self._print_and_log(f"       [Detail]   {detail}")
                                
                            # 3. REMOVED: [Context] dump (too verbose)
                            
                            self._print_and_log(f"       [Needs]    {fix}")
                
                # --- Failsafe: Track banned actions ---
                # UPDATED: Success = Any change detected (events) OR score increase.
                action_succeeded = bool(events) or score_increased
                
                # --- NEW: Track Productive Actions (ACTION6 Only) ---
                if action_succeeded and self.last_action_context and self.last_action_context.startswith('ACTION6'):
                    self.productive_action_types.add(self.last_action_context)
                # -------------------------------------

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
                # --- NEW: Permanent Ignore for Dead Clicks ---
                # If we have clicked this specific object ID before (performed),
                # and it has NEVER caused a change (not productive),
                # we remove it from consideration entirely. It won't even go to the profiler.
                action_key = self._get_learning_key(click_action_template.name, obj['id'])
                
                if (action_key in self.performed_action_types and 
                    action_key not in self.productive_action_types):
                    continue 
                # ---------------------------------------------
                
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
                    # --- NEW: Check for Terminality (The "Safety Layer") ---
                    is_lethal = False
                    for e in predicted_events:
                        if e['type'] == 'TERMINAL' and e.get('outcome') == 'LOSS':
                            is_lethal = True
                            break
                    
                    if is_lethal:
                        # Treat as a catastrophic failure (high cost to sort to bottom)
                        profile['failures'] += 1000 
                    # -------------------------------------------------------
                    else:
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
            self._print_and_log("\n--- Full Profile List (Before Sort) ---")
            if not move_profiles:
                self._print_and_log("  (No moves to profile)")
            
            # Helper for sorting logic (Defined here to use in both display and actual sort)
            def get_sort_key(move_tuple):
                move = move_tuple[0]
                key = self._get_learning_key(move['template'].name, move['object']['id'] if move['object'] else None)
                is_untried_val = 1 if key not in self.performed_action_types else 0
                return (
                    is_untried_val,
                    move_tuple[1]['unknowns'], 
                    move_tuple[1]['discoveries'], 
                    -move_tuple[1]['failures'], 
                    move_tuple[1]['boring']
                )

            # Sort for display matches actual logic
            sorted_for_print = sorted(move_profiles, key=get_sort_key, reverse=True)

            for i, (move, profile, _, _) in enumerate(sorted_for_print):
                action_name = move['template'].name
                target_name = f" on {move['object']['id']}" if move['object'] else ""
                
                prefix = "  -> " if i == 0 else "     " # Highlight the winner
                
                self._print_and_log(f"{prefix}{action_name}{target_name} -> "
                        f"U:{profile['unknowns']} D:{profile['discoveries']} "
                        f"B:{profile['boring']} F:{profile['failures']}")

        # --- Deterministic Priority-Based Sorting ---
        if move_profiles:
            def is_untried(move_tuple):
                move = move_tuple[0]
                key = self._get_learning_key(move['template'].name, move['object']['id'] if move['object'] else None)
                return 1 if key not in self.performed_action_types else 0

            # --- NEW: Recency Score (Fair Rotation) ---
            # If we must repeat an action, pick the one done LONGEST ago.
            # performed_action_types stores {key: turn_id}.
            # We want Smallest Turn ID (Oldest) to be Best.
            # Sort is Reverse=True (Biggest is Best).
            # So we return -TurnID. (-1 is better than -100).
            def get_recency_score(move_tuple):
                move = move_tuple[0]
                key = self._get_learning_key(move['template'].name, move['object']['id'] if move['object'] else None)
                last_turn = self.performed_action_types.get(key, float('inf'))
                return -last_turn

            # 1. Primary Sort
            move_profiles.sort(key=lambda x: (
                is_untried(x),        # 1. Always do new things first
                x[1]['unknowns'],     # 2. Resolve confusion
                x[1]['discoveries'],  # 3. Verify new discoveries
                -x[1]['failures'],    # 4. Avoid failures
                get_recency_score(x), # 5. NEW: Rotate through old actions (LRU)
                x[1]['boring']        # 6. Tie-breaker
            ), reverse=True)
            
            # --- Updated: Recursive Lookahead Tie-Breaker ---
            chosen_move_tuple = None
            
            # Check if the winner is "Boring" (U=0, D=0)
            top_profile = move_profiles[0][1]
            top_score_is_boring = (top_profile['unknowns'] == 0 and 
                                   top_profile['discoveries'] == 0 and
                                   top_profile['failures'] == 0)

            if top_score_is_boring:
                # Find all ties for first place.
                # CRITICAL FIX: The tie definition MUST include 'is_untried'.
                # Since the list is already sorted, we just check items until the sort key changes.
                
                tied_moves = []
                best_sort_key = (
                    is_untried(move_profiles[0]),
                    top_profile['unknowns'],
                    top_profile['discoveries'],
                    -top_profile['failures']
                )
                
                for mp in move_profiles:
                    current_key = (
                        is_untried(mp),
                        mp[1]['unknowns'],
                        mp[1]['discoveries'],
                        -mp[1]['failures']
                    )
                    
                    if current_key == best_sort_key:
                        tied_moves.append(mp)
                    else:
                        # Since list is sorted, as soon as we mismatch, we are done.
                        break
                
                if len(tied_moves) > 1:
                    if self.debug_channels['ACTION_SCORE']: 
                        print(f"\n--- Running Recursive Lookahead (Depth 3) for {len(tied_moves)} Boring Moves ---")
                    
                    best_deep_score = (-999, -999, -999, -999)
                    best_move_idx = 0
                    
                    for i, move_tuple in enumerate(tied_moves):
                        move, _, _, _ = move_tuple
                        deep_score = self._recursive_lookahead_score(current_summary, move['template'], move['object'], 0)
                        
                        if self.debug_channels['ACTION_SCORE']:
                             target = f" on {move['object']['id']}" if move['object'] else ""
                             print(f"  - Move {move['template'].name}{target} -> Chain Score: {deep_score}")

                        if deep_score > best_deep_score:
                            best_deep_score = deep_score
                            best_move_idx = i
                    
                    chosen_move_tuple = tied_moves[best_move_idx]
                else:
                    chosen_move_tuple = tied_moves[0]
            else:
                chosen_move_tuple = move_profiles[0]

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
                self._print_and_log(f"\n--- Discovery Profiler ---")
                
                # UPDATED: Print the Global Step Count (1-based index)
                # self.global_action_counter is 0-based and incremented below, so we add 1 here.
                current_step = self.global_action_counter + 1
                self._print_and_log(f"Chose: {action_name}{target_name} (Step {current_step})")
                
                self._print_and_log(f"Profile: U:{best_profile['unknowns']} D:{best_profile['discoveries']} B:{best_profile['boring']} F:{best_profile['failures']}")
        
        else:
            # Modified: If no moves found, just WAIT. Don't reset.
            action_to_return = None

        # --- Store action for next turn's analysis ---
        if action_to_return:
            learning_key_for_storage = self._get_learning_key(action_to_return.name, chosen_object_id if chosen_object else None)
            # --- UPDATED: Track Turn ID ---
            self.performed_action_types[learning_key_for_storage] = self.global_action_counter
        else:
            learning_key_for_storage = None
        
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
        self.last_grid_size = (height, width)
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
                
                # --- Extract object ID ---
                obj_id_str = ""
                if 'Object id_' in details:
                    # UPDATED: Added REAPPEARED to ID extraction
                    if change_type in ['NEW', 'REMOVED', 'REAPPEARED']: 
                         # Format: "Object id_X (ID ...)"
                        obj_id_str = details.split(' ')[1].replace('id_', '')
                    else:
                        # Format: "Object id_X moved..."
                        obj_id_str = details.split(' ')[1].replace('id_', '')
                
                if obj_id_str.isdigit():
                    event['id'] = f"obj_{obj_id_str}"

                if change_type == 'MOVED':
                    parts = details.split(' moved from ')
                    # --- FIX: Clean the coordinate string of overlap/merge notes ---
                    coord_part = parts[1]
                    if ' (overlapping' in coord_part:
                        coord_part = coord_part.split(' (overlapping')[0]
                    elif ' (merged' in coord_part:
                        coord_part = coord_part.split(' (merged')[0]
                    # ---------------------------------------------------------------
                    pos_parts = coord_part.replace('.', '').split(' to ')
                    start_pos, end_pos = ast.literal_eval(pos_parts[0]), ast.literal_eval(pos_parts[1])
                    event.update({'vector': (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])})
                    events.append(event)

                elif change_type == 'RECOLORED':
                    # Parse "color from 15 to 3" OR "color from 15 to 3, now at..."
                    from_color_str = details.split(' from ')[1].split(' to ')[0]
                    
                    # --- FIX: Robustly handle trailing position info ---
                    to_part = details.split(' to ')[1]
                    to_color_str = to_part.split(',')[0].replace('.', '')
                    # ---------------------------------------------------
                    
                    event.update({'from_color': int(from_color_str), 'to_color': int(to_color_str)})
                    events.append(event)

                elif change_type == 'SHAPE_CHANGED':
                    fp_part = details.split('fingerprint: ')[1]
                    
                    # --- FIX: Stop parsing at the closing parenthesis ---
                    fp_clean = fp_part.split(')')[0]
                    from_fp_str, to_fp_str = fp_clean.split(' -> ')
                    # ---------------------------------------------------
                    
                    event.update({'from_fingerprint': int(from_fp_str), 'to_fingerprint': int(to_fp_str)})
                    events.append(event)

                elif change_type in ['GROWTH', 'SHRINK', 'TRANSFORM']:
                    # Parse "Object id_X at (R, C)..."
                    start_pos_str = details.split(') ')[0] + ')'
                    start_pos = ast.literal_eval(start_pos_str.split(' at ')[1])
                    
                    # --- FIX: Handle optional 'now at' for stationary changes ---
                    if 'now at ' in details:
                        end_pos_str = details.split('now at ')[1].replace('.', '')
                        end_pos = ast.literal_eval(end_pos_str)
                    else:
                        end_pos = start_pos
                    # ------------------------------------------------------------
                    
                    event.update({'start_position': start_pos, 'end_position': end_pos})
                    
                    if '(from ' in details:
                        from_size_str = details.split('(from ')[1].split(' to ')[0]
                        to_size_str = details.split(' to ')[1].split(')')[0]
                        if ')' in to_size_str: to_size_str = to_size_str.split(')')[0]
                        pixel_diff = int(details.split(' by ')[1].split(' pixels')[0])
                        event.update({
                            'from_size': ast.literal_eval(from_size_str.replace('x', ',')),
                            'to_size': ast.literal_eval(to_size_str.replace('x', ',')),
                            'pixel_delta': pixel_diff if change_type == 'GROWTH' else -pixel_diff
                        })
                    
                    if 'shape (' in details:
                        fp_part = details.split('shape (')[1].split(')')[0]
                        from_fp, to_fp = fp_part.split(' -> ')
                        event.update({
                            'from_fingerprint': int(from_fp),
                            'to_fingerprint': int(to_fp)
                        })
                    
                    events.append(event)

                # --- NEW: Handler for REAPPEARED (Same format as NEW/REMOVED) ---
                elif change_type in ['NEW', 'REMOVED', 'REAPPEARED']:
                    id_str = details.split(') ')[0] + ')'
                    id_tuple = ast.literal_eval(id_str.split('ID ')[1])
                    pos_str = '(' + details.split('(')[-1].replace('.', '')
                    position = ast.literal_eval(pos_str)
                    event.update({
                        'position': position, 'fingerprint': id_tuple[0],
                        'color': id_tuple[1], 'size': id_tuple[2], 'pixels': id_tuple[3]
                    })
                    events.append(event)
                
                # --- NEW: Handler for REAPPEARED & TRANSFORMED ---
                elif change_type == 'REAPPEARED & TRANSFORMED':
                    # We just need the position for the physics engine
                    end_pos_str = details.split('now at ')[1].replace('.', '')
                    end_pos = ast.literal_eval(end_pos_str)
                    event.update({'position': end_pos})
                    events.append(event)

            except (ValueError, IndexError, SyntaxError, AttributeError):
                continue
        return events
    
    def _get_concrete_signature(self, event: dict):
        """
        Returns a tuple representing the exact event identity: (Type, ID, Value).
        """
        e_type = event['type']
        obj_id = event.get('id')
        val = None
        
        # --- Logic Fix: Prioritize Mechanism for Shape AND Mass Changes ---
        # If the Physics Engine identified a correlation (e.g. "Reveal & Occlusion"),
        # that explanation IS the consistent result we want to track.
        # We override raw values for Shape, Transform, Growth, and Shrink.
        if e_type in ['SHAPE_CHANGED', 'TRANSFORM', 'GROWTH', 'SHRINK'] and event.get('_physics_note'):
             val = event['_physics_note']
             # --- NEW: Unify Mass Changes under 'SHAPE_CHANGED' ---
             # To the scientific learner, a Reveal is a Reveal, whether it adds pixels (Growth)
             # or just alters the border (Shape Change). We force the type to match.
             e_type = 'SHAPE_CHANGED'
        
        # --- Standard Value Extraction (Fallback) ---
        elif e_type == 'MOVED':
            val = event.get('vector')
        elif e_type == 'RECOLORED':
            val = event.get('to_color')
        elif e_type in ['GROWTH', 'SHRINK']:
            val = event.get('pixel_delta')
        elif e_type in ['TRANSFORM', 'SHAPE_CHANGED']:
            # Fallback: If no physics note, the Fingerprint IS the result that matters.
            val = event.get('to_fingerprint')
        elif e_type == 'NEW':
            obj_id = event.get('position') 
            val = (event.get('color'), event.get('size'))
        elif e_type == 'REMOVED':
            val = 'removed'
        elif e_type == 'TERMINAL': 
            val = event.get('outcome')

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
        elif e_type == 'TERMINAL': # <--- NEW: Handle Terminal Events
            val = event.get('outcome')

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
        # --- NEW: Trigger Relational Learning Module ---
        self._update_relational_constraints(action_key, events, full_context)
        # -----------------------------------------------
        
        # 1. Create Fingerprint (Empty tuple = Failure)
        hashable_events = []
        if events:
            for event in events:
                # --- FIX: Ensure all values are hashable ---
                # The '_physics_agitators' field is a list, which breaks hashing.
                # We must convert any list values to tuples.
                safe_items = []
                for k, v in event.items():
                    if isinstance(v, list):
                        safe_items.append((k, tuple(v)))
                    elif isinstance(v, set):
                        # Just in case we use sets later (though we usually use frozenset)
                        safe_items.append((k, tuple(sorted(list(v)))))
                    else:
                        safe_items.append((k, v))
                
                stable_event_tuple = tuple(sorted(safe_items))
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
    
    def _analyze_global_patterns(self):
        """
        The Global Forecaster.
        Detects inevitable events driven by Time (Cycles) or State (Precursors).
        Updates self.global_cycles and self.global_precursors.
        """
        min_occurrences_for_pattern = 2  # Scientific Rigor: Need 2 confirmations
        
        for sig, turn_ids in self.global_event_history.items():
            if len(turn_ids) < min_occurrences_for_pattern:
                continue
                
            # --- 1. The Chronometer (Periodicity Check) ---
            turns = sorted(turn_ids)
            intervals = [turns[i+1] - turns[i] for i in range(len(turns)-1)]
            
            # Check if all intervals are identical (Perfect Cycle)
            if len(set(intervals)) == 1:
                period = intervals[0]
                self.global_cycles[sig] = period
                if self.debug_channels['HYPOTHESIS']:
                    print(f"  [Global Forecaster] Discovered Cycle: Event {sig} happens every {period} turns.")
            
            # --- 2. The Oracle (Precursor Signal Check) ---
            # We need the contexts from the turn *before* each event.
            relevant_contexts = []
            valid_signal = True
            
            for t_id in turn_ids:
                # Turn ID 1 is index 0 in history. 
                # Pre-event context for Turn T is history[T-2].
                ctx_idx = t_id - 2 
                if ctx_idx >= 0 and ctx_idx < len(self.level_state_history):
                    relevant_contexts.append(self.level_state_history[ctx_idx])
                else:
                    valid_signal = False; break
            
            if valid_signal and relevant_contexts:
                # Find the intersection of all pre-event states
                common_precursor = self._find_common_context(relevant_contexts)
                
                # If specific, we have a signal
                if common_precursor and (common_precursor.get('adj') or common_precursor.get('match') or 
                                         common_precursor.get('align') or common_precursor.get('rels')):
                    self.global_precursors[sig] = common_precursor
                    if self.debug_channels['HYPOTHESIS']:
                        print(f"  [Global Forecaster] Discovered Signal: Event {sig} is preceded by a specific state.")

    def _predict_outcome(self, hypothesis_key: tuple, current_context: dict) -> tuple[list|None, int]:
        """
        Scientific Prediction. Only predicts if a Certified Law exists.
        """
        action_name, target_id = hypothesis_key
        
        # 1. Identify the Object and its Scientific State
        target_obj = None
        if target_id:
            target_obj = next((o for o in current_context['summary'] if o['id'] == target_id), None)
        
        if not target_obj:
            return None, 0
        
        state_sig = self._get_scientific_state(target_obj, current_context)
        
        # 2. Check for Certified Laws
        law = self.certified_laws.get((state_sig, action_name)) # Direct
        if not law:
            law = self.certified_laws.get((state_sig, 'ANY'))   # Global

        # 3. Return Prediction
        if law:
            # --- CASE A: Standard Global/Direct Law ---
            if 'result' in law:
                result_sig = law['result']
                return self._expand_result(result_sig, target_obj), 100
            
            # --- CASE B: Sequential Global Law ---
            elif law.get('type') == 'GLOBAL_SEQUENCE':
                # We need to construct a 'result_sig' from the sequence prediction
                # Format: (Type, ID, Value)
                # Note: Sequence laws usually predict the VALUE, but apply to the CURRENT object
                # if the sequence logic was based on values. 
                # However, our sequence detector tracks TARGET IDs.
                
                # If the sequence predicts the next Target ID, we only act if THIS object is the target.
                # But _predict_outcome is called FOR a specific object.
                
                # Actually, our sequence detector stored 'next_prediction' which is the Target ID.
                next_target_id = law.get('next_prediction')
                
                # If this object is the one predicted to change:
                if next_target_id is not None:
                     # Check if the current object's ID matches the prediction
                     # We stored numeric IDs in the sequence logic (e.g. 7).
                     # target_obj['id'] is 'obj_7'.
                     current_numeric_id = int(target_obj['id'].split('_')[-1])
                     
                     if current_numeric_id == next_target_id:
                         # This object is CHOSEN.
                         # Construct the result signature using the stored type/value
                         r_type = law['result_type']
                         r_val = law['result_value']
                         
                         # (Type, ID, Value)
                         # We use the object's own ID because we are predicting FOR it.
                         result_sig = (r_type, target_obj['id'], r_val)
                         return self._expand_result(result_sig, target_obj), 100
                     else:
                         # This object is NOT the one in the sequence.
                         # It should NOT change (regarding this law).
                         return [], 100
            
            # --- NEW CASE C: Sequential Direct Law (Value Cycles) ---
            elif law.get('type') == 'DIRECT_SEQUENCE':
                r_type = law['result_type']
                seq = law['sequence']
                
                # 1. Determine current value based on the law type
                current_val = None
                if r_type == 'RECOLORED': current_val = target_obj['color']
                elif r_type in ['SHAPE_CHANGED', 'TRANSFORM']: current_val = target_obj['fingerprint']
                elif r_type in ['GROWTH', 'SHRINK']: current_val = target_obj['pixels'] # Or size, depending on tracking
                
                # 2. Find next step in cycle
                if current_val in seq:
                    idx = seq.index(current_val)
                    next_val = seq[(idx + 1) % len(seq)]
                    
                    # Construct result signature
                    result_sig = (r_type, target_obj['id'], next_val)
                    return self._expand_result(result_sig, target_obj), 100
                else:
                    # Current value not in known cycle? Assume cycle break or start of cycle.
                    # Default to first in sequence or fail. Let's return None (Unknown).
                    return None, 0

        return None, 0

    def _expand_result(self, result_sig, target_obj):
        """Converts signature back to event list."""
        
        # Handle Abstract Signature (Type, Val) - Used for Global Laws
        if len(result_sig) == 2:
             r_type, r_val = result_sig
             r_id = target_obj['id'] # Global laws apply to the object in question
        else:
             # Handle Concrete Signature (Type, ID, Val) - Used for Direct Laws
             r_type, r_id, r_val = result_sig
        
        if r_type == 'NO_CHANGE': return []
        
        event = {'id': target_obj['id'], 'type': r_type}
        if r_type == 'MOVED': event['vector'] = r_val
        elif r_type == 'RECOLORED': event['to_color'] = r_val
        elif r_type in ['GROWTH', 'SHRINK']: event['pixel_delta'] = r_val
        elif r_type in ['TRANSFORM', 'SHAPE_CHANGED']: event['to_fingerprint'] = r_val
        elif r_type == 'TERMINAL': event['outcome'] = r_val
        
        return [event]

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
            return [], []

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
                    # --- NEW: Strict Overlap Veto ---
                    # Logic: If the object at this position looks completely different (properties changed),
                    # AND there exists a perfect match for this new look elsewhere in the old frame,
                    # we assume the perfect match MOVED here (Overlap), rather than this object transforming.
                    old_stable_id = self._get_stable_id(old_obj)
                    new_stable_id = self._get_stable_id(new_obj)
                    
                    if old_stable_id != new_stable_id:
                        perfect_match_exists_elsewhere = False
                        for other_old in old_summary:
                            if other_old['id'] == old_obj['id']: continue # Skip self
                            # If we find someone else who matches the new blob perfectly...
                            if self._get_stable_id(other_old) == new_stable_id:
                                perfect_match_exists_elsewhere = True
                                break
                        
                        if perfect_match_exists_elsewhere:
                            # VETO: We found a better candidate (a mover). 
                            # Skip this position match so Pass 2 (Move Detection) can claim it.
                            continue 
                    # --------------------------------

                    # Propagate the persistent ID
                    new_obj['id'] = old_obj['id']
                    
                    color_changed = old_obj['color'] != new_obj['color']
                    shape_changed = (old_obj['fingerprint'] != new_obj['fingerprint'] or
                                     old_obj['size'] != new_obj['size'] or
                                     old_obj['pixels'] != new_obj['pixels'])
                    
                    # --- Explicit Growth/Shrink with Shape Tracking ---
                    pixel_diff = new_obj['pixels'] - old_obj['pixels']
                    size_str_old = f"{old_obj['size'][0]}x{old_obj['size'][1]}"
                    size_str_new = f"{new_obj['size'][0]}x{new_obj['size'][1]}"
                    
                    shape_str = f"shape ({old_obj['fingerprint']} -> {new_obj['fingerprint']})"
                    
                    if pixel_diff > 0:
                        changes.append(
                            f"- GROWTH: Object {old_obj['id'].replace('obj_', 'id_')} at {old_obj['position']} "
                            f"grew by {pixel_diff} pixels (from {size_str_old} to {size_str_new}) and {shape_str}."
                        )
                    elif pixel_diff < 0:
                        changes.append(
                            f"- SHRINK: Object {old_obj['id'].replace('obj_', 'id_')} at {old_obj['position']} "
                            f"shrank by {abs(pixel_diff)} pixels (from {size_str_old} to {size_str_new}) and {shape_str}."
                        )
                    else:
                        # Mass conserved
                        color_changed = old_obj['color'] != new_obj['color']
                        shape_changed = old_obj['fingerprint'] != new_obj['fingerprint']
                        
                        if color_changed and shape_changed:
                            changes.append(f"- TRANSFORM: Object {old_obj['id'].replace('obj_', 'id_')} changed shape and color.")
                        elif color_changed:
                            changes.append(f"- RECOLORED: Object {old_obj['id'].replace('obj_', 'id_')} changed color from {old_obj['color']} to {new_obj['color']}.")
                        elif shape_changed:
                            changes.append(f"- SHAPE_CHANGED: Object {old_obj['id'].replace('obj_', 'id_')} changed shape (fingerprint: {old_obj['fingerprint']} -> {new_obj['fingerprint']}).")

                    matches_to_remove.append((old_obj, new_obj))
                    processed_new_objs_in_pass1.add(id(new_obj))
                    break  # Move to the next old_obj
        
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
                new_inst['id'] = old_inst['id']
                if old_inst['position'] != new_inst['position']:
                    # --- NEW: Detect Overlap in Move Log ---
                    # If this move landed on a spot occupied by an 'old_unexplained' object 
                    # (one that hasn't moved or matched), it is an overlap/occlusion.
                    overlap_note = ""
                    for incumbent in old_unexplained:
                        # Don't check against self (though self should already be popped from instances, 
                        # checking ID ensures safety if logic shifts)
                        if incumbent['id'] == old_inst['id']: continue 

                        if incumbent['position'] == new_inst['position']:
                            overlap_note = f" (overlapping id_{incumbent['id'].replace('obj_', '')})"
                            # We found the victim.
                            break
                    
                    changes.append(f"- MOVED: Object {old_inst['id'].replace('obj_', 'id_')} moved from {old_inst['position']} to {new_inst['position']}{overlap_note}.")
                    # ---------------------------------------

                moves_to_remove.append((old_inst, new_inst))
        
        matched_old_in_pass2 = {id(o) for o, n in moves_to_remove}
        matched_new_in_pass2 = {id(n) for o, n in moves_to_remove}
        old_unexplained = [obj for obj in old_unexplained if id(obj) not in matched_old_in_pass2]
        new_unexplained = [obj for obj in new_unexplained if id(obj) not in matched_new_in_pass2]

        # --- NEW: Merger Detection (Before Fuzzy Match) ---
        # Detects if a "New" object is actually an Old object + a Revealed Ghost
        # Case: obj_35 moves, reveals obj_32, and they fuse into obj_46.
        mergers_to_remove = []
        
        for new_obj in new_unexplained:
             # We are looking for: New = Old + Ghost
             # Candidates must match color
             potential_movers = [o for o in old_unexplained if o['color'] == new_obj['color']]
             
             match_found = False
             for mover in potential_movers:
                  mass_diff = new_obj['pixels'] - mover['pixels']
                  if mass_diff <= 0: continue
                  
                  # Look for a ghost in memory that matches the missing mass/color
                  ghost_candidate_id = None
                  ghost_stable_id = None
                  
                  for stable_id in list(self.removed_objects_memory.keys()):
                       # stable_id format: (fingerprint, color, size, pixels)
                       # Check Color (idx 1) and Pixels (idx 3)
                       if stable_id[1] == new_obj['color'] and stable_id[3] == mass_diff:
                            ghost_stable_id = stable_id
                            if self.removed_objects_memory[stable_id]:
                                ghost_candidate_id = self.removed_objects_memory[stable_id][0] # Peek ID
                            break
                  
                  if ghost_candidate_id:
                       # Match Found! 
                       # 1. Log the requested MERGED event (Narrative Only - No ID claim)
                       # NEW: Stop after the ingredients.
                       changes.append(f"- MERGED: Object {mover['id'].replace('obj_', 'id_')} moved and merged with revealed Object {ghost_candidate_id.replace('obj_', 'id_')}.")
                       
                       # 2. Log a MOVED event for the learner (Scientific)
                       changes.append(f"- MOVED: Object {mover['id'].replace('obj_', 'id_')} moved from {mover['position']} to {new_obj['position']} (merged).")
                       
                       mergers_to_remove.append((mover, new_obj, ghost_stable_id))
                       match_found = True
                       break
             
             if match_found: continue
             
        # Cleanup merged objects so they don't trigger SHRINK/NEW in later passes
        for m, n, g_sid in mergers_to_remove:
             if m in old_unexplained: old_unexplained.remove(m)
             if n in new_unexplained: new_unexplained.remove(n)
             # Consume the ghost from memory
             if g_sid in self.removed_objects_memory:
                  self.removed_objects_memory[g_sid].popleft()
                  if not self.removed_objects_memory[g_sid]:
                       del self.removed_objects_memory[g_sid]
        # --------------------------------------------------

        # --- Pass 3: Fuzzy Matching for GROWTH and SHRINK events (Moved Objects) ---
        if old_unexplained and new_unexplained:
            potential_pairs = []
            for old_obj in old_unexplained:
                for new_obj in new_unexplained:
                    pos1 = old_obj['position']
                    pos2 = new_obj['position']
                    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    potential_pairs.append({'old': old_obj, 'new': new_obj, 'dist': distance})
            
            potential_pairs.sort(key=lambda p: p['dist'])
            
            matched_old = set()
            matched_new = set()
            for pair in potential_pairs:
                old_obj, new_obj = pair['old'], pair['new']
                if id(old_obj) not in matched_old and id(new_obj) not in matched_new:
                    new_obj['id'] = old_obj['id']
                    old_pixels = old_obj['pixels']
                    new_pixels = new_obj['pixels']
                    
                    if new_pixels > old_pixels:
                        event_type = "GROWTH"
                    elif old_pixels > new_pixels:
                        event_type = "SHRINK"
                    else:
                        event_type = "TRANSFORM" # Default

                    pixel_diff = abs(new_obj['pixels'] - old_obj['pixels'])
                    old_size_str = f"{old_obj['size'][0]}x{old_obj['size'][1]}"
                    new_size_str = f"{new_obj['size'][0]}x{new_obj['size'][1]}"
                    
                    if event_type == "GROWTH":
                        details = f"grew by {pixel_diff} pixels (from {old_size_str} to {new_size_str})"
                    elif event_type == "SHRINK":
                        details = f"shrank by {pixel_diff} pixels (from {old_size_str} to {new_size_str})"
                    else:
                        # Mass conserved
                        changed_parts = []
                        if old_obj['fingerprint'] != new_obj['fingerprint']:
                            changed_parts.append("shape")
                        if old_obj['color'] != new_obj['color']:
                            changed_parts.append(f"color from {old_obj['color']} to {new_obj['color']}")
                        
                        size_changed = old_obj['size'] != new_obj['size']
                        size_details = f" (size {old_size_str} -> {new_size_str})" if size_changed else ""
                        
                        # --- NEW: Specific Labels for Pass 3 ---
                        if len(changed_parts) == 1 and "color" in changed_parts[0] and not size_changed:
                             event_type = "RECOLORED"
                             details = changed_parts[0] # "color from X to Y"
                        elif len(changed_parts) == 1 and "shape" in changed_parts[0] and not size_changed:
                             event_type = "SHAPE_CHANGED"
                             details = f"shape (fingerprint: {old_obj['fingerprint']} -> {new_obj['fingerprint']})"
                        else:
                             details = f"transformed{size_details}" if not changed_parts else f"changed { ' and '.join(changed_parts) }{size_details}"
                        # ---------------------------------------

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

            for obj in still_unmatched:
                self.object_id_counter += 1
                new_id = f'obj_{self.object_id_counter}'
                obj['id'] = new_id
                stable_id = self._get_stable_id(obj)
                changes.append(f"- NEW: Object {new_id.replace('obj_', 'id_')} (ID {stable_id}) appeared at {obj['position']}.")

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
            if self.debug_channels['CONTEXT_DETAILS']:
                self._print_and_log("\n--- Relationship Change Log ---")
                for line in sorted(output_lines):
                    self._print_and_log(line)
                self._print_and_log("")

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
            if self.debug_channels['CONTEXT_DETAILS']:
                self._print_and_log("\n--- Adjacency Change Log ---")
                for line in output_lines:
                    self._print_and_log(line)
                self._print_and_log("")

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
            if self.debug_channels['CONTEXT_DETAILS']:
                self._print_and_log("\n--- Diagonal Adjacency Change Log ---")
                for line in output_lines:
                    self._print_and_log(line)
                self._print_and_log("")

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
            if self.debug_channels['CONTEXT_DETAILS']:
                title = "Diagonal Alignment Change Log" if is_diagonal else "Alignment Change Log"
                self._print_and_log(f"\n--- {title} ---")
                for line in sorted(output_lines):
                    self._print_and_log(line)
                self._print_and_log("")

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
            if self.debug_channels['CONTEXT_DETAILS']:
                self._print_and_log("\n--- Match Type Change Log ---")
                for line in sorted(output_lines):
                    self._print_and_log(line)
                self._print_and_log("")
            
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

    def _classify_event_stream(self, current_events: list[dict], action_key: str, prev_context: dict) -> tuple[list[dict], list[dict], list[dict]]:
        """
        The Scientific Method Loop (2-Pass Inheritance).
        Updated with RUN-AWARE Cycle Detection to handle Resets.
        """
        direct_events = []
        global_events = []
        ambiguous_events = []

        prev_obj_map = {o['id']: o for o in prev_context['summary']}
        processed_ids = set()
        
        # Temp storage for Pass 2
        deferred_physics_events = [] 
        id_classification_map = {} 

        # --- Step 1: Update Truth Table ---
        for event in current_events:
            obj_id = event.get('id')
            if not obj_id or obj_id not in prev_obj_map: continue
            
            processed_ids.add(obj_id)
            prev_obj = prev_obj_map[obj_id]
            
            state_sig = self._get_scientific_state(prev_obj, prev_context)
            result_sig = self._get_concrete_signature(event)
            self._update_truth_table(state_sig, action_key, result_sig, obj_id=obj_id)

        # Ingest Negative Data
        for obj in prev_context['summary']:
            if obj['id'] not in processed_ids:
                state_sig = self._get_scientific_state(obj, prev_context)
                self._update_truth_table(state_sig, action_key, ('NO_CHANGE', None, None), obj_id=obj['id'])

        # Certify Laws
        affected_states = {self._get_scientific_state(prev_obj_map[id], prev_context) for id in processed_ids}
        for obj in prev_context['summary']:
            affected_states.add(self._get_scientific_state(obj, prev_context))
        for state_sig in affected_states:
            self._verify_and_certify(state_sig)

        # --- Step 4: Classification ---
        
        # PASS 1: Standard Events
        for event in current_events:
            if event.get('_physics_note'):
                deferred_physics_events.append(event)
                continue
                
            classification = 'AMBIGUOUS'
            obj_id = event.get('id')
            if not obj_id or obj_id not in prev_obj_map: continue

            prev_obj = prev_obj_map[obj_id]
            state_sig = self._get_scientific_state(prev_obj, prev_context)
            current_result_sig = self._get_concrete_signature(event)
            
            # --- Analysis: Count Observations ---
            obj_result_turns = set()
            obj_any_result_turns = set() 
            
            if state_sig in self.truth_table and action_key in self.truth_table[state_sig]:
                for r_sig, entry_list in self.truth_table[state_sig][action_key].items():
                    for entry in entry_list:
                        # Entry can be (turn, oid) [Legacy] or (run, turn, oid) [New]
                        # Normalize to (run, turn, oid)
                        if len(entry) == 3:
                            r_id, t_id, o_id = entry
                        else:
                            r_id, t_id, o_id = 0, entry[0], entry[1]

                        if o_id == obj_id:
                            # Unique ID for counting is now (Run, Turn)
                            uniq_key = (r_id, t_id)
                            obj_any_result_turns.add(uniq_key)
                            if r_sig == current_result_sig:
                                obj_result_turns.add(uniq_key)

            is_direct_mature = len(obj_result_turns) >= 2
            
            direct_rule = self.certified_laws.get((state_sig, action_key))
            global_rule = self.certified_laws.get((state_sig, 'ANY'))
            
            # --- DECISION TREE ---
            
            # 1. Check Global Laws
            if global_rule:
                if global_rule['type'] == 'GLOBAL_SEQUENCE':
                    event_abs = self._get_abstract_signature(event)
                    if global_rule['result_type'] == event_abs[0]:
                        classification = 'GLOBAL'
                        global_events.append(event)
                elif global_rule['type'] == 'GLOBAL':
                    if is_direct_mature: 
                        classification = 'GLOBAL'
                        global_events.append(event)
                    else:
                        reason = "Unverified Global Application"
                        fix = "Test DIFFERENT actions to confirm Global Independence."
                        detail = f"Matches Global Law, but observed only {len(obj_result_turns)} times on this object."
                        ambiguous_events.append({'event': event, 'reason': reason, 'fix': fix, 'detail': detail})
                        id_classification_map[obj_id] = 'AMBIGUOUS'
                        continue

            # 2. Check Direct Laws
            if classification == 'AMBIGUOUS' and direct_rule:
                 if direct_rule['type'] == 'DIRECT':
                     if is_direct_mature:
                        classification = 'DIRECT'
                        direct_events.append(event)
                     else:
                        reason = "Unverified Direct Law"
                        fix = "Repeat SAME action to confirm Direct Causality."
                        detail = f"Matches Direct Law, but observed only {len(obj_result_turns)} times."
                        ambiguous_events.append({'event': event, 'reason': reason, 'fix': fix, 'detail': detail})
                        id_classification_map[obj_id] = 'AMBIGUOUS'
                        continue
                 elif direct_rule['type'] == 'DIRECT_SEQUENCE':
                     classification = 'DIRECT'
                     direct_events.append(event)

            # 3. True Ambiguity / Deviation
            if classification == 'AMBIGUOUS':
                fix = "Repeat SAME action to test Direct Causality."
                
                # Intelligent Status Reporting
                if len(obj_any_result_turns) > 1:
                    if len(obj_result_turns) == 1:
                         reason = "New Outcome Detected"
                         detail = f"Action repeated {len(obj_any_result_turns)} times. Previous outcomes differed."
                    else:
                         # --- PENDING CYCLE CHECK (RUN-AWARE) ---
                         is_cyclic = False
                         cycle_vals = []
                         confidence = ""
                         
                         history_entries = []
                         
                         # Collect Cross-State History
                         for s_sig_iter, acts in self.truth_table.items():
                             if action_key in acts:
                                 for r_sig_iter, t_list in acts[action_key].items():
                                     if r_sig_iter[0] == 'NO_CHANGE': continue
                                     val = r_sig_iter[2]
                                     for entry in t_list:
                                         # Normalize Entry
                                         if len(entry) == 3: r_id, t_id, o_id = entry
                                         else: r_id, t_id, o_id = 0, entry[0], entry[1]

                                         if o_id == obj_id:
                                             history_entries.append({'run': r_id, 'turn': t_id, 'val': val})
                         
                         # Sort by (Run, Turn)
                         history_entries.sort(key=lambda x: (x['run'], x['turn']))
                         
                         # Check Cycle Consistency (Respecting Resets)
                         consistent_flips = 0
                         last_val = None
                         last_run = None
                         
                         for entry in history_entries:
                             curr_val = entry['val']
                             curr_run = entry['run']
                             
                             if last_val is not None:
                                 # Only check flip if we are in the SAME run
                                 if curr_run == last_run:
                                     if curr_val != last_val:
                                         consistent_flips += 1
                                     else:
                                         # Same value in same run -> Breaks the "Simple Flip" hypothesis
                                         # (Unless period > 2, but for ambiguity check we look for flips)
                                         consistent_flips = -999 
                                 else:
                                     # Reset detected! We ignore the transition.
                                     pass
                             
                             last_val = curr_val
                             last_run = curr_run

                         if consistent_flips >= 2: # At least A->B->A
                             is_cyclic = True
                             confidence = "Strong"
                         elif consistent_flips == 1: # A->B
                             is_cyclic = True
                             confidence = "Weak"
                         
                         if is_cyclic:
                             reason = f"Potential Cycle Detected ({confidence})"
                             detail = "Outcomes alternate consistently within runs."
                         else:
                             reason = "Conflicting Results"
                             detail = "Outcomes vary within a single run without a clear pattern."
                         # ---------------------------
                else:
                    # N=1 Logic (Unchanged)
                    total_history = 0
                    for s_sig_iter, acts in self.truth_table.items():
                        if action_key in acts:
                             for res_map in acts[action_key].values():
                                 for entry in res_map: # Iterating list of tuples
                                    # We don't need to parse tuple here, just find obj_id
                                    o_id = entry[-1] # Obj ID is always last
                                    if o_id == obj_id: total_history += 1
                    
                    if total_history > 1:
                        reason = "New State Detected"
                        detail = "First observation for this Object in this specific State."
                    else:
                        reason = "Hypothesis Created (N=1)"
                        detail = "First observation. Causality is unknown."

                if direct_rule or global_rule:
                     reason = "Deviation from Law"
                     detail = "Observed event differs from the certified law for this state."
                     fix = "Analyze context for new Splitter variables."

                ambiguous_events.append({'event': event, 'reason': reason, 'fix': fix, 'detail': detail})
            
            if classification != 'AMBIGUOUS':
                id_classification_map[obj_id] = classification

        # PASS 2: Physics (Unchanged)
        pending_physics = list(deferred_physics_events)
        MAX_PHYSICS_PASSES = 3
        for _ in range(MAX_PHYSICS_PASSES):
            if not pending_physics: break
            next_pending = []
            progress_made = False
            for event in pending_physics:
                agitators = event.get('_physics_agitators', [])
                has_direct_cause = False
                has_global_cause = False
                has_ambiguous_cause = False
                all_unknown = True
                for ag_id in agitators:
                    ag_class = id_classification_map.get(ag_id)
                    if ag_class: all_unknown = False
                    if ag_class == 'DIRECT': has_direct_cause = True
                    elif ag_class == 'GLOBAL': has_global_cause = True
                    elif ag_class == 'AMBIGUOUS': has_ambiguous_cause = True
                
                if has_direct_cause:
                    direct_events.append(event)
                    id_classification_map[event['id']] = 'DIRECT'
                    progress_made = True
                elif has_ambiguous_cause:
                    next_pending.append(event)
                elif has_global_cause and not all_unknown:
                    global_events.append(event)
                    id_classification_map[event['id']] = 'GLOBAL'
                    progress_made = True
                else:
                    next_pending.append(event)
            pending_physics = next_pending
            if not progress_made: break
        
        for event in pending_physics:
            agitators = event.get('_physics_agitators', [])
            ambiguous_events.append({'event': event, 'reason': "Ambiguous Causality", 'fix': "Confirm movement rules of agitators.", 'detail': f"Change correlated with ambiguous movement of {agitators}."})

        return direct_events, global_events, ambiguous_events

    def _update_truth_table(self, state_sig, action_key, result_sig, obj_id=None):
        """
        Records an observation in the Truth Table with Run ID + Turn ID.
        """
        if state_sig not in self.truth_table:
            self.truth_table[state_sig] = {}
        
        if action_key not in self.truth_table[state_sig]:
            self.truth_table[state_sig][action_key] = {}
            
        if result_sig not in self.truth_table[state_sig][action_key]:
            self.truth_table[state_sig][action_key][result_sig] = []
            
        # Store (RunID, TurnID, ObjID)
        # We use getattr/default just in case frame number isn't set yet
        turn_num = getattr(self, 'last_frame_number', 0)
        entry = (self.run_counter, turn_num, obj_id)
        
        self.truth_table[state_sig][action_key][result_sig].append(entry)

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
        STRICT MODE: Disabled heuristic promotion. 
        """
        return []

    def _get_invariant_properties(self, state_sig, action_key, result_sig):
        """
        Scans the history of a specific Outcome to find properties that are 
        CONSTANT across all instances.
        
        UPDATED: Contrastive Analysis + Run-Aware Unpacking.
        Only analyzes context from the CURRENT RUN, as previous run contexts are discarded.
        """
        if state_sig not in self.truth_table: return []
        if action_key not in self.truth_table[state_sig]: return []
        
        # 1. Robust State Signature Unpacking
        if len(state_sig) == 2 and isinstance(state_sig[0], tuple):
            base_sig = state_sig[0]
        else:
            base_sig = state_sig
            
        try:
            color, fp, size = base_sig
        except ValueError:
            return ["State Error"]

        # --- STEP 1: Fetch and Normalize History (Current Run Only) ---
        # We effectively flatten the 3-tuple (run, turn, id) back to (turn, id)
        # BUT only for the current run, so downstream logic works safely.
        
        raw_pos_history = self.truth_table[state_sig][action_key].get(result_sig, [])
        pos_history = []
        for entry in raw_pos_history:
            if len(entry) == 3: run, turn, oid = entry
            else: run, turn, oid = 0, entry[0], entry[1]
            
            if run == self.run_counter:
                pos_history.append((turn, oid))
        
        # If we have no history in this run (e.g. just reset), return basic stats
        if not pos_history: 
            fp_str = str(fp)
            short_fp = fp_str[:6] + ".." if len(fp_str) > 6 else fp_str
            return [f"Color={color}", f"Size={size}", f"Shape={short_fp}"]
        
        # Get Negative History (No Change) for Contrast - Current Run Only
        neg_history = []
        if ('NO_CHANGE', None, None) in self.truth_table[state_sig][action_key]:
            raw_neg = self.truth_table[state_sig][action_key][('NO_CHANGE', None, None)]
            for entry in raw_neg:
                if len(entry) == 3: run, turn, oid = entry
                else: run, turn, oid = 0, entry[0], entry[1]
                
                if run == self.run_counter:
                    neg_history.append((turn, oid))

        # --- Base Invariants (Always True) ---
        conditions = []
        conditions.append(f"Color={color}")
        conditions.append(f"Size={size}")
        fp_str = str(fp)
        short_fp = fp_str[:6] + ".." if len(fp_str) > 6 else fp_str
        conditions.append(f"Shape={short_fp}")

        # --- Contextual Invariants (The "Why") ---
        
        # Helper to extract ALL features for an object in a context
        def extract_features(ctx, oid):
            feats = {}
            if 'adj' in ctx and oid in ctx['adj']:
                feats['adj'] = ctx['adj'][oid] 
            if 'diag_adj' in ctx and oid in ctx['diag_adj']:
                feats['diag_adj'] = ctx['diag_adj'][oid]
            feats['align'] = set()
            if 'align' in ctx:
                for a_type, groups in ctx['align'].items():
                    for coord, ids in groups.items():
                        if oid in ids: feats['align'].add((a_type, coord))
            feats['diag_align'] = set()
            if 'diag_align' in ctx:
                 for a_type, groups in ctx['diag_align'].items():
                    for line_idx, ids in enumerate(groups):
                        if oid in ids: feats['diag_align'].add((a_type, line_idx))
            return feats

        # 1. Compute Intersection of POSITIVES
        # Validation: Check if history index exists
        if not (0 <= pos_history[0][0] - 1 < len(self.level_state_history)): 
            return conditions
            
        first_ctx = self.level_state_history[pos_history[0][0] - 1]
        common_feats = extract_features(first_ctx, pos_history[0][1])

        for i in range(1, len(pos_history)):
            tid, oid = pos_history[i]
            if not (0 <= tid - 1 < len(self.level_state_history)): continue
            ctx = self.level_state_history[tid - 1]
            curr_feats = extract_features(ctx, oid)
            
            # Intersect Adj
            if 'adj' in common_feats:
                if 'adj' not in curr_feats: del common_feats['adj']
                else:
                    new_adj = []
                    for k in range(4):
                        if common_feats['adj'][k] == curr_feats['adj'][k]: new_adj.append(common_feats['adj'][k])
                        else: new_adj.append(None)
                    if all(x is None for x in new_adj): del common_feats['adj']
                    else: common_feats['adj'] = new_adj

            # Intersect Diag Adj
            if 'diag_adj' in common_feats:
                if 'diag_adj' not in curr_feats: del common_feats['diag_adj']
                else:
                    new_dadj = []
                    for k in range(4):
                        if common_feats['diag_adj'][k] == curr_feats['diag_adj'][k]: new_dadj.append(common_feats['diag_adj'][k])
                        else: new_dadj.append(None)
                    if all(x is None for x in new_dadj): del common_feats['diag_adj']
                    else: common_feats['diag_adj'] = new_dadj
            
            # Intersect Sets
            for key in ['align', 'diag_align']:
                if key in common_feats:
                    if key in curr_feats:
                        common_feats[key] = common_feats[key].intersection(curr_feats[key])
                    else:
                        del common_feats[key]

        # 2. Check Contrast with NEGATIVES
        neg_features_union = {'adj': set(), 'diag_adj': set(), 'align': set(), 'diag_align': set()}
        
        for i in range(len(neg_history)):
            tid, oid = neg_history[i]
            if not (0 <= tid - 1 < len(self.level_state_history)): continue
            ctx = self.level_state_history[tid - 1]
            nf = extract_features(ctx, oid)
            
            if 'adj' in nf:
                for k in range(4): neg_features_union['adj'].add((k, nf['adj'][k]))
            if 'diag_adj' in nf:
                for k in range(4): neg_features_union['diag_adj'].add((k, nf['diag_adj'][k]))
            if 'align' in nf:
                neg_features_union['align'].update(nf['align'])
            if 'diag_align' in nf:
                neg_features_union['diag_align'].update(nf['diag_align'])

        # 3. Format Output
        dirs = ['top', 'right', 'bottom', 'left']
        if 'adj' in common_feats:
            for i, neighbor in enumerate(common_feats['adj']):
                if neighbor and neighbor != 'na':
                    n_clean = neighbor.replace('obj_', 'id_')
                    feat_str = f"Adj({dirs[i]}={n_clean})"
                    if (i, neighbor) not in neg_features_union['adj']:
                        feat_str = f"**{feat_str}**"
                    conditions.append(feat_str)
                    
        diag_dirs = ['top_right', 'bottom_right', 'bottom_left', 'top_left']
        if 'diag_adj' in common_feats:
            for i, neighbor in enumerate(common_feats['diag_adj']):
                if neighbor and neighbor != 'na':
                    n_clean = neighbor.replace('obj_', 'id_')
                    feat_str = f"DiagAdj({diag_dirs[i]}={n_clean})"
                    if (i, neighbor) not in neg_features_union['diag_adj']:
                        feat_str = f"**{feat_str}**"
                    conditions.append(feat_str)

        if 'align' in common_feats:
            for (a_type, coord) in common_feats['align']:
                feat_str = f"{a_type}={coord}"
                if (a_type, coord) not in neg_features_union['align']:
                    feat_str = f"**{feat_str}**"
                conditions.append(feat_str)
                
        if 'diag_align' in common_feats:
            for (a_type, line_idx) in common_feats['diag_align']:
                feat_str = f"{a_type}(Line {line_idx})"
                if (a_type, line_idx) not in neg_features_union['diag_align']:
                    feat_str = f"**{feat_str}**"
                conditions.append(feat_str)

        return conditions
    
    def _format_rule_description(self, rule_type, action_key, state_sig, result_sig):
        """
        Returns a scientific description of the rule.
        STRICTLY LIMITS context to what the Scientific Splitter has deemed necessary.
        """
        # 1. Get raw invariants (Color, Size, Shape, Neighbors, etc.)
        raw_conditions = self._get_invariant_properties(state_sig, action_key, result_sig)
        
        # 2. Filter based on Splitter Refinements
        refined_conditions = []
        
        # Always include Intrinsic Properties (Color, Size, Shape) as they define the 'State'
        # We identify them because _get_invariant_properties returns strings like "Color=5"
        intrinsic_prefixes = ["Color=", "Size=", "Shape="]
        
        # Get the list of critical context keys for this state (e.g. ['adj_top'])
        critical_keys = self.state_refinements.get(state_sig, [])
        
        for cond in raw_conditions:
            is_intrinsic = any(cond.startswith(p) for p in intrinsic_prefixes)
            
            if is_intrinsic:
                refined_conditions.append(cond)
            else:
                # It's context (Adj, Align, etc.). Only keep if critical.
                # format of cond is like "**Adj(top=id_5)**" or "Adj(top=id_5)"
                clean_cond = cond.replace('*', '')
                
                keep = False
                # Check simple mapping
                if "Adj(top=" in clean_cond and "adj_top" in critical_keys: keep = True
                elif "Adj(right=" in clean_cond and "adj_right" in critical_keys: keep = True
                elif "Adj(bottom=" in clean_cond and "adj_bottom" in critical_keys: keep = True
                elif "Adj(left=" in clean_cond and "adj_left" in critical_keys: keep = True
                elif "Align" in clean_cond:
                    # e.g. "center_x=10" -> check if "align_center_x" is in critical_keys
                    for k in critical_keys:
                        if k.startswith('align_') and k.replace('align_', '') in clean_cond:
                            keep = True
                            break
                elif "Match" in clean_cond: # e.g. Match(Color=..)
                     for k in critical_keys:
                        if k.startswith('match_') and k.replace('match_', '') in clean_cond:
                            keep = True; break
                
                if keep:
                    refined_conditions.append(cond)

        if not refined_conditions:
            return "Universal Rule"
            
        return " AND ".join(refined_conditions)

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
    
    def get_pass_action(self) -> Optional[GameAction]:
        """
        Determines the safest action to take solely to refresh the game state.
        Strategy:
        1. Try an action that is NOT valid for the current state (1-7).
        2. If all are valid, click (ACTION6) an object that has never caused a change.
        """
        latest_frame = self.frames[-1]
        available_names = {a.name for a in latest_frame.available_actions}
        
        # --- Strategy 1: Use an Invalid Action ---
        for i in range(1, 8):
            try:
                candidate = GameAction.from_id(i)
                if candidate.name not in available_names:
                    # Invalid actions are often treated as No-Ops
                    if candidate.name == 'ACTION6':
                         candidate.set_data({'x': 0, 'y': 0})
                    return candidate
            except ValueError:
                continue

        # --- Strategy 2: Click a "Safe" Object ---
        current_summary = self.last_object_summary
        for obj in current_summary:
            obj_id = obj['id']
            action_key = self._get_learning_key('ACTION6', obj_id)
            
            # If this object has never been productive, clicking it is likely safe
            if action_key not in self.productive_action_types:
                action = GameAction.ACTION6
                r, c = obj['position']
                action.set_data({'x': c, 'y': r}) 
                return action
        
        # --- Fallback: Click Top-Left ---
        action = GameAction.ACTION6
        action.set_data({'x': 0, 'y': 0})
        return action
    
    def _find_color_source(self, target_obj_id: str, target_color: int, full_context: dict) -> set:
        """
        Scans the existing adjacency and alignment maps to find which object 'donated' the color.
        Uses pre-computed context (adj, diag_adj, align) instead of raw grid scanning.
        """
        sources = set()
        
        objects = full_context.get('summary', [])
        obj_map = {o['id']: o for o in objects}
        
        # 1. Check ID-Based (Global Link)
        for obj in objects:
            if obj['id'] != target_obj_id and obj['color'] == target_color:
                sources.add(f"Source: ID({obj['id']})")

        # 2. Check Adjacency (Cardinal) - Uses pre-computed 'adj' map
        # adj_map format: {obj_id: [Top, Right, Bottom, Left]}
        adj_map = full_context.get('adj', {})
        if target_obj_id in adj_map:
            neighbors = adj_map[target_obj_id]
            # Offsets (dx, dy): Top(0,-1), Right(1,0), Bottom(0,1), Left(-1,0)
            offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            
            for i, neighbor_id in enumerate(neighbors):
                if neighbor_id not in ['na', 'x']:
                    neighbor = obj_map.get(neighbor_id)
                    if neighbor and neighbor['color'] == target_color:
                        dx, dy = offsets[i]
                        sources.add(f"Source: Adjacency({dx},{dy})")

        # 3. Check Diagonal Adjacency - Uses pre-computed 'diag_adj' map
        # diag_map format: {obj_id: [TR, BR, BL, TL]}
        diag_map = full_context.get('diag_adj', {})
        if target_obj_id in diag_map:
            neighbors = diag_map[target_obj_id]
            # Offsets (dx, dy): TR(1,-1), BR(1,1), BL(-1,1), TL(-1,-1)
            offsets = [(1, -1), (1, 1), (-1, 1), (-1, -1)]
            
            for i, neighbor_id in enumerate(neighbors):
                if neighbor_id not in ['na', 'x']:
                    neighbor = obj_map.get(neighbor_id)
                    if neighbor and neighbor['color'] == target_color:
                        dx, dy = offsets[i]
                        sources.add(f"Source: Adjacency({dx},{dy})")

        # 4. Check Alignment (Infinite Axis) - Uses pre-computed 'align' map
        # We check if the target shares an alignment group (row/col) with a matching color object
        align_map = full_context.get('align', {})
        for align_type, groups in align_map.items():
            # align_type is e.g., 'center_x' (Vertical), 'top_y' (Horizontal)
            axis = 'x' if 'x' in align_type else 'y'
            
            for coord_val, group_ids in groups.items():
                if target_obj_id in group_ids:
                    # Target is in this alignment line. Check other members.
                    for other_id in group_ids:
                        if other_id == target_obj_id: continue
                        other_obj = obj_map.get(other_id)
                        if other_obj and other_obj['color'] == target_color:
                            # MODIFIED: Now includes the ID of the aligned object.
                            # "I matched color because I am aligned on Y=5 with Object 12"
                            sources.add(f"Source: Alignment({axis}={coord_val}, ID={other_id})")

        return sources

    def _update_relational_constraints(self, action_key: tuple, events: list[dict], full_context: dict):
        """
        Identifies if a color change was caused by a specific neighbor or relation.
        
        UPDATED: Disabled "for now" as requested.
        """
        return

    def _get_relational_classification(self, action_name: str, target_id: str) -> str:
        """
        Determines if a relational rule is DIRECT (Action-Specific) or GLOBAL (Universal).
        Scientific Method:
        1. Consistency: N >= 2 (proven via _update_relational_constraints intersection).
        2. Exclusivity: Does this rule appear in Control Groups (other actions)?
           - If YES -> GLOBAL.
           - If NO -> DIRECT (but ONLY if a Control Group actually exists).
        """
        key = (action_name, target_id)
        if key not in self.relational_constraints:
            return None
            
        my_data = self.relational_constraints[key]
        my_rules = my_data['rules']
        
        # 1. Consistency Test
        if not my_rules: return None # Falsified
        if my_data['count'] < 2: return "HYPOTHESIS" # Not enough data yet
        
        # 2. Exclusivity Test (Compare against Control Groups)
        is_found_elsewhere = False
        
        for other_key, other_data in self.relational_constraints.items():
            other_action, other_id = other_key
            
            # We are looking for the SAME object, but a DIFFERENT action
            if other_id == target_id and other_action != action_name:
                
                # Check if the rules overlap (e.g. both found "Adjacency(0,1)")
                # If they share a mechanism, the mechanism is likely Global.
                common_rules = my_rules & other_data['rules']
                if common_rules:
                    is_found_elsewhere = True
                    break
        
        if is_found_elsewhere:
            return "GLOBAL"
        else:
            # --- NEW: Zero-Assumption Check ---
            # We can only claim DIRECT exclusivity if we have actually tried other actions.
            # If we've only ever done ONE thing, we can't be sure it's not Global.
            
            # self.performed_action_types includes the current action, so we need > 1
            if len(self.performed_action_types) > 1:
                return "DIRECT"
            else:
                # We have consistency (N>=2) but no control group.
                # It is likely Direct, but scientifically unproven.
                return "HYPOTHESIS"
        
    def _analyze_stop_condition(self, start_pos: tuple, end_pos: tuple, last_summary: list[dict]) -> list[str]:
        """
        Determines if a move stopped due to the Grid Edge or an Object.
        Returns a LIST of potential reasons (e.g. ['UNTIL_OBJ_1', 'UNTIL_OBSTRUCTION_LEFT']).
        """
        r1, c1 = start_pos
        r2, c2 = end_pos
        
        # Calculate Direction
        dr, dc = r2 - r1, c2 - c1
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
        
        # Determine Direction Name
        dir_name = "UNKNOWN"
        if step_r == -1 and step_c == 0: dir_name = "TOP"
        elif step_r == 1 and step_c == 0: dir_name = "BOTTOM"
        elif step_r == 0 and step_c == -1: dir_name = "LEFT"
        elif step_r == 0 and step_c == 1: dir_name = "RIGHT"
        elif step_r == -1 and step_c == -1: dir_name = "TOP_LEFT"
        elif step_r == -1 and step_c == 1: dir_name = "TOP_RIGHT"
        elif step_r == 1 and step_c == -1: dir_name = "BOTTOM_LEFT"
        elif step_r == 1 and step_c == 1: dir_name = "BOTTOM_RIGHT"
        
        # The "Shadow Step" (1 step past the destination)
        shadow_r, shadow_c = r2 + step_r, c2 + step_c
        
        # 1. Check Grid Edge
        max_h, max_w = getattr(self, 'last_grid_size', (64, 64))
        if shadow_r < 0 or shadow_r >= max_h or shadow_c < 0 or shadow_c >= max_w:
            return ["UNTIL_EDGE", "UNTIL_OBSTRUCTION", f"UNTIL_OBSTRUCTION_{dir_name}"]
            
        # 2. Check Object Obstruction
        # Find who occupies the shadow spot
        for obj in last_summary:
            if 'pixel_coords' in obj:
                if (shadow_r, shadow_c) in obj['pixel_coords']:
                    # Found the obstacle! Generate all property hypotheses.
                    reasons = []
                    reasons.append(f"UNTIL_OBJ_{obj['id']}")           # Specific ID
                    reasons.append(f"UNTIL_COLOR_{obj['color']}")       # Specific Color
                    reasons.append(f"UNTIL_SHAPE_{obj['fingerprint']}") # Specific Shape
                    reasons.append("UNTIL_OBSTRUCTION")                 # Generic Obstruction
                    reasons.append(f"UNTIL_OBSTRUCTION_{dir_name}")     # Directional Obstruction
                    return reasons
                    
        return []

    def _calculate_growth_shadow(self, prev_obj: dict, event: dict) -> tuple:
        """
        Determines the direction of growth by comparing old and new boundaries.
        Returns a (Start, End) tuple representing the 'Expansion Vector' of the edge.
        Used to feed _analyze_stop_condition.
        """
        old_r, old_c = prev_obj['position']
        old_h, old_w = prev_obj['size']
        
        new_pos = event.get('end_position', prev_obj['position'])
        new_size = event.get('to_size', prev_obj['size'])
        
        new_r, new_c = new_pos
        new_h, new_w = new_size
        
        # Detect Horizontal Growth
        if new_w > old_w:
            if new_c < old_c: # Grew Left
                # Edge moved from old_c to new_c.
                # Vector is (0, -1). 
                return (old_r, old_c), (old_r, old_c - 1)
            else: # Grew Right
                # Edge moved from (old_c + old_w - 1) to right.
                right_edge = old_c + old_w - 1
                return (old_r, right_edge), (old_r, right_edge + 1)

        # Detect Vertical Growth
        if new_h > old_h:
            if new_r < old_r: # Grew Up
                return (old_r, old_c), (old_r - 1, old_c)
            else: # Grew Down
                bottom_edge = old_r + old_h - 1
                return (bottom_edge, old_c), (bottom_edge + 1, old_c)
                
        return None, None

    def _check_interaction_physics(self, target_id: str, changes: list[dict], prev_context: dict, curr_summary: list[dict]) -> tuple[str, list[str]] | None:
        """
        Verifies if a Shape/Size change matches an Interaction (Occlusion/Reveal).
        Uses Strict Occupancy Logic: identifies exactly WHO is standing on the lost/gained pixels.
        """
        # 1. Get Target Data
        prev_target = next((o for o in prev_context['summary'] if o['id'] == target_id), None)
        curr_target = next((o for o in curr_summary if o['id'] == target_id), None)
        
        if not prev_target or not curr_target: return None
        
        old_pixels = set(prev_target['pixel_coords'])
        new_pixels = set(curr_target['pixel_coords'])
        
        gained_pixels = new_pixels - old_pixels
        lost_pixels = old_pixels - new_pixels
        
        if not gained_pixels and not lost_pixels:
            return None 

        agitator_ids = set()
        
        # 2. Identify Movers (The Suspects)
        # Anyone who moved, appeared, or merged is a potential agitator
        moved_ids = set()
        for event in changes:
            # We explicitly include NEW and REAPPEARED to catch spawns causing occlusion
            if event['type'] in ['MOVED', 'NEW', 'REAPPEARED', 'REAPPEARED & TRANSFORMED']:
                if event.get('id'): moved_ids.add(event['id'])
        
        # 3. Check for OCCLUSION (Lost Pixels)
        # Who is standing where I used to be?
        if lost_pixels:
            for obj in curr_summary:
                if obj['id'] == target_id: continue
                
                # Check intersection with lost spots
                overlap = lost_pixels.intersection(obj['pixel_coords'])
                if overlap:
                    # If the intruder Moved (or is New), they are the Cause.
                    if obj['id'] in moved_ids:
                        agitator_ids.add(obj['id'])

        # 4. Check for REVEAL (Gained Pixels)
        # Who used to stand where I am now?
        if gained_pixels:
            for obj in prev_context['summary']:
                if obj['id'] == target_id: continue
                
                overlap = gained_pixels.intersection(obj['pixel_coords'])
                if overlap:
                    # If the previous occupant moved away (is in moved_ids) or was Removed
                    is_removed = any(e['type'] == 'REMOVED' and e.get('id') == obj['id'] for e in changes)
                    if obj['id'] in moved_ids or is_removed:
                        agitator_ids.add(obj['id'])

        if not agitator_ids:
            return None

        # 5. Success
        agitator_list = sorted(list(agitator_ids))
        mover_str = ", ".join(agitator_list[:3]) 
        if len(agitator_list) > 3: mover_str += "..."
        
        explanation_parts = []
        if gained_pixels: explanation_parts.append("Reveal")
        if lost_pixels: explanation_parts.append("Occlusion")
        type_str = " & ".join(explanation_parts)
        
        return (f"CORRELATION: {type_str} caused by movement/removal of {mover_str}", agitator_list)
    
    def _get_scientific_state(self, obj: dict, context: dict) -> tuple:
        """
        Constructs the State Signature. Starts with Intrinsic properties.
        If the Splitter has activated for this object type, it appends specific Context features.
        """
        # 1. Intrinsic Properties (The Base)
        base_sig = (obj['color'], obj['fingerprint'], obj['size'])
        
        # 2. Check for Refinements (Context)
        # Does this object type require checking specific attributes?
        required_context_keys = self.state_refinements.get(base_sig, [])
        
        context_features = []
        for key in required_context_keys:
            # key format examples: 'adj_top', 'align_center_x', 'match_Color'
            val = None
            
            if key.startswith('adj_'):
                # Adjacency check
                direction = key.split('_')[1] # top, right, etc.
                d_idx = {'top':0, 'right':1, 'bottom':2, 'left':3}.get(direction)
                adj_list = context.get('adj', {}).get(obj['id'], ['na']*4)
                if d_idx is not None:
                    # We store if a neighbor exists (True) or is empty (False)
                    # For higher rigor, we could store the neighbor's color, but we start binary.
                    val = (adj_list[d_idx] != 'na')

            elif key.startswith('align_'):
                # Alignment check
                align_type = key.replace('align_', '')
                groups = context.get('align', {}).get(align_type, {})
                val = False
                for coord, ids in groups.items():
                    if obj['id'] in ids:
                        val = True
                        break
            
            elif key.startswith('match_'):
                # Match Group check
                m_type = key.replace('match_', '')
                groups = context.get('match', {}).get(m_type, {})
                val = False
                for props, ids in groups.items():
                    if obj['id'] in ids:
                        val = True
                        break

            context_features.append((key, val))

        # Return (Base, Context_Tuple)
        return (base_sig, tuple(sorted(context_features)))
    
    def _detect_sequential_global_laws(self, state_sig, actions_data):
        """
        Detects if a specific event type (e.g. 'RECOLOR to 3') follows a global repeating
        cycle of Target IDs (e.g. 7, 2, 19, 7...) regardless of the action taken.
        """
        # 1. Gather all events sorted by time
        abstract_timelines = {} 

        for action_name, results in actions_data.items():
            for r_sig, entry_list in results.items():
                if r_sig[0] == 'NO_CHANGE': continue
                
                # Abstract Signature: (Type, Value) - ignoring ID for grouping
                abs_key = (r_sig[0], r_sig[2]) 
                
                if abs_key not in abstract_timelines:
                    abstract_timelines[abs_key] = []
                
                for entry in entry_list:
                    # Handle unpacking
                    if len(entry) == 3:
                        run, turn, obj_id_str = entry
                    else:
                        run, turn, obj_id_str = 0, entry[0], entry[1]

                    try:
                        if isinstance(obj_id_str, str) and '_' in obj_id_str:
                            numeric_id = int(obj_id_str.split('_')[-1])
                            abstract_timelines[abs_key].append((run, turn, numeric_id, action_name))
                    except ValueError:
                        continue

        # 2. Analyze timelines for CYCLE PATTERNS
        for abs_key, timeline in abstract_timelines.items():
            # Sort by RUN then TURN
            timeline.sort(key=lambda x: (x[0], x[1]))
            
            ids = [x[2] for x in timeline]
            n = len(ids)

            # --- Scientific Rigor: Decoupling Check ---
            action_outcomes = {}
            for _, _, obj_id, act_key in timeline:
                if act_key not in action_outcomes: 
                    action_outcomes[act_key] = set()
                action_outcomes[act_key].add(obj_id)
            
            if all(len(outcomes) == 1 for outcomes in action_outcomes.values()):
                continue
            # ------------------------------------------

            # Check for Arbitrary Cycle (e.g., 7, 2, 19, 7, 2...)
            found_cycle = None
            
            for p in range(1, n // 2 + 1):
                matches = True
                
                # Verify consistency for the entire history
                for i in range(n - p):
                    if ids[i] != ids[i+p]:
                        matches = False
                        break
                
                if matches:
                    found_cycle = ids[:p]
                    if len(set(found_cycle)) > 1:
                        break
                    else:
                        found_cycle = None 

            if found_cycle:
                current_idx_in_cycle = (n - 1) % len(found_cycle)
                next_val = found_cycle[(current_idx_in_cycle + 1) % len(found_cycle)]
                
                self.certified_laws[(state_sig, 'ANY')] = {
                    'type': 'GLOBAL_SEQUENCE',
                    'pattern': 'CYCLE',
                    'sequence': found_cycle,
                    'result_type': abs_key[0],
                    'result_value': abs_key[1],
                    'next_prediction': next_val
                }

    def _detect_direct_sequential_laws(self, state_sig, action_key, results_map):
        """
        Detects if a specific Action causes a property (Value) to cycle 
        (e.g. Color 9 -> 8 -> 3 -> 9...).
        Requires observation of at least 2 full cycles.
        """
        # 1. Flatten history into a timeline: (run, turn, type, value)
        timeline = []
        for r_sig, entry_list in results_map.items():
            if r_sig[0] == 'NO_CHANGE': continue
            
            # r_sig is (Type, ID, Value)
            r_type = r_sig[0]
            r_val = r_sig[2]
            
            for entry in entry_list:
                # Handle legacy (turn, id) vs new (run, turn, id)
                if len(entry) == 3:
                    run, turn, obj_id = entry
                else:
                    run, turn, obj_id = 0, entry[0], entry[1]
                
                timeline.append((run, turn, r_type, r_val))
        
        if len(timeline) < 3: return 
        
        # Sort by RUN then TURN to preserve chronological order across resets
        timeline.sort(key=lambda x: (x[0], x[1]))
        
        # 2. Verify Homogeneity (Must be the same TYPE of change, e.g. all RECOLORED)
        primary_type = timeline[0][2]
        if not all(t[2] == primary_type for t in timeline): return
        
        # 3. Extract the Sequence of Values
        # Note: We treat the timeline as continuous for cycle detection, 
        # even if broken by resets, assuming the law holds globally.
        values = [t[3] for t in timeline]
        n = len(values)
        found_cycle = None
        
        # 4. Cycle Detection
        # Try periods from 2 up to n // 2
        for p in range(2, n // 2 + 1):
            matches = True
            # Check consistency: v[i] must equal v[i+p]
            for i in range(n - p):
                if values[i] != values[i+p]:
                    matches = False
                    break
            
            # Additional Check: The cycle must actually CHANGE values.
            cycle_candidate = values[:p]
            if matches and len(set(cycle_candidate)) > 1:
                found_cycle = cycle_candidate
                break
                
        if found_cycle:
             self.certified_laws[(state_sig, action_key)] = {
                'type': 'DIRECT_SEQUENCE',
                'pattern': 'CYCLE',
                'sequence': found_cycle,
                'result_type': primary_type
            }

    def _verify_and_certify(self, state_sig):
        """
        Runs the logic: Is this Consistent? Is it Direct or Global?
        Generates 'Candidate Laws' stored in certified_laws.
        Note: Classification strictly enforces ID checks before applying these laws.
        """
        if state_sig not in self.truth_table: return
        
        actions_data = self.truth_table[state_sig]

        # Step 1: Filter for MATURE actions only (At the State Level)
        # We look for patterns that exist generally, even if specific objects aren't there yet.
        mature_results = {} 
        
        global_ballots = {} 
        mature_direct_candidates = {}

        for action, results in actions_data.items():
            
            # --- NEW: Check for Direct Cycles (e.g. Toggle Switches / Color Cycles) ---
            self._detect_direct_sequential_laws(state_sig, action, results)
            # --------------------------------------------------------------------------

            # Group by Concrete Sig (Type, ID, Value)
            # FIX: We now include the ID (index 1) in the signature.
            # "Recolor Obj 7" and "Recolor Obj 8" are DIFFERENT events and cannot
            # combine to form a Global Law.
            abstract_groups = {}
            for r_sig, entry_list in results.items():
                if r_sig[0] == 'NO_CHANGE': continue
                
                # OLD: abs_sig = (r_sig[0], r_sig[2]) 
                # NEW: Full signature including ID
                abs_sig = r_sig 
                
                if abs_sig not in abstract_groups:
                    abstract_groups[abs_sig] = set()
                turns = set()
                for entry in entry_list:
                    if len(entry) == 3: turns.add((entry[0], entry[1])) # (run, turn)
                    else: turns.add((0, entry[0])) # Legacy fallback
                abstract_groups[abs_sig].update(turns)

            if len(abstract_groups) > 1:
                # If we detected a cycle earlier, don't treat this as a Splitter anomaly!
                # Direct Sequence laws supersede static consistency checks.
                if (state_sig, action) not in self.certified_laws:
                    self._trigger_splitter(state_sig, action, results)
                continue 

            if abstract_groups:
                single_abs_sig = list(abstract_groups.keys())[0]
                change_turn_ids = abstract_groups[single_abs_sig]
                
                no_change_turn_ids = set()
                if ('NO_CHANGE', None, None) in results:
                     for entry in results[('NO_CHANGE', None, None)]:
                        if len(entry) == 3: no_change_turn_ids.add((entry[0], entry[1]))
                        else: no_change_turn_ids.add((0, entry[0]))
                
                historical_failure = no_change_turn_ids - change_turn_ids
                if historical_failure:
                    # Similarly, ignore splitter trigger if cycle exists
                    if (state_sig, action) not in self.certified_laws:
                        self._trigger_splitter(state_sig, action, results)
                    continue

                if single_abs_sig not in global_ballots:
                    global_ballots[single_abs_sig] = set()
                global_ballots[single_abs_sig].add(action)

                if len(change_turn_ids) >= 2:
                    mature_direct_candidates[action] = single_abs_sig

        # --- Step 2: Global Certification (Local Coalition) ---
        # "Different ACTIONS, same State -> Global Candidate"
        for abs_result, agreeing_actions in global_ballots.items():
            if len(agreeing_actions) >= 2:
                self.certified_laws[(state_sig, 'ANY')] = {'type': 'GLOBAL', 'result': abs_result}

        # Step 2.5: Detect Round-Robin / Cycle Patterns
        self._detect_sequential_global_laws(state_sig, actions_data)

        # --- Step 3: Direct Certification ---
        for action, abstract_result in mature_direct_candidates.items():
            # If we already found a Sequence Law or Global Law, skip static Direct
            if self.certified_laws.get((state_sig, 'ANY')): continue
            if self.certified_laws.get((state_sig, action)): continue
            
            concrete_sig = None
            raw_results = actions_data[action]
            for r_sig in raw_results:
                if r_sig[0] == 'NO_CHANGE': continue
                
                # FIX: abstract_result is now the full tuple (Type, ID, Value)
                # so we compare directly instead of constructing a subset tuple.
                if r_sig == abstract_result:
                    concrete_sig = r_sig
                    break
            
            if concrete_sig:
                self.certified_laws[(state_sig, action)] = {'type': 'DIRECT', 'result': concrete_sig}

        # --- Step 4: Universal Law Generalization ---
        # "Different STATES, same Result -> Universal Global"
        
        candidate_result = None
        if (state_sig, 'ANY') in self.certified_laws:
            # FIX: Check if 'result' exists (Sequence laws use different keys)
            law = self.certified_laws[(state_sig, 'ANY')]
            if 'result' in law:
                candidate_result = law['result']
        elif mature_direct_candidates:
             candidate_result = list(mature_direct_candidates.values())[0]
             
        if candidate_result:
            supporting_states = 0
            for other_sig in self.truth_table:
                if other_sig == state_sig: continue
                other_actions = self.truth_table[other_sig]
                found_match = False
                for act, res_dict in other_actions.items():
                    # Skip if this action has a sequence law
                    if (other_sig, act) in self.certified_laws and self.certified_laws[(other_sig, act)].get('type') == 'DIRECT_SEQUENCE':
                        continue

                    for r_sig in res_dict:
                        if r_sig[0] == 'NO_CHANGE': continue
                        
                        # FIX: Compare full tuple here as well
                        if r_sig == candidate_result:
                            found_match = True
                            break
                    if found_match: break
                
                if found_match:
                    supporting_states += 1
            
            if supporting_states >= 2:
                self.certified_laws[(state_sig, 'ANY')] = {'type': 'GLOBAL', 'result': candidate_result}

    def _get_feature_status(self, contexts: list[dict], base_sig: tuple, feature_type: str, sub_key: str) -> str:
        """
        Analyzes a set of historical contexts to see if a feature is consistent.
        Returns: 'ALL' (100% Present), 'NONE' (0% Present), or 'MIXED'.
        """
        count = 0
        total = 0
        
        for ctx in contexts:
            # Find the object matching base_sig in this historical context
            # (Logic copied to ensure we are looking at the specific actor in the specific past)
            target_obj = None
            for obj in ctx['summary']:
                if (obj['color'], obj['fingerprint'], obj['size']) == base_sig:
                    target_obj = obj
                    break
            
            if not target_obj: continue
            total += 1
            
            has_feature = False
            
            if feature_type == 'adj':
                # sub_key is direction (e.g. 'top')
                d_idx = {'top':0, 'right':1, 'bottom':2, 'left':3}.get(sub_key)
                adj_list = ctx.get('adj', {}).get(target_obj['id'], ['na']*4)
                if adj_list[d_idx] != 'na': has_feature = True
                
            elif feature_type == 'align':
                # sub_key is type (e.g. 'center_x')
                groups = ctx.get('align', {}).get(sub_key, {})
                for coord, ids in groups.items():
                    if target_obj['id'] in ids: has_feature = True; break
            
            elif feature_type == 'match':
                 # sub_key is type (e.g. 'Color')
                 groups = ctx.get('match', {}).get(sub_key, {})
                 for props, ids in groups.items():
                    if target_obj['id'] in ids: has_feature = True; break

            if has_feature: count += 1
            
        if total == 0: return 'NONE' # Should not happen if contexts are valid
        
        if count == total: return 'ALL'
        if count == 0: return 'NONE'
        return 'MIXED'

    def _trigger_splitter(self, state_sig, action, conflicting_results: dict):
        """
        The Scientific Splitter.
        Refines the State definition only when two DIFFERENT, REPEATABLE outcomes
        can be perfectly separated by a context feature.
        
        Logic:
        1. Anomaly Filter: Ignore conflicting results until they are Mature (N >= 2).
        2. Hypothesis Search: Find a feature that is 'ALL' in Outcome A and 'NONE' in Outcome B.
        3. Splitting: Only split if a perfect separator is found.
        """
        base_sig = state_sig[0] # (Color, Fingerprint, Size)
        
        # --- 1. MATURITY CHECK (The "Anomaly" Filter) ---
        # We only attempt to split if we have at least 2 distinct observations 
        # for at least 2 different outcomes.
        # This prevents splitting logic based on a single fluke (N=1).
        
        valid_result_keys = []
        
        for res_sig, entry_list in conflicting_results.items():
            # Count unique turns for this result (Handling Run ID)
            unique_turns = set()
            for entry in entry_list:
                if len(entry) == 3: unique_turns.add((entry[0], entry[1])) # (Run, Turn)
                else: unique_turns.add((0, entry[0]))
            
            if len(unique_turns) >= 2:
                valid_result_keys.append(res_sig)
        
        # If we don't have at least 2 mature outcomes, we are not ready to do science.
        # We just wait. The anomaly stays in the truth table as "noise" for now.
        if len(valid_result_keys) < 2:
            return

        # Gather contexts only for the valid (mature) results
        contexts_by_result = {}
        for res in valid_result_keys: 
            entry_list = conflicting_results[res]
            contexts = []
            for entry in entry_list:
                # Unpack safely
                if len(entry) == 3: run, tid, oid = entry
                else: run, tid, oid = 0, entry[0], entry[1]

                # Note: We only look up history if it matches the current run context
                # (Assuming level_state_history is reset per run)
                if 0 <= tid - 1 < len(self.level_state_history):
                    contexts.append(self.level_state_history[tid-1])
            contexts_by_result[res] = contexts

        # Select the first two mature outcomes to compare (Outcome A vs Outcome B)
        # (In complex cases with 3+ outcomes, solving the first pair usually helps resolve the rest iteratively)
        res_A = valid_result_keys[0]
        res_B = valid_result_keys[1]
        
        ctx_group_A = contexts_by_result[res_A]
        ctx_group_B = contexts_by_result[res_B]
        
        candidate_split_key = None
        
        # --- 2. HYPOTHESIS SEARCH (The Separator) ---
        # We look for a feature that is 100% Present in A and 0% in B (or vice versa).
        # We do NOT accept "Mixed" results.
        
        # A. Check Adjacencies
        dirs = ['top', 'right', 'bottom', 'left']
        for d in dirs:
            status_A = self._get_feature_status(ctx_group_A, base_sig, 'adj', d)
            status_B = self._get_feature_status(ctx_group_B, base_sig, 'adj', d)
            
            # Valid Split: ALL vs NONE
            if (status_A == 'ALL' and status_B == 'NONE') or \
               (status_A == 'NONE' and status_B == 'ALL'):
                candidate_split_key = f"adj_{d}"
                break
        
        # B. Check Alignments
        if not candidate_split_key:
            align_types = ['center_x', 'center_y', 'top_y', 'left_x']
            for a_t in align_types:
                status_A = self._get_feature_status(ctx_group_A, base_sig, 'align', a_t)
                status_B = self._get_feature_status(ctx_group_B, base_sig, 'align', a_t)
                
                if (status_A == 'ALL' and status_B == 'NONE') or \
                   (status_A == 'NONE' and status_B == 'ALL'):
                    candidate_split_key = f"align_{a_t}"
                    break

        # C. Check Match Groups
        if not candidate_split_key:
             match_types = ['Exact', 'Color', 'Shape', 'Size']
             for m_t in match_types:
                status_A = self._get_feature_status(ctx_group_A, base_sig, 'match', m_t)
                status_B = self._get_feature_status(ctx_group_B, base_sig, 'match', m_t)
                
                if (status_A == 'ALL' and status_B == 'NONE') or \
                   (status_A == 'NONE' and status_B == 'ALL'):
                    candidate_split_key = f"match_{m_t}"
                    break

        # --- 3. APPLY SPLIT ---
        if candidate_split_key:
            if base_sig not in self.state_refinements:
                self.state_refinements[base_sig] = []
            
            # Avoid duplicate splits
            if candidate_split_key not in self.state_refinements[base_sig]:
                if self.debug_channels['STATE_GRAPH']:
                    print(f"  [Scientific Splitter] Refined State {base_sig} using '{candidate_split_key}' to separate conflicting outcomes.")
                
                self.state_refinements[base_sig].append(candidate_split_key)
                
                # RESET Truth Table for this State Signature
                # This forces the agent to re-sort all history into the new, specific buckets
                # (State + Feature) vs (State + No Feature).
                del self.truth_table[state_sig]

    def _check_feature_presence(self, contexts, base_sig, feature_type, sub_key):
        """
        Helper: Returns True ONLY if the feature is present in ALL contexts (100%).
        Scientific Rigor: We don't guess based on majority. A single mismatch disqualifies the feature.
        """
        count = 0
        total = 0
        
        for ctx in contexts:
            # Find the object matching base_sig in this historical context
            target_obj = None
            for obj in ctx['summary']:
                if (obj['color'], obj['fingerprint'], obj['size']) == base_sig:
                    target_obj = obj
                    break
            
            if not target_obj: continue
            total += 1
            
            has_feature = False
            
            if feature_type == 'adj':
                # sub_key is direction (e.g. 'top')
                d_idx = {'top':0, 'right':1, 'bottom':2, 'left':3}.get(sub_key)
                adj_list = ctx.get('adj', {}).get(target_obj['id'], ['na']*4)
                if adj_list[d_idx] != 'na': has_feature = True
                
            elif feature_type == 'align':
                # sub_key is type (e.g. 'center_x')
                groups = ctx.get('align', {}).get(sub_key, {})
                for coord, ids in groups.items():
                    if target_obj['id'] in ids: has_feature = True; break
            
            elif feature_type == 'match':
                 # sub_key is type (e.g. 'Color')
                 groups = ctx.get('match', {}).get(sub_key, {})
                 for props, ids in groups.items():
                    if target_obj['id'] in ids: has_feature = True; break

            if has_feature: count += 1
            
        if total == 0: return False
        
        # STRICT 100% CHECK
        return count == total