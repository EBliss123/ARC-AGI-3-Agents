from .agent import Agent, FrameData
from .structs import GameAction, GameState
import random
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
        self.actions_printed = False

        # --- Debug Channels ---
        # Set these to True or False to control the debug output.
        self.debug_channels = {
            'PERCEPTION': True,      # Object finding, relationships, new level setup
            'CHANGES': True,         # All "Change Log" prints
            'STATE_GRAPH': True,     # State understanding
            'HYPOTHESIS': True,      # "Initial Hypotheses", "Refined Hypothesis"
            'FAILURE': True,         # "Failure Analysis", "Failure Detected"
            'WIN_CONDITION': True,   # "LEVEL CHANGE DETECTED", "Win Condition Analysis"
            'ACTION_SCORE': True,    # All scoring prints
        }

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
            if self.debug_channels['PERCEPTION']: print("\n--- New Level Detected: Resetting history. ---")
            self.is_new_level = False
            self.final_summary_before_level_change = None
            self.current_level_id_map = {}
            self.last_action_context = None
            self.success_contexts = {}
            self.failure_contexts = {}
            self.failure_patterns = {}
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

            # (Future: Add level change detection here)

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
                prev_action_name, prev_target_id = self.last_action_context
                learning_key = self._get_learning_key(prev_action_name, prev_target_id)
                
                # This is the full context of the *previous* state, when the action was taken.
                prev_context = {
                    'summary': prev_summary,
                    'rels': self.last_relationships,
                    'adj': self.last_adjacencies,
                    'diag_adj': self.last_diag_adjacencies,
                    'diag_align': self.last_diag_alignments
                }

                if changes:
                    # --- SUCCESS ---
                    self.success_contexts.setdefault(learning_key, []).append(prev_context)
                
                else:
                    # --- FAILURE ---
                    if self.debug_channels['FAILURE']:
                        print(f"\n--- Failure Detected for Action {learning_key} ---")
                    
                    self.failure_contexts.setdefault(learning_key, []).append(prev_context)
                    
                    # Get all history for this action
                    successes = self.success_contexts.get(learning_key, [])
                    failures = self.failure_contexts.get(learning_key, [])

                    # Perform the differential analysis
                    self._analyze_failures(learning_key, successes, failures, prev_context)

        # --- 3. Update Memory For Next Turn ---
        # This runs every frame, saving the state we just analyzed
        self.last_object_summary = current_summary
        self.last_adjacencies = current_adjacencies
        self.last_diag_adjacencies = current_diag_adjacencies
        self.last_relationships = current_relationships
        self.last_alignments = current_alignments
        self.last_diag_alignments = current_diag_alignments
        self.last_match_groups = current_match_groups

# --- 4. Choose an Action (Temporary Random Logic) ---
        action_to_return = None
        chosen_object = None
        chosen_object_id = None
        
        # Build a list of all possible moves (global actions + one click per object)
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

        if all_possible_moves:
            # --- THIS IS THE TEMPORARY RANDOM CHOICE ---
            chosen_move = random.choice(all_possible_moves)
            action_to_return = chosen_move['template']
            chosen_object = chosen_move['object']
            
            action_name = action_to_return.name
            target_name = ""
            
            if chosen_object:
                chosen_object_id = chosen_object['id']
                pos = chosen_object['position']
                action_to_return.set_data({'x': pos[1], 'y': pos[0]})
                target_name = f" on {chosen_object_id.replace('obj_', 'id_')}"

            if self.debug_channels['ACTION_SCORE']:
                print(f"\n####### TEMPORARY RANDOM AGENT #######")
                print(f"Chose random action: {action_name}{target_name}")
                print(f"########################################")

        else:
            # Fallback if no actions are possible
            action_to_return = GameAction.RESET

        # --- Store action for next turn's analysis ---
        self.last_action_context = (action_to_return.name, chosen_object_id)
        
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
        Finds the common context across a list of attempts, using wildcard 'x' for
        inconsistent adjacency properties.
        """
        if not contexts:
            return {'adj': {}, 'rels': {}}

        # --- Adjacency Analysis with Wildcards ---
        master_adj = {obj_id: list(contacts) for obj_id, contacts in contexts[0].get('adj', {}).items()}
        for i in range(1, len(contexts)):
            next_adj = contexts[i].get('adj', {})
            for obj_id in list(master_adj.keys()):
                master_pattern = master_adj[obj_id]
                next_contacts = next_adj.get(obj_id)
                if not next_contacts:
                    del master_adj[obj_id]
                    continue
                for i in range(4):
                    if master_pattern[i] == 'x':
                        continue
                    if master_pattern[i] != next_contacts[i]:
                        master_pattern[i] = 'x'
                if all(d == 'x' for d in master_pattern):
                    del master_adj[obj_id]
        common_adj = {obj_id: tuple(pattern) for obj_id, pattern in master_adj.items()}

        # --- Relationship Analysis (Strict Intersection) ---
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
            if self.debug_channels['FAILURE']:
                print("\n--- Failure Analysis ---")
                print("Cannot perform differential analysis: insufficient history of successes or failures.")
            return

        if self.debug_channels['FAILURE']: print("\n--- Failure Analysis: Consistent Differentiating Conditions ---")
        
        common_success_context = self._find_common_context(all_success_contexts)
        common_failure_context = self._find_common_context(all_failure_contexts)
        
        observed_in_any_success_adj = set()
        observed_in_any_success_rels = set()
        for context in all_success_contexts:
            for obj_id, contacts in context.get('adj', {}).items():
                observed_in_any_success_adj.add((obj_id, tuple(contacts)))
            for rel_type, groups in context.get('rels', {}).items():
                for value, ids in groups.items():
                    observed_in_any_success_rels.add((rel_type, value, frozenset(ids)))

        diffs_found = False

        adj_s = common_success_context['adj']
        adj_f = common_failure_context['adj']
        all_adj_ids = set(adj_s.keys()) | set(adj_f.keys())

        for obj_id in sorted(list(all_adj_ids), key=lambda x: int(x.split('_')[1])):
            contacts_s = tuple(adj_s.get(obj_id, ['na']*4))
            contacts_f = tuple(adj_f.get(obj_id, ['na']*4))

            if contacts_s != contacts_f:
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
                    if self.debug_channels['FAILURE']: print(f"- Adjacency Difference for {clean_id}: Failures consistently exhibit pattern {pattern_str}, which has never occurred in a success.")

        rels_s = common_success_context['rels']
        rels_f = common_failure_context['rels']
        all_rel_types = set(rels_s.keys()) | set(rels_f.keys())

        for rel_type in all_rel_types:
            groups_s, groups_f = rels_s.get(rel_type, {}), rels_f.get(rel_type, {})
            all_values = set(groups_s.keys()) | set(groups_f.keys())
            for value in all_values:
                ids_s, ids_f = groups_s.get(value, set()), groups_f.get(value, set())
                if ids_s != ids_f:
                    if (rel_type, value, frozenset(ids_f)) not in observed_in_any_success_rels:
                        diffs_found = True
                        value_str = f"{value[0]}x{value[1]}" if rel_type == 'Size' else value
                        if self.debug_channels['FAILURE']: print(f"- {rel_type} Group ({value_str}) Difference: Failures consistently have members {sorted(list(ids_f))}, which has never occurred in a success.")
        
        if diffs_found:
            self.failure_patterns[action_key] = common_failure_context

        if not diffs_found:
            if self.debug_channels['FAILURE']: print("No conditions found that are both consistent across all failures and unique to them.")

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

        partial_match_definitions = {
            "Color":       lambda o: (o['fingerprint'], o['size'], o['pixels']),
            "Fingerprint": lambda o: (o['color'], o['size'], o['pixels']),
            "Size":        lambda o: (o['color'], o['fingerprint'], o['pixels']),
            "Pixels":      lambda o: (o['color'], o['fingerprint'], o['size']),
        }
        for match_type, key_func in partial_match_definitions.items():
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
        
        for old_stable_id in self.removed_objects_memory.keys():
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