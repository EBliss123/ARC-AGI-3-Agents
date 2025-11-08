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

        # --- Debug Channels ---
        # Set these to True or False to control the debug output.
        self.debug_channels = {
            'PERCEPTION': False,      # Object finding, relationships, new level setup
            #other debug logic here
        }

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """
        Replace with actual action selection logic.
        """
        # If the game is over or hasn't started, the correct action is to reset.
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET
        
        # Perceive all objects and their relationships
        current_summary = self._perceive_objects(latest_frame)
        
        # Analyze the perceived objects to find all relationships, alignments, etc.
        relationships, adjacencies, diag_adjacencies, match_groups, alignments, diag_alignments, conjunctions = self._analyze_perception(current_summary)

        # ( logic will go here)

        # Just pick the first action (replace with real logic)
        if latest_frame.available_actions:
            return latest_frame.available_actions[0]
        
        # Fallback
        return GameAction.RESET

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
    
    def _analyze_perception(self, object_summary: list[dict]) -> tuple:
        """Calls all analysis functions and returns their results."""
        
        relationships, adjacencies, diag_adjacencies, match_groups = self._analyze_relationships(object_summary)
        alignments = self._analyze_alignments(object_summary)
        diag_alignments = self._analyze_diagonal_alignments(object_summary)
        conjunctions = self._analyze_conjunctions(relationships, alignments)
        
        return relationships, adjacencies, diag_adjacencies, match_groups, alignments, diag_alignments, conjunctions

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
            obj_id = obj['id']
            
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
            temp_groups.setdefault(exact_match_key(obj), []).append(obj['id'])
        
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
            unprocessed_objects = [o for o in object_summary if o['id'] not in processed_ids]
            
            for obj in unprocessed_objects:
                temp_groups.setdefault(key_func(obj), []).append(obj['id'])
            
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
                obj_id = obj['id']
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

        coord_map = {(obj['center_y'], obj['center_x']): obj['id'] for obj in object_summary}
        processed_ids = set()
        final_alignments = {
            'top_left_to_bottom_right': [],
            'top_right_to_bottom_left': [],
        }

        # Iterate through every unique pair of objects
        for i in range(len(object_summary)):
            obj_a = object_summary[i]
            id_a = obj_a['id']
            if id_a in processed_ids:
                continue

            for j in range(i + 1, len(object_summary)):
                obj_b = object_summary[j]
                id_b = obj_b['id']
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