import collections

class RelationalTracker:
    def __init__(self):
        self.history = {}
        self.dormant_signatures = []

    def build_dense_footprints(self, grid_tensor):
        """1. Dense Initialization: Maps every coordinate to its relative distances to the same color."""
        color_map = collections.defaultdict(list)
        max_y, max_x = grid_tensor.shape[-2:]
        
        for y in range(max_y):
            for x in range(max_x):
                color = int(grid_tensor[0, y, x].item())
                color_map[color].append((y, x))
                
        dense_footprints = {}
        for color, coords in color_map.items():
            for (y, x) in coords:
                # Calculate (dy, dx) to all other pixels of the same color
                anchors = {(cy - y, cx - x) for (cy, cx) in coords if (cy, cx) != (y, x)}
                dense_footprints[(y, x)] = {"color": color, "anchors": anchors}
                
        return dense_footprints

    def match_and_prune(self, prev_signatures, current_dense):
        """Matches relational identities across frames, prunes inconsistent anchors, 
        and flags PERSISTENT, DISAPPEARED, REAPPEARED, and SPAWNED pixels."""
        matched_results = {}
        available_current_coords = set(current_dense.keys())
        unmatched_prev_coords = set(prev_signatures.keys())

        # 1. Match active previous signatures to current frame
        for prev_coord, prev_data in prev_signatures.items():
            prev_color = prev_data["color"]
            prev_anchors = prev_data["anchors"]
            prev_id = prev_data.get("id")

            candidates = [
                coord for coord in available_current_coords
                if current_dense[coord]["color"] == prev_color
            ]

            if not candidates:
                continue

            best_coord = None
            best_overlap = -1
            best_consistent_anchors = set()

            for cand_coord in candidates:
                cand_anchors = current_dense[cand_coord]["anchors"]
                overlap = prev_anchors.intersection(cand_anchors)
                if len(overlap) > best_overlap:
                    best_overlap = len(overlap)
                    best_coord = cand_coord
                    best_consistent_anchors = overlap

            if best_coord is not None and (best_overlap > 0 or len(prev_anchors) == 0):
                available_current_coords.remove(best_coord)
                unmatched_prev_coords.remove(prev_coord)
                dy = best_coord[0] - prev_coord[0]
                dx = best_coord[1] - prev_coord[1]

                matched_results[best_coord] = {
                    "color": prev_color,
                    "anchors": best_consistent_anchors,
                    "id": prev_id,
                    "prev_coord": prev_coord,
                    "translation": (dy, dx),
                    "presence": "PERSISTENT"
                }

        # 2. Track disappeared signatures and archive to dormant registry
        disappeared = []
        for prev_coord in unmatched_prev_coords:
            lost_data = prev_signatures[prev_coord]
            lost_data["last_coord"] = prev_coord
            lost_data["presence"] = "DISAPPEARED"
            disappeared.append(lost_data)
            self.dormant_signatures.append(lost_data)

        # 3. Check unclaimed current coordinates against dormant signatures (Reappearance) or flag as Spawned
        spawned = {}
        reappeared = {}
        for cand_coord in list(available_current_coords):
            cand_data = current_dense[cand_coord]
            cand_color = cand_data["color"]
            cand_anchors = cand_data["anchors"]

            best_dormant_idx = None
            best_dormant_overlap = -1
            best_consistent_anchors = set()

            for idx, dormant in enumerate(self.dormant_signatures):
                if dormant["color"] == cand_color:
                    overlap = dormant["anchors"].intersection(cand_anchors)
                    if len(overlap) > best_dormant_overlap:
                        best_dormant_overlap = len(overlap)
                        best_dormant_idx = idx
                        best_consistent_anchors = overlap

            if best_dormant_idx is not None and best_dormant_overlap > 0:
                dormant = self.dormant_signatures.pop(best_dormant_idx)
                reappeared[cand_coord] = {
                    "color": cand_color,
                    "anchors": best_consistent_anchors,
                    "id": dormant.get("id"),
                    "last_known_coord": dormant["last_coord"],
                    "presence": "REAPPEARED"
                }
            else:
                spawned[cand_coord] = {
                    "color": cand_color,
                    "anchors": cand_anchors,
                    "id": None,
                    "presence": "SPAWNED"
                }

        return {
            "persistent": matched_results,
            "disappeared": disappeared,
            "reappeared": reappeared,
            "spawned": spawned
        }
    
    def apply_uniqueness_filter(self, footprints):
        """3. The Uniqueness Filter & Tie-Breaker."""
        color_groups = collections.defaultdict(dict)
        for coord, data in footprints.items():
            color_groups[data["color"]][coord] = data["anchors"]
            
        final_signatures = {}
        for color, coords_dict in color_groups.items():
            # Count occurrences of each (dy, dx) vector within this color group
            vector_counts = collections.Counter()
            for anchors in coords_dict.values():
                vector_counts.update(anchors)
                
            instance_counter = 1
            for coord, anchors in coords_dict.items():
                # Find vectors unique to this specific pixel
                unique_vectors = {v for v in anchors if vector_counts[v] == 1}
                
                if unique_vectors:
                    # Drop the rest, keep only the unique identifiers
                    final_signatures[coord] = {"color": color, "anchors": unique_vectors, "id": None}
                else:
                    # Tie-breaker logic for perfectly symmetrical/identical footprints
                    final_signatures[coord] = {"color": color, "anchors": anchors, "id": f"Instance_{instance_counter}"}
                    instance_counter += 1
                    
        return final_signatures