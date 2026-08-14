import torch
from collections import defaultdict
from wake_phase.relational_tracker import RelationalTracker

class DimensionalOverlay:
    def __init__(self):
        self.tracker = RelationalTracker()
        self.prev_signatures = None

    def synthesize_step(self, s_t: torch.Tensor, s_next: torch.Tensor, action_id: int = None) -> dict:
        """
        Overlays Physical, Relational, and Presence dimensions for a single transition.
        Returns a structured dictionary of deltas across all three perspectives.
        """
        # 1. Physical Dimension: Static Coordinate Subtractions (In-place recoloring)
        diff_mask = s_t != s_next
        diff_coords = torch.nonzero(diff_mask)
        
        physical_deltas = []
        for idx in diff_coords:
            y, x = idx[-2].item(), idx[-1].item()
            c_old = int(s_t[..., y, x].item())
            c_new = int(s_next[..., y, x].item())
            physical_deltas.append({
                "coord": (y, x),
                "transition": (c_old, c_new)
            })

        # 2. Relational & Presence Dimensions: Signature Tracking
        dense_current = self.tracker.build_dense_footprints(s_next)
        
        if self.prev_signatures is None:
            # First frame initialization: build dense footprints, apply uniqueness filter
            dense_prev = self.tracker.build_dense_footprints(s_t)
            self.prev_signatures = self.tracker.apply_uniqueness_filter(dense_prev)

        match_results = self.tracker.match_and_prune(self.prev_signatures, dense_current)
        
        # Extract Relational Translations (persisting pixels with non-zero movement)
        relational_deltas = []
        for new_coord, data in match_results["persistent"].items():
            dy, dx = data["translation"]
            if dy != 0 or dx != 0:
                relational_deltas.append({
                    "new_coord": new_coord,
                    "prev_coord": data["prev_coord"],
                    "color": data["color"],
                    "translation": (dy, dx),
                    "anchors": data["anchors"],
                    "id": data["id"]
                })

        # Extract Presence Deltas
        presence_deltas = {
            "disappeared": match_results["disappeared"],
            "reappeared": match_results["reappeared"],
            "spawned": match_results["spawned"]
        }

        # 3. Aggregate Shared Commonality Summaries
        relational_move_groups = defaultdict(list)
        for r in relational_deltas:
            key = (r["color"], r["translation"])
            relational_move_groups[key].append(r)

        physical_color_groups = defaultdict(list)
        for p in physical_deltas:
            physical_color_groups[p["transition"]].append(p["coord"])

        # Update persistent signatures for the next timeline step
        all_active_current = {}
        all_active_current.update(match_results["persistent"])
        all_active_current.update(match_results["reappeared"])
        all_active_current.update(match_results["spawned"])
        self.prev_signatures = self.tracker.apply_uniqueness_filter(all_active_current)

        return {
            "action_id": action_id,
            "physical_deltas": physical_deltas,
            "relational_deltas": relational_deltas,
            "presence_deltas": presence_deltas,
            "summaries": {
                "relational_move_groups": dict(relational_move_groups),
                "physical_color_groups": dict(physical_color_groups)
            }
        }