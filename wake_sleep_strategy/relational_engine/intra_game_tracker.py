from typing import Dict, List
from wake_phase.primitives import ASTNode

class TrajectoryTracker:
    """
    Milestone 5: Intra-Game Trajectories.
    Tracks the chronological sequence of win conditions within a single game
    to predict how the rules will mutate in the next level.
    """
    def __init__(self):
        # Key: Game ID, Value: Chronological list of winning ASTs
        self.game_trajectories: Dict[str, List[ASTNode]] = {}

    def record_level_goal(self, game_id: str, goal_ast: ASTNode):
        """Micro-Step 5.1: Build Goal Trajectories."""
        if game_id not in self.game_trajectories:
            self.game_trajectories[game_id] = []
        self.game_trajectories[game_id].append(goal_ast)

    def predict_next_level_goal(self, game_id: str) -> List[ASTNode]:
        """
        Micro-Step 5.3: Seeding the Next Level (Forward Prediction).
        Returns the goal from the most recently beaten level to seed the next level's mutator.
        Returns a list to match the SeedGenerator's expected input format.
        """
        if game_id in self.game_trajectories and len(self.game_trajectories[game_id]) > 0:
            # Return the most recent goal as the primary hypothesis
            return [self.game_trajectories[game_id][-1]]
        return []

if __name__ == "__main__":
    from wake_phase.primitives import Variable, Constant, Operator
    import operator
    
    print("--- Testing Intra-Game Tracker ---")
    tracker = TrajectoryTracker()
    game = "gravity_blocks"
    
    # Mock beating Level 1
    level_1_goal = Operator('==', operator.eq, Variable('z'), Constant(1))
    tracker.record_level_goal(game, level_1_goal)
    print(f"Recorded Level 1 Goal: {level_1_goal}")
    
    # Agent steps into Level 2 and asks for a prediction
    predictions = tracker.predict_next_level_goal(game)
    print(f"Entering Level 2. Seed Generator Prediction: {predictions}")