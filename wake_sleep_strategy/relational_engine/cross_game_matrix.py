from typing import List, Dict, Set
from wake_phase.primitives import ASTNode

class RelationalMatrix:
    """
    Milestone 4: Cross-Game Relational Pairing.
    Maps the teleology of the game: which physical mechanics historically 
    co-occur with specific goals.
    """
    def __init__(self):
        # Key: Game ID, Value: Dict with 'physics' (Set of fn names) and 'goal' (ASTNode)
        self.solved_games: Dict[str, Dict] = {}

    def record_game_solution(self, game_id: str, physics_ast_names: List[str], win_condition_ast: ASTNode):
        """Micro-Step 4.1: Extracting Game-Specific Pairs."""
        self.solved_games[game_id] = {
            "physics": set(physics_ast_names),
            "goal": win_condition_ast
        }

    def query_historical_goals(self, active_physics_names: List[str]) -> List[ASTNode]:
        """
        Micro-Step 4.2 & 4.4: Relational Seeding.
        Returns historically paired goals based on the overlapping physics mechanics.
        """
        current_physics = set(active_physics_names)
        suggested_goals = []
        
        for game_id, data in self.solved_games.items():
            historical_physics = data["physics"]
            # If there is any intersection in the physics used, suggest the historical goal
            if current_physics.intersection(historical_physics):
                suggested_goals.append(data["goal"])
                
        return suggested_goals

if __name__ == "__main__":
    from wake_phase.primitives import Variable, Constant, Operator
    import operator

    print("--- Testing Relational Matrix ---")
    matrix = RelationalMatrix()
    
    # Mocking a solved Game A (Physics: fn_0, fn_1 -> Goal: z == 1)
    goal_a = Operator('==', operator.eq, Variable('z'), Constant(1))
    matrix.record_game_solution("game_A", ["fn_0", "fn_1"], goal_a)
    print(f"Recorded Game A. Goal: {goal_a}")
    
    # Mocking a solved Game B (Physics: fn_2 -> Goal: color == 4)
    goal_b = Operator('==', operator.eq, Variable('color'), Constant(4))
    matrix.record_game_solution("game_B", ["fn_2"], goal_b)
    print(f"Recorded Game B. Goal: {goal_b}")
    
    # Agent enters Game C, discovers it uses 'fn_1' (from Game A) and 'fn_3' (New)
    print("\nAgent encounters new game with physics: ['fn_1', 'fn_3']")
    predictions = matrix.query_historical_goals(["fn_1", "fn_3"])
    
    print(f"Suggested Goals based on Relational Overlap: {predictions}")