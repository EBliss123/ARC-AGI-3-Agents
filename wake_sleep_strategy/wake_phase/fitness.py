import torch
from typing import List, Dict
from wake_phase.primitives import ASTNode

def apply_proposed_deltas(s_t: torch.Tensor, proposed_deltas: List[Dict], ast_tree: ASTNode = None) -> torch.Tensor:
    """
    The Simulator: Applies the agent's theoretical changes to a copy of the starting board.
    Now supports evaluating algebraic AST math trees dynamically across the grid.
    """
    s_pred = s_t.clone()
    
    # 1. Execute the algebraic tree (if it exists) across every pixel
    if ast_tree is not None:
        max_z, max_y, max_x = s_t.shape
        for z in range(max_z):
            for y in range(max_y):
                for x in range(max_x):
                    current_color = int(s_t[z, y, x].item())
                    context = {"z": z, "y": y, "x": x, "color": current_color}
                    
                    try:
                        # If the tree evaluates to True for this pixel, apply a transition
                        # (Hardcoded to color 4 temporarily for structural testing)
                        if ast_tree.evaluate(context) == True:
                            s_pred[z, y, x] = 4
                    except Exception:
                        pass # Ignore invalid math like division by zero
                        
    # 2. Apply literal coordinates (used primarily by the seed population)
    for delta in proposed_deltas:
        z, y, x = delta["coord"]
        s_pred[z, y, x] = delta["new_color"]
        
    return s_pred

def calculate_pixel_error(s_pred: torch.Tensor, s_next: torch.Tensor) -> int:
    """Calculates the absolute algebraic truth of how many pixels the agent guessed wrong."""
    return int(torch.sum(s_pred != s_next).item())

def evaluate_fitness(s_next: torch.Tensor, s_pred: torch.Tensor, rule_complexity: int, complexity_penalty: float = 0.1) -> float:
    """
    Scores how well the agent's theoretical rules predicted reality.
    A score of 0.0 is perfect.
    
    Occam's Razor: We add a small mathematical penalty for rule complexity to mathematically
    force the agent to favor simple, universal laws of physics over convoluted guesses.
    """
    base_error = calculate_pixel_error(s_pred, s_next)
    total_score = base_error + (rule_complexity * complexity_penalty)
    
    return total_score

def evaluate_goal_fitness(s_t: torch.Tensor, is_win: bool, ast_tree: ASTNode) -> float:
    """Scores how accurately an AST predicts the Win State flag."""
    if ast_tree is None:
        return float('inf')
        
    # Testing a single coordinate context for the structural skeleton
    context = {"z": 0, "y": 0, "x": 0, "color": int(s_t[0, 0, 0].item())}
    try:
        prediction = bool(ast_tree.evaluate(context))
        error = 0.0 if prediction == is_win else 1.0
        return error + (ast_tree.get_complexity() * 0.1)
    except Exception:
        return float('inf')

if __name__ == "__main__":
    # Test block to verify the Simulator and Occam's Razor penalty
    t1 = torch.zeros((1, 3, 3), dtype=torch.int8)
    t2 = torch.zeros((1, 3, 3), dtype=torch.int8)
    
    # Reality: A pixel moves from (0,0) to (0,1)
    t2[0, 0, 1] = 4
    
    # Agent's Theory: Proposes that the pixel moved correctly
    agent_theory = [{"coord": (0, 0, 1), "new_color": 4}]
    
    print("--- Testing Fitness Engine ---")
    
    # Simulate the theory
    s_pred = apply_proposed_deltas(t1, agent_theory)
    
    # Evaluate a simple rule (complexity 1) vs a convoluted rule (complexity 10)
    score_simple = evaluate_fitness(t2, s_pred, rule_complexity=1)
    score_complex = evaluate_fitness(t2, s_pred, rule_complexity=10)
    
    print(f"Error for simple rule: {score_simple}")
    print(f"Error for complex rule: {score_complex}")
    print("The simple rule wins!")