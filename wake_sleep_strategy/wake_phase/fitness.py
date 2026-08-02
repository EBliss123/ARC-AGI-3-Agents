import torch
from typing import List, Dict

def apply_proposed_deltas(s_t: torch.Tensor, proposed_deltas: List[Dict]) -> torch.Tensor:
    """
    The Simulator: Applies the agent's theoretical changes to a copy of the starting board.
    Expects proposed_deltas format: [{"coord": (z, y, x), "new_color": int}]
    """
    s_pred = s_t.clone()
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