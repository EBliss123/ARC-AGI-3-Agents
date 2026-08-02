import random
import torch
from typing import List, Dict
from wake_phase.fitness import apply_proposed_deltas, evaluate_fitness

class EvolutionaryRule:
    def __init__(self, proposed_deltas: List[Dict], complexity: int):
        self.proposed_deltas = proposed_deltas
        self.complexity = complexity
        self.fitness_score = float('inf')

def generate_seed_population(raw_deltas: List[Dict[str, tuple]]) -> List[EvolutionaryRule]:
    """
    Generation 0: The literal truth.
    High accuracy, but terrible complexity (1 point of complexity per literal pixel).
    """
    # Convert raw deltas into the format expected by the Simulator
    proposed = [{"coord": d["coord"], "new_color": d["transition"][1]} for d in raw_deltas]
    
    # The literal rule is as complex as the number of pixels changed
    seed_rule = EvolutionaryRule(proposed_deltas=proposed, complexity=len(proposed))
    return [seed_rule]

def mutate(rule: EvolutionaryRule) -> EvolutionaryRule:
    """
    Placeholder for the deep-logic Mutator.
    Eventually, this will use the Probing API to convert absolute coordinates
    into relative logic (e.g., "turn color 4 if next to color 8").
    For this structural skeleton, it simulates compression by randomly dropping a 
    delta to test if the rule still perfectly predicts reality with less data.
    """
    new_deltas = list(rule.proposed_deltas)
    if len(new_deltas) > 1:
        new_deltas.pop(random.randrange(len(new_deltas)))
    
    # Reduced complexity because the rule requires fewer explicit instructions
    new_complexity = max(1, rule.complexity - 1)
    return EvolutionaryRule(proposed_deltas=new_deltas, complexity=new_complexity)

def evolve(s_t: torch.Tensor, s_next: torch.Tensor, raw_deltas: List[Dict[str, tuple]], generations: int = 5) -> EvolutionaryRule:
    """
    The Arena: Runs the natural selection cycle to find the most compressed, accurate rule.
    """
    population = generate_seed_population(raw_deltas)
    
    for gen in range(generations):
        # 1. Mutate to create new hypotheses
        offspring = []
        for rule in population:
            offspring.append(rule) # Keep parent
            offspring.append(mutate(rule)) # Create mutated child
            
        # 2. Evaluate Fitness in the Arena
        for rule in offspring:
            if rule.fitness_score == float('inf'): # Only score un-evaluated rules
                s_pred = apply_proposed_deltas(s_t, rule.proposed_deltas)
                rule.fitness_score = evaluate_fitness(s_next, s_pred, rule.complexity)
        
        # 3. Survival of the Fittest (Sort by lowest fitness score)
        offspring.sort(key=lambda x: x.fitness_score)
        
        # 4. Keep the top 50% for the next generation
        population = offspring[:max(1, len(offspring) // 2)]
        
    return population[0]

if __name__ == "__main__":
    # Test block to verify the Evolution Engine
    t1 = torch.zeros((1, 3, 3), dtype=torch.int8)
    t2 = torch.zeros((1, 3, 3), dtype=torch.int8)
    
    # Reality: Two pixels change to color 4
    t2[0, 0, 1] = 4
    t2[0, 1, 1] = 4
    
    # The raw extraction from primitives.py
    raw_extraction = [
        {"coord": (0, 0, 1), "transition": (0, 4)},
        {"coord": (0, 1, 1), "transition": (0, 4)}
    ]
    
    print("--- Testing Evolution Engine ---")
    best_rule = evolve(t1, t2, raw_extraction, generations=3)
    
    print(f"Winning Rule Fitness Score: {best_rule.fitness_score:.2f}")
    print(f"Winning Rule Complexity: {best_rule.complexity}")
    print(f"Predictions preserved: {len(best_rule.proposed_deltas)}")