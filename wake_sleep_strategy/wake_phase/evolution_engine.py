import random
import torch
from typing import List, Dict
from wake_phase.primitives import ASTNode, Variable, Constant, Operator, Parameter, FunctionCall, BASE_OPERATORS, BASE_VARIABLES
from wake_phase.fitness import apply_proposed_deltas, evaluate_fitness, evaluate_goal_fitness

class EvolutionaryRule:
    def __init__(self, proposed_deltas: List[Dict], complexity: int, ast_tree: ASTNode = None):
        self.proposed_deltas = proposed_deltas
        self.complexity = complexity
        self.ast_tree = ast_tree
        self.fitness_score = float('inf')

def generate_seed_population(raw_deltas: List[Dict[str, tuple]]) -> List[EvolutionaryRule]:
    """
    Generation 0: The literal truth.
    High accuracy, but terrible complexity (1 point of complexity per literal pixel).
    """
    # Convert raw deltas into the format expected by the Simulator
    proposed = [{"coord": d["coord"], "new_color": d["transition"][1]} for d in raw_deltas]
    
    # The literal rule is as complex as the number of pixels changed
    seed_rule = EvolutionaryRule(proposed_deltas=proposed, complexity=len(proposed), ast_tree=None)
    return [seed_rule]

def generate_random_tree(max_depth: int = 2, cache=None) -> ASTNode:
    """Generates a random AST mathematical tree, utilizing the memory cache if available."""
    if max_depth <= 0 or random.random() < 0.4:
        if random.random() < 0.5:
            return Variable(random.choice(BASE_VARIABLES))
        else:
            return Constant(random.randint(-1, 2))
    else:
        # If the Sleep Phase cache has active functions, 30% chance to reuse a concept
        if cache is not None and len(cache.get_active_functions()) > 0 and random.random() < 0.3:
            active_fns = [name for name, fn in cache.functions.items() if fn.is_active]
            chosen_fn = random.choice(active_fns)
            args = [Variable(random.choice(BASE_VARIABLES)), Constant(random.randint(-1, 2))]
            return FunctionCall(chosen_fn, args)
            
        op_name = random.choice(list(BASE_OPERATORS.keys()))
        func = BASE_OPERATORS[op_name]
        left = generate_random_tree(max_depth - 1, cache)
        right = generate_random_tree(max_depth - 1, cache)
        return Operator(op_name, func, left, right)

def mutate(rule: EvolutionaryRule, cache=None) -> EvolutionaryRule:
    """Mutates the evolutionary rule by generating a new AST branch or simplifying."""
    new_deltas = list(rule.proposed_deltas)
    if len(new_deltas) > 1 and random.random() < 0.5:
        new_deltas.pop(random.randrange(len(new_deltas)))
    
    # Pass the memory cache down to the generator
    mutated_tree = generate_random_tree(max_depth=2, cache=cache)
    new_complexity = rule.complexity + mutated_tree.get_complexity() - 1
    
    return EvolutionaryRule(proposed_deltas=new_deltas, complexity=max(1, new_complexity), ast_tree=mutated_tree)

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
                s_pred = apply_proposed_deltas(s_t, rule.proposed_deltas, ast_tree=rule.ast_tree)
                rule.fitness_score = evaluate_fitness(s_next, s_pred, rule.complexity)
        
        # 3. Survival of the Fittest (Sort by lowest fitness score)
        offspring.sort(key=lambda x: x.fitness_score)
        
        # 4. Keep the top 50% for the next generation
        population = offspring[:max(1, len(offspring) // 2)]
        
    return population[0]

def evolve_win_condition(s_t: torch.Tensor, is_win: bool, generations: int = 5) -> EvolutionaryRule:
    """The Arena for evolving boolean logic that isolates the win state."""
    # Seed population with random ASTs instead of literal pixel deltas
    population = [
        EvolutionaryRule(proposed_deltas=[], complexity=1, ast_tree=generate_random_tree(max_depth=2)) 
        for _ in range(10)
    ]
    
    for gen in range(generations):
        offspring = list(population)
        # Create mutations
        for _ in range(len(population)):
            offspring.append(EvolutionaryRule([], 1, generate_random_tree(max_depth=2)))
            
        for rule in offspring:
            if rule.fitness_score == float('inf'):
                rule.fitness_score = evaluate_goal_fitness(s_t, is_win, rule.ast_tree)
                
        offspring.sort(key=lambda x: x.fitness_score)
        population = offspring[:10]
        
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