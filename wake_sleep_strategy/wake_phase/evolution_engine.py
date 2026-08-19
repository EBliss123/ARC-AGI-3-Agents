import random
import torch
from typing import List, Dict
from wake_phase.fitness import apply_proposed_deltas, evaluate_fitness, evaluate_goal_fitness
from wake_phase.primitives import ASTNode, Variable, Constant, Operator, Parameter, FunctionCall, BASE_OPERATORS, BASE_VARIABLES, ReadColor, And, RelationalCondition
from typing import Set, Tuple

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
    elif random.random() < 0.2:  # 20% chance to evolve a sensory node
        dy = generate_random_tree(max_depth - 1, cache)
        dx = generate_random_tree(max_depth - 1, cache)
        return ReadColor(dy, dx)
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
    
    # Take a leap of faith: 20% chance to completely wipe literal memory and trust the AST math
    if random.random() < 0.2:
        new_deltas = []
    elif len(new_deltas) > 0 and random.random() < 0.5:
        new_deltas.pop(random.randrange(len(new_deltas)))
    
    # Pass the memory cache down to the generator
    mutated_tree = generate_random_tree(max_depth=2, cache=cache)
    
    # Complexity is the remaining literal pixels PLUS the size of the math tree
    new_complexity = len(new_deltas) + mutated_tree.get_complexity()
    
    return EvolutionaryRule(proposed_deltas=new_deltas, complexity=max(1, new_complexity), ast_tree=mutated_tree)

def extract_pixel_invariants(grid: torch.Tensor, y: int, x: int, max_radius: int = 5) -> Set[Tuple[int, int, int]]:
    """Extracts all relative in-bounds (dy, dx, color) features for a single coordinate."""
    invariants = set()
    max_y, max_x = grid.shape
    for dy in range(-max_radius, max_radius + 1):
        for dx in range(-max_radius, max_radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < max_y and 0 <= nx < max_x:
                invariants.add((dy, dx, int(grid[ny, nx].item())))
    return invariants

def find_common_invariants(grid: torch.Tensor, positive_coords: List[Tuple[int, int]], max_radius: int = 5) -> Set[Tuple[int, int, int]]:
    """Intersects relative features across all changing coordinates."""
    if not positive_coords:
        return set()
    common = extract_pixel_invariants(grid, positive_coords[0][0], positive_coords[0][1], max_radius)
    for y, x in positive_coords[1:]:
        common &= extract_pixel_invariants(grid, y, x, max_radius)
        if not common:
            break
    return common

def sift_invariants(grid: torch.Tensor, positive_coords: List[Tuple[int, int]], candidate_pool: Set[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Greedily selects conditions from the invariant pool that eliminate all static false positives."""
    max_y, max_x = grid.shape
    pos_set = set(positive_coords)
    active_negatives = [(y, x) for y in range(max_y) for x in range(max_x) if (y, x) not in pos_set]
    
    selected_conditions = []
    available = set(candidate_pool)

    while active_negatives and available:
        best_cond = None
        best_eliminated_count = -1
        
        for dy, dx, target_c in available:
            eliminated = 0
            for ny, nx in active_negatives:
                py, px = ny + dy, nx + dx
                if not (0 <= py < max_y and 0 <= px < max_x) or int(grid[py, px].item()) != target_c:
                    eliminated += 1
            if eliminated > best_eliminated_count:
                best_eliminated_count = eliminated
                best_cond = (dy, dx, target_c)
        
        if best_cond is None or best_eliminated_count == 0:
            break
            
        selected_conditions.append(best_cond)
        available.remove(best_cond)
        
        # Filter remaining negatives
        dy, dx, target_c = best_cond
        active_negatives = [
            (ny, nx) for ny, nx in active_negatives
            if (0 <= ny + dy < max_y and 0 <= nx + dx < max_x) and int(grid[ny + dy, nx + dx].item()) == target_c
        ]
        
    return selected_conditions

def build_ast_from_conditions(conditions: List[Tuple[int, int, int]]) -> ASTNode:
    """Chains condition tuples into an AST of RelationalCondition and And nodes."""
    if not conditions:
        return Constant(1)
    nodes = [RelationalCondition(dy, dx, c) for dy, dx, c in conditions]
    tree = nodes[0]
    for next_node in nodes[1:]:
        tree = And(tree, next_node)
    return tree

def evolve(s_t: torch.Tensor, s_next: torch.Tensor, dynamic_mask: torch.Tensor = None, action_id: int = None) -> EvolutionaryRule:
    """
    Direct relational invariant synthesis for dynamic change masks.
    """
    if dynamic_mask is None:
        dynamic_mask = (s_t != s_next)
        
    pos_coords = [tuple(c.tolist()) for c in dynamic_mask.nonzero()]
    
    if not pos_coords:
        return EvolutionaryRule(proposed_deltas=[], complexity=1, ast_tree=Constant(0))
        
    candidate_invariants = find_common_invariants(s_t, pos_coords, max_radius=5)
    winning_conditions = sift_invariants(s_t, pos_coords, candidate_invariants)
    ast_tree = build_ast_from_conditions(winning_conditions)
    
    rule = EvolutionaryRule(
        proposed_deltas=[],
        complexity=ast_tree.get_complexity(),
        ast_tree=ast_tree
    )
    s_pred = apply_proposed_deltas(s_t, [], ast_tree=rule.ast_tree)
    rule.fitness_score = evaluate_fitness(dynamic_mask.int(), s_pred, rule.complexity)
    return rule

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