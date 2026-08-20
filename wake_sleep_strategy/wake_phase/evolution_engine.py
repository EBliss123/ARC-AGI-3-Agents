import random
import torch
from typing import List, Dict
from wake_phase.fitness import apply_proposed_deltas, evaluate_fitness, evaluate_goal_fitness
from wake_phase.primitives import ASTNode, Variable, Constant, Operator, Parameter, FunctionCall, BASE_OPERATORS, BASE_VARIABLES, ReadColor, And, RelationalCondition, ActionCondition
from typing import Set, Tuple, Dict
import numpy as np

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

def extract_shifted_view(grid_np: np.ndarray, dy: int, dx: int, oob_val: int = -1) -> np.ndarray:
    """Returns the grid shifted by (dy, dx) with out-of-bounds padded to oob_val."""
    h, w = grid_np.shape
    shifted = np.full((h, w), oob_val, dtype=np.int16)
    
    y_start_src = max(0, dy)
    y_end_src = min(h, h + dy)
    x_start_src = max(0, dx)
    x_end_src = min(w, w + dx)
    
    y_start_dst = max(0, -dy)
    y_end_dst = min(h, h - dy)
    x_start_dst = max(0, -dx)
    x_end_dst = min(w, w - dx)
    
    if y_end_src > y_start_src and x_end_src > x_start_src:
        shifted[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = grid_np[y_start_src:y_end_src, x_start_src:x_end_src]
    return shifted

def minimize_conditions(s_t_np: np.ndarray, selected_conds: List[RelationalCondition], pos_mask: np.ndarray) -> Tuple[List[RelationalCondition], List[RelationalCondition]]:
    """Non-destructive backward minimization: retains minimal active set while archiving backup qualifiers."""
    if len(selected_conds) <= 1:
        return selected_conds, []

    active_conds = list(selected_conds)
    archived_backup = []

    # Attempt to deactivate redundant conditions in reverse order (except the base self check at index 0)
    for i in range(len(selected_conds) - 1, 0, -1):
        candidate_active = [c for idx, c in enumerate(active_conds) if idx != i]
        
        # Test if remaining active conditions alone produce 0 false positives
        combined_mask = np.ones_like(pos_mask, dtype=bool)
        for cond in candidate_active:
            shifted = extract_shifted_view(s_t_np, cond.dy, cond.dx)
            combined_mask &= (shifted == cond.target_color)
            
        # If removing it introduces no false positives, move it to backup archive
        if np.all(combined_mask == pos_mask):
            archived_backup.append(active_conds[i])
            active_conds = candidate_active

    return active_conds, archived_backup

def synthesize_transition_group(s_t_np: np.ndarray, c_before: int, pos_mask: np.ndarray, action_id: int = None) -> ASTNode:
    """Synthesizes minimal relative offset conditions with non-destructive condition preservation."""
    h, w = s_t_np.shape
    pos_coords = np.argwhere(pos_mask)
    if len(pos_coords) == 0:
        return Constant(0)

    # 1. Base precondition: self color match
    current_active_mask = (s_t_np == c_before)
    base_conds = [RelationalCondition(0, 0, c_before)]
    
    # 2. Extract shared invariants across positive coordinates
    p_first_y, p_first_x = pos_coords[0]
    candidate_offsets = []
    search_offsets = [(dy, dx) for dy in range(-8, 9) for dx in range(-8, 9) if not (dy == 0 and dx == 0)]
    
    for dy, dx in search_offsets:
        target_y = p_first_y + dy
        target_x = p_first_x + dx
        if 0 <= target_y < h and 0 <= target_x < w:
            target_color = int(s_t_np[target_y, target_x])
            shifted = extract_shifted_view(s_t_np, dy, dx)
            if np.all(shifted[pos_mask] == target_color):
                candidate_offsets.append((dy, dx, target_color))

    # 3. Vectorized greedy negative sifter
    selected_conds = list(base_conds)
    current_negatives = current_active_mask & (~pos_mask)

    while np.any(current_negatives) and candidate_offsets:
        best_cond = None
        best_eliminated = -1
        
        for dy, dx, c in candidate_offsets:
            shifted = extract_shifted_view(s_t_np, dy, dx)
            matching = (shifted == c)
            eliminated = np.sum(current_negatives & (~matching))
            if eliminated > best_eliminated:
                best_eliminated = eliminated
                best_cond = (dy, dx, c)
                
        if best_cond is None or best_eliminated <= 0:
            break
            
        dy, dx, c = best_cond
        selected_conds.append(RelationalCondition(dy, dx, c))
        candidate_offsets.remove(best_cond)
        
        shifted = extract_shifted_view(s_t_np, dy, dx)
        current_negatives = current_negatives & (shifted == c)

    # 4. Backward Minimization (Keep active minimal, archive the rest)
    minimal_conds, archived_conds = minimize_conditions(s_t_np, selected_conds, pos_mask)

    tree = minimal_conds[0]
    for cond in minimal_conds[1:]:
        tree = And(tree, cond)
        
    tree._archived_qualifiers = archived_conds
    if action_id is not None:
        tree = And(ActionCondition(action_id), tree)
    return tree

def evolve(s_t: torch.Tensor, s_next: torch.Tensor, dynamic_mask: torch.Tensor = None, action_id: int = None) -> EvolutionaryRule:
    """Direct synthesis with minimal qualifiers and non-destructive archive."""
    if dynamic_mask is None:
        dynamic_mask = (s_t != s_next)
        
    if not dynamic_mask.any():
        zero_node = Constant(0)
        zero_node._action_id = action_id
        return EvolutionaryRule(proposed_deltas=[], complexity=1, ast_tree=zero_node)

    s_t_np = s_t.detach().cpu().numpy().astype(np.int16)
    s_next_np = s_next.detach().cpu().numpy().astype(np.int16)
    dyn_mask_np = dynamic_mask.detach().cpu().numpy().astype(bool)
    
    transitions: Dict[Tuple[int, int], np.ndarray] = {}
    for y, x in np.argwhere(dyn_mask_np):
        c_b = int(s_t_np[y, x])
        c_a = int(s_next_np[y, x])
        key = (c_b, c_a)
        if key not in transitions:
            transitions[key] = np.zeros_like(dyn_mask_np, dtype=bool)
        transitions[key][y, x] = True

    group_rules = []
    for (c_b, c_a), group_mask in transitions.items():
        rule_ast = synthesize_transition_group(s_t_np, c_b, group_mask, action_id=action_id)
        group_rules.append(rule_ast)

    final_ast = group_rules[0]
    for next_rule in group_rules[1:]:
        final_ast = Operator("or", lambda a, b: bool(a) or bool(b), final_ast, next_rule)

    final_ast._action_id = action_id
    rule = EvolutionaryRule(
        proposed_deltas=[],
        complexity=final_ast.get_complexity(),
        ast_tree=final_ast
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