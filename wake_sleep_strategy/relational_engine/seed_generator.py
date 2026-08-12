from typing import List
from wake_phase.evolution_engine import EvolutionaryRule, generate_random_tree
from wake_phase.primitives import ASTNode

def generate_smart_population(suggested_goals: List[ASTNode], population_size: int = 10) -> List[EvolutionaryRule]:
    """
    Micro-Step 4.4: Relational Seeding.
    Seeds the evolutionary arena with historical best-guesses before falling back 
    to random mutations to maintain diversity.
    """
    population = []
    
    # 1. Inject the historical goal hypotheses first
    for goal_ast in suggested_goals:
        if len(population) < population_size:
            # Create a rule using the historical AST
            rule = EvolutionaryRule(
                proposed_deltas=[], 
                complexity=goal_ast.get_complexity(), 
                ast_tree=goal_ast
            )
            population.append(rule)
            
    # 2. Fill the remaining slots with random math to ensure genetic diversity
    while len(population) < population_size:
        random_tree = generate_random_tree(max_depth=2)
        rule = EvolutionaryRule(
            proposed_deltas=[], 
            complexity=1, 
            ast_tree=random_tree
        )
        population.append(rule)
        
    return population

if __name__ == "__main__":
    from wake_phase.primitives import Variable, Constant, Operator
    import operator
    
    print("--- Testing Seed Generator ---")
    # Mock a suggested goal returned from the Relational Matrix
    mock_goal = Operator('==', operator.eq, Variable('z'), Constant(1))
    
    # Generate a small population of 3 to easily view the output
    smart_pop = generate_smart_population([mock_goal], population_size=3)
    
    print(f"Generated Population Size: {len(smart_pop)}")
    print(f"Slot 0 (Seeded History): {smart_pop[0].ast_tree}")
    print(f"Slot 1 (Random Mutation): {smart_pop[1].ast_tree}")
    print(f"Slot 2 (Random Mutation): {smart_pop[2].ast_tree}")