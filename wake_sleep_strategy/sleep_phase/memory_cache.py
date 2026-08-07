from typing import Dict, List
from wake_phase.primitives import ASTNode

class CachedFunction:
    """A wrapper for a parameterized blueprint that tracks its evolutionary success."""
    def __init__(self, name: str, blueprint: ASTNode):
        self.name = name
        self.blueprint = blueprint
        self.usage_score = 0
        self.is_active = True

    def promote(self):
        self.usage_score += 1

    def demote(self):
        self.usage_score -= 1
        # If a rule fails too often, it is banished to Deep Storage
        if self.usage_score < -2:
            self.is_active = False

class HierarchicalCache:
    """Micro-Step 3.4: Safe Memory Management."""
    def __init__(self):
        self.functions: Dict[str, CachedFunction] = {}
        self.next_fn_id = 0

    def store(self, blueprint: ASTNode) -> str:
        """Saves a newly parameterized tree into memory as an anonymous function (e.g., fn_0)."""
        fn_name = f"fn_{self.next_fn_id}"
        self.functions[fn_name] = CachedFunction(fn_name, blueprint)
        self.next_fn_id += 1
        return fn_name

    def get_active_functions(self) -> List[ASTNode]:
        """Returns only the highly successful blueprints for the Wake Phase mutator to use."""
        return [fn.blueprint for fn in self.functions.values() if fn.is_active]

    def record_usage(self, fn_name: str, success: bool):
        """Updates the hierarchical standing of a tool based on cross-frame testing."""
        if fn_name in self.functions:
            if success:
                self.functions[fn_name].promote()
            else:
                self.functions[fn_name].demote()

if __name__ == "__main__":
    from wake_phase.primitives import Parameter, Operator
    import operator

    print("--- Testing Hierarchical Cache ---")
    cache = HierarchicalCache()
    
    # Simulate storing our newly parameterized blueprint: (arg_0 + arg_1)
    mock_blueprint = Operator('+', operator.add, Parameter(0), Parameter(1))
    fn_name = cache.store(mock_blueprint)
    print(f"Stored generalized blueprint as: {fn_name}")
    
    # Simulate the agent using it successfully 3 times across different frames
    for _ in range(3): 
        cache.record_usage(fn_name, success=True)
    print(f"After 3 wins -> Score: {cache.functions[fn_name].usage_score} | Active: {cache.functions[fn_name].is_active}")
    
    # Simulate the agent applying it to a new game where the physics changed, failing 6 times
    for _ in range(6): 
        cache.record_usage(fn_name, success=False)
    print(f"After 6 losses -> Score: {cache.functions[fn_name].usage_score} | Active: {cache.functions[fn_name].is_active}")