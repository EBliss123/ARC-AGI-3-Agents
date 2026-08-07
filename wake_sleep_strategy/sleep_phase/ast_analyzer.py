from wake_phase.primitives import ASTNode, Operator, Variable, Constant

def are_structurally_equivalent(node1: ASTNode, node2: ASTNode) -> bool:
    """
    Micro-Step 3.1: Semantic Equivalence Testing.
    Checks if two AST trees have the exact same structural shape (operator topology).
    It ignores the specific variable names or constant values at the leaves, setting 
    the stage for Auto-Parameterization.
    """
    # If both are operators, their operation and children's structure must match identically
    if isinstance(node1, Operator) and isinstance(node2, Operator):
        if node1.op_name != node2.op_name:
            return False
        return (are_structurally_equivalent(node1.left, node2.left) and 
                are_structurally_equivalent(node1.right, node2.right))
    
    # If both are leaves (Variable or Constant), they are structurally equivalent 
    # in shape, even if the actual values (like 'x' vs 'y') differ.
    is_leaf1 = isinstance(node1, (Variable, Constant))
    is_leaf2 = isinstance(node2, (Variable, Constant))
    
    return is_leaf1 and is_leaf2

if __name__ == "__main__":
    import operator
    
    print("--- Testing Semantic Equivalence in AST Analyzer ---")
    # Tree 1: (x + 1)
    tree1 = Operator('+', operator.add, Variable('x'), Constant(1))
    
    # Tree 2: (y + 1)
    tree2 = Operator('+', operator.add, Variable('y'), Constant(1))
    
    # Tree 3: (z == 1)
    tree3 = Operator('==', operator.eq, Variable('z'), Constant(1))
    
    print(f"Comparing {tree1} and {tree2} -> Match: {are_structurally_equivalent(tree1, tree2)}")
    print(f"Comparing {tree1} and {tree3} -> Match: {are_structurally_equivalent(tree1, tree3)}")