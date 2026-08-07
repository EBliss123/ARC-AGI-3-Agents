from wake_phase.primitives import ASTNode, Operator, Variable, Constant, Parameter

def parameterize_tree(node: ASTNode, current_args: list = None) -> ASTNode:
    """
    Micro-Step 3.2: Auto-Parameterization.
    Walks down the tree and replaces all leaf nodes (Variables/Constants) 
    with blank Parameter slots to create a generalized blueprint.
    """
    if current_args is None:
        current_args = []
        
    # Keep structural operators intact, but parameterize their children
    if isinstance(node, Operator):
        left_param = parameterize_tree(node.left, current_args)
        right_param = parameterize_tree(node.right, current_args)
        return Operator(node.op_name, node.func, left_param, right_param)
        
    # If it's a leaf node, rip it out and replace with a parameter slot
    if isinstance(node, (Variable, Constant)):
        arg_index = len(current_args)
        current_args.append(node) # Store original for reference if needed
        return Parameter(arg_index)
        
    return node

if __name__ == "__main__":
    import operator
    
    print("--- Testing Auto-Parameterization ---")
    # Original: (x + 1)
    original_tree = Operator('+', operator.add, Variable('x'), Constant(1))
    print(f"Original Tree: {original_tree}")
    
    parameterized_tree = parameterize_tree(original_tree)
    print(f"Parameterized Blueprint: {parameterized_tree}")