import torch
from typing import Tuple, List, Dict, Any, Callable
import operator
import random

def get_pixel_state(tensor: torch.Tensor, z: int, y: int, x: int) -> int:
    """Returns the absolute integer color value at a specific coordinate."""
    return tensor[z, y, x].item()

def get_deltas(s_t: torch.Tensor, s_next: torch.Tensor) -> List[Dict[str, tuple]]:
    """
    Extracts the absolute truth of what changed between two states.
    Returns a list of dictionaries containing the coordinate and the algebraic transition.
    """
    # Find all coordinates where the tensors mathematically differ
    difference_mask = s_t != s_next
    changed_indices = torch.nonzero(difference_mask)
    
    deltas = []
    for idx in changed_indices:
        z, y, x = idx.tolist()
        color_before = s_t[z, y, x].item()
        color_after = s_next[z, y, x].item()
        
        deltas.append({
            "coord": (z, y, x),
            "transition": (color_before, color_after)
        })
        
    return deltas

def calculate_distance(coord_a: Tuple[int, int, int], coord_b: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Calculates the strict algebraic distance (dz, dy, dx) between two coordinates.
    This provides the raw geometric scaffolding for future learning algorithms.
    """
    z_a, y_a, x_a = coord_a
    z_b, y_b, x_b = coord_b
    
    dz = z_b - z_a
    dy = y_b - y_a
    dx = x_b - x_a
    
    return (dz, dy, dx)

if __name__ == "__main__":
    # Quick mathematical verification of the primitives
    t1 = torch.zeros((1, 3, 3), dtype=torch.int8)
    t2 = torch.zeros((1, 3, 3), dtype=torch.int8)
    
    # Simulate a single pixel changing color from 0 to 4
    t2[0, 1, 1] = 4
    
    print("--- Testing Primitives ---")
    changes = get_deltas(t1, t2)
    print(f"Deltas detected: {changes}")
    
    if changes:
        origin = (0, 0, 0)
        target = changes[0]["coord"]
        dist = calculate_distance(origin, target)
        print(f"Distance from origin to changed pixel: {dist}")

class ASTNode:
    """Base class for all mathematical nodes in the tree."""
    def evaluate(self, context: Dict[str, Any]) -> Any:
        raise NotImplementedError
        
    def get_complexity(self) -> int:
        raise NotImplementedError

class Variable(ASTNode):
    """Represents an absolute coordinate or color (z, y, x, color)."""
    def __init__(self, name: str):
        self.name = name
        
    def evaluate(self, context: Dict[str, Any]) -> Any:
        return context[self.name]
        
    def get_complexity(self) -> int:
        return 1
        
    def __repr__(self) -> str:
        return self.name

class Constant(ASTNode):
    """Represents a raw integer (-1, 0, 1, 2, etc.)."""
    def __init__(self, value: int):
        self.value = value
        
    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.value
        
    def get_complexity(self) -> int:
        return 1
        
    def __repr__(self) -> str:
        return str(self.value)

class Operator(ASTNode):
    """Combines two nodes using strict algebraic logic (+, -, ==)."""
    def __init__(self, op_name: str, func: Callable, left: ASTNode, right: ASTNode):
        self.op_name = op_name
        self.func = func
        self.left = left
        self.right = right
        
    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.func(self.left.evaluate(context), self.right.evaluate(context))
        
    def get_complexity(self) -> int:
        return 1 + self.left.get_complexity() + self.right.get_complexity()
        
    def __repr__(self) -> str:
        return f"({self.left} {self.op_name} {self.right})"

class FunctionCall(ASTNode):
    """Executes a parameterized blueprint from the Sleep Phase Cache."""
    def __init__(self, fn_name: str, args: list):
        self.fn_name = fn_name
        self.args = args
        
    def evaluate(self, context: Dict[str, Any]) -> Any:
        # 1. Resolve arguments using the current coordinates
        resolved_args = [arg.evaluate(context) for arg in self.args]
        
        # 2. Fetch the blueprint from the context's cache
        blueprint = context['cache'].functions[self.fn_name].blueprint
        
        # 3. Create a sub-context and execute the cached function
        sub_context = context.copy()
        sub_context['args'] = resolved_args
        return blueprint.evaluate(sub_context)
        
    def get_complexity(self) -> int:
        # Reusing a function is mathematically cheaper than evolving new math!
        return 1 + sum(arg.get_complexity() for arg in self.args)
        
    def __repr__(self) -> str:
        args_str = ", ".join(map(str, self.args))
        return f"{self.fn_name}({args_str})"

BASE_OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '==': operator.eq,
    '!=': operator.ne
}

BASE_VARIABLES = ['y', 'x', 'color']

class Parameter(ASTNode):
    """An empty slot for auto-parameterized functions."""
    def __init__(self, arg_index: int):
        self.arg_index = arg_index
        
    def evaluate(self, context: Dict[str, Any]) -> Any:
        # Pulls from an 'args' list that the generalized function will provide
        return context['args'][self.arg_index]
        
    def get_complexity(self) -> int:
        return 1
        
    def __repr__(self) -> str:
        return f"arg_{self.arg_index}"

class ReadColor(ASTNode):
    """Reads the color of a pixel at a relative offset (dy, dx)."""
    def __init__(self, dy: ASTNode, dx: ASTNode):
        self.dy = dy
        self.dx = dx
        
    def evaluate(self, context: Dict[str, Any]) -> Any:
        target_y = context["y"] + int(self.dy.evaluate(context))
        target_x = context["x"] + int(self.dx.evaluate(context))
        grid = context["grid"]
        
        # Return -1 if the agent tries to look out of bounds
        max_y, max_x = grid.shape
        if 0 <= target_y < max_y and 0 <= target_x < max_x:
            return int(grid[target_y, target_x].item())
        return -1
        
    def get_complexity(self) -> int:
        return 1 + self.dy.get_complexity() + self.dx.get_complexity()
        
    def __repr__(self) -> str:
        return f"ReadColor({self.dy}, {self.dx})"