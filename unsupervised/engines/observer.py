"""
Purpose: The Ground-Truth Generator.
Responsibilities:
1. Takes grid_before and grid_after.
2. Uses PyTorch meshgrids to output a unified [x, y, old_color, new_color] tensor.
3. Completely stateless. No object recognition or physics concepts here.
"""