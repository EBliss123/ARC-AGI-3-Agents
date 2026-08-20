from typing import Dict, List, Any, Tuple
from wake_phase.primitives import ASTNode, Parameter, FunctionCall, Constant, RelationalCondition, ActionCondition, And, ConditionalColor

class CachedFunction:
    def __init__(self, name: str, blueprint: ASTNode, arity: int):
        self.name = name
        self.blueprint = blueprint
        self.arity = arity
        self.usage_count = 0
        self.is_active = True

class HierarchicalCache:
    """Stores reusable abstracted physics blueprints parameterized during Sleep Phase."""
    def __init__(self):
        self.functions: Dict[str, CachedFunction] = {}
        self._fn_counter = 1

    def get_active_functions(self) -> List[str]:
        return [k for k, v in self.functions.items() if v.is_active]

    def _extract_constants(self, node: ASTNode) -> Tuple[ASTNode, List[int]]:
        """Replaces concrete constants with Parameter slots and extracts argument values."""
        args = []

        def recurse(curr: ASTNode) -> ASTNode:
            if isinstance(curr, ActionCondition):
                idx = len(args)
                args.append(curr.target_action)
                class ParamActionCondition(ASTNode):
                    def __init__(self, arg_idx: int):
                        self.arg_idx = arg_idx
                    def evaluate(self, ctx):
                        val = ctx.get('args', [])[self.arg_idx] if 'args' in ctx else ctx.get('action_id')
                        return ctx.get('action_id') == val
                    def get_complexity(self):
                        return 1
                    def __repr__(self):
                        return f"(Action == arg_{self.arg_idx})"
                return ParamActionCondition(idx)

            elif isinstance(curr, RelationalCondition):
                idx_dy = len(args)
                args.append(curr.dy)
                idx_dx = len(args)
                args.append(curr.dx)
                idx_c = len(args)
                args.append(curr.target_color)

                class ParamRelationalCondition(ASTNode):
                    def __init__(self, p_dy: int, p_dx: int, p_c: int):
                        self.p_dy = p_dy
                        self.p_dx = p_dx
                        self.p_c = p_c
                    def evaluate(self, ctx):
                        args_list = ctx.get('args', [])
                        dy = args_list[self.p_dy]
                        dx = args_list[self.p_dx]
                        tc = args_list[self.p_c]
                        target_y = ctx["y"] + dy
                        target_x = ctx["x"] + dx
                        grid = ctx["grid"]
                        max_y, max_x = grid.shape
                        if 0 <= target_y < max_y and 0 <= target_x < max_x:
                            return int(grid[target_y, target_x].item()) == tc
                        return False
                    def get_complexity(self):
                        return 2
                    def __repr__(self):
                        return f"(S[y+arg_{self.p_dy}, x+arg_{self.p_dx}] == arg_{self.p_c})"
                return ParamRelationalCondition(idx_dy, idx_dx, idx_c)

            elif isinstance(curr, Constant):
                idx = len(args)
                args.append(curr.value)
                return Parameter(idx)

            elif isinstance(curr, And):
                return And(recurse(curr.left), recurse(curr.right))

            elif isinstance(curr, ConditionalColor):
                return ConditionalColor(recurse(curr.condition), recurse(curr.output_val))

            return curr

        blueprint = recurse(node)
        return blueprint, args

    def induce_function(self, atomic_rules: List[ConditionalColor]) -> Tuple[List[ASTNode], int]:
        """Compresses recurring atomic rules into parameterized functions."""
        compressed_rules = []
        structures: Dict[str, List[Tuple[int, ASTNode, List[int]]]] = {}

        for i, rule in enumerate(atomic_rules):
            blueprint, args = self._extract_constants(rule)
            struct_key = repr(blueprint)
            if struct_key not in structures:
                structures[struct_key] = []
            structures[struct_key].append((i, blueprint, args))

        # Register functions for structural templates that occur >= 2 times
        for struct_key, instances in structures.items():
            if len(instances) >= 2:
                fn_name = f"fn_{self._fn_counter}"
                self._fn_counter += 1
                blueprint = instances[0][1]
                arity = len(instances[0][2])
                self.functions[fn_name] = CachedFunction(fn_name, blueprint, arity)
                self.functions[fn_name].usage_count = len(instances)

        for rule in atomic_rules:
            blueprint, args = self._extract_constants(rule)
            struct_key = repr(blueprint)
            matched_fn = None
            for name, fn_obj in self.functions.items():
                if repr(fn_obj.blueprint) == struct_key:
                    matched_fn = name
                    break

            if matched_fn is not None:
                arg_nodes = [Constant(a) for a in args]
                compressed_rules.append(FunctionCall(matched_fn, arg_nodes))
            else:
                compressed_rules.append(rule)

        return compressed_rules, len(self.functions)