import re
from sympy import symbols, Eq, solve, sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


x = symbols('x')

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def parse_expression(expr_string):
    expr_string = expr_string.strip()
    expr_string = re.sub(r'\s+', '', expr_string)
    expr_string = expr_string.replace('×', '*').replace('·', '*')
    return expr_string


def safe_parse(expr_str):
    local_dict = {'x': x}
    try:
        return parse_expr(expr_str, local_dict=local_dict, transformations=TRANSFORMATIONS)
    except Exception:
        return sympify(expr_str, locals=local_dict)


def evaluate_expression(expr_string):
    expr_string = parse_expression(expr_string)

    if not expr_string:
        return "Empty expression"

    has_variable = 'x' in expr_string

    if '=' in expr_string:
        parts = expr_string.split('=')

        if len(parts) != 2:
            return "Invalid equation: multiple '=' signs"

        left_str = parts[0].strip()
        right_str = parts[1].strip()

        if not left_str and not right_str:
            return "Invalid equation"

        if left_str and not right_str:
            if has_variable:
                return "Cannot solve: incomplete equation (nothing after '=')"
            else:
                try:
                    result = safe_parse(left_str)
                    return f"= {result}"
                except Exception:
                    return f"Cannot evaluate: {left_str}"

        if not left_str and right_str:
            return f"= {right_str}"

        if has_variable:
            try:
                left_expr = safe_parse(left_str)
                right_expr = safe_parse(right_str)
                equation = Eq(left_expr, right_expr)
                solution = solve(equation, x)
                if solution:
                    if len(solution) == 1:
                        return f"x = {solution[0]}"
                    else:
                        return f"x = {solution}"
                else:
                    return "No solution found"
            except Exception as e:
                return f"Cannot solve: {str(e)}"
        else:
            try:
                left_val = safe_parse(left_str)
                right_val = safe_parse(right_str)
                if left_val == right_val:
                    return "True (equation is valid)"
                else:
                    return f"False ({left_val} != {right_val})"
            except Exception as e:
                return f"Cannot evaluate: {str(e)}"
    else:
        if has_variable:
            return f"Expression: {expr_string} (need '=' to solve for x)"
        else:
            try:
                result = safe_parse(expr_string)
                return f"= {result}"
            except Exception as e:
                return f"Cannot evaluate: {str(e)}"
