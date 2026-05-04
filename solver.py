import re
from sympy import sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def parse_expression(expr_string):
    expr_string = expr_string.strip()
    expr_string = re.sub(r'\s+', '', expr_string)
    return expr_string


def safe_parse(expr_str):
    try:
        return parse_expr(expr_str, transformations=TRANSFORMATIONS)
    except Exception:
        return sympify(expr_str)


def evaluate_expression(expr_string):
    expr_string = parse_expression(expr_string)
    if not expr_string:
        return "Empty expression"
    if '=' in expr_string:
        parts = expr_string.split('=')
        if len(parts) != 2:
            return "Invalid: multiple '=' signs"
        left = parts[0].strip()
        right = parts[1].strip()
        if left and not right:
            try:
                return f"= {safe_parse(left)}"
            except Exception:
                return f"Cannot evaluate: {left}"
        if not left and right:
            return f"= {right}"
        if left and right:
            try:
                left_val = safe_parse(left)
                right_val = safe_parse(right)
                if left_val == right_val:
                    return "True"
                return f"False ({left_val} != {right_val})"
            except Exception as e:
                return f"Cannot evaluate: {e}"
        return "Invalid equation"
    try:
        return f"= {safe_parse(expr_string)}"
    except Exception as e:
        return f"Cannot evaluate: {e}"


def evaluate_for_canvas(expr_string):
    expr_string = parse_expression(expr_string)
    if not expr_string:
        return "?"
    try:
        result = safe_parse(expr_string)
        return str(result)
    except Exception:
        return "?"
