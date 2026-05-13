import re
from sympy import symbols, Eq, solve, sympify
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# All single-letter variables we support
VARIABLE_LETTERS = set('abcdefghijklmnopqrstuvwxyz')


def parse_expression(expr_string):
    """Clean and normalise a recognised expression string."""
    expr_string = expr_string.strip()
    expr_string = re.sub(r'\s+', '', expr_string)
    return expr_string


def _find_variables(expr_str):
    """Return the set of single-letter variables in the expression."""
    found = set()
    for ch in expr_str:
        if ch in VARIABLE_LETTERS:
            found.add(ch)
    return found


def _insert_implicit_mul(expr_str):
    """Insert '*' for implicit multiplication patterns.

    Handles:
        2x   -> 2*x
        x2   -> x*2
        xy   -> x*y
        2(   -> 2*(
        )2   -> )*2
        )x   -> )*x
        x(   -> x*(
    """
    result = []
    for i, ch in enumerate(expr_str):
        if i > 0:
            prev = expr_str[i - 1]
            # digit followed by letter or '('
            if prev.isdigit() and (ch.isalpha() or ch == '('):
                result.append('*')
            # letter followed by digit or '('
            elif prev.isalpha() and (ch.isdigit() or ch == '('):
                result.append('*')
            # letter followed by letter (different variables)
            elif prev.isalpha() and ch.isalpha():
                result.append('*')
            # ')' followed by digit, letter or '('
            elif prev == ')' and (ch.isdigit() or ch.isalpha() or ch == '('):
                result.append('*')
            # digit or letter followed by '(' already handled above
        result.append(ch)
    return ''.join(result)


def safe_parse(expr_str):
    """Parse an expression string into a SymPy expression."""
    expr_str = _insert_implicit_mul(expr_str)
    try:
        return parse_expr(expr_str, transformations=TRANSFORMATIONS)
    except Exception:
        return sympify(expr_str)


def evaluate_expression(expr_string):
    """Evaluate an expression or solve an equation.

    Returns a human-readable result string.

    Supported forms:
        '2+3='        -> '= 5'
        '2x+9=9'      -> 'x = 0'
        '3x-6=0'      -> 'x = 2'
        '2+3'         -> '= 5'
    """
    expr_string = parse_expression(expr_string)
    if not expr_string:
        return "Empty expression"

    variables = _find_variables(expr_string)

    if '=' in expr_string:
        parts = expr_string.split('=')
        # Filter out empty parts (trailing '=')
        parts = [p for p in parts if p.strip()]

        if len(parts) == 0:
            return "Invalid equation"

        # --- Pure arithmetic: "2+3=" ---
        if len(parts) == 1 and not variables:
            try:
                return f"= {safe_parse(parts[0])}"
            except Exception:
                return f"Cannot evaluate: {parts[0]}"

        # --- Equation with variable(s): "2x+9=9" ---
        if variables:
            if len(parts) == 1:
                # "2x+9=" => treat right side as 0
                lhs_str = parts[0]
                rhs_str = '0'
            elif len(parts) == 2:
                lhs_str = parts[0]
                rhs_str = parts[1]
            else:
                return "Invalid: multiple '=' signs"

            try:
                lhs = safe_parse(lhs_str)
                rhs = safe_parse(rhs_str)
                equation = Eq(lhs, rhs)

                # Pick the variable to solve for (prefer x, then alphabetical)
                if len(variables) == 1:
                    var = symbols(variables.pop())
                else:
                    if 'x' in variables:
                        var = symbols('x')
                    else:
                        var = symbols(sorted(variables)[0])

                solutions = solve(equation, var)
                if not solutions:
                    return "No solution"
                if len(solutions) == 1:
                    return f"{var} = {solutions[0]}"
                return ', '.join(f"{var} = {s}" for s in solutions)
            except Exception as e:
                return f"Cannot solve: {e}"

        # --- Pure arithmetic equality check: "5=5" ---
        if len(parts) == 2 and not variables:
            try:
                left_val = safe_parse(parts[0])
                right_val = safe_parse(parts[1])
                if left_val == right_val:
                    return "True"
                return f"False ({left_val} ≠ {right_val})"
            except Exception as e:
                return f"Cannot evaluate: {e}"

        return "Invalid equation"

    # --- No '=' sign at all ---
    if variables:
        return "Draw = to solve"
    try:
        return f"= {safe_parse(expr_string)}"
    except Exception as e:
        return f"Cannot evaluate: {e}"


def evaluate_for_canvas(expr_string):
    """Evaluate for display on the drawing canvas.

    For equations: returns 'x = 0'
    For arithmetic: returns the numeric result
    """
    expr_string = parse_expression(expr_string)
    if not expr_string:
        return "?"

    variables = _find_variables(expr_string)

    # Equation with variable
    if variables:
        # Split on '=' and solve
        parts = expr_string.split('=')
        parts = [p for p in parts if p.strip()]

        if len(parts) == 1:
            lhs_str = parts[0]
            rhs_str = '0'
        elif len(parts) == 2:
            lhs_str = parts[0]
            rhs_str = parts[1]
        else:
            return "?"

        try:
            lhs = safe_parse(lhs_str)
            rhs = safe_parse(rhs_str)
            equation = Eq(lhs, rhs)

            if len(variables) == 1:
                var = symbols(variables.pop())
            elif 'x' in variables:
                var = symbols('x')
            else:
                var = symbols(sorted(variables)[0])

            solutions = solve(equation, var)
            if not solutions:
                return "No solution"
            if len(solutions) == 1:
                return f"{var} = {solutions[0]}"
            return ', '.join(f"{var} = {s}" for s in solutions)
        except Exception:
            return "?"

    # Pure arithmetic
    try:
        result = safe_parse(expr_string)
        return str(result)
    except Exception:
        return "?"
