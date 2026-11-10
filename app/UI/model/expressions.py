"""Safe arithmetic for numeric model input."""
import ast
import math
import operator

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_expression(expression, variables=None):
    variables = variables or {}
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 256:
        raise ValueError('Enter an arithmetic expression of at most 256 characters.')
    try:
        tree = ast.parse(expression, mode='eval')
    except (SyntaxError, ValueError) as exc:
        raise ValueError('Invalid arithmetic expression.') from exc
    if len(list(ast.walk(tree))) > 64:
        raise ValueError('Expression is too complex.')
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.Name) and node.id in variables:
            value = variables[node.id]
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            value = _UNARY[type(node.op)](visit(node.operand))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 20:
                raise ValueError('Exponent is too large.')
            try:
                value = _BINARY[type(node.op)](left, right)
            except (ArithmeticError, OverflowError, TypeError) as exc:
                raise ValueError('Cannot calculate expression.') from exc
        else:
            raise ValueError('Only numeric arithmetic and known parameters are allowed.')
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError('Expression result must be finite.')
        return value
    return float(visit(tree.body))
