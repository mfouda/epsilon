"""Create certain equality constraints from indicator functions."""

from epsilon.compiler import validate
from epsilon.expression import *
from epsilon.expression_pb2 import Expression, Problem, Cone

def is_equality_constraint(expr):
    return (expr.expression_type == Expression.INDICATOR and
            expr.cone.cone_type == Cone.ZERO )
    
def has_non_scalar_constant(expr):
    if (expr.expression_type == Expression.CONSTANT and
        dimension(expr) > 1):
        return True

    for arg in expr.arg:
        if has_non_scalar_constant(arg):
            return True
            
    return False
    
def transform(input):
    validate.check_sum_of_prox(input)
    
    prox = []
    constraints = [f for f in input.constraint]
    for f in input.objective.arg:
        if (is_equality_constraint(f) and not has_non_scalar_constant(f)):
            constraints.append(f)
        else:
            prox.append(f)

    if prox:
        objective = add(*prox)
    else:
        objective = add(constant(1, 1, 0))

    return Problem(objective=objective, constraint=constraints)

