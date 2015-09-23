"""Finalize various things before passing to solver."""

from epsilon.expression import *

def linear_equality_to_constraint(problem):
    """Move all linear equalities to constraints.

    TODO(mwytock): This can likely go away if we rationalize constraints vs
    indicator functions, i.e. by unifying separate/recombine/linearize
    phases."""

    obj_terms = []
    for f in problem.objective.arg:
        if (f.expression_type == Expression.INDICATOR and
            f.cone.cone_type == Cone.ZERO):
            problem.constraint.add().CopyFrom(f)
        else:
            obj_terms.append(f)

    problem.objective.CopyFrom(add(*obj_terms))

def transform(problem):
    linear_equality_to_constraint(problem)
    return problem
