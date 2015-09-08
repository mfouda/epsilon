"""Build prox problem formulation."""

from collections import defaultdict

from epsilon import expression_pb2
from epsilon.expression_pb2 import Expression, Curvature
from epsilon.prox_pb2 import ConsensusVariable, ProxFunction, ProxProblem
from epsilon.expression import *

def is_sum_squares(expr):
    return (expr.expression_type == Expression.POWER and
            expr.p == 2 and
            expr.arg[0].expression_type == Expression.P_NORM and
            expr.arg[0].p == 2)

def is_affine_elementwise(expr):
    return (expr.curvature.curvature_type == Curvature.AFFINE and
            expr.curvature.elementwise)

def add_expr(a, b):
    if a.expression_type == Expression.UNKNOWN:
        a.CopyFrom(b)
    elif a.expression_type == Expression.ADD:
        a.arg.add().CopyFrom(b)
    else:
        a.CopyFrom(add(a, b))

def vstack_expr(a, b):
    if a.expression_type == Expression.UNKNOWN:
        a.CopyFrom(b)
    elif a.expression_type == Expression.VSTACK:
        a.arg.add().CopyFrom(b)
    else:
        a.CopyFrom(vstack(a, b))

def function_vars(function):
    retval = {}
    for arg in function.arg:
        retval.update(expr_vars(arg))
    if function.HasField("affine"):
        retval.update(expr_vars(function.affine))
    if function.HasField("regularization"):
        retval.update(expr_vars(function.regularization))
    return retval

def rename_function_var(old_var_id, new_var_id, function):
    for arg in function.arg:
        rename_var(old_var_id, new_var_id, arg)
    if function.HasField("affine"):
        rename_var(old_var_id, new_var_id, function.affine)
    if function.HasField("regularization"):
        rename_var(old_var_id, new_var_id, function.regularization)

class State(object):
    """Represents the internal state in transforming expression tree to
    ProxProblem or list of ProxProblems."""

    def __init__(self):
        self.prox_prob = ProxProblem()
        self.dist_prox_probs = []
        self.alpha = 1
        self.prox_vars = 0
        self.affine_exprs = []
        self.regularization_exprs = []

    def add_affine(self, expr):
        self.affine_exprs.append(expr)

    def add_function(self, function, arg):
        f = self.prox_prob.prox_function.add()
        f.function = function
        f.alpha = self.alpha
        f.arg.add().CopyFrom(arg)
        self.alpha = 1

    def add_variable(self, size):
        variable_id = "prox:" + str(self.prox_vars)
        self.prox_vars += 1
        return variable(size.dim[0], size.dim[1], variable_id)

    def add_equality_constraint(self, expr):
        self.prox_prob.equality_constraint.add().CopyFrom(expr)

class ProxHandler(object):
    def match(self, expr):
        return False

    def arg(self, expr):
        return expr.arg[0]

    def accept_arg(self, arg):
        return is_affine_elementwise(arg)

class Norm1Handler(ProxHandler):
    function = ProxFunction.NORM_1
    def match(self, expr):
        return (expr.expression_type == Expression.P_NORM and
                expr.p == 1)

class Norm2Handler(ProxHandler):
    function = ProxFunction.NORM_2
    def match(self, expr):
        return (expr.expression_type == Expression.P_NORM and
                expr.p == 2)

class Norm12Handler(ProxHandler):
    function = ProxFunction.NORM_1_2
    def match(self, expr):
        return (expr.expression_type == Expression.SUM and
                expr.arg[0].expression_type == Expression.NORM_2_ELEMENTWISE)

    def arg(self, expr):
        m = dimension(expr.arg[0].arg[0])
        return hstack(*(reshape(arg, m, 1) for arg in expr.arg[0].arg))

class SumSquaresHandler(ProxHandler):
    function = ProxFunction.SUM_SQUARES
    def match(self, expr):
        return is_sum_squares(expr)

    def arg(self, expr):
        return expr.arg[0].arg[0]

    def accept_arg(self, arg):
        return arg.curvature.curvature_type == Curvature.AFFINE

class NegativeLogDetHandler(ProxHandler):
    function = ProxFunction.NEGATIVE_LOG_DET
    def match(self, expr):
        return (expr.expression_type == Expression.NEGATE and
                expr.arg[0].expression_type == Expression.LOG_DET)

    def arg(self, expr):
        return expr.arg[0].arg[0]

OBJECTIVE_HANDLERS = (
    Norm1Handler(),
    Norm2Handler(),
    Norm12Handler(),
    SumSquaresHandler(),
    NegativeLogDetHandler())

def convert_objective(expr, state):
    assert dimension(expr) == 1

    if (expr.curvature.curvature_type == Curvature.AFFINE or
        expr.curvature.curvature_type == Curvature.CONSTANT):
        state.add_affine(expr)

    elif expr.expression_type == Expression.MULTIPLY:
        num_constants = 0
        for i, arg in enumerate(expr.arg):
            if arg.curvature.curvature_type == Curvature.CONSTANT:
                num_constants += 1
                state.alpha *= scalar_constant(arg)
            else:
                idx = i
        assert num_constants == len(expr.arg) - 1
        convert_objective(expr.arg[idx], state)

    elif expr.expression_type == Expression.ADD:
        for arg in expr.arg:
            convert_objective(arg, state)

    else:
        handler_found = False
        for handler in OBJECTIVE_HANDLERS:
            if handler.match(expr):
                arg = handler.arg(expr)
                if not handler.accept_arg(arg):
                    t = state.add_variable(arg.size)
                    state.add_equality_constraint(add(arg, negate(t)))
                    arg = t

                state.add_function(handler.function, arg)
                handler_found = True
                break

        if not handler_found:
            print expr
            raise RuntimeError("No handler found")

def convert_constraint(constraint, state):
    if constraint.cone == expression_pb2.NON_NEGATIVE:
        assert len(constraint.arg) == 1
        neg = Expression(expression_type=Expression.NEGATE)
        neg.size.CopyFrom(constraint.arg[0].size)
        neg.arg.add().CopyFrom(constraint.arg[0])
        state.add_prox_function(ProxFunction.INDICATOR_NON_NEGATIVE, [neg])
    elif constraint.cone == expression_pb2.ZERO:
        assert len(constraint.arg) == 1
        state.add_equality_constraint(constraint.arg[0])

def is_elementwise(expr):
    if not expr.curvature.curvature_type == Curvature.AFFINE:
        return False

    if expr.expression_type == Expression.VARIABLE:
        return True

    if expr.expression_type in (
            Expression.MULTIPLY_ELEMENTWISE,
            Expression.ADD):
        return all(arg.curvature.elementwise or
                   arg.curvature.curvature_type == Curvature.CONSTANT
                   for arg in expr.arg)

    return False

def add_attributes(expr):
    """Add expression attributes helpful for translation."""
    for arg in expr.arg:
        add_attributes(arg)

    expr.curvature.elementwise = is_elementwise(expr)

def all_expressions(prob):
    return [prob.objective] + [arg for c in prob.constraint for arg in c.arg]

def finalize_flexible_terms(state):
    """Finalize affine, regularization terms that can be added to other prox
    functions."""

    # Add each affine/reg term to prox function with greatest intersection
    fs = state.prox_prob.prox_function
    for expr in state.affine_exprs:
        v = set(expr_vars(expr).keys())
        f = max(
            fs, key=lambda f: len(v.intersection(set(function_vars(f).keys()))))
        add_expr(f.affine, expr)

def add_local_copies(state):
    """If two prox functions refer to the same variable, add a copy."""

    fs = state.prox_prob.prox_function
    var_function_map = defaultdict(list)
    orig_vars = {}

    for f in state.prox_prob.prox_function:
        f_vars = function_vars(f)
        orig_vars.update(f_vars)
        for var_id in f_vars:
            var_function_map[var_id].append(f)

    for var_id, fs in var_function_map.iteritems():
        # Enumerate the functions backwards so that the last prox function keeps
        # the original variable id.
        for i, f in enumerate(fs[-2::-1]):
            new_var_id = "%s:%d" % (var_id, i)
            old_var = orig_vars[var_id]
            new_var = Expression()
            new_var.CopyFrom(old_var)
            new_var.variable.variable_id = new_var_id

            state.add_equality_constraint(add(old_var, negate(new_var)))
            rename_function_var(var_id, new_var_id, f)

def split_problem(state):
    """Split the problem for distributed solve.

    Currently, use simple algorithm that solves a single prox function with a
    consensus variable.
    """
    assert not state.prox_prob.HasField("equality_constraint")

    fs = state.prox_prob.prox_function
    var_function_map = defaultdict(list)

    for f in state.prox_prob.prox_function:
        f_vars = function_vars(f)
        for var_id in f_vars:
            var_function_map[var_id].append(f)

    for i, f in enumerate(state.prox_prob.prox_function):
        prob = ProxProblem(prox_function=[f])

        for var_id, var in function_vars(f).iteritems():
            num_instances = len(var_function_map[var_id])
            if num_instances == 1:
                continue

            # Use the original var as the consensus var
            consensus_var = Expression()
            consensus_var.CopyFrom(var)
            new_var_id = "%s:%d" % (var_id, i)
            var.variable.variable_id = new_var_id

            # Add equality constraint
            prob.equality_constraint.CopyFrom(
                add(var, negate(consensus_var)))

            # Add consensus variable
            prob.consensus_variable.add(
                variable_id=var_id,
                num_instances=num_instances)

            # Rename variable
            rename_function_var(var_id, new_var_id, prob.prox_function[0])

        state.dist_prox_probs.append(prob)

def convert_problem(prob, distributed=False):
    state = State()

    for expr in all_expressions(prob):
        add_attributes(expr)

    convert_objective(prob.objective, state)
    for constraint in prob.constraint:
        convert_constraint(constraint, state)

    finalize_flexible_terms(state)

    if distributed:
        split_problem(state)
        return state.dist_prox_probs
    else:
        add_local_copies(state)
        return [state.prox_prob]
