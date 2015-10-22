"""Implements the linear canonicalize transforms on the AST."""


# Helper functions
# TODO(mwytock): Move to cannicalize_util.py
def is_constant(expr):
    pass

def is_linear(expr):
    pass

def only_arg(expr):
    if len(expr.arg) != 1:
        raise CanonicalizeError("wrong number of args", expr)
    return expr.arg[0]

def dim(expr, index=None):
    if len(expr.size.dim) != 2:
        raise CanonicalizeError("wrong number of dimensions", expr)
    if index is None:
        return expr.size.dim[0]*expr.size.dim[1]
    else:
        return expr.size.dim[index]

# Transforms on the AST
def transform_variable(expr):
    return expr

def transform_add(expr):
    return expression.add(*(transform_expr(e) for e in expr.arg))

def transform_transpose(expr):
    return expression.linear_map(
        linear_map.transpose(dim(expr,0), dim(expr,1)),
        only_arg(expr))

def transform_index(expr):
    return expression.linear_map(
        linear_map.kronecker_product(
            linear_map.index(expr.key[1], dim(only_arg(expr),1)),
            linear_map.index(expr.key[0], dim(only_arg(expr),0))))

def transform_multiply_generic(expr, const_transform):
    if len(expr.arg) != 2:
        raise CanonicalizeError("wrong number of args", expr)

    if is_constant(expr.arg(0)):
        return expression.linear_map(
            linear_map.left_matrix_product(...),
            expr.arg(1))
    elif is_constant(expr.arg(1)):
        return expression.linear_map(
            linear_map.right_matrix_product(...),
            expr.arg(0))

    raise CanonicalizeError("multiplying two non constants", expr)

def transform_multiply(expr):
    return transform_multiply_generic(expr, linear_map.xx())

def transform_multiply_elementwise(expr):
    return transform_multiply_generic(expr, linear_map.yy())

def transform_negate(expr):
    return expression.linear_map(linear_map.negate(dim(expr)), only_arg(expr))

def transform_sum(expr):
    return expression.linear_map(linear_map.sum(dim(expr)), only_arg(expr))

def transform_linear_expr(expr):
    transform = locals()["transform_" + expr.expression_type().lower()]
    return transform(expr)

def transform_expr(expr):
    if is_linear(expr):
        return transform_linear_expr(expr)
    else:
        for arg in expr.arg:
            arg.CopyFrom(transform_expr(arg))
        return expr

def transform_problem(problem):
    validate.check_sum_of_prox(problem)
    f = [transform_expr(e) for e in problem.objective.arg]
    C = [transform_expr(e) for e in problem.constraint]
    return Problem(objective=add(f), constraint=C)
