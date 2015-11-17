"""Operations on LinearMaps."""

import numpy as np
import scipy.sparse as sp

from epsilon import linear_map
from epsilon.expression_pb2 import Expression, LinearMap
from epsilon.expression_util import *

class LinearMapType(object):
    """Handle type conversion for linear maps."""
    def __init__(self, linear_map):
        self.linear_map = linear_map

    @property
    def basic(self):
        return self.linear_map.linear_map_type in (
            LinearMap.DENSE_MATRIX,
            LinearMap.SPARSE_MATRIX,
            LinearMap.DIAGONAL_MATRIX,
            LinearMap.SCALAR)

    @property
    def diagonal(self):
        return self.linear_map.linear_map_type in (
            LinearMap.DIAGONAL_MATRIX,
            LinearMap.SCALAR)

    @property
    def scalar(self):
        return self.linear_map.linear_map_type == LinearMap.SCALAR

    def copy(self):
        linear_map = LinearMap()
        linear_map.CopyFrom(self.linear_map)
        return LinearMapType(linear_map)

    def promote(self, other):
        assert self.basic
        assert other.basic
        if self.linear_map.linear_map_type > other.linear_map.linear_map_type:
            self.linear_map.linear_map_type = other.linear_map.linear_map_type

    def __add__(self, B):
        print "LinearMapType __add__"
        assert isinstance(B, LinearMapType)

        C = self.copy()
        C.promote(B)
        return C

    def __mul__(self, B):
        print "LinearMapType __mul__"
        if isinstance(B, AffineExpression):
            return B.__rmul__(self)
        assert isinstance(B, LinearMapType)

        C = self.copy()
        C.promote(B)
        return C


CONSTANT = "constant"
class AffineExpression(object):
    def __init__(self, linear_maps):
        self.linear_maps = linear_maps

    @property
    def diagonal(self):
        return all(A.diagonal for A in self.linear_maps.values())

    @property
    def scalar(self):
        return all(A.scalar for A in self.linear_maps.values())

    def __rmul__(self, A):
        assert isinstance(A, LinearMapType)

        print "AffineExpression __rmul__"
        C = AffineExpression(self.linear_maps.copy())
        for var_id, Bi in self.linear_maps.items():
            C.linear_maps[var_id] = A*Bi
        return C

    def __add__(self, B):
        assert isinstance(B, AffineExpression)

        print "AffineExpression __add__"
        C = AffineExpression(self.linear_maps.copy())
        for var_id, Bi in B.linear_maps.items():
            if var_id not in self.linear_maps:
                C.linear_maps[var_id] = Bi
            else:
                C.linear_maps[var_id] += Bi
        return C

# TODO(mwytock): memoize
def get_affine_expr(expr):
    if (expr.expression_type == Expression.CONSTANT or
        expr.expression_type == Expression.VARIABLE):
        var_id = (CONSTANT if expr.expression_type == Expression.CONSTANT else
                  expr.variable.variable_id)
        return AffineExpression({
            var_id: LinearMapType(linear_map.identity(dim(expr)))})

    elif expr.expression_type == Expression.ADD:
        return reduce(lambda A,B: A+B,
                      (get_affine_expr(arg) for arg in expr.arg))

    elif expr.expression_type == Expression.LINEAR_MAP:
        A = LinearMapType(expr.linear_map)
        return A*get_affine_expr(only_arg(expr))

    raise ExpressionError("unkonwn expr type", expr)


def is_diagonal(expr):
    return get_affine_expr(expr).diagonal

def is_scalar(expr):
    return get_affine_expr(expr).scalar
