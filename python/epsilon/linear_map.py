"""Operations on LinearMaps."""

import scipy.sparse as sp

from epsilon.expression_pb2 import LinearMap
from epsilon import constants

# Atomic linear maps
def kronecker_product(A, B):
    return LinearMap(
        linear_map_type=LinearMap.KRONECKER_PRODUCT,
        m=A.m*B.m,
        n=A.n*B.n,
        A=A, B=B)

def dense_matrix(constant, m, n):
    return LinearMap(
        linear_map_type=LinearMap.DENSE_MATRIX,
        m=m,
        n=n,
        constant=constant)

def sparse_matrix(constant, m, n):
    return LinearMap(
        linear_map_type=LinearMap.SPARSE_MATRIX,
        m=m,
        n=n,
        constant=constant)

def diagonal_matrix(constant, n):
    return LinearMap(
        linear_map_type=LinearMap.DIAGONAL_MATRIX,
        m=n,
        n=n,
        constant=constant)

def scalar(alpha, n):
    return LinearMap(
        linear_map_type=LinearMap.SCALAR,
        m=n,
        n=n,
        scalar=alpha)

# Operations on linear maps
def transpose(A):
    return LinearMap(
        linear_map_type=LinearMap.TRANSPOSE,
        m=A.n,
        n=A.m,
        A=A)

# Implementation of various linear maps in terms of atoms
def identity(n):
    return scalar(1, n)

def index(slice, n):
    m = slice.stop - slice.start
    A = sp.coo_matrix(
        np.ones(n),
        (np.arange(m), np.arange(slice.start, m, slice.step)))
    return sparse_matrix(constants.store(A), A.m, A.n)

def one_hot(i, n):
    """[0, ... 0, 1, 0, ...]."""
    a = sp.coo_matrix(([1], ([0], [i])), shape=(1, n))
    return sparse_matrix(constants.store(a), 1, A.n)

def sum(n):
    """All ones vector"""
    return dense_matrix(constants.store(np.ones(n)), 1, A.n)

def negate(n):
    return scalar(-1,n)

def left_matrix_product(A, n):
    if n==1:
        return A
    return kronecker_product(identity(n), A)

def right_matrix_product(B, m):
    if m==1:
        return transpose(B)
    return kronecker_product(transpose(B), identity(m))
