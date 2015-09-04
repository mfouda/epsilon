
import distopt
import cvxpy as cp
import numpy as np

def create(A_, b_):
    A = distopt.cvxpy_expr.DistributedConstant(distopt.Matrix(A_))
    b = distopt.cvxpy_expr.DistributedConstant(distopt.Matrix(b_))
    x = cp.Variable(A.shape.cols)
    return cp.Problem(cp.Minimize(cp.norm(A*x - b)))
