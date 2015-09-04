
import distopt
import cvxpy as cp
import numpy as np

def create(A_, b_, N_, lam_):
    N = int(N_)
    lam = float(lam_)

    def A(i):
        return distopt.cvxpy_expr.DistributedConstant(
            distopt.Matrix(A_ + str(i)))
    def b(i):
        return distopt.cvxpy_expr.DistributedConstant(
            distopt.Matrix(b_ + str(i)))

    m, n = A(0).size
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(
        sum(cp.sum_squares(A(i)*x - b(i)) for i in range(N)) +
        lam*cp.norm(x,1)))
