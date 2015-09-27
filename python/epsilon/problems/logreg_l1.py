
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(m, n):
    if n < 100:
        p = 1
    else:
        p = 100./n

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = A*sp.diags([1 / np.sqrt(np.sum(A**2, 0))], [0])

    x0 = sp.rand(n, 1, p)
    x0.data = np.random.randn(x0.nnz)
    b = np.sign(A*x0 + np.sqrt(0.1)*np.random.randn(m,1))

    lam = 0.1*np.sqrt(n)
    x = cp.Variable(n)

    # TODO(mwytock): This is nasty, need to make this easier to write in CVXPY
    # in a way that the problem description doesnt scale with the number of
    # examples!
    bA = sp.diags([-b.ravel()], [0])*A
    fi = [cp.log_sum_exp(cp.vstack(0, -bA[i,:]*x)) for i in range(m)]
    return cp.Problem(cp.Minimize(sum(fi)))
