import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(n, r=10, density=0.1):
    np.random.seed(0)

    L1 = np.random.randn(n,r)
    L2 = np.random.randn(r,n)
    L0 = L1.dot(L2)

    S0 = sp.rand(n, n, density)
    S0.data = 10*np.random.randn(len(S0.data))

    M = L0 + S0
    kap = np.abs(S0).sum()

    L = cp.Variable(n, n)
    S = cp.Variable(n, n)
    f = cp.norm_nuc.normNuc(L)
    C = [cp.norm1(S) <= kap,
         L + S == M]

    return cp.Problem(cp.Minimize(f), C)
