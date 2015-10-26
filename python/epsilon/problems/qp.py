"""Standard form QP."""

import numpy as np
import numpy.linalg as LA

def create(n):
    np.random.seed(0)

    # Generate a well-conditioned positive definite matrix
    P = np.random.rand(n,n);
    P = P + P.T;
    D, V = LA.eig(P);
    P = (V*(1+np.random.rand(n))).dot(V.T);

    q = np.random.randn(n);
    r = np.random.randn();

    l = np.random.randn(n);
    u = np.random.randn(n);
    lb = np.minimum(l,u);
    ub = np.maximum(l,u);

    x = cp.Variable(n)
    f = cp.Minimize(0.5*cp.quad_form(x, P) + q.T*x + r)
    C = [lb <= x, x <= ub]
    return cp.Problem(f, C)
