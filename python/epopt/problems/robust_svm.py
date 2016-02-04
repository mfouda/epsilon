
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from epopt.problems import problem_util

def create(m, n):
    mu = 1
    rho = 1
    sigma = 0.1

    A = problem_util.normalized_data_matrix(m, n, mu)
    x0 = sp.rand(n, 1, rho)
    x0.data = np.random.randn(x0.nnz)
    x0 = x0.toarray().ravel()

    b = np.sign(A.dot(x0) + sigma*np.random.randn(m))
    A[b>0,:] += 0.7*np.tile([x0], (np.sum(b>0),1))
    A[b<0,:] -= 0.7*np.tile([x0], (np.sum(b<0),1))

    P = la.block_diag(np.random.randn(n-1,n-1), 0)

    lam = 1
    x = cp.Variable(A.shape[1])

    # Straightforward formulation
    # TODO(mwytock): Fix compiler so this works
    # z = 1 - sp.diags([b],[0])*A*x + cp.norm1(P.T*x)
    # f = lam*cp.sum_squares(x) + cp.sum_entries(cp.max_elemwise(z, 0))

    t = cp.Variable(1)
    z = 1 - sp.diags([b],[0])*A*x + t
    f = lam*cp.sum_squares(x) + cp.sum_entries(cp.max_elemwise(z, 0))
    C = [cp.norm1(P.T*x) <= t]
    return cp.Problem(cp.Minimize(f), C)

#return cp.Problem(cp.Minimize(f))

    # ep.solve(prob)
    # print f.args[0].value, f.args[1].value



    # print f.value

    # return cp.Problem(cp.Minimize(f))



    # # Direct epigraph

    # # Generate data
    # X = np.hstack([np.random.randn(m,n), np.ones((m,1))])
    # theta0 = np.random.randn(n+1)
    # y = np.sign(X.dot(theta0) + 0.1*np.random.randn(m))
    # X[y>0,:] += np.tile([theta0], (np.sum(y>0),1))
    # X[y<0,:] -= np.tile([theta0], (np.sum(y<0),1))





    # # TODO(mwytock): write this as:
    # # f = (lam/2*cp.sum_squares(theta) +
    # #      problem_util.hinge(1 - y[:,np.newaxis]*X*theta+cp.norm1(P.T*theta)))

    # # already in prox form
    # t1 = cp.Variable(m)
    # t2 = cp.Variable(1)
    # z = cp.Variable(n+1)
    # f = lam/2*cp.sum_squares(theta) + problem_util.hinge(1-t1)
    # C = [t1 == y[:,np.newaxis]*X*theta - t2,
    #      cp.norm1(z) <= t2,
    #      P.T*theta == z]
    # return cp.Problem(cp.Minimize(f), C)
