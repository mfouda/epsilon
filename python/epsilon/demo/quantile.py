import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(m, n):
    m = int(m)
    n = int(n)

    # Generate data
    x = np.random.rand(m)*2*np.pi - np.pi
    y = np.sin(x) + 0.1*(x+np.pi)*np.random.randn(m)

    # RBF features
    mu_rbf = np.array([np.linspace(-np.pi-1, np.pi+1, n)])
    mu_sig = np.median(np.sqrt((mu_rbf.T - mu_rbf)**2))
    X = np.hstack([np.exp(-(mu_rbf.T - x).T**2/(2*mu_sig**2)), np.ones((m,1))])

    quantile_loss = lambda x,alpha : cp.sum_entries(cp.max_elemwise(-alpha*x, (1-alpha)*x))
    alphas = np.linspace(0.01, 0.99, 99)

    theta = cp.Variable(n+1, len(alphas))
    f = sum([quantile_loss(X*theta[:,i] - y, alpha) for i,alpha in enumerate(alphas)])
    DX = X*(theta[:,1:] - theta[:,:-1])
    C = [DX >= 0]

    return cp.Problem(cp.Minimize(f), C)
