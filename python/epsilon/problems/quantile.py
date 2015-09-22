import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def create(m, n, k):
    np.random.seed(0)

    # Generate data
    x = np.random.rand(m)*2*np.pi - np.pi
    y = np.sin(x) + 0.1*(x+np.pi)*np.random.randn(m)
    alphas = np.linspace(1./k, 1-1./k, k-1)

    # RBF features
    mu_rbf = np.array([np.linspace(-np.pi-1, np.pi+1, n)])
    mu_sig = np.median(np.sqrt((mu_rbf.T - mu_rbf)**2))
    X = np.hstack([np.exp(-(mu_rbf.T - x).T**2/(2*mu_sig**2)), np.ones((m,1))])

    Theta = cp.Variable(n+1, len(alphas))

    quantile_loss = lambda x, alpha : cp.sum_entries(cp.max_elemwise(-alpha*x, (1-alpha)*x))
    f = sum([quantile_loss(X*Theta[:,i] - y, alpha) for i,alpha in enumerate(alphas)])
    DXT = X*(Theta[:,1:] - Theta[:,:-1])
    C = [DXT >= 0]

    return cp.Problem(cp.Minimize(f), C)
