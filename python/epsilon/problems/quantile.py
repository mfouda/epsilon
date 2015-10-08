import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def quantile_loss(x, alpha):
    return cp.sum_entries(cp.max_elemwise(-alpha*x, (1-alpha)*x))

def create(m, n, k):
    np.random.seed(0)

    # Generate data
    x = np.random.rand(m)*2*np.pi - np.pi
    y = np.sin(x) + 0.1*(x+np.pi)*np.random.randn(m)
    alphas = np.linspace(1./(k+1), 1-1./(k+1), k)

    # RBF features
    mu_rbf = np.array([np.linspace(-np.pi-1, np.pi+1, n)])
    mu_sig = np.median(np.sqrt((mu_rbf.T - mu_rbf)**2))
    X = np.exp(-(mu_rbf.T - x).T**2/(2*mu_sig**2))

    Theta = cp.Variable(n,k)
    XT = cp.Variable(m,k)
    f = sum([quantile_loss(XT[:,i] - y, alphas[i]) for i in xrange(k)])
    C = [XT == X*Theta,
         XT[:,:-1] - XT[:,1:] >= 0]
    return cp.Problem(cp.Minimize(f), C)
