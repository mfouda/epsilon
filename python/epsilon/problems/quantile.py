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

    thetas = [cp.Variable(n) for i in xrange(k)]
    yp = [cp.Variable(m) for i in xrange(k)]
    f = sum([quantile_loss(yp[i] - y, alphas[i]) for i in xrange(k)])
    c = [yp[i] == X*thetas[i] for i in xrange(k)]
    c += [yp[i+1] - yp[i] >= 0 for i in xrange(k-1)]
    return cp.Problem(cp.Minimize(f), c)
