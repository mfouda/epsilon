import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def quantile_loss(y, XT, alphas):
    m, k = XT.size
    Y = np.tile(y, (k, 1)).T
    A = np.tile(alphas, (m, 1))
    return cp.sum_entries(
        cp.max_elemwise(
            cp.mul_elemwise( -A, XT - Y),
            cp.mul_elemwise(1-A, XT - Y)))

def create(m, n, k, p, sigma=0.1):
    # Generate data
    x = np.random.rand(m)*2*np.pi*p
    y = np.sin(x) + sigma*np.sin(x)*np.random.randn(m)
    alphas = np.linspace(1./(k+1), 1-1./(k+1), k)

    # RBF features
    mu_rbf = np.array([np.linspace(-1, 2*np.pi*p+1, n)])
    mu_sig = (2*np.pi*p+2)/n
    X = np.exp(-(mu_rbf.T - x).T**2/(2*mu_sig**2))

    # TODO(mwytock): We need extra variable here because we dont have a
    # sylvester solver for equations of the form AXB = C. At some point we'll
    # fix this.
    Theta = cp.Variable(n,k)
    f = quantile_loss(y, X*Theta, alphas)
    C = [X*(Theta[:,:-1] - Theta[:,1:]) >= 0]
    return cp.Problem(cp.Minimize(f), C)
