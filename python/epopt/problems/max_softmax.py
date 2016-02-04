import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util
from epopt.functions import one_hot

def create(**kwargs):
    k = 3  #class
    m = 5 #instance
    n = 10 #dim
    p = 4  #p-largest
    q = 7
    X = problem_util.normalized_data_matrix(m,n,1)
    Y = np.random.randint(0, k-1, (q,m))

    Theta = cp.Variable(n,k)
    t = cp.Variable(q)
    f = cp.sum_largest(t, p)
    C = []
    for i in range(q):
        Yi = one_hot(Y[i], k)
        texp = cp.Variable(m)
        C.append(cp.log_sum_exp(X*Theta, axis=1) <= texp)
        C.append(cp.sum_entries(texp)-cp.sum_entries(cp.mul_elemwise(X.T.dot(Yi), Theta)) <= t[i])
    return cp.Problem(cp.Minimize(f), C)
