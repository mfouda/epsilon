import cvxpy as cp
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from epopt.problems import problem_util
from epopt.functions import one_hot

def create(**kwargs):
    # m>k
    k = kwargs['k']  #class
    m = kwargs['m']  #instance
    n = kwargs['n']  #dim
    p = 5   #p-largest
    q = 10
    X = problem_util.normalized_data_matrix(m,n,1)
    Y = np.random.randint(0, k-1, (q,m))

    Theta = cp.Variable(n,k)
    t = cp.Variable(q)
    texp = cp.Variable(m)
    f = cp.sum_largest(t, p)+cp.sum_entries(texp) + cp.sum_squares(Theta)
    C = []
    C.append(cp.log_sum_exp(X*Theta, axis=1) <= texp)
    for i in range(q):
        Yi = one_hot(Y[i], k)
        C.append(-cp.sum_entries(cp.mul_elemwise(X.T.dot(Yi), Theta)) == t[i])

    t_eval = lambda: np.array([
        -cp.sum_entries(cp.mul_elemwise(X.T.dot(one_hot(Y[i], k)), Theta)).value for i in range(q)])
    f_eval = lambda: cp.sum_largest(t_eval(), p).value \
        + cp.sum_entries(cp.log_sum_exp(X*Theta, axis=1)).value \
        + cp.sum_squares(Theta).value
    
    return cp.Problem(cp.Minimize(f), C), f_eval
