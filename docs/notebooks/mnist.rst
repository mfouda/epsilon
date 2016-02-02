
MNIST image classification
==========================

In this example, we consider the classic machine learning dataset MNIST
and the task of classifying handwritten digits. By modern computer
vision standards this dataset is considered small, yet it is
sufficiently large that many standard classifiers (e.g. those in the
Python package ``sklearn``) require significant time to train a model.
Nonetheless, `Epsilon <http://epopt.io/>`__ is able to fit a model that
achieves near state-of-the-art accuracy in a few minutes.

.. figure:: mnist.png
   :alt: MNIST examples

   MNIST examples

The standard task is to train a multiclass classifier that can correctly
identify digits from their pixel intensity values. We will build a
classifier to perform this task using `mutlticlass hinge
loss <http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf>`__.

.. code:: python

    %matplotlib inline
    import io
    import urllib
    import cvxpy as cp
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.linalg as LA
    import epopt as ep
    
    
    data = "http://epopt.s3.amazonaws.com/mnist.npz"
    mnist = np.load(io.BytesIO(urllib.urlopen(data).read()))

Multiclass hinge loss
---------------------

The multiclass hinge loss is a piecewise linear convex surrogate for the
misclassification error in a multiclass problem; given a feature vector
:math:`x \in \mathbb{R}^n` and label :math:`y \in \{0,\ldots,k\}` we
incur loss

.. math::


   \max_j \; \{\theta_j^Tx + 1 - \delta_{j,y} \} - \theta_y^Tx

where :math:`\theta_j \in \mathbb{R}^{n}` is the weights for class
:math:`j` and :math:`\delta_{p,q}` is equal to :math:`1` if
:math:`p = q` and :math:`0` otherwise.

In order to minimize this function using CVXPY and Epsilon, we must
write down its definition in matrix form. For convenience, Epsilon
provides the ``multiclass_hinge_loss()`` function as well as several
other common loss functions occuring in machine learning, see
`functions.py <https://github.com/mwytock/epsilon/blob/master/python/epopt/functions.py>`__
for details.

.. code:: python

    def multiclass_hinge_loss(Theta, X, y):
        k = Theta.size[1]
        Y = one_hot(y, k)
        return (cp.sum_entries(cp.max_entries(X*Theta + 1 - Y, axis=1)) -
                cp.sum_entries(cp.mul_elemwise(X.T.dot(Y), Theta)))


.. parsed-literal::

    


We will also add a bit of :math:`\ell_2`-regularization on the parameter
vectors :math:`\theta_1, \ldots, \theta_k` to prevent over-fitting. The
final optimization problem is

.. math::


   \DeclareMathOperator{\minimize}{minimize} \minimize \;\; \sum_{i=1}^m \left( \max_j \; \{\theta_j^Tx_i + 1 - \delta_{j,y_i} \} - \theta_{y_i}^Tx_i \right) + \sum_{j=1}^k \lambda \|\theta_j\|_2^2

where the parameter :math:`\lambda > 0` controls the regularization. We
set up the problem in CVXPY and solve with Epsilon as follows:

.. code:: python

    # Problem data
    X = mnist["X"] / 255.   # raw pixel data scaled to [0, 1]
    y = mnist["Y"].ravel()  # labels {0, ..., 9}
    Xtest = mnist["Xtest"] / 255.
    ytest = mnist["Ytest"].ravel()
    
    # Parameters
    m, n = X.shape
    k = 10
    Theta = cp.Variable(n, k)
    lam = 1
    
    # Form problem with CVXPY and solve with Epsilon
    f = ep.multiclass_hinge_loss(Theta, X, y) + lam*cp.sum_squares(Theta)
    prob = cp.Problem(cp.Minimize(f))
    ep.solve(prob, verbose=True)
    
    # Get solution and compute train/test error
    def error(x, y):
        return 1 - np.sum(x == y) / float(len(x))
    
    Theta0 = np.array(Theta.value)
    print "Train error:", error(np.argmax(X.dot(Theta0), axis=1), y)
    print "Test error:", error(np.argmax(Xtest.dot(Theta0), axis=1), ytest)


.. parsed-literal::

    Epsilon 0.2.4
    Compiled prox-affine form:
    objective:
      add(
        affine(dense(A)*var(x)),
        non_negative(var(y)),
        affine(kron(dense(B), dense(C))*diag(D)*var(Z)),
        sum_square(var(W)))
    
    constraints:
      zero(add(add(kron(transpose(dense(B)), scalar(1.00))*var(x), scalar(-1.00)*add(kron(scalar(1.00), dense(K))*var(W), dense(e)*1.00, scalar(-1.00)*const(F))), scalar(-1.00)*var(y)))
      zero(add(var(Z), scalar(-1.00)*var(W)))
    Epsilon compile time: 1.4502 seconds
    
    iter=0 residuals primal=1.29e+05 [1.29e+03] dual=2.24e+02 [1.29e+03]
    iter=40 residuals primal=9.62e+00 [1.02e+01] dual=2.54e+01 [1.29e+03]
    Epsilon solve time: 38.7465 seconds
    Train error: 0.0853166666667
    Test error: 0.0891


Thus, a simple linear classifier on pixel intensities achieves a 8.9%
error rate on this task. This forms a reasonable baseline, but raw pixel
values are in fact poor predictors and we can do much better by
considering a nonlinear decision functions which we explore next.

Non-linear classifier using random Fourier features
---------------------------------------------------

It turns out we can fit a non-linear decision function by approximating
a Gaussian kernel using random Fourier features. In particular if we
transform the input data by

.. math::


   z(x) = \cos(Wx + b)

with :math:`W \in \mathbb{R}^{d \times n}` with elements sampled from a
zero-mean Normal distribution and :math:`b \in \mathbb{R}^d` with chosen
uniformly at random from :math:`[0, 2\pi]`, then

.. math::


   z(x)^Tz(x') \approx \exp \left( \frac{-\|x - x'\|_2^2}{2} \right),

for details see `Rahimi and Recht
(2007) <http://www.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>`__.
We will use this transformation to build a better classifier, with
preprocessing following that of `Agarwal et al.
(2014) <http://arxiv.org/abs/1310.1949>`__, `code available
here <https://github.com/fest/secondorderdemos>`__. This is
straightforward to implement in a few lines of Python:

.. code:: python

    def median_dist(X):
        """Compute the approximate median distance by sampling pairs."""
        k = 1<<20  # 1M random points
        i = np.random.randint(0, X.shape[0], k)
        j = np.random.randint(0, X.shape[0], k)
        return np.sqrt(np.median(np.sum((X[i,:] - X[j,:])**2, axis=1)))
        
    def pca(X, dim):
        """Perform centered PCA."""
        X = X - X.mean(axis=0)
        return LA.eigh(X.T.dot(X))[1][:,-dim:]
    
    # PCA and median trick
    np.random.seed(0)
    V = pca(mnist["X"], 50)
    X = mnist["X"].dot(V)
    sigma = median_dist(X)
    
    # Random features
    n = 4000
    W = np.random.randn(X.shape[1], n) / sigma
    b = np.random.uniform(0, 2*np.pi, n)
    X = np.cos(X.dot(W) + b)
    Xtest = np.cos(mnist["Xtest"].dot(V).dot(W) + b)

Given our transformed dataset we now have significantly more features
(the feature matrix, :math:`X \in \mathbb{R}^{60000 \times 4000}`) but
we still fit the model using the same method CVXPY/Epsilon and the same
method as before:

.. code:: python

    # Parameters
    m, n = X.shape
    k = 10
    Theta = cp.Variable(n, k)
    lam = 10
    
    # Form problem with CVXPY and solve with Epsilon
    f = ep.multiclass_hinge_loss(Theta, X, y) + lam*cp.sum_squares(Theta)
    prob = cp.Problem(cp.Minimize(f))
    ep.solve(prob, verbose=True)
    
    # Get solution and compute train/test error
    Theta0 = np.array(Theta.value)
    print "Train error:", error(np.argmax(X.dot(Theta0), axis=1), y)
    print "Test error:", error(np.argmax(Xtest.dot(Theta0), axis=1), ytest)


.. parsed-literal::

    Epsilon 0.2.4
    Compiled prox-affine form:
    objective:
      add(
        affine(dense(A)*var(x)),
        non_negative(var(y)),
        affine(kron(dense(B), dense(C))*diag(D)*var(Z)),
        sum_square(var(W)))
    
    constraints:
      zero(add(add(kron(transpose(dense(B)), scalar(1.00))*var(x), scalar(-1.00)*add(kron(scalar(1.00), dense(K))*var(W), dense(e)*1.00, scalar(-1.00)*const(F))), scalar(-1.00)*var(y)))
      zero(add(var(Z), scalar(-1.00)*var(W)))
    Epsilon compile time: 9.8725 seconds
    
    iter=0 residuals primal=7.12e+05 [7.12e+03] dual=2.71e+02 [7.12e+03]
    iter=30 residuals primal=6.94e+00 [7.43e+00] dual=1.70e+01 [7.12e+03]
    Epsilon solve time: 196.5668 seconds
    Train error: 0.00501666666667
    Test error: 0.0157


Our classifier now achieves an error rate of 1.57% improving
significantly over the baseline.

Critically, it only takes <3.5 minutes to train this classifier which is
significantly faster than many of the dedicated Python machine learning
packages (e.g. those provided by ``sklearn``).
