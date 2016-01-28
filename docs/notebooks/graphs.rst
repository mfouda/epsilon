
Optimization over graphs
========================

This example is from `Hallac, Leskovec and Boyd, "Network Lasso:
Clustering and Optimization in Large Graphs"
(2015) <http://web.stanford.edu/~hallac/Network_Lasso.pdf>`__ which
considers a general class of optimization problems over a graphs (with
vertices :math:`\mathcal{V}` and edges :math:`\mathcal{E}`)

.. math::


   \DeclareMathOperator{\minimize}{minimize} \minimize \;\; \sum_{i \in \mathcal{V}}f_i(x_i) + \sum_{(j,k) \in \mathcal{E}} g_{jk}(x_j, x_k)

where the optimization variable :math:`x_i \in \mathbb{R}^p` is
associated with the graph vertex :math:`i`.

In particular, with each node we will associate a vector
:math:`a_i \in \mathbb{R}^{500}` and solve the problem

.. math::


   \minimize \;\; \sum_{i \in \mathcal{V}}\|x_i - a_i\|_2^2 + \lambda \sum_{(j,k) \in \mathcal{E}} \|x_j - x_k\|_2.

Conceptually, each node would like to have its :math:`x_i` variable
match :math:`a_i` but by regularizing the variables across the graph we
encourage adjacent :math:`x_j`, :math:`x_k` to be similar. The
regularization penalty :math:`\|x_j - x_k\|_2` (which is referred to as
"sum-of-norms" regularization or the "group fused lasso") will in
actually create a clustering effect, encouraging many of the weights to
be the *same* across neighbors.

.. code:: python

    import cvxpy as cp
    import epopt as ep
    import numpy as np
    import scipy.sparse as sp
    import snap

We generate a 3-regular random graph (every vertex has 3 neighbors)
using `SNAP for Python <http://snap.stanford.edu/snappy/index.html>`__:

.. code:: python

    # Generate a random graph
    N = 2000
    K = 3
    graph = snap.GenRndDegK(N, K)

Then, we write this problem in matrix form by introducing the
differencing operator
:math:`D \in \{-1,0,1\}^{|\mathcal{E}| \times |\mathcal{N}|}`; for each
edge between vertices :math:`j` and :math:`k`, we add the following row
to :math:`D`:

.. math::


   (0, \ldots \underset{\substack{\;\;\uparrow \\ \;\;j}}{-1},
   \ldots \underset{\substack{\uparrow \\ k}}{1}, \ldots 0).

This allows us to form the problem as

.. math::


   \minimize \;\; \|X - A\|_F^2 + \|DX\|_{2,1}

with :math:`X, A \in \mathbb{R}^{|\mathcal{N}| \times 500}`. Here
:math:`\|\cdot\|_F` denotes the Frobenius norm (the :math:`\ell_2`-norm
applied to the elements of a matrix) and :math:`\|\cdot\|_{2,1}` the
:math:`\ell_2/\ell_1` mixed norm:

.. math::


   \|A\|_{2,1} = \sum_{i=1}^m \left( \sum_{j=1}^n A_{ij}^2 \right)^{1/2}

for :math:`A \in \mathbb{R}^{m \times n}`.

In Python:

.. code:: python

    # Parameters
    N = 2000
    K = 3
    p = 500
    lam = 1
    
    # Generate random graph
    E = graph.GetEdges()
    
    # Construct differencing operator over graph
    data = np.hstack((np.ones(E), -np.ones(E)))
    i = np.hstack((np.arange(E), np.arange(E)))
    j = ([e.GetSrcNId() for e in graph.Edges()] +
         [e.GetDstNId() for e in graph.Edges()])
    D = sp.coo_matrix((data, (i, j)))
    
    # Formulate problem
    X = cp.Variable(N, p)
    A = np.random.randn(N, p)
    f = cp.sum_squares(X-A) + lam*cp.sum_entries(cp.pnorm(D*X, 2, axis=1))
    prob = cp.Problem(cp.Minimize(f))
    
    # Solve with Epsilon
    ep.solve(prob, verbose=True)


.. parsed-literal::

    Epsilon 0.2.4
    Compiled prox-affine form:
    objective:
      add(
        sum_square(add(var(X), scalar(-1.00)*const(A))),
        affine(dense(b)*var(y)),
        second_order_cone(var(z), var(w)))
    
    constraints:
      zero(add(kron(scalar(1.00), sparse(C))*var(X), scalar(-1.00)*var(w)))
      zero(add(var(y), scalar(-1.00)*var(z)))
    Epsilon compile time: 0.0402 seconds
    
    iter=0 residuals primal=4.64e+02 [6.67e+00] dual=6.56e+02 [8.21e+00]
    iter=40 residuals primal=1.60e-02 [2.46e+01] dual=1.33e-01 [1.86e-01]
    Epsilon solve time: 42.9604 seconds




.. parsed-literal::

    ('optimal', 94711.639709220675)



Thus, we are able to solve this problem with 2000 x 500 = 1M variables
regularized over a graph in about 40 seconds. Even more importantly,
this graph-based optimization framework can easily be modified to
incorporate many varieties of convex functions associated with nodes and
edges to model many interesting problem, refer to `the full
paper <http://web.stanford.edu/~hallac/Network_Lasso.pdf>`__ for more
examples.
