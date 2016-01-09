.. Epsilon documentation master file, created by
   sphinx-quickstart on Fri Jan  8 11:13:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Scalable Convex Programming
===========================

Epsilon is a system for general convex programming using fast linear and
proximal operators.

As with existing convex programming frameworks (e.g. `CVX
<http://cvxr.com/cvx/>`_, `CVXPY <http://cvxpy.org/>`_, `Convex.jl
<http://convexjl.readthedocs.org/en/latest/>`_, etc.), users specify convex
optimization problems using a natural grammar for mathematical expressions,
composing functions in a way that is guaranteed to be convex by the rules of
disciplined convex programming. Given such an input, the Epsilon compiler
transforms the optimization problem into a mathematically equivalent form
consisting only of functions with efficient proximal operators---an intermediate
representation we refer to as *prox-affine form*. By reducing problems to this
form, Epsilon enables solving general convex problems using a large library of
fast proximal and linear operators and is often faster than existing approaches
by an order of magnitude or more.

In order to use Epsilon, form an optimization problem using CVXPY in the
usual way but solve it using Epsilon.

.. code:: python

   import numpy as np
   import cvxpy as cp
   import epopt as ep

   # Form lasso problem with CVXPY
   m = 5
   n = 10
   A = np.random.randn(m,n)
   b = np.random.randn(m)
   x = cp.Variable(n)
   f = cp.sum_squares(A*x - b) + cp.norm1(x)
   prob = cp.Problem(cp.Minimize(f))

   # Solve with Epsilon
   ep.solve(prob, verbose=True)

   # Print solution value
   print
   print "solution:"
   print x.value

..

Behind the scenes, the Epsilon compiler recognizes this problem is composed of
two functions which have efficient proximal operators and applies an operator
splitting algorithm to find the solution.

The above code snippet produces the following output:

.. code::

   Epsilon 0.2.4, prox-affine form
   objective:
      add(
       sum_square(add(dense(A)*var(x), scalar(-1.00)*const(b))),
       norm_1(var(y)))

   constraints:
     zero(add(var(x), scalar(-1.00)*var(y)))
   Epsilon compile time: 0.0047 seconds

   iter=0 residuals primal=1.50e+00 [1.54e-02] dual=0.00e+00 [2.17e-02]
   iter=20 residuals primal=1.25e-02 [1.57e-02] dual=9.54e-03 [3.41e-02]
   Epsilon solve time: 0.0039 seconds

   solution:
   [[ 0.        ]
    [-0.70320791]
    [-0.        ]
    [ 0.53035546]
    [ 0.        ]
    [-0.        ]
    [-0.10162891]
    [ 0.        ]
    [-0.64069419]
    [ 1.08194075]]
..

For further details on the design and implementation of the Epsilon compiler and
solver, refer to the full paper [#epsilon]_.

.. [#epsilon] `Convex programming with fast proximal and linear operators
	      <http://arxiv.org/abs/1511.04815>`_. Matt Wytock, Po-Wei Wang and
	      J. Zico Kolter, 2015.

.. toctree::
   :maxdepth: 2
   :hidden:

   quick_start
   proximal_operators
   examples
   benchmarks
