Examples
========

Examples of using Epsilon to solve various optimization problems, especially
those arising in statistical machine learning. On :ref:`classic ML datasets <classic>`, Epsilon
reproduces (near) state-of-the-art results within the rapid prototyping
environment provided by CVXPY and numerical Python. On :ref:`more complex convex
models <complex>`, we demonstrate the expressive power of declarative
programming in building objective functions and constraints extending beyond the
standard regularized loss models typical to classical machine learning.


.. _classic:

Classic ML datasets
-------------------

.. toctree::
   :maxdepth: 1

   notebooks/mnist
   notebooks/newsgroups

.. _complex:

More complex convex models
--------------------------

.. toctree::
   :maxdepth: 1

   notebooks/ercot
   notebooks/graphs
