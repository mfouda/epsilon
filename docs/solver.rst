Epsilon Solver
==============

Once the `compiler <compiler.html>`_ has put a convex problem into *separable
prox-affine form*, the solver applies the ADMM-based operator splitting
algorithm using the library of proximal operators. The implementation details of
each operator are abstracted from the high-level algorithm which applies
operators only through a common interface providing the basic mathematical
operators required.

Separable input
---------------

The separable prox-affine form is

.. math::
   \DeclareMathOperator{\subjectto}{subject\;to}
   \DeclareMathOperator{\minimize}{minimize}

   \begin{split}
    \minimize \;\; & \sum_{i=1}^N f_i(H_i(x_i)) \\
    \subjectto \;\; & \sum_{i=1}^N A_i(x_i) = 0.
   \end{split}
..

where :math:`A_i` denotes an affine transformation.

Internally, the compiler represents problem as an AST; for example, the lasso
problem in separable form (printed by ``epopt.text_format.format_problem``) is

.. code::

   objective:
      add(
       sum_square(add(dense(A)*var(x), scalar(-1.00)*const(b))),
       norm_1(var(y)))

   constraints:
     zero(add(var(x), scalar(-1.00)*var(y)))

..

Mathametically, this AST represents the problem

.. math::

   \begin{split}
   \minimize \;\; & \|Ax - b\|_2^2 + \|y\|_1  \\
   \subjectto \;\; & x - y = 0
   \end{split}
..

which is in the separable prox-affine form.


ADMM algorithm
--------------

The Epsilon solver employs a variant of ADMM to solve problems in the separable
prox-affine form. This approach can be motivated by considering the augmented
Lagrangian

.. math::

   L_\lambda(x_1,\ldots,x_N,y) = \sum_{i=1}^N f_i(H(x_i)) + y^T(Ax - b) +
  (1/2\lambda) \| Ax - b \|_2^2
..

where :math:`y` is the dual variable, :math:`\lambda \ge 0` is the augmented Lagrangian
penalization parameter, and :math:`Ax = \sum_{i=1}^NA_i(x_i)`. The ADMM method applied
here results in the Gauss-Seidel updates with

.. math::

   \DeclareMathOperator*{\argmin}{argmin}
  \begin{split}
    x_i^{k+1} &:=  \argmin_{x_i} \lambda f_i(H_i(x_i)) + \frac{1}{2} \left \|
    \sum_{j < i}A_j(x_j^{k+1}) + A_i(x_i) + \sum_{j > i} A_j(x_j^{k}) - b + u^k \right\|_2^2 \\
    u^{k+1}   &:= u^k + Ax^{k+1} - b
  \end{split}
..

where we have :math:`u = \lambda y` is the scaled dual variable. Critically, the
:math:`x_i`-updates are applied using the (generalized) proximal operator, let

.. math::

   v_i^k = b - u^k - \sum_{j < i}A_j(x_j^{k+1}) - \sum_{j > i}
   A_j(x_j^k)
..

and we have

.. math::
   x_i^{k+1} :=  \argmin_{x_i} \lambda f_i(H_i(x_i)) + (1/2)\|A(x_i) - v_i^k
    \|_2^2
..

The ability of the solver to evaluate the generalized proximal operator
efficiently will depend on :math:`f_i` and :math:`A_i` (in the most common case
:math:`A_i^TA_i = \alpha I`, a scalar matrix, which can be handled by any
proximal operator); it is the responsibility of the compiler to ensure
that the prox-affine problem has been put in the required form such that
these evaluations map to efficient implementations from the proximal operator
library.
