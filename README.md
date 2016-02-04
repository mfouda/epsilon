# Epsilon [![Circle CI](https://circleci.com/gh/mwytock/epsilon.svg?style=svg)](https://circleci.com/gh/mwytock/epsilon)

Epsilon is a system for general convex programming using fast linear and
proximal operators.

As with existing convex programming frameworks
(e.g. [CVX](http://cvxr.com/cvx/), [CVXPY](http://cvxpy.org/),
[Convex.jl](http://convexjl.readthedocs.org/en/latest/), etc.), users specify
convex
optimization problems using a natural grammar for mathematical expressions,
composing functions in a way that is guaranteed to be convex by the rules of
disciplined convex programming. Given such an input, the Epsilon compiler
transforms the optimization problem into a mathematically equivalent form
consisting only of functions with efficient proximal operators---an intermediate
representation we refer to as *prox-affine form*. By reducing problems to this
form, Epsilon enables solving general convex problems using a large library of
fast proximal and linear operators and is often faster than existing approaches
by an order of magnitude or more.

**For more information, refer to the [Epsilon documentation](http://epopt.io/).**
