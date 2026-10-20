Motivation and audience
=======================

Background
----------
Working as a mathematician and algorithms engineer, I often prototyped
mathematical models, control algorithms and simulations in Python, usually
starting with NumPy. I would then encounter several limits:

1. I wanted symbolic analysis or simplification. I had to copy and modify my
   code for SymPy: converting SymPy expressions to NumPy was straightforward,
   but the reverse was not.
2. I wanted faster numerical execution. CasADi was often my preferred tool for
   simulation, but adapting existing Python code to its interfaces took work.
3. I wanted tools that used automatic differentiation at a higher level of
   abstraction than CasADi and SymPy exposed: a CVXPY-like optimisation problem
   builder, a rigid-body toolkit, and dynamics and control tools such as
   observability analysis.
4. I wanted sparse matrices and tensors to take precedence over dense arrays,
   since sparse representations suited most of my problems.

These tasks usually involve different tools and different representations of
the same equations. I wanted to keep one Python model across those workflows:
NumPy for reference simulation, SymPy for symbolic analysis, and CasADi for
parameter fitting.

I built Coker for roboticists, control engineers and applied mathematicians
working with these models. Its design brings together:

- A sparse symbolic representation of mathematical functions, constructed by
  tracing a Python callable.
- Backends that translate the recorded computation into their own representations
  for execution or analysis.
- Tools for common robotics, dynamics and optimisation tasks.

For worked examples, see the tutorial and cookbook recipes on
:doc:`evaluating one model across backends <../tutorials/backends>`,
:doc:`simulating a dynamical system <../how_to/simulation>`, and
:doc:`fitting a dynamical model to data <../how_to/system_identification>`.
For symbolic analysis, see the
:doc:`system-analysis guide <../dynamics/system_analysis>`.

Coker relies on supporting backends for automatic differentiation; it is not a
general automatic differentiation framework. Backend capabilities differ, so
one model is not guaranteed to work unchanged with every backend. Coker is also
not an all-in-one modelling suite: plotting and visualisation remain out of
scope.

Why ``coker``?
--------------
``coker`` is common mathematical shorthand for
`cokernel <https://en.wikipedia.org/wiki/Cokernel>`_.

.. Author notes

   - Make the target audience explicit, building on the roles described above.
   - Explain the value of one Python model with multiple execution backends.
   - Clarify the distinction between supporting automatic differentiation and
     being a general automatic differentiation framework.
   - Retain the non-goals and avoid promising backend equivalence.
