Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install coker

To use the CasADi backend (required for trajectory optimisation)::

   pip install coker casadi

Basic usage
-----------

Define a function by decorating a Python callable with :func:`coker.function`,
providing the argument spaces. Coker traces the implementation and compiles it
for the chosen backend.

.. code-block:: python

   import numpy as np
   import coker
   from coker import function, Scalar, VectorSpace

   # Scalar function
   f = function(
       arguments=[Scalar("x")],
       implementation=lambda x: 2 * x + 1,
       backend="numpy",
   )
   print(f(3))  # 7

   # Vector function
   A = np.array([[1, 0], [0, -1]], dtype=float)

   g = function(
       arguments=[VectorSpace("x", 2)],
       implementation=lambda x: A @ x,
       backend="numpy",
   )
   print(g(np.array([1.0, 2.0])))  # [ 1. -2.]

Switching backends
------------------

The same implementation can be compiled for any supported backend by changing
the ``backend`` argument:

.. code-block:: python

   f_casadi = function(
       arguments=[Scalar("x")],
       implementation=lambda x: x ** 2,
       backend="casadi",
   )

Available backends: ``"numpy"``, ``"casadi"``, ``"sympy"``, ``"coker"``.
