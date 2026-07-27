Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install coker

Optional extras are available for backend-specific dependencies:

.. code-block:: bash

   pip install "coker[casadi]"
   pip install "coker[jax]"

Basic usage
-----------

Compile a Python callable with :func:`coker.function`, providing the argument
spaces explicitly. Coker traces the implementation and compiles it for the
chosen backend.

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

Available backend names in the current package are ``"numpy"``, ``"casadi"``,
``"sympy"``, ``"coker"``, and ``"jax"``.

Next steps
----------

- :doc:`backends` for backend-selection guidance and the capability matrix.
- :doc:`dynamics` for ODE construction and variational optimisation.
- :doc:`toolkits` for robotics, system modelling, and codesign helpers.
- :doc:`examples` for the main repository entry points.
