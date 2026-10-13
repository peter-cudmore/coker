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

Bundled backend names are ``"numpy"``, ``"casadi"``, ``"sympy"``, ``"jax"``,
and ``"pytorch"``; optional dependencies must be installed where required.
Additional plugins can register their own names.

Both :func:`coker.function` and the implicit current-backend fallback use
``"numpy"``. Pass ``backend=`` explicitly for portable examples. The native
``"coker"`` backend is supplied by the separate ``coker_backend`` package.
Ordinary evaluation and lowering do not change the current selection.

Migrating to 0.4
----------------

``coker.algebra.kernel`` has been removed. Public imports from :mod:`coker`
remain supported. Code using internal algebra types must import from their
owning modules: ``function`` and ``Function`` from
``coker.algebra.function``; ``Tape``, ``Tracer``, and ``TraceContext`` from
``coker.algebra.graph``; dimensions from ``coker.algebra.dimensions``; and
``OP``/``Noop`` from ``coker.algebra.ops``.

Next steps
----------

- :doc:`public_api` for canonical imports and the 0.5 migration contracts.
- :doc:`backends` for backend-selection guidance and the capability matrix.
- :doc:`dynamics` for ODE construction and variational optimisation.
- :doc:`toolkits` for robotics, system modelling, and codesign helpers.
- :doc:`examples` for the main repository entry points.
