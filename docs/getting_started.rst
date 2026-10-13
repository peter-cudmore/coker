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
   pip install "coker[pytorch]"

JAX requires Python 3.11 or newer. NumPy and SymPy are included in the base
installation; the other built-in backend entry points need their corresponding
extras before use.

The kinematics visualiser has a separate optional Matplotlib dependency:

.. code-block:: bash

   pip install "coker[plotting]"

Plotting is not required for non-visual kinematics or other Coker APIs. See
:doc:`examples` for interactive and headless double-pendulum commands.

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

   from coker import Scalar, function

   f_casadi = function(
       arguments=[Scalar("x")],
       implementation=lambda x: x ** 2,
       backend="casadi",
   )

The package declares backend entry points for ``"numpy"``, ``"casadi"``,
``"sympy"``, ``"jax"``, and ``"pytorch"``. Install the optional dependencies
listed above before selecting CasADi, JAX, or PyTorch.

In Coker 0.4, :func:`coker.function` defaults to ``"numpy"``. Pass
``backend=`` explicitly for portable examples. The native ``"coker"`` backend
is not a built-in entry point or an extra: it is registered by the separately
installed ``coker_backend`` plugin.

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

- :doc:`backends` for backend-selection guidance and the capability matrix.
- :doc:`dynamics` for ODE construction and variational optimisation.
- :doc:`toolkits` for robotics, system modelling, and codesign helpers.
- :doc:`examples` for the main repository entry points.
