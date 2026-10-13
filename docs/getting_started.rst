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

Basic usage
-----------

Compile a Python callable with :func:`coker.function`, providing the argument
spaces explicitly. Coker traces the implementation and compiles it for the
chosen backend.

.. doctest::

   >>> import numpy as np
   >>> from coker import function, Scalar, VectorSpace
   >>> f = function(
   ...     arguments=[Scalar("x")],
   ...     implementation=lambda x: 2 * x + 1,
   ...     backend="numpy",
   ... )
   >>> assert f(3) == 7
   >>> A = np.array([[1, 0], [0, -1]], dtype=float)
   >>> g = function(
   ...     arguments=[VectorSpace("x", 2)],
   ...     implementation=lambda x: A @ x,
   ...     backend="numpy",
   ... )
   >>> np.testing.assert_allclose(g(np.array([1.0, 2.0])), [1., -2.])

Switching backends
------------------

The same implementation can be compiled for any supported backend by changing
the ``backend`` argument:

The following example requires ``pip install "coker[casadi]"``:

.. doctest::

   >>> from coker import function, Scalar
   >>> f_casadi = function(
   ...     arguments=[Scalar("x")],
   ...     implementation=lambda x: x ** 2,
   ...     backend="casadi",
   ... )
   >>> assert f_casadi(3.) == 9.

Available backend names in the current package are ``"numpy"``, ``"casadi"``,
``"sympy"``, ``"coker"``, ``"jax"`` and ``"pytorch"``. The backend capability
matrix and ownership contracts are in :doc:`api/backends`.

In Coker 0.4, :func:`coker.function` defaults to ``"numpy"``. Pass
``backend=`` explicitly for portable examples. The native ``"coker"`` backend
is supplied by the separate ``coker_backend`` package.

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
