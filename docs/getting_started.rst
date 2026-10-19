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

The double-pendulum visualisation script declares its own NumPy, Matplotlib,
and local Coker dependencies in inline ``uv`` metadata. From the repository
root, run:

.. code-block:: bash

   uv run scripts/double_pendulum.py

Matplotlib is not a package dependency or extra and is not required for
non-visual kinematics or other Coker APIs. See :doc:`examples` for interactive
and headless double-pendulum commands.

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
listed above before selecting CasADi, JAX, or PyTorch. Additional plugins can
register their own names.

Both :func:`coker.function` and the implicit current-backend fallback use
``"numpy"``. Pass ``backend=`` explicitly for portable examples.
Ordinary evaluation and lowering do not change the current selection.

Prefer CasADi for optimisation, simulation, and high-performance numerical
workloads; NumPy remains the reference implementation and small-system fallback.
The NumPy construction default does not change that backend policy. See
:doc:`backends` for current capability limits and installation requirements.

Migrating to 0.5
----------------

- Import dynamics construction from ``coker.dynamics``. General systems use
  ``DynamicsSpec`` with ``create_dynamics_from_spec``; controlled ODEs use
  ``create_control_system``. Use ``None`` for absent spec inputs, constraints,
  and quadratures instead of importing an internal sentinel.
- Unpack ``system, projections = direct_sum(...)``. Projection fields and
  numeric/structured parameter behavior are described in :doc:`dynamics`.
- Replace program tuple attributes with ``program.input_shape()`` and
  ``program.output_shape()``. The latter excludes the objective;
  ``program.result_shape`` includes it. Replace ``system.backend()`` with the
  read-only ``system.backend`` property.
- Import callable parameters from ``coker.parameters.function_parameters``,
  programs/solve status from ``coker.toolkits.codesign``, ODE option interfaces
  from ``coker.interfaces``, and signatures/lowering options from
  ``coker.backends.lowered``. Removed facade exports have no compatibility shims.
- ``get_current_backend()`` falls back to NumPy. ``function()`` continues to
  default to NumPy even after explicit ambient selection.
- ``get_backend_by_name(name)`` no longer changes ambient selection. Use
  ``get_backend_by_name(name, set_current=True)`` only to deliberately select
  the backend consumed by later implicit builders/evaluator calls. Prefer
  explicit ``backend=`` for portable workflows. Variational problems retain
  their domain-specific CasADi solver default and require the CasADi extra.
- ``Function.lower(options)`` returns a ``LoweredFunction`` without changing
  the function's backend or ambient selection. ``MathematicalProgram.lower()``
  instead returns a solver-pinned ``Function``; calling it solves the program
  and returns only the objective and requested outputs. Imported native
  callables remain pinned to their owning backend.

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

- :doc:`api/index` for canonical imports, extension interfaces, and internals.
- :doc:`backends` for backend-selection guidance and the capability matrix.
- :doc:`dynamics` for ODE construction and variational optimisation.
- :doc:`toolkits` for robotics and codesign helpers.
- :doc:`examples` for the main repository entry points.
