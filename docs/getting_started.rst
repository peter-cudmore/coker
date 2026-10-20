Usage and migration
===================

This page summarises setup, basic calls, and migration from earlier versions.
For a first application, start with :doc:`basics/getting_started`. Follow
:doc:`tutorials/index` for step-by-step lessons.

Installation
------------

.. code-block:: bash

   pip install coker

Install the extra for each optional backend you need:

.. code-block:: bash

   pip install "coker[casadi]"
   pip install "coker[jax]"
   pip install "coker[pytorch]"

NumPy and SymPy are included in the base installation. CasADi, JAX, and PyTorch
require their corresponding extras; JAX requires Python 3.11 or newer.
See :doc:`basics/installation` for environment setup.

To run the double-pendulum visualisation from the repository root:

.. code-block:: bash

   uv run scripts/double_pendulum.py

The script's inline ``uv`` metadata declares NumPy, Matplotlib, and local Coker
dependencies. Matplotlib is not a package dependency or extra; non-visual
kinematics and other Coker APIs do not require it. See :doc:`examples` for
interactive and headless commands.

Basic usage
-----------

Declare argument spaces and pass a Python callable to :func:`coker.function`.
Coker traces it and compiles it for the chosen backend:

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

Select a backend with the ``backend`` argument:

.. code-block:: python

   from coker import Scalar, function

   f_casadi = function(
       arguments=[Scalar("x")],
       implementation=lambda x: x ** 2,
       backend="casadi",
   )

The bundled backend names are ``"numpy"``, ``"casadi"``, ``"sympy"``,
``"jax"``, and ``"pytorch"``. Plugins can register additional names.

Both :func:`coker.function` and the implicit current-backend fallback use
``"numpy"``. Pass ``backend=`` explicitly for portable examples.
Evaluation and lowering do not change the current selection. See
:doc:`backends` for supported capabilities and :doc:`tutorials/backends` for
backend selection and native function extraction.

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

- :doc:`tutorials/index` for the ordered basic-usage lessons.
- :doc:`how_to/index` for task-focused recipes.
- :doc:`api/index` for canonical imports, extension interfaces, and internals.
- :doc:`backends` for backend capabilities and selection rules.
- :doc:`dynamics` for ODE construction and variational optimisation.
- :doc:`toolkits` for robotics and codesign helpers.
- :doc:`examples` for repository programs and model examples.
