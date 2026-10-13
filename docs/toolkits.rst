Toolkit guides
==============

Beyond the core symbolic API, Coker ships several domain-focused toolkits under
``src/coker/toolkits/``. They share the same tracing and backend model, so the
same function/back-end ideas apply across robotics, optimisation, and
system-modelling workflows.

Spatial algebra
---------------

Import rigid-motion values from ``coker.toolkits.spatial``. An ``Isometry3``
maps child-frame coordinates into the parent frame, and ``a @ b`` applies
``b`` first. Angles are radians; vector inputs normally have shape ``(3,)``.
The capitalized ``SE3Adjoint``/``SE3CoAdjoint`` act on finite transforms;
lowercase ``se3*`` operators take screws and are not interchangeable.

See :doc:`api/robotics` for explicit value/member contracts, a complete spatial
composition doctest, and limitations of currently exported helpers.

Rigid-body kinematics and dynamics
----------------------------------

Import ``RigidBody``, joint types and ``Inertia`` from
``coker.toolkits.kinematics``. Add links parent-first with local rest transforms,
then attach end effectors. Numeric and traced generalized coordinates share
the flat ``(n,)`` shape, ordered by links and then each joint's axes.

Use the existing ``RigidBody.to_function()`` bridge for world-frame Cartesian
positions and point Jacobians. It returns a traceable callable; pass that
callable to ``coker.function`` with explicit backend selection when a compiled
function is needed. See :doc:`api/robotics` for a complete construction and
evaluation doctest, Jacobian conventions, and dynamics contracts.

Plotting requires optional ``matplotlib`` and the explicit import
``from coker.toolkits.kinematics.visualiser import KinematicsVisualiser``.
The visualiser is not exported from the kinematics package root and accepts
numeric configurations, not traced plotting operations.

System modelling
----------------

The working system-modelling contract is component metadata and port wiring,
not a released simulation/compiler pipeline. Import ordinary builders from
``coker.toolkits.system_modelling`` and standard blocks from
``coker.toolkits.system_modelling.std_lib.common.signal_processing``.

Discovery uses the explicit import
``from coker.toolkits.system_modelling.api import list_components``.
Import component modules first: discovery lists registered classes, not
filesystem contents. It returns ``(name, slash_separated_module_path, None)``.

See :doc:`api/modelling` for an executable discovery/wiring doctest, supported
user and extension members, and incomplete component/simulation paths.

Codesign-style mathematical programs
------------------------------------

:mod:`coker.toolkits.codesign` provides the ordinary user entry points
``ProblemBuilder``, ``Minimise``, ``MathematicalProgram``, ``bounded``, ``norm``,
``weighted_norm``, ``SolveInfo``, and ``SolveFailure``. Import them explicitly
from that module, not from root ``coker``. ``OptimisationModule`` is an advanced
backend extension interface, not needed to construct or solve a program.
See :doc:`api/optimisation` for the complete public contract and solver limits.

Declare runtime inputs separately from optimisation decisions, then set a
scalar objective and a nonempty list of symbolic outputs. Entering or leaving
the builder context does not solve or consume it; ``build`` constructs the
solver, and calling the returned program solves for the supplied runtime inputs.
With no backend argument, ``build`` uses the current backend. Use an explicit
backend for reproducible solver selection.

This complete example requires CasADi/IPOPT. It disables solver printing so
only deterministic result assertions are checked:

.. doctest::

   >>> import numpy as np
   >>> from coker import VectorSpace
   >>> from coker.backends.casadi import CasadiNLPSolverOptions
   >>> from coker.toolkits.codesign import Minimise, ProblemBuilder, bounded
   >>> options = CasadiNLPSolverOptions(optimiser_options={
   ...     "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False,
   ...     "ipopt.max_iter": 50, "ipopt.max_cpu_time": 30.0,
   ... })
   >>> with ProblemBuilder(arguments=[VectorSpace("target", 2)], solver_options=options) as builder:
   ...     (target,) = builder.arguments
   ...     x = builder.new_variable(name="x", shape=(2,), initial_value=np.ones(2))
   ...     delta = x - target
   ...     builder.objective = Minimise(np.dot(delta, delta))
   ...     builder.constraints = [bounded(x[0], target[0], target[0])]
   ...     builder.outputs = [x]
   ...     problem = builder.build("casadi")
   >>> assert problem.parameters == {} and problem.solve_info is None
   >>> objective, x_value = problem(np.array([3.0, -1.0]))
   >>> assert isinstance(objective, float) and abs(objective) < 1e-6
   >>> np.testing.assert_allclose(x_value, [3.0, -1.0], atol=1e-6)
   >>> np.testing.assert_allclose(problem.parameters["x"], x_value, atol=1e-6)
   >>> assert problem.solve_info.success

Use ``bounded(residual, lower, upper)`` for inclusive componentwise bounds.
Equal bounds express equality, and infinite bounds leave a side unbounded.
Bounds may depend on runtime parameters; CasADi evaluates them as residual
constraints at solve time. Symbolic ``<`` and ``>`` constraints are implemented
as non-strict numerical bounds.

Function-valued decisions use ``new_function_parameter`` with a declared
``FunctionSpace`` and a finite ``FunctionParameter`` realisation. Import
declarations from their canonical ``coker.parameters.function_parameters``
module, not ``coker.dynamics``:

.. doctest::

   >>> import numpy as np
   >>> from coker import FunctionSpace, Scalar
   >>> from coker.backends.casadi import CasadiNLPSolverOptions
   >>> from coker.parameters.function_parameters import FittedFunction, RadialBasisFunction
   >>> from coker.toolkits.codesign import Minimise, ProblemBuilder
   >>> rate = FunctionSpace("rate", [Scalar("t")], [Scalar("rate")])
   >>> declaration = RadialBasisFunction([0.0], 1.0, name="response")
   >>> options = CasadiNLPSolverOptions(optimiser_options={
   ...     "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False,
   ... })
   >>> with ProblemBuilder(solver_options=options) as builder:
   ...     response = builder.new_function_parameter(rate, declaration)
   ...     value = response(0.0)
   ...     builder.objective = Minimise((value - 0.5) ** 2 + (response(1.0) - 0.25) ** 2)
   ...     builder.outputs = [value]
   ...     problem = builder.build("casadi")
   >>> objective, value = problem()
   >>> assert abs(objective) < 1e-6 and abs(value.item() - 0.5) < 1e-6
   >>> fitted_response = problem.parameters["response"]
   >>> assert isinstance(fitted_response, FittedFunction)
   >>> assert fitted_response.specification is declaration
   >>> assert fitted_response.space is rate
   >>> assert abs(float(fitted_response(1.0)) - 0.25) < 1e-6
   >>> assert problem.solve_info.success

After a successful numeric solve, ``problem.parameters`` is replaced with the
named decisions: floats for scalars, shaped NumPy arrays for vectors/tensors,
and ``FittedFunction`` instances for function decisions. A fitted function
retains its declaration in ``specification``, its declared space in ``space``,
its native callable in ``function``, and its fitted basis in ``parameters``.
Hidden solver captures do not alter the public ``(objective, *outputs)`` tuple.
The current mapping and result arrays have no copy/isolation guarantee; copy
values if retaining or mutating a snapshot.

``problem.solve_info`` is initially ``None``. Solver-reported failure raises
``SolveFailure`` with matching failure metadata and no result tuple.
``parameters`` retains the last successful solution (or stays empty), so it is
not a solution to the failed call. Configuration or argument errors may leave
the preceding status unchanged. Solver success is not a global-optimality
guarantee.

CasADi uses flat IPOPT constructor settings through ``solver_options`` as in the
first example. NumPy uses SciPy ``trust-constr`` and does not accept solver
options. PyTorch has its own NLP options type; JAX and SymPy currently have no
standalone codesign NLP implementation. ``MathematicalProgram.lower()`` returns
a Coker ``Function`` on the program's backend, unlike ``Function.lower()``,
which returns a backend execution handle. Nested optimiser execution remains
backend-dependent.

Where to continue
-----------------

- :doc:`examples` for the main repository entry points.
- ``tests/toolkits/kinematics/test_kinematics_examples.py`` for richer robotics
  models.
- ``tests/toolkits/test_codesign.py`` for more codesign scenarios, including
  constraint handling and failure cases.
