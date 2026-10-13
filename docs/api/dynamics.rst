Dynamics contracts
==================

This reference describes the **current** public surface, not the pending API
cleanup in issue #88. Ordinary applications import modelling, composition,
analysis, and variational objects from :mod:`coker.dynamics`. General and
controlled construction currently require their owning module::

   from coker.dynamics import (
       DynamicsSpec, DynamicalSystem, create_autonomous_ode, direct_sum,
       VariationalProblem, VariationalProblemBuilder, VariationalSolution,
       TranscriptionOptions,
   )
   from coker.dynamics.system import (
       create_dynamics_from_spec, create_control_system,
   )

Neither ``create_dynamics_from_spec`` nor ``create_control_system`` is currently
re-exported by ``coker.dynamics``. General finite decisions belong to
:mod:`coker.parameters`; function realizations belong to
:mod:`coker.parameters.function_parameters`. They are not dynamics namespace exports.

Construct and evaluate
----------------------

The autonomous constructor defaults to the NumPy backend. Its callbacks are
``x0(p)``, ``xdot(x, p)``, and optional ``output(x, p)``; a constant initial
state is also accepted. ``p_init`` is a shape-check value, not a parameter
binding retained for evaluation. Output is not necessarily state.

.. doctest::

   >>> import numpy as np
   >>> from coker.dynamics import create_autonomous_ode, direct_sum
   >>> first = create_autonomous_ode([1.0, 2.0], lambda x, p: -x, backend="numpy")
   >>> second = create_autonomous_ode([3.0, 4.0], lambda x, p: 0 * x, backend="numpy")
   >>> times = np.array([0.0, 0.5, 1.0])
   >>> assert first(times).shape == (3, 2)
   >>> assert np.allclose(first(times), np.exp(-times[:, None]) * np.array([1.0, 2.0]), atol=1e-3)
   >>> combined, projections = direct_sum(first, second)
   >>> values = combined(times)
   >>> assert values.shape == (3, 4)
   >>> components = projections.outputs.inverse(values[-1])
   >>> assert np.allclose(components[1], [3.0, 4.0])
   >>> extract_first, extract_second = projections.outputs.as_matrices()
   >>> assert extract_first.shape == (2, 4)
   >>> assert np.allclose(extract_second @ values[-1], [3.0, 4.0])

``direct_sum`` returns a **pair**, not just a system. Projection fields are
``state``, ``algebraic``, ``quadratures``, ``outputs``, ``controls``, and
``parameters``, ordered exactly like the supplied subsystems. Numeric fields
have ``inverse(vector)`` and ``as_matrices()``. Absent blocks split to ``None``;
their extraction matrices have zero rows. Numeric parameters are concatenated
into one flat vector. When any parameter is function-valued, the composed
parameter declaration is positional and ``projections.parameters.split(tuple)``
returns each subsystem's tuple instead of numeric extraction matrices.

Current scalar-output limitation: concatenation requires array-valued component
outputs. Components restored as zero-dimensional scalars fail in ``direct_sum``
evaluation with ``ValueError: zero-dimensional arrays cannot be concatenated``.
The example uses multi-component vector outputs, which support both sampling
and projection extraction.

For controlled ODEs, the convenience constructor evaluates ``u(t)`` before
calling ``xdot(t, x, u_value, p)`` or ``output(t, x, u_value, p)``:

.. doctest::

   >>> import numpy as np
   >>> from coker import FunctionSpace, Scalar, VectorSpace
   >>> from coker.dynamics.system import create_control_system
   >>> control_space = FunctionSpace("u", [Scalar("t")], [VectorSpace("u_value", 1)])
   >>> controlled = create_control_system(
   ...     [0.0], lambda t, x, u, p: u, control_space, backend="numpy"
   ... )
   >>> assert np.allclose(controlled(1.0, lambda t: np.array([2.0])), [2.0])

For general ODE/DAE models use ``DynamicsSpec``. Its initial callback is
``initial_conditions(z, u, p) -> (x0, z0)``: the first argument is the algebraic
slot, **not initial time**. Other callbacks receive ``(t, x, z, u, p)``;
outputs additionally receive accumulated ``q``. Here ``u`` is the control
function, so the callback explicitly evaluates ``u(t)``. Use ``Noop()`` for
absent inputs, constraints, and quadratures; ``None`` for absent algebraic or
parameter spaces. Algebraic variables require residual constraints. State and
derivative shapes must agree.

.. doctest::

   >>> import numpy as np
   >>> from coker import Scalar
   >>> from coker.algebra.ops import Noop
   >>> from coker.dynamics import DynamicsSpec
   >>> from coker.dynamics.system import create_dynamics_from_spec
   >>> spec = DynamicsSpec(
   ...     inputs=Noop(), parameters=(Scalar("rate"),), algebraic=None,
   ...     initial_conditions=lambda z, u, p: (np.array([0.0]), None),
   ...     dynamics=lambda t, x, z, u, p: np.array([p[0]]),
   ...     constraints=Noop(), outputs=lambda t, x, z, u, p, q: x,
   ...     quadratures=Noop(),
   ... )
   >>> general = create_dynamics_from_spec(spec, backend="numpy")
   >>> assert np.allclose(general(1.0, 3.0), [3.0])

A single parameter declaration reaches callbacks directly as ``p``. Tuple/list
declarations reach callbacks as one positional tuple ``p`` (including a
one-element tuple). Direct calls use ``system(time, *parameters)`` without
inputs and ``system(time, input_function, *parameters)`` with inputs; do not add
an input placeholder. Function slots accept matching Coker ``Function`` or
``FittedFunction`` objects or traceable Python callables; Coker functions must
have the system's backend.

Direct endpoint evaluation returns the output's declared shape. Sampled
multi-component vector output has shape ``(n_times, n_output)``. The sampled
scalar/one-component branch attempts to flatten to ``(n_times,)``, but currently
fails if the backend restores its individual outputs as zero-dimensional
scalars. Evaluate scalar endpoints individually and collect them explicitly;
use array-valued outputs for composition. Direct evaluation integrates from
time zero. NumPy supports ODEs; CasADi supports ODEs and algebraic residuals;
PyTorch ODE integration requires the optional ``torchdiffeq`` dependency.
SymPy does not numerically integrate; JAX has no dynamics integrator.

.. autofunction:: coker.dynamics.create_autonomous_ode

.. autofunction:: coker.dynamics.system.create_control_system

.. autofunction:: coker.dynamics.system.create_dynamics_from_spec

.. autoclass:: coker.dynamics.DynamicsSpec

.. autoclass:: coker.dynamics.DynamicalSystem
   :members: __call__, backend, get_state_dimensions, output_as_function_space

.. autofunction:: coker.dynamics.direct_sum

Build, solve, and inspect
-------------------------

Use the builder inside ``with``. ``state(time)`` is differential state;
``output(time)`` is observed output and can include algebraic or integral
channels. Time bindings are ``builder.t`` (default), numeric ``0``, and
``builder.t_final``; arbitrary numeric times and foreign tracers are rejected.
``integrate(expr)`` registers a scalar physical-time integral accumulated from
zero, not numerical sampling. Comparisons depending on path time become path
constraints, initial-bound expressions become initial constraints, and remaining
expressions (including parameter-only comparisons) become terminal constraints.
Path constraints are enforced at transcription nodes, not certified everywhere
between them.

``build(Minimise(expr), subject_to=[...])`` creates a problem but does not solve.
After context exit the problem remains valid but builder symbolic access raises
``RuntimeError``. A numeric horizon is fixed; a ``BoundedVariable`` horizon is
optimized and must have a strictly positive lower bound.

For a finite scalar/vector parameter space, direct ``VariationalProblem``
construction accepts ordered scalar decisions (or an explicit
``system_parameter_map``). For tuple parameter spaces, supply one declaration
per slot: scalar ``BoundedVariable``/``UnboundedVariable`` (or fixed scalar),
vector ``BoundVector``/``DenseTensorVariable``, and named ``FunctionParameter``
realizations for function slots. Specialization flattens solver decisions while
retaining the original public parameter layout.

``problem()`` creates and solves a new backend solver. Retain
``solver = problem.get_solver()`` for repeated solves; ``solver.parameters``
lists free scalar fixing names and ``solver.solve(**fixed_parameters)`` fixes
specified values by name. CasADi is the default solver and supports constrained
collocation, control declarations, algebraic states, quadratures, and free
horizons. PyTorch supports fixed-horizon, bound-only ODE parameter fitting and
quadratures, without control declarations, algebraic states, or comparison
constraints. NumPy, SymPy, and JAX do not implement variational solving. The
model's construction backend and problem's solve backend are separate choices.

This complete fitting example requires CasADi/IPOPT in addition to NumPy.
Solver chatter is redirected; numerical values are checked with tolerances.

.. doctest::

   >>> import contextlib
   >>> import io
   >>> import numpy as np
   >>> from coker import VectorSpace
   >>> from coker.dynamics import create_autonomous_ode, VariationalProblemBuilder, TranscriptionOptions
   >>> from coker.parameters import BoundedVariable
   >>> from coker.toolkits.codesign import Minimise
   >>> fitted_system = create_autonomous_ode(
   ...     lambda p: p[:1], lambda x, p: 0 * x,
   ...     parameters=VectorSpace("p", 1), backend="numpy",
   ... )
   >>> options = TranscriptionOptions(minimum_n_intervals=2, minimum_degree=3, verbose=False)
   >>> with VariationalProblemBuilder(
   ...     fitted_system, t_final=1.0,
   ...     parameters=[BoundedVariable("value", lower_bound=0.5, upper_bound=3.0, guess=1.0)],
   ...     transcription_options=options,
   ... ) as builder:
   ...     problem = builder.build(Minimise((builder.output(builder.t_final)[0] - 2.0) ** 2))
   >>> with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
   ...     solution = problem()
   >>> assert solution.solve_info.success
   >>> assert solution.cost < 1e-6
   >>> assert abs(solution.parameters["value"] - 2.0) < 1e-3
   >>> assert solution.state(0.5).shape == (1,)
   >>> assert solution.algebraic(0.5) is None
   >>> assert solution.quadratures(0.5) is None
   >>> assert solution.control_law(0.5) is None
   >>> assert np.allclose(solution(0.5), [2.0], atol=1e-3)
   >>> sampled = np.vstack([solution(t) for t in [0.0, 0.5, 1.0]])
   >>> assert sampled.shape == (3, 1)
   >>> assert solution.as_raw().shape[0] == 2

``VariationalSolution`` accessors take **scalar physical time** in
``[0, solution.t_final]``. ``state(t)``, ``algebraic(t)``, and ``quadratures(t)``
return flat ``(n_x,)``, ``(n_z,)``, and ``(n_q,)`` arrays; absent algebraic or
quadrature channels return ``None``. ``control_law(t)`` is an ordered
``(n_controls,)`` array or ``None``. Piecewise controls have half-open bins and
may not be evaluated at the terminal boundary. ``solution(t)`` evaluates the
observed output, not the concatenated state. Sample multiple times explicitly.
``as_raw()`` contains time in row zero and concatenated path knots below,
not output samples.

``parameters`` maps names to floats, shaped NumPy arrays, or ``FittedFunction``
objects. ``solve_info`` carries backend-neutral success/status metadata; failed
solves raise ``SolveFailure`` with that information. Iteration callback
solutions are candidates, not success certificates. ``path_constraints`` and
``terminal_constraints`` inspect only retained compiled half-space expressions;
they are not full transcription feasibility certificates.

.. autoclass:: coker.dynamics.VariationalProblemBuilder
   :members: t, t_final, state, input, output, parameters, integrate, build

.. autoclass:: coker.dynamics.VariationalProblem
   :members: get_solver, __call__, horizon_decision, decision_declarations

.. autoclass:: coker.dynamics.TranscriptionOptions

.. autoclass:: coker.dynamics.VariationalSolution
   :members: __call__, state, algebraic, quadratures, control_law, as_raw, path_constraints, terminal_constraints

.. autoclass:: coker.dynamics.ConstantControlVariable

.. autoclass:: coker.dynamics.PiecewiseConstantVariable

.. autoclass:: coker.dynamics.SpikeVariable

Quadratures and the pending conversion route
--------------------------------------------

A ``DynamicalSystem`` with ``dqdt`` explicitly rejects direct ``system(time, ...)``
evaluation with ``NotImplementedError`` even when its backend integrator can
handle quadratures. The residual-system conversion route mentioned by issue #89
is **absent in this checkout**; there is no canonical conversion function to
import or execute. This documentation does not invent that API or implement the
pending cutover. Use the existing variational transcription for integral
channels, or the advanced backend integration interface with explicit initial
quadrature values (see :doc:`../backends`).

This complete example uses a model quadrature in the existing variational route
(NumPy and CasADi/IPOPT dependencies). Model quadratures start at zero. For
builder-owned integrals, use ``builder.integrate(scalar_expression)`` instead:

.. doctest::

   >>> import contextlib
   >>> import io
   >>> import numpy as np
   >>> from coker import Scalar
   >>> from coker.algebra.ops import Noop
   >>> from coker.dynamics import DynamicsSpec, VariationalProblemBuilder, TranscriptionOptions
   >>> from coker.dynamics.system import create_dynamics_from_spec
   >>> from coker.parameters import BoundedVariable
   >>> from coker.toolkits.codesign import Minimise
   >>> quadrature_system = create_dynamics_from_spec(
   ...     DynamicsSpec(
   ...         inputs=Noop(), parameters=(Scalar("value"),), algebraic=None,
   ...         initial_conditions=lambda z, u, p: (np.array([p[0]]), None),
   ...         dynamics=lambda t, x, z, u, p: 0 * x,
   ...         constraints=Noop(),
   ...         outputs=lambda t, x, z, u, p, q: x + q,
   ...         quadratures=lambda t, x, z, u, p: x,
   ...     ), backend="numpy",
   ... )
   >>> rejected = False
   >>> try:
   ...     quadrature_system(1.0, 2.0)
   ... except NotImplementedError:
   ...     rejected = True
   >>> assert rejected
   >>> with VariationalProblemBuilder(
   ...     quadrature_system, t_final=1.0,
   ...     parameters=[BoundedVariable("value", lower_bound=0.5, upper_bound=3.0, guess=1.0)],
   ...     transcription_options=TranscriptionOptions(minimum_n_intervals=2, minimum_degree=3),
   ... ) as builder:
   ...     terminal_error = builder.output(builder.t_final)[0] - 4.0
   ...     integral_problem = builder.build(
   ...         Minimise(terminal_error ** 2)
   ...     )
   >>> with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
   ...     integral_solution = integral_problem()
   >>> assert integral_solution.solve_info.success
   >>> assert integral_solution.quadratures(1.0).shape == (1,)
   >>> assert np.allclose(integral_solution.quadratures(1.0), [2.0], atol=1e-3)
   >>> assert np.allclose(integral_solution(1.0), [4.0], atol=1e-3)

Analysis and advanced extensions
--------------------------------

The ordinary analysis functions return generic **local** rank conclusions, not
global reachability or practical estimation guarantees. Read
:doc:`../dynamics/system_analysis` for scope, rank conditions, and a complete
example. Unsupported models return ``INCONCLUSIVE`` with a reason; low-level
symbolic lowering instead raises ``UnsupportedSystemError``.

.. autofunction:: coker.dynamics.analyse_controllability

.. autofunction:: coker.dynamics.analyse_observability

.. autofunction:: coker.dynamics.analyse_identifiability

.. autoclass:: coker.dynamics.AnalysisStatus

.. autoclass:: coker.dynamics.AnalysisResult

.. autoclass:: coker.dynamics.ControllabilityResult

.. autoclass:: coker.dynamics.ObservabilityResult

.. autoclass:: coker.dynamics.IdentifiabilityResult

Extension hooks include ``VariationalIterationCallback`` (return ``True`` to
continue), ``SolverParameters`` and backend-specific settings, and
``VariationalSolution.from_solver`` for backend implementations. Advanced
symbolic inspection uses ``SymbolicSystem``, ``SymbolicDAESystem``,
``lower_system`` and ``lower_dae_system``. Returned projection helpers live in
``coker.dynamics.system``; they are not root namespace constructor APIs.

.. autoclass:: coker.dynamics.VariationalIterationCallback
   :members: __call__

.. autoclass:: coker.dynamics.ConstraintSpec
   :members: from_expression

.. autoclass:: coker.dynamics.QuadratureSpec

.. autoclass:: coker.dynamics.SymbolicSystem

.. autoclass:: coker.dynamics.SymbolicDAESystem

.. autofunction:: coker.dynamics.lower_system

.. autofunction:: coker.dynamics.lower_dae_system

.. autoexception:: coker.dynamics.UnsupportedSystemError

.. autoclass:: coker.dynamics.system.ProjectionSet

.. autoclass:: coker.dynamics.system.CompositionOperator
   :members: inverse, as_matrices


Current exports also include ``InterpolatingPoly``,
``InterpolatingPolyCollection``, ``lgr_points``, ``legendre_coefficient``,
``evaluate_legendre_polynomial``, ``expand_coefficients``,
``generate_discritisation_operators`` (current spelling), and
``split_at_non_differentiable_points``. These are advanced collocation/mesh
helpers, not the recommended model-construction route. ``ControlLaw``,
``ControlVariable``, ``ControlSolution``, ``Solution``, and ``LossFunction`` are
annotation aliases. Private trace adapters, parameter partitions, and backend
transcription factories are implementation internals; no all-members autodoc
contract is intended for them.
