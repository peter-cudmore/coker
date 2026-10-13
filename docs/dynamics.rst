Dynamics and optimisation
=========================

Coker's dynamics layer sits on top of the same symbolic function machinery used
by :func:`coker.function`. It adds typed descriptions for state, control,
parameter, and transcription data so you can define ODE systems and then solve
parameter-fitting or optimal-control style problems against them.

Core entry points
-----------------

Ordinary dynamics APIs are re-exported from :mod:`coker.dynamics`. General and
controlled constructors currently live in :mod:`coker.dynamics.system`:

- :func:`coker.dynamics.create_autonomous_ode`
- :func:`coker.dynamics.direct_sum`
- :class:`coker.dynamics.VariationalProblem`
- :class:`coker.parameters.BoundedVariable`
- :class:`coker.parameters.UnboundedVariable`
- :class:`coker.dynamics.TranscriptionOptions`
- :class:`coker.toolkits.codesign.SolveInfo` and :class:`coker.toolkits.codesign.SolveFailure`
- :class:`coker.parameters.BoundVector`
- :class:`coker.parameters.DenseTensorVariable`
- :class:`coker.parameters.function_parameters.FunctionParameter`
- :class:`coker.parameters.function_parameters.ClosureParameter`
- :class:`coker.parameters.function_parameters.MonotonePiecewiseLinear`
- :class:`coker.parameters.function_parameters.DenseLayer`
- :class:`coker.parameters.function_parameters.RadialBasisFunction`


For general models import ``create_dynamics_from_spec`` from
``coker.dynamics.system`` and describe callbacks with ``DynamicsSpec``. For a
controlled ODE import ``create_control_system`` from that same module; it is
not a ``coker.dynamics`` re-export. These are the actual construction paths in
this checkout, not the pending export/default changes in issue #88. See
:doc:`api/dynamics` for canonical imports, direct-sum projections, callback
shapes, and executable construction examples.

``create_autonomous_ode()`` builds a :class:`~coker.dynamics.DynamicalSystem`
from an initial-condition function and an ``xdot`` function. If you pass a
parameter space, the system becomes directly usable inside a
:class:`~coker.dynamics.VariationalProblem`.

Functional problem building
---------------------------

New code should use :class:`coker.dynamics.VariationalProblemBuilder`. Declare
controls and parameters at construction time, then pass a
:class:`coker.toolkits.codesign.Minimise` objective and ``subject_to``
comparisons to ``build``.

For example, a fixed-horizon path-planning problem can constrain the input and
terminal state without constructing a loss callback:

.. code-block:: python

   from coker.dynamics import VariationalProblemBuilder
   from coker.toolkits.codesign import Minimise

   with VariationalProblemBuilder(
       system,
       t_final=5.0,
       control=[control],
   ) as problem:
       x = problem.state(problem.t)
       u = problem.input(problem.t)
       x_goal = problem.state(problem.t_final)
       built = problem.build(
           Minimise(problem.integrate(u @ u) + x_goal @ x_goal),
           subject_to=[u <= 1, u >= -1, x_goal == goal],
       )

For data fitting, declare the unknown as a bounded parameter and use the
symbolic observed output directly in the objective:

.. code-block:: python

   with VariationalProblemBuilder(
       system,
       t_final=1.0,
       parameters=[value],
   ) as problem:
       observed = 2.0
       built = problem.build(
           Minimise((problem.output(problem.t_final)[0] - observed) ** 2)
       )

Function-valued parameters
--------------------------

Systems may declare a parameter as a :class:`coker.FunctionSpace`. Supply a
:class:`~coker.parameters.function_parameters.FunctionParameter` realization at problem
construction; Coker replaces its concrete parameter blocks with scalar solver decisions before
solving. The function space and realization must agree on their argument and
output shapes.
Solver-bound function parameters require a non-empty ``name``; Coker uses it
to name their concrete parameter blocks.
``DenseLayer`` realizes ``activation(weights @ x + bias)`` for vector targets
and ``activation(weight * x + bias)`` for scalar targets. Vector activations
and targets use vector inputs and outputs. Scalar activations and targets use
two unbounded scalar decisions, ``weight`` and ``bias``.

``ClosureParameter`` binds explicitly declared scalar or dense parameter blocks
to a Coker function. Its function receives those blocks first, then the target
function arguments. Use ``ClosureParameter.bind_callable`` to trace a Python
implementation with that calling convention.

For example, a one-component vector response over a two-component state can use
a dense layer with an identity activation:

.. code-block:: python

   from coker import FunctionSpace, VectorSpace, function
   from coker.dynamics import VariationalProblemBuilder
   from coker.parameters.function_parameters import DenseLayer
   from coker.toolkits.codesign import Minimise

   activation = function(
       [VectorSpace("hidden", 1)],
       lambda hidden: hidden,
   )
   response = FunctionSpace(
       "response",
       arguments=[VectorSpace("state", 2)],
       output=[VectorSpace("rate", 1)],
   )
   # The system must declare ``parameters=(response,)`` and call ``p[0](x)``.
   with VariationalProblemBuilder(
       system,
       t_final=1.0,
       parameters=[DenseLayer(2, activation, name="response")],
   ) as problem:
       built = problem.build(
           Minimise((problem.output(problem.t_final)[0] - 0.5) ** 2)
       )

``DenseLayer`` weights and biases, and ``RadialBasisFunction`` coefficients,
are unbounded by default. Pass both ``lower_bound`` and ``upper_bound`` to
``RadialBasisFunction`` to constrain every coefficient.
``MonotonePiecewiseLinear`` has unbounded internal decisions, but its declared
output bounds and monotonicity are enforced by its parameterization.

The builder classifies comparisons by their time binding: expressions at
``t`` are path constraints, expressions at ``0`` are initial point
constraints, expressions at ``t_final`` (and parameter-only expressions) are
terminal point constraints.  A numeric ``t_final`` is fixed; a
``BoundedVariable`` declaration makes the horizon a free decision.  The
``integrate`` operation denotes the dynamic quadrature
``\int_0^T expr(t) dt`` rather than numerical sampling in the builder.
``state(time)`` exposes the differential state, while ``output(time)`` is the
observed system output (and may include algebraic or quadrature channels).
CasADi is the default variational backend and supports constrained collocation,
algebraic states, controls, quadratures, and free horizons. PyTorch additionally
supports fixed-horizon, bound-only ODE parameter fitting and quadratures, without
declared controls, algebraic states, or comparison constraints. NumPy, JAX, and
SymPy do not implement variational solving. The model construction backend is
independent of the problem's solve backend.

Adaptive CasADi transcription
-----------------------------

Use :class:`~coker.backends.casadi.CasadiVariationalOptions` through
``TranscriptionOptions.backend_options`` to enable adaptive collocation. The
initial mesh uses ``minimum_n_intervals`` intervals at ``minimum_degree``.
After each solve, CasADi estimates a local state defect. An interval first has
its polynomial degree increased up to ``maximum_degree``; it is bisected only
when that limit is reached. Refinement stops when every interval is at or below
``mesh_tolerance`` or raises :class:`RuntimeError` after
``maximum_iterations`` refinements.

.. code-block:: python

   from coker.backends.casadi import CasadiVariationalOptions
   from coker.dynamics import TranscriptionOptions

   transcription_options = TranscriptionOptions(
       minimum_n_intervals=4,
       minimum_degree=3,
       backend_options=CasadiVariationalOptions(
           refinement_enabled=True,
           mesh_tolerance=1e-5,
           maximum_degree=10,
           maximum_iterations=6,
       ),
   )
   problem = VariationalProblem(
       system=system,
       loss=loss,
       t_final=1.0,
       backend="casadi",
       transcription_options=transcription_options,
   )

``mesh_tolerance`` is the maximum scaled relative state defect per interval.
``minimum_interval_duration`` prevents h-refinement from splitting an interval
below that normalized width. ``maximum_degree`` must be no less than
``minimum_degree``. Leave ``refinement_enabled`` at its default ``False`` for
a single transcription solve.

``absolute_tolerance`` bounds initial-condition, continuity, and algebraic
residuals. ``derivative_defect_tolerance`` independently bounds collocation
derivative defects; ``None`` inherits ``absolute_tolerance`` and ``0.0`` requires
an exact NLP equality. ``segment_defect_tolerance`` records the reference
tolerance for physical post-solve segment-defect diagnostics; it does not add
NLP constraints.

During one adaptive solve, the CasADi model setup is retained while only
mesh-dependent NLPs are compiled. Each refined mesh interpolates the preceding
path and seeds compatible controls, free parameters, and a free horizon from
the preceding solution. Explicit parameters passed to ``solve`` override that
carried parameter value. Successful adaptive solutions expose
``adaptive_refinement_rounds`` (the number of mesh updates after the initial
solve) and ``adaptive_maximum_defect`` (the final maximum scaled state defect).
Previously visited mesh signatures are reused by a bounded cache. Cache reuse
is an implementation detail: callers should treat adaptive solving as
deterministic for a fixed problem, options, and parameter values, rather than
rely on a cache lifetime across separately created problems.

Heterogeneous and function-valued parameters
--------------------------------------------

``DynamicsSpec.parameters`` may be one declaration or a positional tuple of
``Scalar``, ``VectorSpace``, and ``FunctionSpace`` declarations. A dynamics
callback receives that same positional layout as one final ``p`` tuple.

Direct trajectory evaluation retains that declaration order. For a system
without declared inputs, call ``system(time, *parameters)``. For a system with
declared inputs, call ``system(time, input_value, *parameters)``. Do not add an
input placeholder to the no-input form.

For example, first declare the function-valued slot:

.. code-block:: python

   import numpy as np

   from coker import FunctionSpace, Scalar, VectorSpace

   gain_curve = FunctionSpace(
       "gain_curve",
       arguments=[Scalar("time")],
       output=[Scalar("gain")],
   )
   # ``system`` was declared with:
   # parameters=(Scalar("offset"), VectorSpace("bias", 2), gain_curve)

   time = np.array([0.0, 0.5, 1.0])
   offset = 2.0
   bias = np.array([0.25, -0.25])
   gain_curve_value = lambda time: 1.0 + time

   trajectory = system(time, offset, bias, gain_curve_value)

For an otherwise equivalent system with a declared input, put its input value
immediately after time:

.. code-block:: python

   trajectory = system(time, input_value, offset, bias, gain_curve_value)

Values for a declared ``FunctionSpace`` parameter may be ordinary Python
callables, :class:`coker.Function` objects, or
:class:`~coker.parameters.function_parameters.FittedFunction` objects. Coker normalizes
these values once before integration, so callers do not provide backend-native
or lowered handles. No ``Noop`` placeholder is used for direct evaluation.

The variational builder receives one declaration for each positional parameter.
Use ``BoundVector`` for a bounded vector decision block, and
``DenseTensorVariable`` for an unbounded dense block reconstructed with the
shape of its initial guess:

.. code-block:: python

   from coker.dynamics import VariationalProblemBuilder
   from coker.parameters.function_parameters import MonotonePiecewiseLinear
   from coker.parameters import BoundVector

   with VariationalProblemBuilder(
       system,
       t_final=1.0,
       parameters=[
           MonotonePiecewiseLinear(
               domain_knots=[0.0, 0.5, 1.0],
               lower_bound=0.0,
               upper_bound=2.0,
               name="response",
           ),
           BoundVector(
               "offset",
               lower_bound=[-1.0, -1.0],
               upper_bound=[1.0, 1.0],
               guess=[0.0, 0.0],
           ),
       ],
   ) as builder:
       problem = builder.build(Minimise(builder.output(builder.t_final)[0] ** 2))

Function-valued parameters use named concrete parameter blocks.
``DenseLayer`` expands its weights and bias, while
``MonotonePiecewiseLinear`` and ``RadialBasisFunction`` each expand one vector
block. The solver flattens those blocks into scalar decisions, then restores
their original shapes when rebuilding a callable parameter. The returned
``solution.parameters`` mapping restores scalar decisions as ``float`` values
and vector or dense tensor decisions as shaped NumPy arrays. Function decisions
are ``FittedFunction`` objects, retaining their specification, space, callable,
and structured fitted parameters.

``MonotonePiecewiseLinear.domain_knots`` should span the normalized integration
domain from ``0`` to ``1``; use ``0`` and ``1`` as the endpoint knots and add
interior knots where the fitted response needs more resolution. This is the
solver's normalized integration coordinate, not physical time. A system whose
parameter is expressed in physical time should pass its normalized time
coordinate to the fitted function. The temporal transcription mesh is
independent of the selected interior knots.

Worked parameter-fitting example
--------------------------------

The following complete example fits an unknown constant initial state. It
requires NumPy and CasADi/IPOPT. It uses the builder so the objective's time
binding is explicit; constructing the problem does not solve it.

.. doctest::

   >>> import contextlib
   >>> import io
   >>> import numpy as np
   >>> from coker import VectorSpace
   >>> from coker.dynamics import create_autonomous_ode, VariationalProblemBuilder, TranscriptionOptions
   >>> from coker.parameters import BoundedVariable
   >>> from coker.toolkits.codesign import Minimise
   >>> system = create_autonomous_ode(
   ...     lambda p: p[:1], lambda x, p: 0 * x,
   ...     parameters=VectorSpace("p", 1), backend="numpy",
   ... )
   >>> with VariationalProblemBuilder(
   ...     system, t_final=1.0,
   ...     parameters=[BoundedVariable("value", lower_bound=0.5, upper_bound=3.0, guess=1.0)],
   ...     transcription_options=TranscriptionOptions(minimum_n_intervals=2, minimum_degree=3),
   ... ) as builder:
   ...     problem = builder.build(Minimise((builder.output(builder.t_final)[0] - 2.0) ** 2))
   >>> with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
   ...     solution = problem()
   >>> assert solution.solve_info.success
   >>> assert abs(solution.parameters["value"] - 2.0) < 1e-3
   >>> assert solution.cost < 1e-6
   >>> assert solution.state(0.5).shape == (1,)
   >>> assert np.allclose(solution(0.5), [2.0], atol=1e-3)

``solution(t)`` returns observed output, while ``state(t)``, ``algebraic(t)``,
and ``quadratures(t)`` return flat state channels at scalar physical time.
Absent algebraic, quadrature, and control channels return ``None``. Stack
individual evaluations to sample a trajectory; the solution accessors do not
accept a vector of times. ``solution.parameters`` retains named fitted values
and ``solution.solve_info`` reports status. Retain ``problem.get_solver()`` for
repeated solves and call ``solver.solve(**fixed_parameters)`` to fix named
values; ``problem()`` creates a new solver each time.

Direct ``DynamicalSystem`` evaluation rejects systems containing quadratures.
The residual-system conversion API proposed by issue #88 is absent in this
checkout. Existing variational transcription supports integral channels;
:doc:`api/dynamics` includes an executable quadrature example and makes
this missing conversion dependency explicit.

What the test suite exercises
-----------------------------

The current dynamics coverage includes:

- direct simulation checks for scalar and vector linear systems;
- parameter fitting for constants, lines, and exponential systems;
- path constraints and regularisation terms;
- solver re-entrancy and warm-start behaviour;
- callback support through ``VariationalIterationCallback``.

That makes the tests a useful source of real usage patterns even where the API
surface is broader than this short guide.

Solve status and failures
-------------------------

Both the dynamics layer and the codesign helpers surface solve metadata through
:class:`coker.toolkits.codesign.SolveInfo`. Failed solves raise
:class:`coker.toolkits.codesign.SolveFailure` with the backend status attached.

Next places to read
-------------------

- ``tests/dynamical_systems/test_variational_solver.py`` for fitting and
  constrained-solve examples.
- ``tests/dynamical_systems/test_variational_solver_callback.py`` for iteration
  callback usage.
- :doc:`backends` for backend-selection advice.


System analysis
---------------

.. toctree::
   :maxdepth: 1

   dynamics/system_analysis
