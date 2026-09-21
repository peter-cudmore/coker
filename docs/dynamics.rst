Dynamics and optimisation
=========================

Coker's dynamics layer sits on top of the same symbolic function machinery used
by :func:`coker.function`. It adds typed descriptions for state, control,
parameter, and transcription data so you can define ODE systems and then solve
parameter-fitting or optimal-control style problems against them.

Core entry points
-----------------

The main public surface is re-exported from :mod:`coker.dynamics`:

- :func:`coker.dynamics.create_autonomous_ode`
- :func:`coker.dynamics.direct_sum`
- :class:`coker.dynamics.VariationalProblem`
- :class:`coker.dynamics.BoundedVariable`
- :class:`coker.dynamics.UnboundedVariable`
- :class:`coker.dynamics.TranscriptionOptions`
- :class:`coker.toolkits.codesign.SolveInfo` and :class:`coker.toolkits.codesign.SolveFailure`
- :class:`coker.dynamics.BoundVector`
- :class:`coker.dynamics.DenseTensorVariable`
- :class:`coker.dynamics.FunctionParameter`
- :class:`coker.dynamics.MonotonePiecewiseLinear`
- :class:`coker.dynamics.Perceptron`
- :class:`coker.dynamics.RadialBasisFunction`

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
:class:`~coker.dynamics.FunctionParameter` realization at problem construction;
Coker replaces it with the realization's scalar basis decisions before solving.
The function space and realization must agree on argument and scalar-output
shapes.

``Perceptron`` requires one vector argument and a scalar output.
``RadialBasisFunction`` and ``MonotonePiecewiseLinear`` require one scalar
argument and a scalar output.

For example, a scalar response over a two-component state can use a perceptron:

.. code-block:: python

   from coker import FunctionSpace, Scalar, VectorSpace
   from coker.dynamics import Perceptron, VariationalProblemBuilder
   from coker.toolkits.codesign import Minimise

   response = FunctionSpace(
       "response",
       arguments=[VectorSpace("state", 2)],
       output=[Scalar("rate")],
   )
   # The system must declare ``parameters=(response,)`` and call ``p[0](x)``.
   with VariationalProblemBuilder(
       system,
       t_final=1.0,
       parameters=[Perceptron(2, name="response")],
   ) as problem:
       built = problem.build(
           Minimise((problem.output(problem.t_final)[0] - 0.5) ** 2)
       )

``Perceptron`` and ``RadialBasisFunction`` decisions are unbounded by default.
Pass both ``lower_bound`` and ``upper_bound`` to constrain every basis
coefficient. ``MonotonePiecewiseLinear`` has unbounded internal decisions, but
its declared output bounds and monotonicity are enforced by its
parameterization.

The builder classifies comparisons by their time binding: expressions at
``t`` are path constraints, expressions at ``0`` are initial point
constraints, expressions at ``t_final`` (and parameter-only expressions) are
terminal point constraints.  A numeric ``t_final`` is fixed; a
``BoundedVariable`` declaration makes the horizon a free decision.  The
``integrate`` operation denotes the dynamic quadrature
``\int_0^T expr(t) dt`` rather than numerical sampling in the builder.
``state(time)`` exposes the differential state, while ``output(time)`` is the
observed system output (and may include algebraic or quadrature channels).
The current executable lowering boundary is the CasADi backend; symbolic
construction is backend-independent, but solving a built problem currently
requires CasADi.

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
callback receives that same positional layout as one final ``p`` tuple:


.. code-block:: python

   from coker import FunctionSpace, Scalar, VectorSpace
   from coker.algebra.ops import Noop
   from coker.dynamics import DynamicsSpec, create_dynamics_from_spec

   gain_curve = FunctionSpace(
       "gain_curve",
       arguments=[Scalar("speed")],
       output=[Scalar("gain")],
   )
   system = create_dynamics_from_spec(
       DynamicsSpec(
           inputs=Noop(),
           parameters=(gain_curve, VectorSpace("offset", 2)),
           algebraic=None,
           initial_conditions=lambda _z, _u, _p: (0.0, None),
           dynamics=lambda _t, x, _z, _u, p: p[0](x[0]) + p[1][0],
           constraints=Noop(),
           outputs=lambda _t, x, _z, _u, _p, _q: x,
           quadratures=Noop(),
       )
   )

The variational builder receives one declaration for each positional parameter.
Use ``BoundVector`` for a bounded vector decision block, and
``DenseTensorVariable`` for an unbounded dense block reconstructed with the
shape of its initial guess:

.. code-block:: python

   from coker.dynamics import (
       BoundVector,
       MonotonePiecewiseLinear,
       VariationalProblemBuilder,
   )

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

Function-valued parameters currently require one scalar argument and one scalar
output. ``MonotonePiecewiseLinear`` expands its monotonic basis into scalar
solver decisions, then reconstructs a callable parameter for the original
system. The returned ``solution.parameters`` mapping restores scalar decisions
as ``float`` values and vector or dense tensor decisions as shaped NumPy
arrays. Function decisions are ``FittedFunction`` objects, retaining their
specification, space, callable, and fitted parameters.

``MonotonePiecewiseLinear.domain_knots`` should span the normalized integration
domain from ``0`` to ``1``; use ``0`` and ``1`` as the endpoint knots and add
interior knots where the fitted response needs more resolution. This is the
solver's normalized integration coordinate, not physical time. A system whose
parameter is expressed in physical time should pass its normalized time
coordinate to the fitted function. The temporal transcription mesh is
independent of the selected interior knots.

Worked parameter-fitting example
--------------------------------

The test suite includes a minimal parameter-identification problem where the
unknown parameter is the constant state value of a one-dimensional system.

.. code-block:: python

   import numpy as np
   from coker import VectorSpace
   from coker.dynamics import (
       BoundedVariable,
       VariationalProblem,
       create_autonomous_ode,
   )

   def x0(p):
       return p[0]

   def xdot(x, p):
       return 0

   measured_value = np.array([2.0])

   system = create_autonomous_ode(
       parameters=VectorSpace("p", 1),
       x0=x0,
       xdot=xdot,
       backend="numpy",
   )

   def loss(solution, p_inner):
       total_error = 0.0
       for t_i in np.arange(0, 1, 0.1):
           truth = measured_value[0]
           estimate = solution(t_i, p_inner)
           total_error += (truth - estimate) ** 2
       return total_error

   problem = VariationalProblem(
       loss=loss,
       system=system,
       parameters=[
           BoundedVariable("value", upper_bound=3, lower_bound=0.5, guess=2)
       ],
       t_final=1,
       backend="casadi",
   )

   solution = problem()
   print(solution.parameters["value"])
   print(solution.solve_info.success)

This is the same shape used throughout ``tests/dynamical_systems/``: build a
system, define a loss against the system's callable output, and then solve a
:class:`~coker.dynamics.VariationalProblem`.

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
