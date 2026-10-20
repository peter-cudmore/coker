Dynamics and optimisation
=========================

Coker's dynamics layer uses the symbolic machinery of :func:`coker.function`
to describe states, controls, parameters, and transcriptions. Use it to define
ODEs and DAEs, fit parameters, and formulate optimal-control problems.

Core entry points
-----------------

Public entry points include:

- :func:`coker.dynamics.create_autonomous_ode`
- :func:`coker.dynamics.create_control_system`
- :class:`coker.dynamics.DynamicsSpec` with :func:`coker.dynamics.create_dynamics_from_spec`
- :class:`coker.dynamics.ResidualDynamicalSystem` with :func:`coker.dynamics.to_residual_dynamical_system`
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

``create_autonomous_ode()`` builds a :class:`~coker.dynamics.DynamicalSystem`
from initial-condition and ``xdot`` functions. Declare a parameter space to
use the system in a :class:`~coker.dynamics.VariationalProblem`.

``create_control_system(x0, xdot, control, ...)`` builds a controlled ODE with
the callback ``xdot(t, x, u, p)``, where ``u`` is the control value.
Use ``DynamicsSpec`` for general systems; it requires no internal absence
sentinel:

.. code-block:: python

   import numpy as np
   from coker import FunctionSpace, Scalar, VectorSpace
   from coker.dynamics import (
       DynamicsSpec, create_control_system, create_dynamics_from_spec, direct_sum,
   )

   control = FunctionSpace("u", [Scalar("t")], [VectorSpace("u", 1)])
   controlled = create_control_system(
       [0.0], lambda t, x, u, p: u, control,
   )
   general = create_dynamics_from_spec(DynamicsSpec(
       inputs=None,
       parameters=None,
       algebraic=None,
       initial_conditions=lambda z, u, p: (np.array([1.0]), None),
       dynamics=lambda t, x, z, u, p: -x,
       constraints=None,
       outputs=lambda t, x, z, u, p, q: x,
       quadratures=None,
   ))

``DynamicsSpec.initial_conditions(z, u, p)`` returns ``(x0, z0)``. Its dynamics,
constraints, and quadrature callbacks receive ``(t, x, z, u, p)``; outputs
receive ``(t, x, z, u, p, q)``. Tuple parameters arrive as one final ``p`` tuple
(before ``q`` for outputs). ``None`` declares absent inputs, constraints, or
quadratures. ``system.backend`` exposes the backend name as a read-only property.

Composition and projections
---------------------------

``direct_sum(*systems, backend=None)`` returns
``(DynamicalSystem, ProjectionSet)`` in subsystem order. Projections are returned
separately, not attached to the system. The composed system uses the first
system's backend unless ``backend=`` selects another.

.. code-block:: python

   combined, projections = direct_sum(general, general)
   output = combined(0.25)
   left, right = projections.outputs.inverse(output)
   left_matrix, right_matrix = projections.outputs.as_matrices()

``ProjectionSet`` exposes ``state``, ``algebraic``, ``quadratures``, ``outputs``,
``controls``, and ``parameters``. Numeric projections are
``CompositionOperator`` values: ``inverse(value)`` splits a flat vector,
``as_matrices()`` returns one projection matrix per subsystem, and
``offsets()``/``sizes()`` describe the partition. Absent channels produce
``None`` entries in ``inverse`` and zero-row matrices. Scalar outputs occupy
one flat slot.

When any subsystem declares a function-valued parameter, ``parameters`` is a
``StructuredParameterProjection`` instead. ``split(parameter_tuple)`` returns
one positional tuple per subsystem, and ``widths`` reports declaration counts,
not flattened scalar widths. Numeric-only parameters retain their flat-vector
``CompositionOperator`` projection.

See :doc:`api/index` for the ordinary/extension/internal API boundary.

Residual differential-algebraic systems
---------------------------------------

:class:`~coker.dynamics.ResidualDynamicalSystem` represents an implicit system:

.. math::

   F(t, w, \dot{w}, z, u, p) = 0,

where ``w`` combines differential and quadrature state, and ``z`` is the
algebraic state. Use it when the model is inherently implicit; use
:func:`~coker.dynamics.to_residual_dynamical_system` to convert an existing
semi-explicit :class:`~coker.dynamics.DynamicalSystem`.

The NumPy backend integrates square index-one residual systems with a fixed-step
implicit method. The residual must provide one row for each differential or
quadrature rate and algebraic variable. Initial algebraic values are required
when algebraic variables are declared. CasADi provides two explicitly selected
residual solver paths through
:class:`~coker.backends.casadi.CasadiResidualSolverOptions`:
:attr:`~coker.backends.casadi.CasadiResidualSolver.IDAS` uses the native
SUNDIALS plugin, and
:attr:`~coker.backends.casadi.CasadiResidualSolver.VARIATIONAL` uses a direct
collocation feasibility solve. IDAS options and IPOPT options remain separate
in that options object.
This interface does not discretise PDEs. A finite-element, finite-volume, or
weak-form model may supply its residual DAE.

This CasADi example solves the coupled residual equations
``xdot + z - 1 = 0`` and ``x + 2 * xdot - z = 0``:

.. doctest:: residual_dae

   >>> import numpy as np
   >>> from coker import Dimension, Scalar, VectorSpace, function
   >>> from coker.dynamics import ResidualDynamicalSystem
   >>> residual = ResidualDynamicalSystem(
   ...     inputs=None,
   ...     parameters=None,
   ...     x0=function(
   ...         [VectorSpace("z", 1), None, None],
   ...         lambda z, _u, _p: (np.array([0.0]), z),
   ...         backend="casadi",
   ...     ),
   ...     F=function(
   ...         [
   ...             Scalar("t"),
   ...             VectorSpace("w", 1),
   ...             VectorSpace("wdot", 1),
   ...             VectorSpace("z", 1),
   ...             None,
   ...             None,
   ...         ],
   ...         lambda _t, w, wdot, z, _u, _p: np.array(
   ...             [wdot[0] + z[0] - 1.0, w[0] + 2.0 * wdot[0] - z[0]]
   ...         ),
   ...         backend="casadi",
   ...     ),
   ...     y=function(
   ...         [Scalar("t"), VectorSpace("x", 1), VectorSpace("z", 1),
   ...          None, None, None],
   ...         lambda _t, x, _z, _u, _p, _q: x,
   ...         backend="casadi",
   ...     ),
   ...     differential=Dimension(1),
   ...     algebraic=Dimension(1),
   ...     quadrature=None,
   ... )
   >>> np.testing.assert_allclose(
   ...     residual(0.25),
   ...     [1.0 - np.exp(-0.25 / 3.0)],
   ...     rtol=1e-4,
   ... )

Keep a semi-explicit model's ODE/DAE declaration and convert it when residual
access is needed:

.. doctest:: residual_dae

   >>> from coker.dynamics import create_autonomous_ode, to_residual_dynamical_system
   >>> explicit = create_autonomous_ode(
   ...     x0=np.array([1.0]),
   ...     xdot=lambda x, _parameters: x,
   ...     backend="numpy",
   ... )
   >>> converted = to_residual_dynamical_system(explicit)
   >>> np.testing.assert_allclose(
   ...     converted.F(
   ...         0.0, np.array([2.0]), np.array([2.0]), np.empty((0,)), None, None
   ...     ),
   ...     [0.0],
   ... )


Functional problem building
---------------------------

Use :class:`coker.dynamics.VariationalProblemBuilder` to declare controls,
parameters, a :class:`coker.toolkits.codesign.Minimise` objective, and
``subject_to`` comparisons.

The optimisation examples require ``pip install "coker[casadi]"``, which
includes NumPy. Symbolic construction is backend-independent; solving a built
problem currently requires CasADi.

This fixed-horizon path-planning example constrains the input and terminal
state. It constructs a symbolic problem only: the current CasADi solve path
cannot lower its control-dependent callbacks.

.. code-block:: python

   import numpy as np
   from coker import FunctionSpace, Scalar
   from coker.dynamics import (
       ConstantControlVariable, DynamicsSpec, VariationalProblemBuilder,
       create_dynamics_from_spec,
   )
   from coker.toolkits.codesign import Minimise

   input_space = FunctionSpace("u", [Scalar("t")], [Scalar("u")])
   system = create_dynamics_from_spec(
       DynamicsSpec(
           inputs=input_space,
           parameters=None,
           algebraic=None,
           initial_conditions=lambda _z, _u, _p: (np.zeros(1), None),
           dynamics=lambda t, _x, _z, u, _p: np.ones(1) * u(t),
           constraints=None,
           outputs=lambda _t, x, _z, _u, _p, _q: x,
           quadratures=None,
       )
   )
   control = ConstantControlVariable("u", lower_bound=-1.0, upper_bound=1.0)
   goal = np.array([1.0])

   with VariationalProblemBuilder(
       system,
       t_final=5.0,
       control=[control],
   ) as problem:
       x = problem.state(problem.t)
       u = problem.input(problem.t)
       x_goal = problem.state(problem.t_final)
       built = problem.build(
           Minimise(problem.integrate(u ** 2) + np.dot(x_goal, x_goal)),
           subject_to=[u <= 1, u >= -1, x_goal == goal],
       )

For data fitting, declare the unknown as a bounded parameter and use the
symbolic observed output directly in the objective:

.. code-block:: python

   from coker import VectorSpace
   from coker.dynamics import VariationalProblemBuilder, create_autonomous_ode
   from coker.parameters import BoundedVariable
   from coker.toolkits.codesign import Minimise

   system = create_autonomous_ode(
       parameters=VectorSpace("value", 1),
       x0=lambda p: p[0],
       xdot=lambda _x, _p: 0.0,
   )
   value = BoundedVariable("value", lower_bound=0.5, upper_bound=3.0, guess=1.0)

   with VariationalProblemBuilder(
       system,
       t_final=1.0,
       parameters=[value],
   ) as problem:
       observed = 2.0
       built = problem.build(
           Minimise((problem.output(problem.t_final)[0] - observed) ** 2)
       )

The builder classifies comparisons by their time binding:

- expressions at ``t`` are path constraints;
- expressions at ``0`` are initial point constraints;
- expressions at ``t_final`` and parameter-only expressions are terminal point
  constraints.

A numeric ``t_final`` fixes the horizon; a ``BoundedVariable`` makes it a solver
decision. ``integrate(expr)`` denotes the dynamic quadrature
``\int_0^T expr(t) dt``, not numerical sampling in the builder.
``state(time)`` exposes the differential state. ``output(time)`` exposes the
observed output, which may include algebraic or quadrature channels.

Function-valued parameters
--------------------------

Systems may declare a :class:`coker.FunctionSpace` parameter. At problem
construction, supply a
:class:`~coker.parameters.function_parameters.FunctionParameter` realisation
with matching argument and output shapes. Coker expands its concrete parameter
blocks into scalar solver decisions. A solver-bound function parameter needs
a non-empty ``name`` for those blocks.

``DenseLayer`` realises ``activation(weights @ x + bias)`` for vector targets
and ``activation(weight * x + bias)`` for scalar targets. Vector activations
and targets use vector inputs and outputs. Scalar activations and targets use
two unbounded scalar decisions, ``weight`` and ``bias``.

``ClosureParameter`` binds declared scalar or dense blocks to a Coker function.
The function receives those blocks before the target function's arguments.
Use ``ClosureParameter.bind_callable`` to trace a Python implementation with
that calling convention.

This dense layer maps a two-component state to a one-component response with
an identity activation:

.. code-block:: python

   import numpy as np
   from coker import FunctionSpace, VectorSpace, function
   from coker.dynamics import (
       DynamicsSpec, VariationalProblemBuilder, create_dynamics_from_spec,
   )
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
   system = create_dynamics_from_spec(
       DynamicsSpec(
           inputs=None,
           parameters=(response,),
           algebraic=None,
           initial_conditions=lambda _z, _u, _p: (np.zeros(2), None),
           dynamics=lambda _t, x, _z, _u, p: np.concatenate([p[0](x), p[0](x)]),
           constraints=None,
           outputs=lambda _t, x, _z, _u, _p, _q: x,
           quadratures=None,
       )
   )
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
output bounds and monotonicity are enforced by its parameterisation.


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
   from coker.dynamics import TranscriptionOptions, VariationalProblem, create_autonomous_ode

   system = create_autonomous_ode(x0=0.0, xdot=lambda _x, _p: 1.0)

   def loss(trajectory, _parameters):
       return (trajectory(1.0) - 1.0) ** 2

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
below that normalised width. ``maximum_degree`` must be no less than
``minimum_degree``. Leave ``refinement_enabled`` at its default ``False`` for
a single transcription solve.

``absolute_tolerance`` bounds initial-condition, continuity, and algebraic
residuals. Set ``segment_defect_tolerance`` and
``derivative_defect_tolerance`` independently to bound the integrated
per-segment defects and the collocation derivative defects. Both default to
``None`` and therefore inherit ``absolute_tolerance``; either may be ``0.0``
to require an exact NLP equality.

During an adaptive solve, CasADi retains the model setup and compiles only the
mesh-dependent NLPs. Each refined mesh interpolates the preceding path and
seeds compatible controls, free parameters, and a free horizon from the
preceding solution. Explicit parameters passed to ``solve`` override the
carried parameter value. Successful solutions expose
``adaptive_refinement_rounds`` (mesh updates after the initial solve) and
``adaptive_maximum_defect`` (the final maximum scaled state defect).

A bounded cache reuses previously visited mesh signatures. Its lifetime is
an implementation detail. Treat adaptive solving as deterministic for a
fixed problem, options, and parameter values, without relying on cache reuse
across separately created problems.

Parameter order and reconstruction
----------------------------------

System parameter declarations may be one declaration or a positional tuple of
``Scalar``, ``VectorSpace``, and ``FunctionSpace`` declarations. A dynamics
callback receives that same positional layout as one final ``p`` tuple.

Direct trajectory evaluation retains that declaration order. For a system
without declared inputs, call ``system(time, *parameters)``. For a system with
declared inputs, call ``system(time, input_value, *parameters)``. Do not add an
input placeholder to the no-input form.

Declare and supply all three parameter slots in the same order:

.. code-block:: python

   import numpy as np

   from coker import FunctionSpace, Scalar, VectorSpace
   from coker.dynamics import DynamicsSpec, create_dynamics_from_spec

   gain_curve = FunctionSpace(
       "gain_curve",
       arguments=[Scalar("time")],
       output=[Scalar("gain")],
   )
   spec = DynamicsSpec(
       inputs=None,
       parameters=(Scalar("offset"), VectorSpace("bias", 2), gain_curve),
       algebraic=None,
       initial_conditions=lambda _z, _u, _p: (np.zeros(2), None),
       dynamics=lambda t, _x, _z, _u, p: p[1] + p[0] + p[2](t),
       constraints=None,
       outputs=lambda _t, x, _z, _u, _p, _q: x,
       quadratures=None,
   )
   system = create_dynamics_from_spec(spec, backend="numpy")

   time = np.array([0.0, 0.5, 1.0])
   offset = 2.0
   bias = np.array([0.25, -0.25])
   gain_curve_value = lambda time: 1.0 + time

   trajectory = system(time, offset, bias, gain_curve_value)

A system with a declared input takes its input value immediately after time:

.. code-block:: python

   from dataclasses import replace

   input_space = FunctionSpace("input", [Scalar("time")], [Scalar("value")])
   input_system = create_dynamics_from_spec(
       replace(
           spec,
           inputs=input_space,
           dynamics=lambda t, _x, _z, u, p: p[1] + p[0] + p[2](t) + u(t),
       ),
       backend="numpy",
   )
   input_value = lambda _time: 0.5
   trajectory_with_input = input_system(time, input_value, offset, bias, gain_curve_value)

Values for a declared ``FunctionSpace`` parameter may be Python callables,
:class:`coker.Function` objects, or
:class:`~coker.parameters.function_parameters.FittedFunction` objects. Coker
normalises them once before integration; do not pass backend-native or lowered
handles. Direct evaluation does not use a ``Noop`` placeholder.

For the original no-input ``system``, pass builder declarations in the same
order: scalar offset, length-two bias, then gain curve. ``BoundVector``
declares a bounded vector decision block. ``DenseTensorVariable`` declares an
unbounded dense block with the shape of its initial guess:

.. code-block:: python

   from coker.dynamics import VariationalProblemBuilder
   from coker.parameters.function_parameters import MonotonePiecewiseLinear
   from coker.parameters import BoundedVariable, BoundVector
   from coker.toolkits.codesign import Minimise

   with VariationalProblemBuilder(
       system,
       t_final=1.0,
       parameters=[
           BoundedVariable("offset", lower_bound=-1.0, upper_bound=1.0, guess=0.0),
           BoundVector(
               "bias",
               lower_bound=[-1.0, -1.0],
               upper_bound=[1.0, 1.0],
               guess=[0.0, 0.0],
           ),
           MonotonePiecewiseLinear(
               domain_knots=[0.0, 0.5, 1.0],
               lower_bound=0.0,
               upper_bound=2.0,
               name="gain_curve",
           ),
       ],
   ) as builder:
       problem = builder.build(Minimise(builder.output(builder.t_final)[0] ** 2))

``DenseLayer`` expands into weight and bias blocks.
``MonotonePiecewiseLinear`` and ``RadialBasisFunction`` each expand into one
vector block. The solver flattens these blocks into scalar decisions and
restores their shapes when reconstructing the fitted function.

``solution.parameters`` maps scalar decisions to ``float`` values and vector
or dense tensor decisions to shaped NumPy arrays. Function decisions become
``FittedFunction`` objects with their specification, space, callable, and
structured fitted parameters.

``MonotonePiecewiseLinear.domain_knots`` should span the normalised integration
domain from ``0`` to ``1``; use these as endpoint knots and add interior knots
where the fitted response needs more resolution. This is the solver's
normalised integration coordinate, not physical time. For a system parameter
expressed in physical time, pass the corresponding normalised time coordinate
to the fitted function. The temporal transcription mesh is independent of the
interior knots.

Worked parameter-fitting example
--------------------------------

This example fits the constant state of a one-dimensional system using the
callback-based :class:`~coker.dynamics.VariationalProblem` interface:

.. code-block:: python

   import numpy as np
   from coker import VectorSpace
   from coker.dynamics import VariationalProblem, create_autonomous_ode
   from coker.parameters import BoundedVariable

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


Solve status and failures
-------------------------

Dynamics and codesign solves expose metadata through
:class:`coker.toolkits.codesign.SolveInfo`. A failed solve raises
:class:`coker.toolkits.codesign.SolveFailure` with the backend status attached.

Further reading
----------------

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
