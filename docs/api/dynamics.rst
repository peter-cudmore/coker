Dynamics API
============

System construction
-------------------

The constructors below return :class:`~coker.dynamics.DynamicalSystem` objects;
they trace callbacks but do not integrate a trajectory. ``backend`` defaults
to ``"numpy"``. See :doc:`../dynamics` for worked models, composition examples,
and the residual-DAE formulation.

.. py:class:: coker.dynamics.DynamicsSpec(inputs, parameters, algebraic, initial_conditions, dynamics, constraints, outputs, quadratures)

   Declare a semi-explicit system before tracing it with
   :func:`~coker.dynamics.create_dynamics_from_spec`.

   ``inputs`` is a :class:`~coker.FunctionSpace` for the input trajectory.
   ``parameters`` is ``None``, a scalar, vector or function-space declaration,
   or an ordered tuple/list of these declarations. ``algebraic`` is a
   :class:`~coker.VectorSpace` for the algebraic state, or ``None``.

   Callbacks have the following positional contracts:

   * ``initial_conditions(z, u, p)`` returns ``(x0, z0)``. Return ``None``
     for ``z0`` when no algebraic state is declared.
   * ``dynamics(t, x, z, u, p)`` returns the differential-state rate with
     the same shape as ``x0``.
   * ``constraints(t, x, z, u, p)`` returns the algebraic residual whose
     value must be zero.
   * ``quadratures(t, x, z, u, p)`` returns one scalar or vector rate.
   * ``outputs(t, x, z, u, p, q)`` returns the declared observable output.

   ``u`` is a trajectory callable: use ``u(t)`` to obtain its current value.
   For tuple/list parameter declarations, ``p`` is one tuple in declaration
   order, including callable values for function-space entries. A single
   declaration supplies its value directly.

   Use ``None`` for absent inputs, parameters, algebraic state, constraints,
   or quadratures. Absent input, parameter, algebraic and quadrature callback
   arguments receive ``None``; keep their positions in the callback signature.
   Algebraic state and constraints must be declared together. Quadrature
   initial values are not part of ``initial_conditions``.

.. py:function:: coker.dynamics.create_dynamics_from_spec(spec, backend="numpy")

   Trace a :class:`~coker.dynamics.DynamicsSpec` and return a
   :class:`~coker.dynamics.DynamicalSystem` containing its initial-condition,
   rate, residual, quadrature and output functions.

   The initial-condition callback must return exactly two values. The
   differential rate must have the initial differential state's shape, and
   the initial algebraic value must match its declaration. Shape mismatches
   raise ``AssertionError``; constraints without an algebraic declaration
   raise ``ValueError``. Tracing requires every callback operation to be
   supported by the selected backend.

.. py:function:: coker.dynamics.create_control_system(x0, xdot, control, parameters=None, output=None, backend="numpy", p_init=None, u_init=None)

   Return a controlled ODE with no algebraic state or quadratures.
   ``control`` is a function space of time with one scalar or vector output;
   ``parameters`` is a scalar/vector declaration or ``None``.

   ``x0`` is a fixed initial value (array, list, tuple or number), or a
   callback ``x0(p)``. ``xdot(t, x, u, p)`` returns a rate with the initial
   state's shape. Optional ``output(t, x, u, p)`` returns the observable;
   otherwise the output is the state. These callbacks receive the current
   **value** ``u``, not the input trajectory callable.
   With no parameters, ``p`` is ``None``.

   ``p_init`` and ``u_init`` supply sample values for construction-time shape
   checks, not parameter or control defaults for trajectory evaluation.
   ``u_init`` is a callable sampled at time zero. Omitted samples use zeros
   for declared parameters and controls. ``p_init`` requires a parameter
   declaration and must match its size.

.. py:function:: coker.dynamics.create_autonomous_ode(x0, xdot, parameters=None, output=None, backend="numpy", p_init=None)

   Return an ODE with no controls, algebraic state or quadratures.
   ``x0`` is a fixed initial value (array, list, tuple or number), or
   ``x0(p)``. ``parameters`` is a scalar/vector declaration or ``None``.

   ``xdot(x, p)`` returns a rate with the initial state's shape.
   Optional ``output(x, p)`` returns the observable; otherwise the output
   is the state, with scalar initial states producing scalar output.
   There is no time argument. ``p`` is ``None`` without a parameter
   declaration. ``p_init`` provides a construction-time parameter sample,
   with the same rules as :func:`~coker.dynamics.create_control_system`.

Trajectory objects
------------------

Call a system as ``system(time, input_trajectory, *parameter_values)``, omitting
the input argument if none is declared and passing one positional value per
parameter declaration. ``time`` is an endpoint or a one-dimensional time grid.
The result is the declared output, not a tuple of internal state channels.
On a grid, scalar and one-element-vector outputs form a one-dimensional
series; larger vectors are stacked with one row per requested time.

``DynamicalSystem.__call__`` currently rejects systems with quadratures with
``NotImplementedError``. Convert them with
:func:`~coker.dynamics.to_residual_dynamical_system` to evaluate a residual
trajectory; nonempty quadratures then start at zero.

Numerical support depends on the integration backend, not just whether
callbacks can be traced:

* NumPy integrates explicit ODEs, including converted ODEs, with SciPy.
  Independent square index-one residual models use a fixed-step implicit
  method. Converted semi-explicit DAEs are not supported by its direct
  integration path.
* PyTorch evaluates explicit ODEs, including converted ODEs,
  but rejects algebraic constraints and independent implicit residual models.
  ODE integration requires the PyTorch extra and ``torchdiffeq``.
* CasADi supports semi-explicit ODEs/DAEs and independent residual models.
  For independent residual models, select the IDAS or variational path with
  :class:`~coker.backends.casadi.CasadiResidualSolverOptions`.
* SymPy provides symbolic lowering and analysis, not numerical integration.

.. autoclass:: coker.dynamics.DynamicalSystem
   :members: backend,get_state_dimensions,output_as_function_space
   :special-members: __call__

.. autoclass:: coker.dynamics.ResidualDynamicalSystem
   :members: backend,get_state_dimensions,output_as_function_space
   :special-members: __call__

.. py:function:: coker.dynamics.to_residual_dynamical_system(system)

   Convert a semi-explicit :class:`~coker.dynamics.DynamicalSystem` into a
   :class:`~coker.dynamics.ResidualDynamicalSystem`. The residual combines
   differential-rate equations, quadrature-rate equations and algebraic
   constraints, retaining the input, parameter and output declarations.
   See :doc:`../dynamics` for the residual equation and worked examples.

Composition
-----------

.. py:function:: coker.dynamics.direct_sum(*systems, backend=None)

   Return ``(system, projections)`` for one or more independent
   :class:`~coker.dynamics.DynamicalSystem` objects. Channels are concatenated
   in subsystem order. ``system`` is a ``DynamicalSystem``;
   ``projections`` is a :class:`~coker.dynamics.ProjectionSet`, returned
   separately rather than attached to the system.

   The composed backend defaults to the first subsystem's backend.
   ``backend=`` traces the composition using another backend, which must
   support every subsystem operation.

   Numeric scalar/vector parameters use a flat-vector
   :class:`~coker.dynamics.CompositionOperator`. If any parameter declaration
   is function-valued, the combined parameter declarations remain an ordered
   tuple and the parameter projection is a
   :class:`~coker.dynamics.StructuredParameterProjection`.

   On the numeric-only path, each subsystem must declare one scalar/vector
   parameter space or no parameters; numeric-only tuples of declarations
   are not supported by this composition path.

.. py:class:: coker.dynamics.ProjectionSet(parameters, state, algebraic, quadratures, outputs, controls)

   Projections returned by :func:`~coker.dynamics.direct_sum`.
   ``state``, ``algebraic``, ``quadratures``, ``outputs`` and ``controls``
   are :class:`~coker.dynamics.CompositionOperator` objects.
   ``parameters`` is either a ``CompositionOperator`` for numeric-only
   declarations or a ``StructuredParameterProjection`` for a heterogeneous
   tuple containing function-valued declarations.

.. py:class:: coker.dynamics.CompositionOperator(*spaces)

   Join and split numeric channels declared by scalar/vector spaces or
   ``None``. Calling the operator with one value per subsystem flattens and
   concatenates present channels; an entirely absent channel returns ``None``.
   A scalar occupies one slot.

   .. py:method:: inverse(value)

      Return a list of flat subsystem slices in declaration order, with
      ``None`` for absent channels. Scalar slices remain one-element vectors.

   .. py:method:: as_matrices()

      Return one NumPy projection matrix per subsystem. Multiplication by
      the combined flat vector selects that subsystem's slice. Absent
      channels have zero-row matrices.

   .. py:method:: offsets()

      Yield each subsystem's starting offset in the combined vector.

   .. py:method:: sizes()

      Yield each subsystem's flattened width, or zero for an absent channel.

   .. py:method:: dim()

      Return the combined vector space, or ``None`` if its total width is zero.

.. py:class:: coker.dynamics.StructuredParameterProjection(widths)

   Partition parameter declarations without flattening their values.
   ``widths`` is a tuple of declaration counts per subsystem, not scalar
   dimensions. Function-valued entries remain callable.

   .. py:method:: split(parameters)

      Return a list containing one positional parameter tuple per subsystem.
      Subsystems without parameters receive an empty tuple.

Symbolic analysis
-----------------

See :doc:`../dynamics/system_analysis` for the equations, rank-test
interpretation and supported model scope. These entry points accept a
``DynamicalSystem``, ``SymbolicSystem`` or ``SymbolicDAESystem``.
They use SymPy lowering regardless of the model's numerical backend.

The supported domain is smooth systems without explicit time dependence,
with scalar or finite one-dimensional-vector states, controls, parameters
and outputs. Semi-explicit DAEs must be index one, with one constraint per
algebraic variable and a generically nonsingular algebraic Jacobian.
Quadratures, function-valued parameters, implicit symbolic functions,
nonsmooth expressions and control-dependent DAE constraints are unsupported.
An independent ``ResidualDynamicalSystem`` is not an analysis input.

Every analysis accepts keyword-only ``max_order=None``: ``None`` allows
expansion until rank closure; a non-negative integer limits expansion order.
An unsupported model, invalid limit or exhausted limit returns
``INCONCLUSIVE`` with a ``reason`` rather than establishing a negative result.
Unsupported symbolic operations can therefore make analysis inconclusive
even when numerical evaluation succeeds.

.. autofunction:: coker.dynamics.analyse_controllability

   Return :class:`~coker.dynamics.ControllabilityResult` from the control-field
   Lie-algebra rank test. Despite the function name, the property tested is
   **generic local accessibility**, not global controllability or reachability
   with constrained inputs. Dynamics must be affine in their controls.

.. autofunction:: coker.dynamics.analyse_observability

   Return :class:`~coker.dynamics.ObservabilityResult` from output
   Lie derivatives. Known controls are treated as arbitrary signals;
   controlled dynamics must be input-affine. The required rank is the
   differential-state dimension, including for supported DAEs.

.. autofunction:: coker.dynamics.analyse_identifiability

   Return :class:`~coker.dynamics.IdentifiabilityResult` by augmenting the
   state with constant unknown parameters and testing observability.
   Controlled systems are unsupported.

   Keyword-only ``parameters`` is ``None`` to treat all model parameters as
   unknown, or a sequence in flattened scalar-parameter order.
   ``BoundedVariable`` and ``UnboundedVariable`` entries denote unknowns;
   real numeric entries denote known values substituted before analysis.
   Bounds are not used by the symbolic rank test. The sequence must match
   the flattened parameter count. Required rank is the differential-state
   dimension plus the number of unknown scalar parameters; this is a joint
   state/parameter rank test, not a numerical fitting result.

.. py:class:: coker.dynamics.AnalysisStatus

   Enum identifying the outcome for the property named by the result type.
   Compare ``result.status`` with an enum member, not with its truth value.

   .. py:attribute:: TRUE

      Full required rank establishes the generic local property.
      The enum value is ``True``.

   .. py:attribute:: FALSE

      Stabilised rank below the required rank gives a negative rank-test
      result. This is not the outcome for unsupported models.
      The enum value is ``False``.

   .. py:attribute:: INCONCLUSIVE

      The supported analysis could not decide, or the model/arguments are
      unsupported. Inspect ``reason``. The enum value is ``None``.

.. py:class:: coker.dynamics.AnalysisResult(status, rank, required_rank, matrix, generators, rank_conditions, reason=None)

   Immutable base result shared by the three analyses.

   ``status`` is an :class:`~coker.dynamics.AnalysisStatus`. ``rank`` is the
   established symbolic matrix rank; ``required_rank`` is the full rank
   needed by the test. ``matrix`` is the final SymPy matrix. Accessibility
   stores vector fields as columns; observability and identifiability store
   output/Lie-derivative gradients as rows. ``generators`` contains the
   vector fields or scalar expressions used to form the matrix.

   .. py:attribute:: rank_conditions

      Tuple of SymPy expressions that must all be nonzero for the reported
      generic rank to apply. Conditions include a nonzero maximal-minor
      witness and, for supported DAE rank results, the algebraic Jacobian
      determinant. No conclusion is given on excluded singular sets.
      A rank-zero ODE has no nonzero-minor witness.

   .. py:attribute:: reason

      Diagnostic string for an inconclusive outcome, otherwise ``None``.
      An inconclusive result can retain partial matrix/rank information.
      Failure before lowering may instead report empty data and
      ``required_rank=0``; this does not establish a zero-dimensional model.

.. autoclass:: coker.dynamics.ControllabilityResult

.. autoclass:: coker.dynamics.ObservabilityResult

.. autoclass:: coker.dynamics.IdentifiabilityResult

Symbolic representations
~~~~~~~~~~~~~~~~~~~~~~~~

These lower-level interfaces expose ordered SymPy data for advanced analysis.
Unlike the analysis entry points, lowering raises
:class:`~coker.dynamics.UnsupportedSystemError` for unsupported models.

.. autofunction:: coker.dynamics.lower_system

   Return :class:`~coker.dynamics.SymbolicSystem` for a supported ODE.
   Passing an existing ``SymbolicSystem`` returns it unchanged.
   Algebraic constraints require ``lower_dae_system`` instead.

.. autofunction:: coker.dynamics.lower_dae_system

   Return :class:`~coker.dynamics.SymbolicDAESystem` for a semi-explicit
   ``DynamicalSystem``, retaining algebraic variables rather than solving
   constraints. Index-one regularity is checked when an analysis constructs
   the constraint-manifold geometry.

.. autoclass:: coker.dynamics.SymbolicSystem

   ``state``, ``parameters`` and ``controls`` are ordered tuples of SymPy
   symbols. ``dynamics`` and ``outputs`` are ordered scalar-expression tuples.
   ``dynamics`` has one entry per differential state.

.. autoclass:: coker.dynamics.SymbolicDAESystem

   Adds ordered ``algebraic`` symbols and ``constraints`` expressions to the
   ODE fields. Constraints must be independent of controls; their Jacobian
   with respect to ``algebraic`` must be square and generically nonsingular
   for analysis.

.. autoexception:: coker.dynamics.UnsupportedSystemError

Variational problems
--------------------

Use ``VariationalProblemBuilder`` to construct symbolic objectives and
constraints. ``VariationalProblem`` also accepts a loss callback.
``TranscriptionOptions`` configures discretisation; backend-specific options
select solver behaviour. See :doc:`../dynamics` for CasADi solve support and
current lowering limits.


.. autoclass:: coker.dynamics.VariationalProblemBuilder
   :members: state,input,output,integrate,build

.. autoclass:: coker.dynamics.TranscriptionOptions

.. autoclass:: coker.dynamics.VariationalProblem
   :members: get_solver
