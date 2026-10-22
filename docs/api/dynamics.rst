Dynamics API
============

System construction
-------------------

The constructors below return :class:`~coker.dynamics.DynamicalSystem` objects;
they trace callbacks but do not integrate a trajectory. ``backend`` defaults
to ``"numpy"``. See :doc:`../dynamics` for worked models, composition examples,
and the residual-DAE formulation.

.. autoclass:: coker.dynamics.DynamicsSpec

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

.. autofunction:: coker.dynamics.create_dynamics_from_spec

   Trace a :class:`~coker.dynamics.DynamicsSpec` and return a
   :class:`~coker.dynamics.DynamicalSystem` containing its initial-condition,
   rate, residual, quadrature and output functions.

   The initial-condition callback must return exactly two values. The
   differential rate must have the initial differential state's shape, and
   the initial algebraic value must match its declaration. Wrong callback
   types raise ``TypeError``; invalid return arity, shapes, or missing
   algebraic/constraint configuration raise ``ValueError``. Dynamics and
   quadrature callbacks each return one output. Tracing requires every
   callback operation to be supported by the selected backend.

.. autofunction:: coker.dynamics.create_control_system

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
   for declared vector parameters and controls. Scalar parameter declarations
   currently require an explicit ``p_init``. ``p_init`` requires a parameter
   declaration and must match its size.
   Wrong callback or sample-callable types raise ``TypeError``. Invalid
   parameter/control sample shapes, a multi-output control declaration,
   a rate shape differing from ``x0``, or an output returning ``None`` raise
   ``ValueError``.


.. autofunction:: coker.dynamics.create_autonomous_ode

   Return an ODE with no controls, algebraic state or quadratures.
   ``x0`` is a fixed initial value (array, list, tuple or number), or
   ``x0(p)``. ``parameters`` is a scalar/vector declaration or ``None``.

   ``xdot(x, p)`` returns a rate with the initial state's shape.
   Optional ``output(x, p)`` returns the observable; otherwise the output
   is the state, with scalar initial states producing scalar output.
   There is no time argument. ``p`` is ``None`` without a parameter
   declaration. ``p_init`` provides a construction-time parameter sample,
   with the same rules as :func:`~coker.dynamics.create_control_system`.
   Wrong callback types raise ``TypeError``. Invalid parameter sample shapes,
   a rate shape differing from ``x0``, or an output returning ``None`` raise
   ``ValueError``.


Trajectory objects
------------------

Call a system as ``system(time, input_trajectory, *parameter_values)``, omitting
the input argument if none is declared and passing one positional value per
parameter declaration. ``time`` is an endpoint or a one-dimensional time grid.
The result is the declared output, not a tuple of internal state channels.
On a supported grid, larger vectors are stacked with one row per requested
time. NumPy squeezes one-component outputs to zero-dimensional values at
Python scalar endpoints. Grid evaluation of these outputs currently raises
``ValueError`` because it attempts to concatenate zero-dimensional values.
Evaluate separate scalar endpoints for one-component NumPy outputs instead;
this limitation does not apply to multi-component vector outputs.

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

.. autofunction:: coker.dynamics.to_residual_dynamical_system

   Convert a semi-explicit :class:`~coker.dynamics.DynamicalSystem` into a
   :class:`~coker.dynamics.ResidualDynamicalSystem`. The residual combines
   differential-rate equations, quadrature-rate equations and algebraic
   constraints, retaining the input, parameter and output declarations.
   See :doc:`../dynamics` for the residual equation and worked examples.

Composition
-----------

.. autofunction:: coker.dynamics.direct_sum

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

.. autoclass:: coker.dynamics.ProjectionSet

   Projections returned by :func:`~coker.dynamics.direct_sum`.
   ``state``, ``algebraic``, ``quadratures``, ``outputs`` and ``controls``
   are :class:`~coker.dynamics.CompositionOperator` objects.
   ``parameters`` is either a ``CompositionOperator`` for numeric-only
   declarations or a ``StructuredParameterProjection`` for a heterogeneous
   tuple containing function-valued declarations.

.. autoclass:: coker.dynamics.CompositionOperator
   :members: inverse,as_matrices,offsets,sizes,dim
   :special-members: __call__

   Join and split numeric channels declared by scalar/vector spaces or
   ``None``. Calling the operator with one value per subsystem flattens and
   concatenates present channels; an entirely absent channel returns ``None``.
   A scalar occupies one slot.

   ``inverse(value)`` returns a list of flat subsystem slices in declaration
   order, with ``None`` for absent channels. Scalar slices remain one-element
   vectors. ``as_matrices()`` returns one NumPy projection matrix per subsystem;
   absent channels have zero-row matrices.

   ``offsets()`` and ``sizes()`` yield each subsystem's starting offset and
   flattened width. ``dim()`` returns the combined vector space, or ``None``
   when the total width is zero.

.. autoclass:: coker.dynamics.StructuredParameterProjection
   :members: split

   Partition parameter declarations without flattening their values.
   ``widths`` is a tuple of declaration counts per subsystem, not scalar
   dimensions. Function-valued entries remain callable.

   ``split(parameters)`` returns one positional tuple per subsystem.
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

.. autoclass:: coker.dynamics.AnalysisStatus

   Enum identifying the outcome for the property named by the result type.
   Compare ``result.status`` with an enum member, not with its truth value.

   ``TRUE`` means full required rank establishes the generic local property;
   its enum value is ``True``. ``FALSE`` means stabilised rank is below the
   required rank; its value is ``False``. Unsupported models do not produce
   a negative rank-test result.

   ``INCONCLUSIVE`` means the supported analysis could not decide, or the
   model/arguments are unsupported. Inspect ``reason``. Its enum value is
   ``None``.

.. autoclass:: coker.dynamics.AnalysisResult

   Immutable base result shared by the three analyses.

   ``status`` is an :class:`~coker.dynamics.AnalysisStatus`. ``rank`` is the
   established symbolic matrix rank; ``required_rank`` is the full rank
   needed by the test. ``matrix`` is the final SymPy matrix. Accessibility
   stores vector fields as columns; observability and identifiability store
   output/Lie-derivative gradients as rows. ``generators`` contains the
   vector fields or scalar expressions used to form the matrix.

   ``rank_conditions`` is a tuple of SymPy expressions that must all be nonzero
   for the reported generic rank to apply. Conditions include a nonzero
   maximal-minor witness and, for supported DAE rank results, the algebraic
   Jacobian determinant. No conclusion is given on excluded singular sets.
   A rank-zero ODE has no nonzero-minor witness.

   ``reason`` is a diagnostic string for an inconclusive outcome, otherwise
   ``None``. An inconclusive result can retain partial matrix/rank information.
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
   :members: t,t_final,parameters,state,input,output,integrate,build
   :special-members: __enter__,__exit__

.. autoclass:: coker.dynamics.TranscriptionOptions

.. autoclass:: coker.dynamics.VariationalProblem
   :members: horizon_decision,decision_declarations,get_solver
   :special-members: __call__

Controls and callbacks
~~~~~~~~~~~~~~~~~~~~~~

Declare scalar control decisions with the classes below. Their coefficients,
sample rates and spike locations use the normalised transcription coordinate,
not physical time. Inspect a solved control in physical time through
:meth:`~coker.dynamics.VariationalSolution.control_law`.

.. autoclass:: coker.dynamics.ConstantControlVariable
   :members: degrees_of_freedom,to_solution

   ``name`` identifies one scalar decision with ``lower_bound`` and
   ``upper_bound``. ``degrees_of_freedom(*interval)`` returns one.
   ``to_solution(value)`` constructs a constant control.

.. autoclass:: coker.dynamics.PiecewiseConstantVariable
   :members: degrees_of_freedom,to_solution

   ``sample_rate`` specifies coefficients per unit normalised time.
   ``degrees_of_freedom(start, end)`` returns
   ``ceil((end - start) * sample_rate)``.
   ``to_solution(value)`` retains the coefficient array in a piecewise control.

.. autoclass:: coker.dynamics.SpikeVariable
   :members: degrees_of_freedom,to_solution

   ``time`` is the normalised spike location.
   ``degrees_of_freedom(*interval)`` returns one.
   ``to_solution(value)`` constructs a spike control.

The corresponding solution objects are callable at **normalised** time.
They are stored in ``VariationalSolution.control_solutions``; they do not
themselves convert physical time.

.. autoclass:: coker.dynamics.ConstantControlSolution
   :special-members: __call__

   Returns its stored ``value`` at any time.

.. autoclass:: coker.dynamics.PiecewiseControlSolution
   :special-members: __call__

   Selects coefficient ``int(t * sample_rate)`` from ``value``.
   A coefficient index beyond the stored array raises ``AssertionError``;
   this object does not implement extrapolation or endpoint clipping.

.. autoclass:: coker.dynamics.SpikeControlSolution
   :special-members: __call__

   Returns ``value`` when ``abs(t - time) < tolerance`` and zero otherwise.
   It is a pointwise control representation, not a distributional impulse.

``ControlVariable`` and ``ControlSolution`` are unions of their three concrete
types above. ``ControlLaw`` is the callable control annotation. ``Solution``
and ``LossFunction`` annotate the trajectory and objective callbacks used by
``VariationalProblem``; they are not additional result containers.

.. autoclass:: coker.dynamics.VariationalIterationCallback
   :special-members: __call__

   Extension point receiving an iteration index and a
   ``VariationalSolution`` iterate plus backend-specific keyword arguments.
   Return ``True`` to continue. Configure it through the existing
   ``TranscriptionOptions.interation_callback`` option or the backend's
   callback option; callback support depends on that backend.

.. _variational-solution:

Solve results
-------------

Calling a variational problem or solver returns
:class:`~coker.dynamics.VariationalSolution`. An unsuccessful solve raises
:class:`~coker.toolkits.codesign.SolveFailure` with ``exception.solve_info``;
the result interface does not turn failure into a successful trajectory.
Native solver, configuration and evaluation exceptions are not necessarily
wrapped. See :doc:`optimisation` for :class:`~coker.toolkits.codesign.SolveInfo`
and :doc:`../dynamics` for a worked solve and result inspection.

``cost`` is the final scalar objective; ``parameters`` maps names to fitted
values; ``solve_info`` holds status metadata (or ``None`` when not supplied).
CasADi reconstructs numeric scalar decisions as floats and vector/tensor
decisions as shaped NumPy arrays. PyTorch can retain tensor values.
Function-valued decisions are
:class:`~coker.parameters.function_parameters.FittedFunction` objects; their
structured coefficients follow the selected backend. Supplied literal
parameter values retain their reconstruction convention.

Public evaluation uses scalar **physical** time on ``[0, t_final]``:

* ``solution(t)`` returns the declared observable output, preserving its
  evaluator's shape and numeric convention.
* ``state(t)`` returns the differential-state NumPy vector.
* ``algebraic(t)`` and ``quadratures(t)`` return NumPy vectors or ``None`` when
  the corresponding projector is absent.
* ``control_law(t)`` stacks controls in declaration order, or returns ``None``
  when no controls are present. Boundary behaviour follows the control type.
* ``path_constraints(t)`` and ``terminal_constraints()`` return flat,
  nonnegative stored half-space violations. Zero means satisfied. An empty
  stored list produces an empty array, not ``None``. These are not full
  feasibility certificates: initial point constraints, collocation equations
  and decision bounds are not included. Two finite bounds on one stored
  component raise ``ValueError``.

The underlying ``path`` contains combined differential, algebraic and
quadrature channels in **normalised** time. ``projectors`` select these
channels; ``control_solutions`` also use normalised time. ``output`` is the
low-level evaluator and ``path_constraint_exprs``/``terminal_constraint_exprs``
are the stored expressions. Prefer the public evaluation methods over calling
these representations directly. Present path channels reject times outside
the solved intervals with ``ValueError``; there is no time-grid or
extrapolation contract. Absent channels need not evaluate the time at all.

``as_raw()`` allocates a sample array whose first row holds physical knot
times and remaining rows hold the combined **internal** channels, not outputs.
It omits each segment's right endpoint, including the final horizon.
``to_poly()`` instead samples the declared output at all interpolation nodes
and returns a new piecewise NumPy output interpolant on physical intervals,
with the original segment degrees. Nonlinear output maps and discontinuous
controls need not agree with direct output evaluation between nodes.
Neither conversion changes the normalised ``path``.

Treat result data as read-only. The result object is mutable and does not
promise defensive copies of stored arrays, mappings or evaluator inputs.
Changing ``parameters`` does not refit the trajectory or synchronise the
numeric parameter vector used by outputs and constraints.

``adaptive_refinement_rounds`` and ``adaptive_maximum_defect`` are ``None``
unless adaptive refinement supplies them. The latter is the final maximum
absolute sampled coupled residual, not an objective or a global error bound.
``segment_defects`` is a tuple, empty when not supplied. CasADi supplies
segment diagnostics even for a fixed mesh; these contain weighted signed
physical residual vectors rather than the maximum used for adaptation.

.. autoclass:: coker.dynamics.VariationalSolution
   :members: from_solver,path_constraints,terminal_constraints,as_raw,state,algebraic,quadratures,control_law,to_poly
   :special-members: __call__

.. autoclass:: coker.dynamics.variational.solution.SegmentDefectDiagnostic

   ``normalized_interval`` and ``physical_interval`` identify the segment,
   ``degree`` identifies its polynomial degree, and ``tolerance`` records the
   transcription's segment-defect tolerance. ``full_residual`` is a flat,
   floating NumPy array over the coupled equations. Field assignment is
   frozen, but its residual array is not made read-only.
