Public API and 0.5 migration
============================

This inventory defines the supported import boundary. Package facades below
are deliberate user APIs; importing an implementation module does not make
its other names public. Backend names remain extensible plugin strings, not
a closed enumeration. See :doc:`backends` for capabilities and optional extras.

Ordinary usage
--------------

- **Core: ``coker``.** Construct computations with ``function`` and inspect
  ``Function``. Declare ``Scalar``, ``VectorSpace``, and ``FunctionSpace``;
  inspect ``Dimension`` metadata. ``zeros``, ``if_then_else``,
  ``SymbolicVector``, ``SparseMatrixBuilder``, and ``SparseMatrixPattern``
  support symbolic construction.
- **Dynamics: ``coker.dynamics``.** Use ``create_autonomous_ode`` for autonomous
  ODEs, ``create_control_system`` for controlled ODEs, or ``DynamicsSpec`` with
  ``create_dynamics_from_spec`` for general dynamics. All return a
  ``DynamicalSystem``. Compose with ``direct_sum`` and inspect its returned
  ``ProjectionSet``, ``CompositionOperator``, or
  ``StructuredParameterProjection``. Build optimisation problems with
  ``VariationalProblemBuilder`` and ``TranscriptionOptions``; the explicit
  callback-based ``VariationalProblem`` construction route remains supported.
  ``VariationalSolution`` and the control solution types report solved paths.
  ``ConstantControlVariable``, ``PiecewiseConstantVariable``, and
  ``SpikeVariable`` declare control decisions. Symbolic analyses use
  ``analyse_controllability``, ``analyse_observability``, and
  ``analyse_identifiability`` with ``AnalysisStatus`` and their corresponding
  result types.
- **Numeric declarations: ``coker.parameters``.** Use ``BoundedVariable``,
  ``UnboundedVariable``, ``BoundVector``, ``DenseTensorVariable``, and
  ``Constant`` for parameters and decisions. ``ParameterVariable`` describes
  their union and ``ValueType`` their value category.
- **Callable declarations: ``coker.parameters.function_parameters``.** Use
  ``ClosureParameter``, ``BoundFunctionParameter``, ``DenseLayer``,
  ``MonotonePiecewiseLinear``, or ``RadialBasisFunction``. Solves reconstruct
  a ``FittedFunction``. There is no ``coker.function_parameters`` package.
- **Mathematical programs: ``coker.toolkits.codesign``.** Use ``ProblemBuilder``,
  ``Minimise``, ``bounded``, ``norm``, and ``weighted_norm``. Inspect
  ``MathematicalProgram``, ``BoundedConstraint``, ``SolverOptions``, ``SolveInfo``,
  and ``SolveFailure`` from this same package. These solve-status types also
  describe variational solves; they are not imported from ``coker.dynamics``.
- **Backend selection: ``coker.backends``.** Use ``get_backend_by_name`` and
  ``get_current_backend``. The lower-level ``evaluate(function, args,
  backend=None)`` helper uses the current selection when omitted; ordinary
  ``Function`` calls instead use their own backend.
- **Signatures and lowering: ``coker.backends.lowered``.** Use
  ``FunctionInputSpec``, ``FunctionOutputSpec``, ``FunctionSignature``, and
  ``LoweringOptions`` for native imports and lowering configuration. Inspect
  a ``LoweredFunction`` and its ``LoweringCapabilities`` there. Backend-specific
  option subclasses live in their backend package, e.g.
  ``coker.backends.casadi.CasadiNLPSolverOptions`` and
  ``coker.backends.pytorch.PytorchNLPSolverOptions``.
- **System modelling: ``coker.toolkits.system_modelling``.** Discover loaded
  components with ``list_components``. Author models with ``model``,
  ``CokerModel``, ``Block``, ``BlockContainer``, ``BlockSpec``, and ``PortBundle``;
  inspect factories with ``is_model`` and ``is_component_def``. Declare ports
  and values with ``Signal``, ``Variable``, ``VectorVariable``, ``Vector``,
  ``Angle``, ``Position3D``, ``Orientation3D``, ``Parameter``,
  ``ConstantParameter``, and ``ValueType``. Import standard-library blocks from
  ``coker.toolkits.system_modelling.std_lib`` to register them for discovery.
- **Robotics:** ``coker.toolkits.spatial`` exposes rigid-motion types,
  quaternion operations, bases, and Lie-algebra helpers;
  ``coker.toolkits.kinematics`` exposes ``RigidBody``, ``Inertia``, ``JointType``,
  and ``Free``, ``Planar``, ``Revolute``, and ``Weld`` joints. These existing
  domain facades are unchanged.

Supported advanced and extension contracts
------------------------------------------

- ``coker.SymbolicCallable`` is the symbolic call/lowering interface.
  ``coker.interfaces.FunctionSignatureValue`` declares callable shape methods;
  ``coker.interfaces.SolverParameters`` is the ODE options base.
- ``coker.backends.register_backend(name, factory)`` and the
  ``coker.backends`` entry-point group register arbitrary plugin names.
  ``coker.backends.backend.Backend`` and ``VariationalSolver`` define backend
  implementation interfaces. Backend implementations return
  ``coker.backends.lowered.LoweredFunction`` handles.
- ``coker.toolkits.codesign.OptimisationModule`` defines numerical solves and
  ``last_solve_info``. ``WeightedNorm`` is the structured objective metadata
  produced by ``weighted_norm`` for backend/QP recognition; ordinary models
  call the helper instead of constructing that metadata.
- ``coker.parameters.ParameterMixin`` and
  ``coker.parameters.function_parameters.FunctionParameter`` define parameter
  extensions. ``coker.dynamics.ControlVariable``, ``ControlLaw``,
  ``ControlSolution``, ``Solution``, ``LossFunction``, and
  ``VariationalIterationCallback`` support custom controls and callbacks.
  Concrete control solution classes remain available from ``coker.dynamics``.
- ``coker.dynamics.lower_system`` and ``lower_dae_system`` return
  ``SymbolicSystem`` or ``SymbolicDAESystem`` for advanced symbolic inspection;
  ``UnsupportedSystemError`` and ``AnalysisResult`` describe this boundary.

Internal representations
------------------------

``Tape``, ``Tracer``, ``TraceContext``, opcodes, and ``Noop`` are compiler
representation helpers, not ordinary construction APIs. ``ConstraintSpec``
and ``QuadratureSpec`` are normalized builder/backend records; express
constraints through comparisons and integrals through the variational builder.
Collocation polynomials, reference operators, and mesh-splitting helpers remain
in their implementation modules, not in the dynamics facade. A solution's
polynomial representation is not a separate stable construction API.

System-modelling lazy handles, block/port indexes, ``PortForwarder``, subclass
registries, and ``get_component_registry`` are connection/discovery machinery.
Use returned ports with model connection methods, and use ``list_components``
instead of importing registry internals.

Shared metadata and intentional domain differences
--------------------------------------------------

``Function.input_shape()`` and ``output_shape()`` return tuples in declaration
order; ``MathematicalProgram`` uses the same method access. A program's
``output_shape()`` describes only the requested outputs. Its ``result_shape``
property additionally prepends the scalar objective, matching its numeric
``(objective, *outputs)`` result. Private decision captures are never public
results, including when a program is lowered. ``ProblemBuilder`` retains
``input_shape`` and ``output_shape`` properties: it is a construction object,
not a callable.

``Function.backend``, ``MathematicalProgram.backend``, and
``DynamicalSystem.backend`` expose backend names without a method call. A
system's property is read-only and derives from its dynamics function. A
manually constructed program may leave ``backend=None``; its lowering then
uses NumPy. Builder-created programs always name their solver backend.

``Function.lower(options=None)`` returns a backend-specific ``LoweredFunction``.
``LoweringOptions(backend="sympy")`` can request an alternate lowering without
changing the function's evaluation backend or the current selection. Imported
native callables remain pinned to their owning backend.

``MathematicalProgram.lower()`` instead returns a traced ``Function`` pinned to
the program's solver backend. Calling it solves the program; call its own
``lower()`` to obtain an execution handle. Embedding the program in a graph
using a different backend is rejected. These domains intentionally do not
share a lowering wrapper.

Migration to 0.5
----------------

- Replace internal dynamics construction imports with ``from coker.dynamics
  import DynamicsSpec, create_dynamics_from_spec, create_control_system``.
  Use ``None`` for absent spec inputs, constraints, and quadratures; ordinary
  construction need not import ``Noop``.
- Unpack ``system, projections = direct_sum(...)``. There is no implicit
  projection attribute on the system. Projection fields and numeric/structured
  parameter behavior are described in :doc:`dynamics`.
- Replace program tuple attributes with ``program.input_shape()`` and
  ``program.output_shape()``. Replace ``system.backend()`` with
  ``system.backend``.
- Import function parameters from ``coker.parameters.function_parameters``;
  import programs and solve status from ``coker.toolkits.codesign``. Import the
  ODE configuration base from ``coker.interfaces``. Removed dynamics aliases
  and internal helper exports have no compatibility shims.
- Import ``list_components`` from ``coker.toolkits.system_modelling``. Load a
  component library before discovery; hints are currently ``None``.
- ``get_current_backend()`` now falls back to the base-install NumPy plugin,
  not the separately installed native ``coker`` plugin. ``function()`` continues
  to default to NumPy, even after explicit ambient selection.
- ``get_backend_by_name(name)`` no longer changes ambient state. If an
  application intentionally selects the backend consumed by subsequent
  implicit builders/evaluator calls, use
  ``get_backend_by_name(name, set_current=True)``. Prefer explicit ``backend=``
  for portable workflows. Evaluation, lowering, and explicit problem builds
  do not perform this ambient selection. Variational problems retain their
  domain-specific CasADi solver default and require the CasADi extra.
