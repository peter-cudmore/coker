API Reference
=============

Look up public interfaces for function construction, modelling, solving and
backend integration. Each page states canonical imports, call/result contracts,
backend support and ownership limits. Importability alone does not make an
implementation helper a supported API.

.. Author notes

   - Audience: users locating canonical public imports and their owning
     reference contracts, separate from extension interfaces and internals.
   - Keep constructor, call, return-value, error and backend-support contracts
     next to their signatures, including function parameters and solve results.
   - Keep ordinary usage distinct from supported extension interfaces.
   - Prefer links to the cookbook over duplicate walkthroughs.

.. toctree::
   :maxdepth: 1

   functions
   spaces
   conditional
   optimisation
   dynamics
   parameters
   backends
   robotics
   modelling

Canonical imports
-----------------

Use these packages for construction and result inspection. Incidental imports
in implementation modules are not public APIs.

- ``coker``: ``function``/``Function``, argument spaces, shape metadata and
  symbolic tensor/sparse construction. See :doc:`functions` and :doc:`spaces`.
- ``coker.dynamics``: ``DynamicsSpec``/``create_dynamics_from_spec``,
  ``create_control_system``, ``create_autonomous_ode``, ``direct_sum`` and its
  ``ProjectionSet``, variational builders/options/results, controls,
  symbolic analyses, ``ResidualDynamicalSystem`` and
  ``to_residual_dynamical_system``. See :doc:`dynamics`.
- ``coker.parameters``: scalar, vector and tensor decision declarations;
  ``coker.parameters.function_parameters``: ``ClosureParameter``,
  ``BoundFunctionParameter``, ``DenseLayer``, ``MonotonePiecewiseLinear``,
  ``RadialBasisFunction`` and ``FittedFunction`` results. See :doc:`parameters`.
- ``coker.toolkits.codesign``: ``ProblemBuilder``, ``MathematicalProgram``,
  ``Minimise``, ``bounded``, ``norm``, ``weighted_norm``, ``SolverOptions``,
  ``SolveInfo`` and ``SolveFailure``. See :doc:`optimisation`.
- ``coker.backends``: plugin lookup and explicit selection;
  ``coker.backends.lowered``: signatures, input/output specs, lowering options,
  execution handles and capabilities. See :doc:`backends`.
- ``coker.toolkits.spatial`` and ``coker.toolkits.kinematics``: rigid-motion
  values, adjoints and articulated models. See :doc:`robotics`.
- ``coker.toolkits.system_modelling``: component metadata, port wiring and
  ``list_components`` discovery. See :doc:`modelling` for the boundary between
  working model construction and unreleased simulation functionality.

Extension interfaces
--------------------

.. toctree::
   :maxdepth: 1

   extensions

- ``coker.SymbolicCallable`` and ``coker.interfaces.FunctionSignatureValue``;
  ``coker.interfaces.SolverParameters`` for ODE options.
- ``coker.backends.register_backend`` and the backend entry-point group;
  ``coker.backends.backend.Backend``/``VariationalSolver`` and
  ``coker.backends.lowered.LoweredFunction`` for backend implementations.
- ``coker.toolkits.codesign.OptimisationModule`` for numerical solvers;
  ``WeightedNorm`` for backend recognition of ``weighted_norm`` objectives.
- ``coker.parameters.ParameterMixin`` and
  ``coker.parameters.function_parameters.FunctionParameter`` for declarations;
  dynamics control interfaces and ``VariationalIterationCallback`` for custom
  controls and callbacks; ``lower_system``/``lower_dae_system`` and their
  symbolic result types for advanced analysis.

``Tape``, ``Tracer``, ``TraceContext``, opcodes and ``Noop`` are compiler
internals. ``ConstraintSpec``, ``QuadratureSpec``, collocation operators and
mesh helpers are builder/backend representations, not ordinary construction
APIs. Use comparisons, integrals and the variational builder instead.
