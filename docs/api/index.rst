API Reference
=============

Look up public interfaces for function construction, modelling, solving, and
backend integration.

.. Author notes

   - List the supported public classes and functions that still need coverage.
   - Add constructor, call, return-value, error, and backend-support contracts.
   - Prioritise function parameters, solve results, analysis, and toolkit APIs.
   - Keep ordinary usage distinct from supported extension interfaces.
   - Request complete executable examples rather than source-code inventories.

.. toctree::
   :maxdepth: 1

   functions
   spaces
   conditional
   optimisation
   dynamics
   parameters
   backends

Canonical imports
-----------------

Use these packages for construction and result inspection. Incidental imports
in implementation modules are not public APIs.

- ``coker``: ``function``/``Function``, argument spaces, shape metadata, and
  symbolic tensor/sparse construction. See :doc:`functions` and :doc:`spaces`.
- ``coker.dynamics``: ``DynamicsSpec``/``create_dynamics_from_spec``,
  ``create_control_system``, ``create_autonomous_ode``, ``direct_sum`` and its
  ``ProjectionSet``, variational builders/options/results, controls, and
  symbolic analyses, ``ResidualDynamicalSystem``, and
  ``to_residual_dynamical_system``. See :doc:`dynamics` and :doc:`../dynamics`.
- ``coker.parameters``: scalar, vector, and tensor decision declarations;
  ``coker.parameters.function_parameters``: callable declarations and
  ``FittedFunction`` results. See :doc:`../dynamics`.
- ``coker.toolkits.codesign``: ``ProblemBuilder``, ``MathematicalProgram``,
  objectives/constraints, solver options, ``SolveInfo``, and ``SolveFailure``.
  See :doc:`optimisation` and :doc:`../toolkits`.
- ``coker.backends``: plugin lookup and explicit selection;
  ``coker.backends.lowered``: signatures, input/output specs, lowering options,
  execution handles, and capabilities. See :doc:`../backends`.
- ``coker.toolkits.spatial`` and ``coker.toolkits.kinematics``: rigid-motion
  primitives and articulated models. See :doc:`../toolkits`.

Extension interfaces
--------------------

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

``Tape``, ``Tracer``, ``TraceContext``, opcodes, and ``Noop`` are compiler
internals. ``ConstraintSpec``, ``QuadratureSpec``, collocation operators, and
mesh helpers are builder/backend representations, not ordinary construction
APIs. Use comparisons, integrals, and the variational builder instead.
Migration notes are in :doc:`../getting_started`.
