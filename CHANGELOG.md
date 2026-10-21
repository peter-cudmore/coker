# Changelog

## Unreleased

### Added
- Contribution and credit policies, with CI passing required before review and a named contributor list in `AUTHORS.md`.
- A model-focused README with executable spring-parameter fitting and symbolic matrix/PyTorch autograd examples, checked directly by documentation CI.
- Five-section documentation with introductory tutorials, a task-focused cookbook, architecture guides, API reference, and source-only author notes.
- Executable examples for installation, scalar and vector functions, symbolic loops and conditionals, backend selection and import/export, simulation, parameter fitting, path planning, robot-arm kinematics, solver configuration, and lowering, with rendered plots and CI-backed numerical checks.
- Installed-wheel CI profiles for isolated base NumPy/SymPy installations on Ubuntu, Windows, and macOS (Python 3.10/3.13), and individual CasADi/JAX/PyTorch extras on Ubuntu (Python 3.11/3.13), retaining the all-extra Ubuntu suite and verifying wheel provenance for copied or symlinked installs; classify CUDA support as experimental and unverified pending GPU release evidence ([#98](https://github.com/peter-cudmore/coker/issues/98)). Broader support-boundary alignment remains tracked in [#89](https://github.com/peter-cudmore/coker/issues/89).
- Distribution metadata links to the public documentation, source repository, issue tracker, and changelog.
- Public API reference for dynamics construction and analysis, optimisation builders, callable parameters, backend lowering, and solver options, separating ordinary usage, supported extensions, and internal helpers.
- Public controlled/general dynamics construction and composition projection types through `coker.dynamics`, component discovery through `coker.toolkits.system_modelling`, and `MathematicalProgram` through `coker.toolkits.codesign`.
- `DynamicsSpec` accepts `None` for absent inputs, constraints, and quadratures.
- Residual differential-algebraic systems through `ResidualDynamicalSystem` and `to_residual_dynamical_system`, with NumPy integration and explicit CasADi IDAS or variational solver options.
- `CasadiVariationalOptions` for CasADi solver policy and adaptive mesh refinement controls.
- Local defect estimation with p-refinement before h-refinement.
- Reuse of the previous path as the refined NLP initial guess.
- Function-valued variational parameters with monotone piecewise-linear, perceptron, and radial-basis realizations.
- `UnboundedVariable` declarations for scalar decisions without user-supplied bounds.
- SymPy symbolic lowering preserves named function parameters and external calls.
- Symbolic local accessibility, observability, and structural identifiability analysis for supported ODE and semi-explicit index-one DAE systems.
- Literal scalar and vector system parameters can be bound without solver decisions.
- CasADi transcriptions remove exact duplicate constraint rows and reject conflicting duplicates before IPOPT.
- Inline `uv` dependencies for the double-pendulum visualisation script, using the local Coker checkout and Matplotlib without adding a package extra.
- Configurable PyTorch variational trajectory degree and interval count through `PytorchVariationalSolverOptions`, defaulting to degree eight and four intervals.

### Changed
- Trim collocation implementation types from the `coker.dynamics` root API while retaining the public `DynamicsSpec` construction route.
- Consolidate onboarding into Basics and tutorials, remove migration advice, and add task-specific reading routes, simpler backend calls, JAX selection guidance, and precise recipe adaptation checks.
- Default implicit backend selection to the base-install NumPy plugin; backend lookup selects ambient state only with explicit `set_current=True`. Plugin names remain extensible.
- Standardize program shape metadata as `input_shape()` / `output_shape()` methods and dynamical-system backend metadata as the `backend` property.
- Keep function parameters at `coker.parameters.function_parameters`, signatures/lowering configuration at `coker.backends.lowered`, and solve status at `coker.toolkits.codesign`; remove obsolete dynamics reexports and system-modelling registry/handle helpers from public facades.
- Relocate dynamical-system analysis to `coker.dynamics` and reusable symbolic rank and DAE-constraint operations to the SymPy backend.
- Rename generic rank witnesses on analysis results to `rank_conditions`.
- Let identifiability analysis classify each flattened scalar parameter as an optimisation variable or a known numeric constant.
- Bump Coker to version 0.4.4.
- Collocation reference-operator caching is scoped to a transcription factory rather than process-global state.
- CasADi transcription compilation reuses bounded cached mesh-specific solvers during parameter sweeps.
- Existing `TranscriptionOptions` CasADi fields remain supported and are translated when no explicit backend policy is provided.
- Exclude CasADi 3.8.x from the optional dependency because the failure reproduces with a CasADi-only replay, indicating a regression in CasADi or its bundled solver stack.

### Fixed
- Missing optional CasADi, JAX, and PyTorch backend dependencies now report the requested backend and installation command, including JAX's Python requirement, while preserving the original exception cause and broken-plugin diagnostics.
- SymPy numerical division returns `NaN` for `ZeroDivisionError`, allowing conditionals to select a valid branch instead of failing on inactive division by zero.
- Preserve node-major value layout when evaluating multidimensional polynomial knot derivatives and coupled residual diagnostics.
- Stabilize PyTorch variational trajectories with low-degree piecewise LGR interpolation, preserving state, quadratures, and physical-time evaluation; restore the trajectory-fit regression.
- Normalize scalar outputs and column-shaped path samples when converting variational solutions with `to_poly()`.
- Evaluate variational solutions in physical time using only `t_final`, with normalized CasADi and PyTorch paths and preserved control/defect semantics.
- Evaluation, alternate lowering, and explicit problem builds no longer change subsequent implicit backend selection.
- Lowered mathematical programs expose only objective and requested outputs while still reconstructing private decision captures.
- Direct-sum composition correctly packs scalar subsystem outputs on NumPy.
- Preserve configured adaptive transcription interval counts.
- Preserve legacy CasADi `TranscriptionOptions` compatibility.
- Direct trajectory evaluation supports heterogeneous scalar, vector, and function-valued parameters.
- Use `magnitude` when constructing results of `se3_bracket` and `se3CoAdjoint.apply`, preserving scaled spatial operations.
- Consolidate CasADi exports, retaining `CasadiVariationalOptions`, and include `MathematicalProgram` in the codesign public inventory.
- Repair function-parameter and codesign documentation imports, complete example setup, and align heterogeneous decisions with the model's scalar/vector/function parameter order.
- Correct backend availability, robotics example paths and coverage claims, and the double-pendulum visualiser import; respect Matplotlib backend selection instead of requiring Qt.
