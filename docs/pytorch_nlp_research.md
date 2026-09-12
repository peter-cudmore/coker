# PyTorch nonlinear-programming backend research

## Goal

Implement a PyTorch backend optimiser for Coker that solves constrained nonlinear mathematical programs and variational problems, including neural-ODE parameter fitting. The public behaviour must be compatible with the existing CasADi backend.

This document is a working design note on branch `pytorch-nlp`. It is intentionally uncommitted.

## Existing Coker contract

The backend method is:

```python
Backend.build_optimisation_problem(
    cost,
    constraints,
    parameters,
    outputs,
    initial_conditions,
)
```

The CasADi implementation:

- flattens all non-parameter tape inputs into one decision vector;
- accepts runtime parameters separately;
- converts each constraint to a lower/upper half-plane bound;
- supports both constant and symbolic constraint bounds;
- returns a callable accepting runtime parameters and returning the declared outputs;
- records `last_solve_info` and raises `SolveFailure` if the solve does not succeed.

The existing variational-solver contract requires:

- `parameters: list[str]`;
- `solve(**fixed_parameters) -> VariationalSolution`;
- bound-aware initial guesses;
- solver-status propagation through `SolveFailure`;
- assembly into the existing state, control, and interpolation solution structures.

The PyTorch backend already lowers Coker functions to eager PyTorch execution with autograd and integrates ODEs through `torchdiffeq`. It can therefore evaluate neural-ODE objectives and their derivatives. As documented by Coker, an optimisation solve is a numerical boundary: differentiation through an argmin/argmax is not required for CasADi API compatibility.

## Library survey

### Rejected: ipax

[`ipax`](https://github.com/wahln/ipax) is a PyTorch-capable primal-dual interior-point solver. Its surface covers nonlinear equality and inequality constraints, two-sided constraints, bounds, automatic differentiation, and dense/matrix-free/sparse linear solves.

It is not suitable as a Coker dependency:

- it is a beta project created in June 2026;
- it has a single owner and very little adoption;
- its public API is not yet stable;
- Coker would inherit a critical dependency for solver correctness and long-term maintenance from an unproven codebase.

The functionality is relevant as design reference only: primal-dual interior-point methods, filter line search, feasibility restoration, L-BFGS Lagrangian Hessian approximations, and explicit convergence diagnostics are all useful ingredients for a Coker-owned solver.

### Reference-only: cyipopt / IPOPT

[`cyipopt`](https://pypi.org/project/cyipopt/) is a mature Python wrapper for IPOPT. IPOPT solves the required general nonlinear-program form:

$$
\min_x f(x) \quad \text{subject to} \quad g_L \le g(x) \le g_U,\quad x_L \le x \le x_U.
$$

It is a strong CPU reference implementation and useful oracle for validation. It is not the desired primary backend because its callback interface is NumPy/CPU-oriented. A PyTorch model or neural ODE on CUDA would repeatedly transfer values and derivative data between device and host. Windows installation also depends on an available IPOPT build.

### Rejected as foundation: pytorch-minimize

[`pytorch-minimize`](https://github.com/rfeinman/pytorch-minimize) provides PyTorch-autograd derivatives for deterministic unconstrained methods. Its constrained `minimize_constr` API delegates to SciPy `trust-constr`; CUDA values move back and forth to CPU. It cannot provide a native PyTorch constrained-NLP path.

### Insufficient alone: torch.optim

[`torch.optim`](https://pytorch.org/docs/stable/optim.html) supplies unconstrained optimisers such as Adam, SGD, and L-BFGS. It has no general nonlinear equality/inequality constraint or bound interface.

It is nevertheless the right low-level execution engine for an initial Coker-owned solver. A barrier or augmented-Lagrangian outer loop can form a differentiable scalar merit objective, and then use `torch.optim.LBFGS` through its closure interface. This preserves PyTorch tensors, devices, and autograd through all objective and constraint evaluations.

### Rejected: PTNL

`ptnl` / `pytorch-nonlinear` advertises Torch-native SQP and interior-point methods, but is proprietary, alpha-stage, and has no meaningful observed adoption. It is not appropriate for a load-bearing dependency.

## Decision

Implement a Coker-owned solver. Do not add `ipax`, `cyipopt`, `pytorch-minimize`, or PTNL as a runtime dependency.

Start with a torch-native constrained optimiser based on a barrier method executed by `torch.optim.LBFGS`. Preserve an internal solver-adapter boundary so that future SQP or primal-dual interior-point implementations can replace the algorithm without changing the public Coker API.

## Initial algorithm

### NLP normal form

Reuse the CasADi backend's input flattening and constraint normalization. Express the problem as:

$$
\min_x f(x; p)
$$

subject to:

$$
h(x; p) = 0,
$$

$$
g(x; p) \le 0,
$$

$$
x_L \le x \le x_U.
$$

For every Coker half-plane constraint $l \le c(x; p) \le u$:

- emit $l - c(x; p) \le 0$ when a finite lower bound exists;
- emit $c(x; p) - u \le 0$ when a finite upper bound exists;
- substitute symbolic bounds into those expressions before evaluating them.

### Equality constraints

A pure log barrier cannot enforce equalities. The first implementation should use a quadratic penalty:

$$
P_\rho(x) = f(x) + \frac{\rho}{2}\lVert h(x) \rVert_2^2 - \mu \sum_i \log(-g_i(x)).
$$

This is simple and fully PyTorch-native, but it can become ill-conditioned as $\rho$ rises. It is suitable for a deliberately bounded initial feature set and test problems, not the final general NLP implementation.

A more robust next step is an augmented-Lagrangian equality term:

$$
L_{\rho}(x, \lambda) = f(x) + \lambda^T h(x) + \frac{\rho}{2}\lVert h(x) \rVert_2^2 - \mu \sum_i \log(-g_i(x)),
$$

with outer updates:

$$
\lambda \leftarrow \lambda + \rho h(x).
$$

The internal solver state should retain primal and multiplier values to make warm starts possible.

### Inequalities and variable bounds

For strict interior iterates, apply a log barrier to each finite inequality slack. Variable bounds are simply additional slacks:

$$
x_L - x < 0, \qquad x - x_U < 0.
$$

The solver must construct a strictly feasible starting point before evaluating any logarithm. If the caller's initial point is on or outside a bound, project it to the bound interior using a configurable positive margin. General nonlinear inequalities cannot be safely projected this way. The initial release should fail explicitly when it cannot find a strictly feasible point rather than silently clipping or accepting NaNs.

### Outer-loop schedule

For each barrier value $\mu$:

1. Minimise the current merit objective using `torch.optim.LBFGS` and a closure that recomputes objective, constraints, and gradients.
2. Assess primal feasibility, stationarity proxy, objective change, and finite values.
3. Update equality multipliers and penalty coefficient when equality residuals do not decrease sufficiently.
4. Decrease $\mu$ only after the inner solve reaches its tolerance.
5. Stop only when the original objective is finite and feasibility and stationarity satisfy explicit tolerances.

Return a structured solve-info record and raise `SolveFailure` for infeasibility, non-finite evaluation, inner-iteration exhaustion, or failure to make progress.

## API design

Keep library-specific types private. The public PyTorch backend should mirror the CasADi callable contract:

```python
solver = pytorch_backend.build_optimisation_problem(
    cost,
    constraints,
    parameters,
    outputs,
    initial_conditions,
)
outputs = solver(*runtime_parameters)
```

The implementation should:

- lower cost, constraints, and outputs once to PyTorch executable handles;
- retain a stable ordering for flattened decisions and runtime parameters;
- use the decision tensor as the only `torch.optim` parameter;
- evaluate all Coker graph values on the decision tensor's device and dtype;
- return the output representation expected by existing mathematical-program callers;
- retain `last_solve_info` with Coker-owned status values.

Do not expose the raw `torch.optim.Optimizer`. Solver options should be Coker dataclasses with stable names, defaults, and validation.

## Variational problems and neural ODE fitting

The initial variational backend should reuse Coker's existing transcription and solution assembly rather than implement a separate neural-ODE-specific API.

For neural-ODE fitting:

- model parameters and any free initial conditions are decision variables;
- `torchdiffeq` evaluates the state trajectory under PyTorch autograd;
- data mismatch, regularisation, and quadrature terms form the objective;
- physical/state/control path constraints use the same nonlinear-constraint machinery;
- solver calls remain numerical boundaries, matching the CasADi backend contract.

Start with unconstrained or bound-only neural-ODE fitting because a barrier method requires a strictly feasible path. Add equality and general path constraints after the generic NLP solver passes parity tests.

## Implementation sequence

1. Add a private PyTorch NLP normalizer that exactly matches the CasADi decision/parameter flattening and half-plane-bound interpretation.
2. Add a `PytorchNlpOptions` dataclass and Coker-owned solve status/result types.
3. Implement and test an unconstrained `torch.optim.LBFGS` solve path. This establishes result mapping, shape validation, output evaluation, and diagnostics.
4. Add finite variable bounds using strict-interior initialization and log barriers.
5. Add nonlinear inequality barriers and clear infeasible-start diagnostics.
6. Add equality constraints through augmented Lagrangian updates.
7. Implement `PytorchBackend.create_variational_solver` by reusing the existing variational transcription and solution assembly contracts.
8. Add warm starts only after the cold-start path has stable convergence and status semantics.
9. Evaluate whether a Coker-owned SQP or primal-dual interior-point implementation is required for robustness, sparse problems, or difficult equality-constrained models.

## Acceptance and validation plan

Compare every supported case against the CasADi/IPOPT backend on the same Coker expression graph:

- unconstrained nonlinear minimisation;
- variable bounds, including boundary initial guesses;
- equality constraints;
- nonlinear inequalities;
- two-sided and parameter-dependent bounds;
- scalar, vector, and matrix decision values;
- runtime-parameter validation and output shape preservation;
- infeasible initial-point and non-finite-objective failures;
- warm-start behaviour once implemented;
- neural-ODE parameter fitting using the existing PyTorch `torchdiffeq` path;
- variational solution structure and solver status.

Measure CPU and CUDA separately. CUDA verification must ensure objective, constraints, decision state, gradients, and output evaluation stay on-device; a host copy is a failure for the native PyTorch path.

## Approved delivery assumptions

- Implement the staged, Coker-owned barrier/augmented-Lagrangian approach in
  this plan. Do not add an off-the-shelf NLP solver dependency.
- CUDA is required for the new NLP and variational-solver execution paths.
  The existing PyTorch evaluator remains device-generic; only an attempt to
  construct or execute the new solver on a non-CUDA device must fail clearly.
- `float32` is the default solver dtype. Solver tolerances, interior margins,
  iteration limits, and schedule values are configuration, not global
  acceptance constants. Callers may select `float64` with correspondingly
  tighter tolerances.
- Initial low-cost defaults are: `inner_max_iter=25`,
  `restoration_max_iter=25`, `barrier_max_stages=8`,
  `augmented_lagrangian_max_stages=10`, `lbfgs_history_size=10`, a
  `0.2` barrier reduction factor, and a `10.0` equality-penalty growth factor.
  The `float32` defaults are `1e-4` for feasibility, gradient, and
  equality tolerances; `1e-5` for relative objective change; and `1e-4` for
  the strict-interior margin. These are provisional defaults to obtain a
  working implementation, not performance or robustness commitments.
- Take the easiest useful variational path first: fixed-horizon neural-ODE
  parameter fitting and bound-only decisions through direct shooting. Add
  controls and general constraints incrementally after that path works.
- Sample path constraints on an explicit, simple transcription grid for the
  initial implementation. This is a discrete constraint approximation and will
  be refined after working end-to-end solves exist.

## Detailed implementation plan

### Phase 1: solver boundary and unconstrained contract

**Files**

- Add `src/coker/backends/pytorch/optimisation.py`.
- Update `src/coker/backends/pytorch/__init__.py`.
- Replace the unsupported-construction test in
  `tests/backends/pytorch/test_backend.py`.
- Add PyTorch-specific mathematical-program tests under
  `tests/backends/pytorch/`.

**Work**

1. Define private `PytorchOptimisationProblem`, `PytorchConstraint`, and
   `PytorchNlpOptions` types in `optimisation.py`. Keep Coker's public
   `SolveInfo`, `SolveFailure`, and callable result contract; do not publish a
   `torch.optim` object or library-specific result type.
2. Make `PytorchNlpOptions` own the CUDA device and numerical dtype, defaulting
   to the current CUDA device and `torch.float32`. Validate that decision
   values and every runtime tensor can be represented on that device and dtype;
   normalize numeric initial conditions and runtime arguments once per solve.
   Reject CPU execution and mixed-device tensors with a clear configuration
   error.
3. Use `build_problem_bindings`, `build_initial_guess`,
   `normalise_runtime_args`, and `materialise_tape_inputs` from
   `coker.backends.optimisation`. This makes PyTorch's decision ordering,
   parameter ordering, shape validation, and initial-condition semantics match
   the existing NumPy implementation.
4. Evaluate cost, declared outputs, and constraint residuals via
   `evaluate_inner` and the PyTorch backend. The flattened decision vector must
   be the sole leaf tensor with `requires_grad=True`.
5. Implement an unconstrained `torch.optim.LBFGS` path. Its closure must clear
   gradients, recompute the scalar cost, reject non-finite values, invoke
   `backward()`, and return the un-detached scalar tensor.
6. Map configured stopping limits, non-finite values, and LBFGS exhaustion into
   Coker `SolveInfo` and `SolveFailure`. LBFGS does not expose a reliable
   KKT-style convergence status, so the initial status must be based on
   configured gradient infinity-norm and relative-objective-change checks after
   the final closure evaluation.
7. Return the cost followed by requested outputs as detached CPU arrays or
   scalars, matching `MathematicalProgram._call_numeric`; keep all work before
   that final boundary on CUDA.

**Acceptance**

- Existing zero-input and runtime-parameter mathematical-program cases pass
  under `backend="pytorch"` when supplied CUDA inputs.
- Cost, decision output shapes, `program.solve_info`, and `SolveFailure`
  semantics match the existing NumPy/CasADi tests within the configured
  `float32` tolerances.
- CUDA is mandatory for solver tests: the closure's decision, cost, and
  gradient remain CUDA `float32` tensors until final result conversion.
- A CPU solver invocation fails with an explicit CUDA-required error.

### Phase 2: finite decision bounds and interior starts

**Files**

- Extend `src/coker/backends/pytorch/optimisation.py`.
- Add bounded-program regression cases to
  `tests/backends/pytorch/test_optimisation.py`.

**Work**

1. Extend constraint normalization to preserve the residual and both bounds
   rather than immediately collapsing them. Evaluate symbolic bounds on every
   solve with the current runtime parameters.
2. Detect constraints affine in the decision vector with
   `is_affine_in_decisions`. Initially use only direct variable-bound
   constraints for strict-interior projection; do not infer variable bounds
   from arbitrary affine expressions.
3. Add a configurable positive interior margin. For finite lower/upper
   decision bounds, project a supplied initial guess to
   `[lower + margin, upper - margin]`; reject empty intervals after margin
   application.
4. Add the bound slacks to the log barrier and reduce the barrier coefficient
   only after the current inner LBFGS solve meets its termination criteria.
5. Treat a minimizer that approaches a boundary as a valid limiting solution:
   return the unmodified decision, not an artificially interior one, only
   after the final barrier stage. Report feasibility against the original
   bounds.

**Acceptance**

- Boundary optima converge from interior and boundary guesses.
- Invalid bound intervals, non-finite slacks, and failed strict-interior
  initialization raise `SolveFailure` with actionable status text.
- No unconstrained case changes behaviour.

### Phase 3: nonlinear inequalities, feasibility restoration, and diagnostics

**Files**

- Extend `src/coker/backends/pytorch/optimisation.py`.
- Add nonlinear and parameter-dependent bound tests to
  `tests/backends/pytorch/test_optimisation.py`.

**Work**

1. Convert each finite lower bound to `lower - residual < 0` and each finite
   upper bound to `residual - upper < 0`. Constant and symbolic bounds use the
   same conversion.
2. Implement a separate feasibility-restoration objective for an initial point
   outside nonlinear inequalities: minimise the squared positive constraint
   violation with LBFGS, while respecting known direct bounds. Do not evaluate
   the log barrier until every slack is strictly negative.
3. Bound restoration iterations and require a concrete maximum violation before
   declaring a strictly feasible point. An infeasible or stalled restoration is
   a solve failure, never a clipped solution.
4. Add barrier outer-loop state: barrier coefficient, objective/feasibility
   history, inner iterations, and termination reason. Final acceptance requires
   finite values, primal feasibility, and the configured gradient/stationarity
   proxy.
5. Add a structured Coker-owned solve-info adapter only if the existing
   `SolveInfo` cannot represent the required status fields. Do not fork status
   semantics by returning PyTorch-specific exceptions.

**Acceptance**

- Two-sided, nonlinear, and parameter-dependent constraints match CasADi on
  the existing codesign cases.
- Contradictory inequalities fail deterministically and retain the failure in
  `MathematicalProgram.solve_info`.
- The solver never evaluates `log` on a non-positive slack.

### Phase 4: equality constraints with an augmented Lagrangian

**Files**

- Extend `src/coker/backends/pytorch/optimisation.py`.
- Add equality-constrained and mixed-constraint regressions to
  `tests/backends/pytorch/test_optimisation.py`.

**Work**

1. Classify equal lower/upper bounds as equality residuals, subject to a
   configurable equality tolerance. Retain unequal two-sided bounds as two
   inequalities.
2. Add multiplier and penalty state to the outer solve:
   `lambda`, penalty coefficient, barrier coefficient, best feasible primal,
   and residual history.
3. Minimise the augmented-Lagrangian-plus-barrier merit function with LBFGS;
   after each successful inner solve update multipliers and increase the
   penalty only when equality residual reduction stalls.
4. Use scaled primal feasibility, equality residual, gradient/Lagrangian
   stationarity proxy, barrier complementarity proxy, and relative objective
   change as stopping conditions. Record every component in the solve result.
5. Preserve the best feasible iterate separately from the final iterate. A
   failed final barrier stage must not return an earlier iterate as a successful
   solve.

**Acceptance**

- Scalar/vector equality constraints and mixed equality/inequality problems
  match CasADi solutions within explicit problem-scaled tolerances.
- Equality-infeasible problems raise `SolveFailure`; near-feasible points are
  not silently classified as success.
- Repeated calls with different runtime parameters do not leak multipliers or
  primal state unless warm starts are explicitly enabled.

### Phase 5: warm starts, options, and public documentation

**Files**

- Extend `src/coker/backends/pytorch/optimisation.py` and
  `src/coker/backends/pytorch/__init__.py`.
- Update `src/coker/toolkits/codesign/__init__.py` only if backend-neutral
  options need an extension point.
- Update `docs/backends.rst` and the appropriate API documentation.

**Work**

1. Add validated options for outer/inner iteration limits, barrier schedule,
   equality penalty schedule, feasibility tolerances, LBFGS settings, and
   warm-start policy.
2. On warm starts, reuse only a prior primal and equality multipliers after
   checking decision dimension, dtype/device compatibility, and current
   runtime-bound feasibility. Re-run restoration when a prior point is no
   longer interior.
3. Document supported NLP classes, strict-interior requirement, expected
   local-solver behaviour, non-differentiability through the returned solve,
   failure semantics, and CPU/CUDA behaviour.
4. Remove the current documentation claim that PyTorch cannot construct or
   solve mathematical programs only after Phases 1–4 pass.

**Acceptance**

- Default behaviour is cold-start and deterministic for a fixed Torch device,
  dtype, and seed.
- Warm starts improve a repeated parameterized solve without changing its
  final feasibility/result contract.
- Documentation describes only shipped capabilities.

### Phase 6: variational solver implementation

**Files**

- Add `src/coker/backends/pytorch/variational/__init__.py`.
- Add `src/coker/backends/pytorch/variational/solver.py`.
- Update `src/coker/backends/pytorch/__init__.py`.
- Add PyTorch variational tests under `tests/backends/pytorch/` and enable
  PyTorch only for the tests that its implementation supports.

**Work**

1. Do not copy CasADi's symbolic direct-collocation implementation. It is
   built around `ca.MX`, CasADi callbacks, and IPOPT-specific multiplier
   inputs. Extract only backend-neutral declaration, validation, and
   `VariationalSolution` assembly helpers where they have no CasADi types.
2. Implement the smallest useful PyTorch direct-shooting transcription first:
   fixed horizon, free parameter declarations, bound-only decisions, and a
   scalar neural-ODE fitting loss. Integrate the state and quadratures through
   the existing PyTorch `torchdiffeq` path and evaluate the loss as a CUDA
   tensor.
3. Add controls, optimized horizon, terminal/initial constraints, and sampled
   path constraints in that order. Flatten each added free declaration into the
   NLP decision vector. Each added capability requires an end-to-end
   variational regression before it is advertised.
4. Use a simple explicit sampling grid in a PyTorch-specific
   transcription-options extension for the first path-constraint
   implementation. A continuous path constraint cannot be guaranteed by a
   finite sample grid; document this numerical limitation and test the chosen
   grid.
5. Implement `PytorchVariationalSolver(VariationalSolver)` with the same
   `parameters` property and `solve(**fixed_parameters)` validation as
   `CasadiVariationalSolver`. Reuse the NLP solver's status and map solved
   decisions through the shared solution assembler.
6. Add warm starts after the direct-shooting cold path is stable. Retain only
   primal/multiplier state through the NLP solver; never retain a Torch autograd
   graph between calls.

**Acceptance**

- A CUDA `float32` PyTorch `torchdiffeq` neural-ODE parameter-fitting test
  recovers known parameters within configured tolerances and verifies gradients
  reach the decision vector.
- Fixed-horizon bound-only fitting returns a valid `VariationalSolution`.
- Terminal, initial, and sampled path constraints each have feasible and
  infeasible end-to-end cases once their incremental implementation is added.
- Returned solution evaluation, fixed-parameter validation, and failure
  propagation match the current `VariationalSolver` contract.

### Phase 7: robustness gate and next solver decision

Run representative nonlinear-program and variational cases on CUDA `float32`;
track objective error, maximum constraint violation, stationarity proxy, wall
time, and peak device memory. Use CasADi/IPOPT CPU `float64` only as an
off-device reference oracle, with comparisons judged against the configured
PyTorch tolerances rather than exact-value parity.

Do not broaden the public feature matrix until the barrier/augmented-Lagrangian
solver meets the defined feasibility and convergence thresholds. If it fails on
ill-conditioned, active-set-changing, or equality-dominated cases, implement a
Coker-owned SQP or primal-dual interior-point solver behind the same private
adapter boundary rather than adding an unvetted external dependency.

## Delivery checklist

### Core NLP

- [ ] Add private solver boundary: `PytorchOptimisationProblem`,
  `PytorchConstraint`, and `PytorchNlpOptions`.
- [ ] Implement CUDA tensor normalization: default CUDA device and `float32`,
  with clear rejection of CPU and mixed-device solver inputs.
- [ ] Implement unconstrained LBFGS solve: closure lifecycle, finite-value
  checks, configured stopping checks, and final result evaluation.
- [ ] Map solver outcomes to Coker: `SolveInfo`, `SolveFailure`, result
  shapes, and `MathematicalProgram.solve_info`.
- [ ] Add unconstrained CUDA regressions: zero-input, runtime-parameter,
  non-finite, CPU-rejection, dtype, device, and output-shape cases.

**Gate:** CUDA `float32` unconstrained mathematical programs satisfy configured
objective/gradient tolerances and preserve the existing public callable
contract.

### Constraints

- [ ] Implement finite decision barriers: strict-interior margin, direct-bound
  projection, and final original-bound feasibility checks.
- [ ] Implement nonlinear feasibility restoration: bounded LBFGS minimization
  of squared positive constraint violation before barrier evaluation.
- [ ] Implement inequality barrier stages: lower/upper bound normalization,
  symbolic-bound evaluation, barrier schedule, and diagnostic history.
- [ ] Implement equality multiplier updates: augmented-Lagrangian state,
  penalty schedule, stationarity proxy, and best-feasible iterate handling.
- [ ] Add constrained CUDA regressions: boundary optima, two-sided bounds,
  symbolic bounds, nonlinear inequalities, equalities, mixed constraints, and
  infeasible failures.

**Gate:** CUDA `float32` constrained solves meet configured feasibility and
stationarity tolerances; failed restoration and exhausted schedules report
`SolveFailure`, never an apparently valid result.

### Variational

- [x] Implement direct-shooting parameter fitting: fixed horizon,
  `BoundedVariable` parameters, CUDA `torchdiffeq` evaluation, and LBFGS.
- [x] Add bound-only variational solves: fixed-parameter validation, solve-info
  propagation, parameter guesses, and returned-solution evaluation.
- [ ] Add controls and horizon decisions: flatten declarations, reconstruct
  controls, then support optimized horizon.
- [ ] Add sampled variational constraints: terminal, initial, and explicit-grid
  path constraints with feasible/infeasible regressions.

**Gate:** a CUDA `float32` neural-ODE fitting problem recovers known parameters
within configured tolerance and returns a valid `VariationalSolution`.

### Hardening

- [x] Implement validated solver options: `SolverOptions` owns
  backend-independent NLP settings and `PytorchNLPSolverOptions` owns CUDA
  float32 LBFGS/barrier configuration. `PytorchODESolverParameters` configures
  ODE initial-value solves; variational fitting has no separate options object.
- [x] Implement warm-start state reuse: validated primal/multiplier reuse
  without retaining autograd graphs or bypassing strict interior restoration.
- [x] Update shipped capability documentation: PyTorch NLP and the supported
  fixed-horizon variational subset are CUDA-only.
- [x] Run CUDA benchmark corpus:
  ``scripts/benchmark_pytorch_nlp.py --run`` executes the NLP and neural-ODE
  fitting corpus on CUDA.
- [ ] Evaluate next solver architecture: retain the private adapter. Whether
  the barrier/augmented-Lagrangian implementation is adequate, or needs
  replacement with Coker-owned SQP/primal-dual interior-point code, remains
  pending empirical CUDA benchmark data.

**Gate:** feature documentation reflects verified behavior; the benchmark
harness must identify whether the barrier/augmented-Lagrangian implementation
is adequate or needs replacement behind the existing private boundary.

## Sources

- [PyTorch optimisers](https://pytorch.org/docs/stable/optim.html)
- [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize)
- [cyipopt](https://pypi.org/project/cyipopt/)
- [IPOPT](https://github.com/coin-or/Ipopt)
- [ipax](https://github.com/wahln/ipax) — rejected dependency; design reference only
