# VariationalProblemBuilder test-driven development plan

## Purpose

Introduce `VariationalProblemBuilder`: a context-managed, symbolic API for
constructing `VariationalProblem` instances from a `DynamicalSystem`.  The
builder must cover constrained trajectory optimisation, path planning, and
data fitting without users manually constructing loss functions, terminal
constraint functions, or quadrature states.

The primary reference is SysOpt's `SolverContext` and `Problem` design:

- [`SolverContext.integral`](https://github.com/csp-at-unimelb/sysopt/blob/main/sysopt/problems/solver.py)
  registers a symbolic quadrature in the dynamic model.
- [`Problem._get_minimisation_specification`](https://github.com/csp-at-unimelb/sysopt/blob/main/sysopt/problems/solver.py)
  separates terminal cost, quadratures, decision bounds, path constraints, and
  point constraints.
- SysOpt's
  [`test_codesign_problem.py`](https://github.com/csp-at-unimelb/sysopt/blob/main/tests/2_solver_interface/test_codesign_problem.py)
  specifies free-horizon decisions, terminal constraints, and control bounds.
- SysOpt's
  [`test_solver.py`](https://github.com/csp-at-unimelb/sysopt/blob/main/tests/2_solver_interface/test_solver.py)
  specifies the quadrature contract: `integral(y ** 2)` adds a state with
  `q(0) = 0` and `q_dot = y ** 2`.

This plan deliberately ports *behavioural contracts*, not SysOpt classes or
its symbolic backend.

## Target public API

```python
with VariationalProblemBuilder(
    system,
    t_final=BoundedVariable("T", lower_bound=0.1, upper_bound=10, guess=1),
    control=[PiecewiseConstantVariable("u", sample_rate=20)],
    parameters=[BoundedVariable("mass", lower_bound=1, upper_bound=10, guess=2)],
) as problem:
    x = problem.state(problem.t)
    u = problem.input(problem.t)
    y = problem.output(problem.t)
    p = problem.parameters()

    x_final = problem.state(problem.t_final)
    running_cost = x @ x + u @ u
    cost = problem.integrate(running_cost) + x_final @ x_final

    compiled = problem.build(
        Minimise(cost),
        subject_to=[
            u <= 1,
            u >= -1,
            x[0] <= 5,
            problem.state(0) == initial_state,
            x_final == goal_state,
            p[0] >= 0,
        ],
    )
```

### API invariants

1. `control=` and `parameters=` are keyword collections of existing Coker
   `ControlVariable`, `BoundedVariable`, and numeric-constant declarations.
2. `t_final` accepts either a positive fixed float or a `BoundedVariable`.
   The latter is a free-horizon decision, as in SysOpt's `Variable('t_f')`.
3. The context owns one trace/tape.  Tracers returned by the builder are valid
   only inside that context and must not silently mix with another builder's
   trace.
4. `t` is the path-time marker.  `t_final` is the terminal-time marker and,
   when declared with `BoundedVariable`, the free-horizon decision.
5. `state(time)` returns differential state `x`; it does not expose algebraic
   or quadrature state.  `input(time)` returns instantaneous input `u`.
   `output(time)` returns system output `y`; `parameters()` returns `p`.
6. `integrate(expr)` accepts scalar symbolic expressions and denotes
   $\int_0^T expr(t)\,dt$.  It is valid for time-dependent and constant
   integrands, and may be nested in any scalar cost or constraint expression.
7. `build(Minimise(cost), subject_to=...)` returns an executable
   `VariationalProblem`, not an intermediate public problem type.
8. Equality and both inequality directions are valid constraints.

## Constraint classification

Classification is inferred from bound time structure, never wrappers such as
`path(...)` or `terminal(...)`.

| Expression time binding | Lowered form |
| --- | --- |
| `t` | path constraint |
| `0` | initial point constraint |
| `t_final` | terminal point constraint |
| no time binding (parameter-only) | terminal point constraint |
| a mix of `t` and endpoint values | path constraint; endpoint values are broadcast |

`state(0)` validates the parameter-dependent initial condition supplied by the
system; it does not introduce a free initial-state decision.  Initial and
terminal input values are valid endpoint expressions.  Times other than `0`,
`t`, and `t_final` are rejected in the initial implementation.

## Test strategy

Every phase begins by adding the named tests.  A phase may not add a feature
until its tests fail for the intended reason.  Tests should use Coker's public
API and assert observable symbolic/lowering/solve behaviour, never internal
node indices except where the test verifies a deliberately public archive or
solver specification.

Use a small controlled scalar integrator fixture throughout:

```python
# x_dot = u, x(0) = p[0], y = x
```

This fixture supports exact terminal values, input bounds, parameter-dependent
initial conditions, data-fitting losses, and a free final time.  Add a second
analytic decay fixture for quadrature accuracy:

```python
# x_dot = -a * x, x(0) = x0, y = x
```

The analytic integral is used as SysOpt uses it: compare a registered
quadrature for `y ** 2` with the closed-form integral.

## Phase 1 — establish failing API contracts

Create `tests/dynamical_systems/test_variational_problem_builder.py` before
implementation changes.

1. **Context exposes a coherent trace.**
   - Inside one builder, `t`, `t_final`, `state(t)`, `input(t)`, `output(t)`,
     and `parameters()` create composable symbolic expressions.
   - A tracer from a different builder is rejected at `build` with an error
     that identifies the foreign trace.
   - Access after context exit is rejected if it would create a new symbol.

2. **Construction declarations are retained.**
   - Fixed `t_final`, controls, bounded parameters, and constants are preserved
     in the built `VariationalProblem`.
   - `BoundedVariable('T', ...)` creates a horizon decision rather than a
     normal system parameter.

3. **`build` is functional.**
   - The old imperative `minimise`/`add_*` API is deprecated but remains
     available during the transition.
   - Missing `Minimise`, a non-scalar cost, non-comparison constraint, and
     foreign tracer each fail deterministically.

4. **Symbolic comparison coverage.**
   - `==`, `<=`, and `>=` survive lowering with their expected residual and
     bounds.
   - Vector comparisons either lower componentwise or fail with an explicit
     shape error; select one policy and lock it down before backend work.

These tests replace the current builder tests, which exercise the obsolete
imperative API.

## Phase 2 — temporal binding and expression lowering

Add focused lowering tests before modifying `VariationalProblem`.

1. `state(builder.t) <= limit` appears in the path-constraint collection.
2. `state(builder.t_final) == goal` appears in terminal point constraints.
3. `state(0) == initial` appears in initial point constraints.
4. `parameters()[0] >= 0` appears in terminal point constraints.
5. `state(builder.t) <= state(builder.t_final)` is path scoped and retains the
   terminal expression as a path-wide value.
6. `input(0)` and `input(t_final)` are accepted in point constraints.
7. `state(0.5)` and expressions that mix unsupported concrete times fail with
   a message naming the allowed bindings.
8. `output(builder.t)` can drive a path constraint and
   `output(builder.t_final)` a terminal/data-fitting objective.

Implement a private, immutable lowering record rather than encoding scope in
ad-hoc Python tuples.  It must record comparison operation, residual,
lower/upper bounds, trace identity, and temporal binding.  Extend
`VariationalProblem` with explicit initial point constraints; do not overload
its terminal list or silently convert initial constraints to path constraints.

## Phase 3 — quadrature model and `integrate`

Treat `integrate` as a dynamic-model operation, following SysOpt exactly.  It
must not numerically sample an expression while the builder is open and must
not synthesize a terminal-only expression.

### Tests first

1. **Registration:** `q = builder.integrate(output(builder.t) ** 2)` returns a
   scalar symbolic expression and appends one quadrature channel to the
   builder's model record.
2. **Dynamic contract:** the lowered model has `q(0) = 0` and
   `q_dot = output(t) ** 2`.
3. **Analytic value:** with the decay fixture, evaluate `q(t_final)` and compare
   it against the closed-form integral.  This replicates SysOpt
   `TestSolverUnconstrained.test_quadrature` and
   `TestSolverCompositeModel.test_quadrature`.
4. **Constant integrand:** `integrate(parameters()[0] ** 2)` returns
   `T * p[0] ** 2` numerically for a fixed horizon and tracks free `T` when
   the horizon is a decision.
5. **Cost composition:** terminal cost plus several integral terms has the
   expected scalar cost and independent quadrature channels.
6. **Constraint composition:** `integrate(input(t) ** 2) <= energy_budget`
   becomes an endpoint constraint on its quadrature state.
7. **Nested arithmetic:** `2 * integrate(x[0] ** 2) + integrate(u ** 2)`
   works; vector-valued integrands fail before lowering.
8. **Deduplication policy:** either two calls with the same expression create
   distinct channels or they intern one channel.  Choose and test one explicit
   policy; SysOpt appends channels, so distinct calls is the conservative
   choice.

### Implementation

Add a builder-owned quadrature registry containing the scalar integrand,
trace identity, and assigned channel.  At `build`, compose registry entries
with any system-defined `dqdt` into one augmented quadrature derivative
function.  Preserve system quadratures; builder quadratures append after them.
The builder's `integrate(expr)` lowers to the associated `q_i(t_final)`
expression.  This is the Coker equivalent of SysOpt's `add_quadrature` and
`Quadrature` mechanism.

## Phase 4 — free horizon and decision mapping

Port the behavioural coverage from SysOpt
`test_codesign_problem_with_path_variable`.

1. A fixed numeric horizon is not a decision variable.
2. `BoundedVariable('T', ...)` is exposed as the first-class horizon decision,
   has its declared lower bound, upper bound, and guess, and is not duplicated
   in the ordinary system-parameter vector.
3. A controlled free-horizon problem retains both `T` and a
   `PiecewiseConstantVariable` in the decision specification.
4. The generated system-parameter map continues to map only declared system
   parameters; adding `T` must not shift parameter columns.
5. A terminal constraint and a path input bound remain distinguishable with a
   free horizon.
6. The solver evaluates a free-horizon solution using its solved `T`, including
   quadratures scaled over `[0, T]`.

Implementation must make the final-time map explicit in `VariationalProblem`.
Do not overload `system_parameter_map` with horizon data.  The solver must
transcribe on a normalized collocation domain or otherwise correctly account
for `dt/dtau = T`; a merely symbolic `T` in the objective is insufficient.

## Phase 5 — repair and extend the CasADi backend

The builder is complete only when CasADi solves the lowered problem.  Current
CasADi lowering consumes terminal constraints but does not add
`path_constraints`; it also has no initial-point collection, builder
quadrature registry, or free-horizon transcription.

Add tests to `tests/dynamical_systems/test_variational_solver.py` first.

1. **Path bounds are enforced.**
   Solve `x_dot = u` with `u <= 1`, `u >= -1` and a terminal target.  Sample
   the returned control/trajectory at collocation points and prove no bound is
   violated beyond solver tolerance.  This is the Coker analogue of SysOpt's
   path-variable bound test.
2. **Path state constraint is enforced.**
   A trajectory that would otherwise exceed `x <= limit` is constrained at
   every collocation knot.
3. **Terminal equality is enforced.**
   `state(t_final) == goal` yields a solution within transcription tolerance.
4. **Initial point constraint validates `x0(p)`.**
   A feasible parameter-dependent initial constraint solves; an incompatible
   one raises `SolveFailure`.
5. **Parameter-only terminal constraint is enforced.**
6. **Quadratic running cost.**
   The optimizer's objective agrees with the analytically known integral on a
   fixed test trajectory.
7. **Integral constraint.**
   An energy bound expressed with `integrate(u ** 2)` changes the solution and
   is satisfied at the terminal quadrature state.
8. **Free horizon.**
   A minimum-time bounded-control transfer solves with `T` in its declared
   range and the expected optimal time.
9. **Data fitting.**
   Fit a parameter from samples represented in a scalar symbolic objective
   using `output(t_final)` and/or registered integral residuals; recover the
   known parameter.
10. **Infeasibility.**
    Contradictory path, point, and integral constraints each produce
    `SolveFailure` with solver status preserved.

### CasADi implementation order

1. Lower initial, terminal, and path comparison records to CasADi residuals
   with explicit lower/upper bounds.  Equality uses identical bounds.
2. Append path residuals at every collocation knot; append initial residuals
   at the initial polynomial value; append terminal residuals at the final
   polynomial value.
3. Augment the collocation state with all registered quadrature channels.
   Enforce their collocation derivatives and continuity exactly as existing
   quadrature states are enforced.
4. Evaluate integral expressions from the final quadrature value, not by
   re-integrating a CasADi expression in the objective or constraints.
5. Introduce a horizon decision and scale collocation derivatives, interval
   lengths, integration weights, control segmentation, and quadrature updates
   by `T`.
6. Keep decision-variable lower/upper bounds sourced from constructor
   declarations.  Do not infer them from `subject_to`; temporal classification
   controls constraint placement.
7. Re-run existing callback, warm-start, parameter-map, and infeasibility
   tests after each lowering change.

## Phase 6 — end-to-end SysOpt parity fixtures

Add Coker equivalents of the two SysOpt references and keep them as durable
contracts.

### Decay/codesign parity

Equivalent to SysOpt `test_codesign_problem_1`:

- model `x_dot = -a*x`, `x(0) = x0`;
- bounded decisions `a`, `x0`;
- `cost = integrate(-output(t) ** 2) - output(t_final)`;
- compare objective to the closed-form integral;
- verify the solved or differentiated result against analytic expectations if
  the selected Coker backend exposes derivatives.

### Path-variable/free-horizon parity

Equivalent to SysOpt `test_codesign_problem_with_path_variable`:

- free `T` and piecewise-constant input;
- terminal state constraint;
- `u <= 1`, `u >= -1` path constraints;
- assert both decisions and their declared bounds are present in lowering;
- solve and assert terminal/path feasibility.

These are parity tests, not copied SysOpt internals.  They must use
`VariationalProblemBuilder` only.

## Phase 7 — documentation and deprecation transition

1. Add one path-planning example and one data-fitting example to the dynamics
   documentation using the final API.
2. Document temporal classification, the fixed/free horizon distinction,
   integral semantics, and the fact that `state()` is differential state while
   `output()` is observed output.
3. Deprecate the old imperative builder API with `DeprecationWarning`, retain
   it as a supported compatibility path, and retain its behavioural tests.
   Do not add aliases between its methods and the new functional API.
4. Document CasADi as the supported execution backend for this API until other
   backends implement the same lowering contract.

## Completion criteria

The feature is complete only when:

- every phase's tests pass;
- the SysOpt-inspired decay/quadrature and path-variable/free-horizon parity
  tests pass against Coker's public API;
- CasADi enforces path, initial, terminal, parameter-only, and integral
  constraints;
- builder-generated quadratures are dynamic states with zero initial values;
- fixed and free horizons both solve correctly; and
- the legacy imperative builder API emits `DeprecationWarning`, remains
  available for existing callers, and retains transition coverage.
