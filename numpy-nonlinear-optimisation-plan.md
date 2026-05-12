# Nonlinear optimisation plan for numpy backend

## Current observed state
- Junior optimisation tasks appear complete in the current tree.
- `python -m pytest` passes: 213 passed, 14 skipped, 1 warning.
- Focused optimisation-related tests also pass: `python -m pytest tests/toolkits/test_codesign.py tests/symbolic_api/test_higher_order.py -vv` -> 26 passed, 14 skipped, 1 warning.
- The numpy backend already contains a partial `trust-constr` path in `src/coker/backends/numpy/core.py`, but it does not match the current backend contract and is not safe to rely on as-is.

## Observed gaps in the current numpy optimisation path
1. `NumpyBackend.build_optimisation_problem(...)` does not accept the required `initial_conditions` parameter from the backend interface.
2. Runtime parameters are modelled with `sympy.MatrixSymbol`, but the generated call signatures mix decision variables and parameter symbols inconsistently.
3. The current objective Jacobian wrapper only accepts one argument and does not follow SciPy's `f(x, *args)` / `jac(x, *args)` calling contract.
4. Constraint lowering mixes linear/nonlinear detection incorrectly by differentiating with respect to `problem_args` instead of decision variables only.
5. `NonlinearConstraint(..., hess="cs")` is not a safe default when the Jacobian is also numerical/derived indirectly; SciPy requires a coherent derivative contract.
6. Solver startup ignores requested initial conditions and always uses zeros.
7. Result reconstruction is wrong for runtime-parameterized solves (`list(*solver_args)`), and the backend does not populate `last_solve_info` or raise `SolveFailure` on unsuccessful solves.
8. There are no numpy-backend optimisation tests, so regressions would be invisible.

## Implementation phases

### Phase 1: Lock the numpy optimisation contract with tests
Files:
- `tests/toolkits/test_codesign.py`
- `tests/conftest.py` only if backend parametrization needs adjustment

Add backend-specific tests that run on `numpy` instead of skipping everything except `casadi`:
- zero-runtime-input solve on numpy
- runtime-parameterized solve on numpy
- codesign `norm(...)` use on numpy
- infeasible solve on numpy, asserting explicit failure instead of plausible output
- consistent single-output and multi-output mapping on numpy

Acceptance:
- Tests fail against the current numpy implementation for the right reasons.
- Tests encode the same user-facing contract already exercised for CasADi.

### Phase 2: Replace the partial solver bridge with a clean trust-constr adapter
Files:
- `src/coker/backends/numpy/core.py`
- `src/coker/optimisation.py`

Refactor `NumpyBackend.build_optimisation_problem(...)` around SciPy's actual contract:
- accept `(cost, constraints, parameters, outputs, initial_conditions)`
- build one flat decision vector `x` from non-parameter tape inputs in declaration order
- build a deterministic unpacker from `x` back to per-variable numpy arrays/scalars
- build a deterministic packer for runtime parameters from `*args` in declared order
- compile objective/constraint/output callables with signature `fn(x, *runtime_args)`
- use `initial_conditions` to build `x0` in decision-variable order

Do not keep mutating `tape.input_indicies` with ad hoc substitution as the main control flow. Isolate symbolic lowering from solver-call assembly so repeated solves stay deterministic.

Acceptance:
- The solver closure is pure with respect to repeated calls.
- Runtime inputs are forwarded as separate SciPy `args`, not as nested tuples.
- Initial guesses reflect builder-specified values.

### Phase 3: Lower constraints in a trust-constr-native way
Files:
- `src/coker/backends/numpy/core.py`

For each half-plane constraint produced by `constraint.as_halfplane_bound()`:
- evaluate the symbolic residual as a function of decision variables and runtime parameters
- differentiate with respect to the flat decision vector only
- construct either:
  - `LinearConstraint` when the Jacobian is constant with respect to decision variables and runtime parameters, or
  - `NonlinearConstraint` otherwise
- represent one-sided bounds with `lb`/`ub` exactly, instead of baking sign conventions into custom lambdas

Derivative policy:
- objective: provide exact symbolic gradient and Hessian when available from SymPy
- nonlinear constraints: start with exact Jacobians; use a quasi-Newton Hessian update strategy such as `scipy.optimize.BFGS()` unless a correct constraint Hessian-of-dot-product is implemented
- prefer correctness over premature second-derivative coverage

Rationale from SciPy docs:
- `minimize(..., method="trust-constr")` expects `fun(x, *args)`, `jac(x, *args)`, and compatible constraint derivative contracts.
- `NonlinearConstraint.hess` must return the Hessian of `dot(fun(x), v)`; a placeholder Hessian is worse than omitting one.

Acceptance:
- Linear and nonlinear inequality constraints both solve through the same backend path.
- The derivative contract matches SciPy's documented API.

### Phase 4: Truthful solve reporting and output reconstruction
Files:
- `src/coker/backends/numpy/core.py`
- `src/coker/optimisation.py`
- possibly `src/coker/toolkits/codesign/__init__.py` only if additive plumbing is needed

Add numpy solve metadata mirroring the CasADi path:
- helper to map `scipy.optimize.OptimizeResult` into `SolveInfo`
- populate `last_solve_info` on every solve attempt
- raise `SolveFailure` when `OptimizeResult.success` is false
- include backend=`"numpy"`, solver=`"trust-constr"`, return status/message, iteration count

Reconstruct outputs by evaluating lowered output expressions against:
- solved decision vector `soln.x`
- runtime parameter arguments in declared order

Keep the existing successful return shape contract: list of outputs, later reshaped by `MathematicalProgram.__call__`.

Acceptance:
- Failures are distinguishable from success.
- Single-output and multi-output programs both return usable values.
- Repeated solves update `problem.solve_info` truthfully.

### Phase 5: Verify end-to-end behavior
Commands:
- `python -m pytest tests/toolkits/test_codesign.py tests/symbolic_api/test_higher_order.py`
- add any new numpy-backend-specific test module if needed and run it directly
- `python -m pytest`

Verification note:
- Run the test suite in an environment with both optional dependencies installed: `jax` and `casadi`.
- Do not treat skipped coverage caused by missing optional dependencies as sufficient verification for this work.

If code formatting changes touch `src/`, run:
- `python -m black src`

## Notes from the SciPy tutorial/docs to follow
- Use `method="trust-constr"` on `scipy.optimize.minimize`.
- Objective signature should be `fun(x, *args)`.
- Provide `jac` explicitly when available.
- Use `LinearConstraint` and `NonlinearConstraint` instead of encoding constraints only through penalty terms.
- For nonlinear constraints, `hess(x, v)` must match SciPy's Hessian contract for `dot(fun(x), v)`; otherwise use a documented approximation strategy.
- `OptimizeResult.success`, `message`, `nit`, and constraint-violation fields should feed backend solve reporting.

## Primary sources
- SciPy optimize tutorial: https://docs.scipy.org/doc/scipy/tutorial/optimize.html#trust-region-constrained-algorithm-method-trust-constr
- trust-constr reference: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
- NonlinearConstraint reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html
