# Optimisation design tasks

## 1. Decide the supported norm contract for optimisation models
- Files: `src/coker/toolkits/codesign/__init__.py`, `src/coker/algebra/ops.py`, `src/coker/backends/casadi/casadi.py`
- Decision: define which vector and matrix norm orders are supported now, and reject unsupported combinations explicitly.
- Why senior-level:
  - this sets user-facing API semantics across backends
  - matrix `ord=2` is not a trivial extension on CasADi and must not be guessed

## 2. Introduce structured solve-status and failure reporting for mathematical programs
- Files: `src/coker/backends/casadi/optimiser.py` and a shared solve-status type location
- Decision: settle the public contract for solve metadata (`SolveInfo`) and failed-solve exceptions (`SolveFailure`).
- Why senior-level:
  - callers need a durable contract for distinguishing success, infeasibility, and backend failure
  - this affects how downstream code reports, retries, or recovers from failed solves

## 3. Propagate solve status through variational optimisation results
- Files: `src/coker/backends/casadi/variational_solver.py`, `src/coker/dynamics/types.py`
- Decision: extend `VariationalSolution` additively so solve metadata is exposed without breaking existing consumers.
- Why senior-level:
  - this touches a public result type
  - callback payloads, final solutions, and failure behavior must stay consistent

## 4. Resolve the initial-condition contract ambiguity in codesign
- Files: `src/coker/toolkits/codesign/__init__.py`, `tests/toolkits/test_codesign.py`
- Decision: either normalize both dict-style and list-style initial conditions or tighten the builder contract and update callers/tests together.
- Why senior-level:
  - this is an API behavior decision, not just a local bug fix
  - different choices change how much legacy usage remains supported

## 5. Add failure-path verification for infeasible solves
- Files: targeted tests under `tests/toolkits/` and `tests/dynamical_systems/`
- Decision: lock in the exact observable contract for infeasible solves.
- Minimum contract to decide:
  - whether failures always raise
  - which status fields are required
  - whether callers should inspect attached metadata on the exception, the problem object, or both
- Why senior-level:
  - these tests define the truthfulness contract for solver failures
  - once merged, downstream users will depend on the chosen behavior

## Recommended direction
- Preserve current success return shapes where possible.
- Attach additive solve metadata instead of wrapping every successful result in a new object.
- Raise explicit failures on unsuccessful solves instead of returning plausible-looking infeasible iterates.
