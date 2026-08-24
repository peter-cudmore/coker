# Composition implementation checklist

## Program contract

- [x] Make `MathematicalProgram` return `(m, *y)`, with the solved objective first.
- [x] Migrate every existing program-result caller to the objective-first contract.
- [x] Add regression tests and commit the completed phase.

## Backend-agnostic composition

- [x] Add statically known `MathematicalProgram` calls to the symbolic graph.
- [x] Add NumPy/SciPy host orchestration for acyclic program-call graphs.
- [x] Test `P(f(r))` and `f(P(p))`; commit the completed phase.

## CasADi composition

- [x] Add CasADi concrete module orchestration while retaining solver calls as numerical boundaries.
- [x] Test post-solve composition; commit the completed phase.

## Coker composition

- [x] Add Python-side `CokerModule` orchestration around prebuilt QP solvers.
- [x] Reuse the existing mapped QP runtime per module call; do not rebuild a QP or solver per invocation.
- [x] Test host composition and prebuilt solver reuse; commit the completed phase.

## Embedded Coker module execution

- [ ] Add a `QpCall` bytecode layer that references a QP executable and binds
  its parameter slices and primal output slice.
- [ ] Require the embedding application to provide one prepared QP context per
  call layer: solver arena, evaluator workspace, coefficient buffer, flat
  parameter buffer, and primal solution buffer.
- [ ] Execute `QpCall` layers through those caller-owned contexts, copying the
  primal solution into the parent program workspace only after coefficient
  evaluation and solving complete.
- [ ] Reject mismatched QP ids, input widths, output widths, missing contexts,
  and duplicate call-context bindings before execution.

## Cleanup

- [x] Update public documentation for objective-first program results and composition semantics.
- [x] Run the targeted composition verification set.
- [x] Commit cleanup.

## Invariants

- Solver calls are numerical boundaries: no derivative is defined through argmin/argmax.
- Program/module call graphs are static and acyclic.
- Coker mapped execution borrows archived bytecode and uses only embedding-provided mutable workspace after setup.
- No compatibility alias preserves the former outputs-only `MathematicalProgram` result contract.
