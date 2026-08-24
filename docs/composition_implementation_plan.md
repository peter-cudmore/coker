# Composition implementation checklist

## Program contract

- [ ] Make `MathematicalProgram` return `(m, *y)`, with the solved objective first.
- [ ] Migrate every existing program-result caller to the objective-first contract.
- [ ] Add regression tests and commit the completed phase.

## Backend-agnostic composition

- [ ] Add statically known `MathematicalProgram` calls to the symbolic graph.
- [ ] Add NumPy/SciPy host orchestration for acyclic program-call graphs.
- [ ] Test `P(f(r))`, `f(P(p))`, and nesting; commit the completed phase.

## CasADi composition

- [ ] Add CasADi concrete module orchestration while retaining solver calls as numerical boundaries.
- [ ] Test pre- and post-solve composition; commit the completed phase.

## Coker composition

- [ ] Add Python-side `CokerModule` orchestration and composite artifact export.
- [ ] Add a mapped-runtime QP-call layer with caller-provided nested workspaces.
- [ ] Test host and mapped execution, then commit the completed phase.

## Cleanup

- [ ] Update public documentation and examples for objective-first program results and composition semantics.
- [ ] Run the full targeted composition verification set.
- [ ] Commit cleanup.

## Invariants

- Solver calls are numerical boundaries: no derivative is defined through argmin/argmax.
- Program/module call graphs are static and acyclic.
- Coker mapped execution borrows archived bytecode and uses only embedding-provided mutable workspace after setup.
- No compatibility alias preserves the former outputs-only `MathematicalProgram` result contract.
