# Phase 01 — Freeze contracts and build the test oracle

## Goal

Create a reliable differential-test baseline before replacing any lowering or runtime code. No production lowering behavior changes in this phase.

## Prerequisites

- Read `docs/lowering_bytecode_rewrite_plan.md` completely.
- Treat existing Python backend evaluation as the high-precision semantic oracle.
- Preserve all existing tests unchanged.

## Tasks

- [x] Define the initial runtime-observation oracle.
  - Record finite `f32` primal and directional-derivative results from the current desktop runtime against the existing Python Coker graph reference.
  - Use explicitly registered operation-specific absolute/relative tolerances.
  - Record transcendentals and special floating-point results as desktop observations; do not assume NumPy and runtime math libraries are bitwise equivalent.
  - Defer fast-math, FMA, flush-to-zero, SIMD numerical-policy, and target-device profiling decisions.
- [x] Register initial tolerances in one policy fixture.
  - Start with conservative operation-specific finite-value tolerances and tighten them only from measured desktop results.
  - Do not scatter numeric tolerances through individual tests.
- [x] Create deterministic tape fixtures covering:
  - scalars, vectors, reshape, concatenate, static transpose/slice/repetition;
  - constants, linear maps, quadratic products, dot, matrix multiplication;
  - every runtime-supported source scalar opcode; record unsupported or bytecode-only opcode gaps separately;
  - fork/join DAGs, unused branches, nested calls, and deep acyclic calls;
  - near-identity maps such as `1.01*x` and unsafe cyclic residual candidates.
- [x] Add a single differential-test helper.
  - Evaluate the existing Python Coker graph reference.
  - Accept a runtime result and tangent result.
  - Use the fixture's finite-domain precondition and registered tolerance.
  - Let the fixture test name and `assert_allclose` diagnostic identify the failing observation.
  - For special values, report classification only; do not assert bitwise equality.
- [x] Define one test-local runtime-observation policy fixture.
  - Do not scatter numeric literals through tests.
  - Test an explicitly supplied policy by applying it to the near-identity fixture.
- [x] Capture baseline assertions for every fixture:
  - compare current Python graph primal/tangent output with the desktop runtime;
  - apply the fixture's finite-input domain and operation tolerance;
  - exercise desktop special floating-point values without requiring equality.

## Acceptance criteria

- [x] Existing Python test suite remains green.
- [x] Every finite fixture has an explicit primal/tangent tolerance.
- [x] Special-value fixtures assert no crash and record desktop behavior without bitwise-equality claims.
- [x] At least one fixture exercises each runtime-supported source scalar opcode, and unsupported or bytecode-only opcode gaps are regression-tested.
- [x] Tests distinguish Python graph reference behavior from current desktop-runtime observations.

## Do not do

- Do not change bytecode structures.
- Do not delete Python lowering or JSON export.
- Do not add target-specific optimization claims.
