# Phase 01 — Freeze contracts and build the test oracle

## Goal

Create a reliable differential-test baseline before replacing any lowering or runtime code. No production lowering behavior changes in this phase.

## Prerequisites

- Read `docs/lowering_bytecode_rewrite_plan.md` completely.
- Treat existing Python backend evaluation as the high-precision semantic oracle.
- Preserve all existing tests unchanged.

## Tasks

- [ ] Define the initial runtime-observation oracle.
  - Record finite `f32` primal and directional-derivative results from the current desktop runtime against the Python semantic oracle.
  - Use explicitly registered operation-specific absolute/relative tolerances.
  - Record transcendentals and special floating-point results as desktop observations; do not assume NumPy and runtime math libraries are bitwise equivalent.
  - Defer fast-math, FMA, flush-to-zero, SIMD numerical-policy, and target-device profiling decisions.
- [ ] Register initial tolerances in one policy fixture.
  - Start with conservative operation-specific finite-value tolerances and tighten them only from measured desktop results.
  - Do not scatter numeric tolerances through individual tests.
- [ ] Create deterministic tape fixtures covering:
  - scalars, vectors, reshape, concatenate, static transpose/slice/repetition;
  - constants, linear maps, quadratic products, dot, matrix multiplication;
  - transcendental operations, division, powers, comparisons, and `case`;
  - fork/join DAGs, unused branches, nested calls, and deep acyclic calls;
  - near-identity maps such as `1.01*x` and unsafe cyclic residual candidates.
- [ ] Add a single differential-test helper.
  - Evaluate the Python high-precision oracle.
  - Later accept a runtime result and tangent result.
  - Use the fixture's finite-domain precondition and registered tolerance.
  - Report fixture name, operation, inputs, primal mismatch, tangent mismatch, and tolerance.
  - For special values, report classification only; do not assert bitwise equality.
- [ ] Define one `LoweringPolicy` test fixture with the documented defaults.
  - Do not scatter numeric literals through tests.
  - Test that an explicitly supplied policy overrides the default.
- [ ] Record baselines for every fixture:
  - current Python output and tangent output;
  - finite-input domain and operation tolerance;
  - desktop special-value observations when applicable;
  - node count, live-output count, and known operation classes.

## Acceptance criteria

- [ ] Existing Python test suite remains green.
- [ ] Every finite fixture has an explicit primal/tangent tolerance.
- [ ] Special-value fixtures assert no crash and record desktop behavior without bitwise-equality claims.
- [ ] At least one fixture exercises each current scalar opcode.
- [ ] Tests distinguish high-precision semantic behavior from current desktop-runtime observations.

## Do not do

- Do not change bytecode structures.
- Do not delete Python lowering or JSON export.
- Do not add target-specific optimization claims.
