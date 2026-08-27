# Phase 06 — Nonlinear frames, iterative calls, and tangents

## Goal

Add N-phase execution and retained nested calls without recursion. Achieve primal/tangent parity for the existing scalar opcode set and acyclic function tables.

## Tasks

- [ ] Implement N-phase lowering.
  - Keep B-eligible operations in B phases; degree overflow causes B→B, never N.
  - Emit only intrinsically N-required operations: transcendental, variable division, variable/fractional/negative power, comparisons, `case`, and calls.
  - Fold compile-time constants before N emission.
- [ ] Implement the local N frame.
  - Copy the contiguous input span into caller-provided frame slots `0..input_len`.
  - Reserve `input_len..frame_len` for temporaries/results.
  - Emit output frame range and copy it to the declared contiguous output span.
- [ ] Validate N frames statically.
  - Every destination is writable, in range, and written once.
  - Every argument is an input or an earlier defined temporary.
  - Every declared output-frame slot is defined.
- [ ] Implement primal N kernels for every supported scalar opcode.
  - Match finite inputs against the Phase 01 current desktop-runtime contract and fixture tolerance.
  - Record special floating-point behavior as desktop-local rather than requiring bitwise parity.
- [ ] Implement push-forward N kernels.
  - Define and test every derivative rule, including branch selection semantics for `case` and comparisons.
  - Return a structured compile error for an opcode lacking a valid derivative rule.
- [ ] Build the retained function call graph iteratively.
  - Reject recursive cycles and compute maximum call depth and cumulative workspace/frame requirements.
- [ ] Implement iterative `CallFrame` execution.
  - Caller provides frame-stack storage and primal/tangent workspace ranges.
  - A frame records callee, next phase, bindings, outputs, and subranges.
  - Push callee frame, run phases, bind outputs, pop, and resume caller; do not call Rust/C recursively.

## Required tests

- [ ] Every scalar opcode matches f32 primal/tangent oracle including edge cases.
- [ ] Deep acyclic nested calls use the explicit frame stack and no recursion.
- [ ] Recursive call graph and insufficient caller frame storage fail before execution.
- [ ] N-frame forward reference, input overwrite, and undefined output frame tests fail validation.

## Exit criterion

Ordinary functions using B, gather, N, and acyclic calls execute from mapped bytecode with primal/tangent parity.
