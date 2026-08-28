# Phase 06 — Vector N phases, iterative calls, and tangents

## Goal
Add vectorized N-phase execution and retained nested calls without recursion. An N phase applies an opcode descriptor stream to immutable input coordinates and writes a separate output vector. Achieve primal/tangent parity for the existing scalar opcode set and acyclic function tables.


## CI recovery re-entry: ordinary-phase ping-pong banking

Phase 05 should have established physical disjointness for ordinary B and
gather phases: it owns workspace layout and static mapping materialization.
Phase 06 is the correct recovery re-entry because it extends the same
workspace contract to N, Case, and Call phases. Phase 06 was the last green
gate; do not advance to Phase 07 reduction or Phase 08 cleanup until this
checklist and the Phase 06 gate are green.

### Required checklist

- [ ] Remove temporary-output and restore-gather emission introduced as an
  overlap workaround. Ordinary phases write directly to their final bank.
- [ ] Reserve two disjoint absolute workspace bank ranges in the compiler;
  retain the current artifact fields and concrete-offset runtime interface.
- [ ] At every scheduled-phase boundary, gather live-through values from the
  prior read bank into the next read bank, then swap bank roles.
- [ ] Emit B, N, Case, Call, and ordinary gather phases with source offsets in
  the read bank and destination offsets in the opposite bank.
- [ ] Keep a single compiler-side placement table for resolved static mappings,
  phase inputs, outputs, and live-through values; do not rewrite spans during
  individual emission branches.
- [ ] Reject an ordinary phase whose concrete read and write ranges overlap;
  retain this strict mapped-artifact validation.
- [ ] Permit same-span execution only for a certified residual phase after
  Phase 07 checks linearity, update order, no later source consumers, and no
  separately requested source output. Transfer span ownership from source to
  result on success.
- [ ] Derive every phase header from the row-table length immediately before
  and after that phase's rows are appended. Do not synthesize post-write
  restore phases.
- [ ] Validate primal and tangent use the identical bank layout and gather
  mapping without allocation.
- [ ] Add focused tests for B, N, Case, Call, and gather read/write bank
  disjointness; live-through preservation across a swap; and residual
  in-place rejection when the input remains observable.
- [ ] Run `cargo test -p coker-compiler`, rebuild both wheels, then run the
  isolated full Python suite with `PYTHONPATH=target/wheel_extract` and
  `-o pythonpath=`. This is the Phase 06 recovery gate.

## Tasks

- [ ] Implement N-phase lowering.
  - Emit only intrinsically unary or binary N-required operations: transcendental, variable division, and variable/fractional/negative power.
  - Lower `case` to a dedicated ternary Case phase and nested functions to retained Call phases.
  - Fold compile-time constants before N emission.
- [ ] Implement non-in-place vector N rows in caller workspace.
  - An N phase is `y = N(o, x, x)`, where row `o_i = (opcode, j, k)` defines `y_i = apply(opcode, x_j, x_k)`.
  - The output coordinate is the row position `i`; unary operations use an opcode-defined reserved operand index for the unused coordinate.
  - Each row is intrinsically nonlinear; do not emit B-eligible arithmetic as an N temporary.
  - N reads only its declared immutable input span and writes its direct result only to a compiler-proven-disjoint mutable output span.
  - The N output is its final destination in the opposite phase bank. Only
    live-through inputs are gathered at phase boundaries; N results are never
    relocated by a post-execution restore gather.
- [ ] Validate N rows statically.
  - Every N output row is writable, in range, and written once.
  - Every operand coordinate is within the immutable input span or is the opcode-defined sentinel.
  - The N output span is disjoint from all N operands and belongs to the
    opposite phase bank; validation rejects any overlap.
- [ ] Implement a dedicated ternary Case phase.
  - A Case row carries condition, then, and else input coordinates, with its output coordinate implicit from row position.
  - Primal execution selects `then` when the condition is nonzero and `else` otherwise; push-forward propagates only through the selected branch.
- [ ] Implement primal N kernels for every supported scalar opcode over the N descriptor stream.
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
- [ ] N descriptor rows with out-of-range operands, invalid unary sentinel/opcode arity, overlapping input/output spans, or malformed descriptor/output lengths fail validation.

## Exit criterion

Ordinary functions using B, gather, N, and acyclic calls execute from mapped bytecode with primal/tangent parity.
