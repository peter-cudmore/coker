# Phase 02 — Maturin builder and typed tape ingestion

## Goal

Replace Python-to-JSON compiler input with a capacity-preallocated PyO3 builder that owns a validated Rust `TypedDag`. It must not lower or emit bytecode yet.

## Tasks

- [ ] Create the `coker_compiler` maturin package boundary.
  - Python backend imports this package; do not expose Rust compiler internals through existing runtime modules.
  - Document package build/install commands next to its Python integration test.
- [ ] Implement `Builder` construction with explicit capacities:
  - node count;
  - flattened operand count;
  - constant count and scalar count;
  - labelled input/output count;
  - nested-function/call metadata count.
- [ ] Implement ordered ingestion methods.
  - `push_constant(...)` copies compiler-owned constant data and validates shape/type immediately.
  - `push_node(...)` accepts one node with operation tag, operand indices, dimension metadata, and optional constant/function reference.
  - Reject skipped node indices, forward references, operand-capacity overflow, invalid dimensions, and unsupported operation tags at the push boundary.
  - Add explicit input/output-label methods; preserve their source order.
- [ ] Make ingestion allocation behavior explicit.
  - Reserve all declared capacities at construction.
  - A capacity shortfall is a structured compile error; never silently grow in the hot ingestion loop.
  - Python references must not remain in Rust after a `push_*` call.
- [ ] Implement `finish_tape()`.
  - Verify declared counts match received data.
  - Return a pure Rust `TypedDag` with index-based storage only.
  - Do not yet run scheduling or bytecode emission.
- [ ] Add PyO3 integration tests that build fixtures node-by-node and compare the resulting typed DAG to a Rust-only fixture builder.

## Acceptance criteria

- [ ] Python can construct every Phase 01 fixture through `Builder`.
- [ ] Rust-only and PyO3-built `TypedDag` values compare identically.
- [ ] Malformed ingestion fails at the first invalid push with node index and field context.
- [ ] `finish_tape()` leaves no Python-owned object reachable from compiler state.

## Do not do

- Do not retain JSON compilation as a second production input path.
- Do not emit an archive from `build()` until Phase 04.
- Do not traverse the tape recursively.
