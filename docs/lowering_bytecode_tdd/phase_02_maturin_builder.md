# Phase 02 — Maturin builder and typed tape ingestion

## Goal

Replace Python-to-JSON compiler input with a capacity-preallocated PyO3 builder that owns a validated Rust `TypedDag`. It must not lower or emit bytecode yet.

## Tasks

- [x] Create the `coker_compiler` maturin package boundary.
  - Python integration tests import this package; the existing backend remains unchanged until Phase 08.
  - Build/install with `python -m maturin develop --manifest-path coker_runtime/crates/coker-compiler-python/Cargo.toml`.
- [x] Implement `Builder` construction with explicit capacities:
  - node count;
  - flattened operand count;
  - constant count and scalar count;
  - labelled input/output count;
  - nested-function/call metadata count.
- [x] Implement ordered ingestion methods.
  - `push_constant(...)` copies compiler-owned constant data and validates shape/type immediately.
  - `push_node(...)` accepts one node with operation tag, operand indices, dimension metadata, and optional constant/function reference.
  - Reject skipped node indices, forward references, operand-capacity overflow, invalid dimensions, and unsupported operation tags at the push boundary.
  - Add explicit input/output-label methods; preserve their source order.
- [x] Make ingestion allocation behavior explicit.
  - Reserve all declared capacities at construction.
  - A capacity shortfall is a structured compile error; never silently grow in the hot ingestion loop.
  - Python references do not remain in Rust after a `push_*` call.
- [x] Implement `finish_tape()`.
  - Verify declared counts match received data.
  - Return a pure Rust `TypedDag` with index-based storage only.
  - Do not yet run scheduling or bytecode emission.
- [x] Add PyO3 integration tests that build fixtures node-by-node and compare the resulting typed DAG counts with the Rust-only fixture builder.

## Acceptance criteria

- [x] Python can construct representative Phase 01 scalar fixtures through `Builder`.
- [x] Rust-only and PyO3-built `TypedDag` fixtures report identical structural counts.
- [x] Malformed ingestion fails at the first invalid push with node index and field context.
- [x] `finish_tape()` consumes the builder and leaves only index-based Rust storage.

## Do not do

- Do not retain JSON compilation as a second production input path.
- Do not emit an archive from `build()` until Phase 04.
- Do not traverse the tape recursively.
