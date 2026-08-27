# Phase 09 — QP lowering migration and embedded parity

## Goal

Complete the deferred QP migration after ordinary-tape parity is accepted. Remove Python QP compilation logic while preserving mapped artifacts, caller-owned buffers, and no-std constraints.

## Prerequisite

Phase 08 is complete. Do not begin this phase while ordinary lowering still depends on Python `SparseNet` or `BilinearWeights` metadata.

## Tasks

- [ ] Define the Rust QP handoff from the ordinary compiler.
  - Specify labelled coefficient outputs, ordering, scalar widths, and input binding contract.
  - Reuse the Rust typed/lowered representation; do not reconstruct dense Python arrays.
- [ ] Move coefficient evaluator extraction to Rust.
  - Build the coefficient function through the ordinary compiler pipeline.
  - Preserve output order for `P`, `A`, `q`, lower bounds, upper bounds, and all solver update slices.
- [ ] Move sparse CSC pattern derivation to Rust.
  - Keep pattern tables in the same mapped artifact as the coefficient evaluator.
  - Validate dimensions, row pointers, index bounds, nonzero counts, and compatibility with update slices.
- [ ] Move embedded solver-plan construction/validation to Rust.
  - Preserve pointer-free QP plan data, QDLDL symbolic metadata, arena layout, and OSQP profile restrictions.
  - Keep host conveniences behind `std`; embedded execution remains no-std capable.
- [ ] Implement Python QP builder integration.
  - Python supplies symbolic QP model/tape data through the same explicit builder pattern.
  - Rust returns an aligned `CompiledArtifact`/QP owner; Python bytes remain persistence only.
- [ ] Execute QP with caller-provided buffers.
  - Borrow mapped coefficient evaluator and CSC tables.
  - Evaluate coefficients into caller buffers, scatter directly into caller OSQP update buffers, and reuse caller-provided prepared solver state.
  - Never deserialize QP vectors, rebuild dense matrices, construct a solver per solve, or allocate solve results.
- [ ] Remove the temporary Python QP extraction adapter after parity is proven.

## Required tests

- [ ] Rust/Python QP coefficient parity across representative parameter values.
- [ ] CSC pattern and coefficient-slice ordering parity.
- [ ] Mapped QP archive alignment and plan validation failures.
- [ ] Repeated solve test proves no allocation and prepared-solver reuse.
- [ ] Embedded/no-std integration test with caller-owned arenas and OSQP workspace.

## Exit criterion

The Coker backend has no Python compilation logic: ordinary and QP artifacts are lowered in Rust, execute from mapped archives, and satisfy the repository embedded-runtime constraints.
