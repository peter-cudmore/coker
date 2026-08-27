# Phase 08 — Ordinary backend parity, performance gates, and cleanup

## Goal

Make the Rust compiler/runtime the production ordinary-tape path, achieve feature parity with the existing Python test suite, then remove obsolete Python lowering infrastructure.

## Tasks

- [ ] Route the Coker Python backend through `coker_compiler.Builder` and `CompiledArtifact`.
  - Keep a temporary test-only switch to compare legacy and Rust paths.
  - Do not expose JSON lowering as a public ordinary compilation API.
- [ ] Port or adapt every existing ordinary backend test.
  - Classify failures as unsupported source operation, extraction defect, lowering defect, archive validation defect, primal mismatch, or tangent mismatch.
  - Add regression fixtures before fixing each discovered defect.
  - Compare Python high-precision semantics where appropriate.
  - Compare finite-domain primal/tangent results under the Phase 01 current desktop-runtime tolerance contract.
  - Record special-value behavior as desktop-local; do not require bitwise runtime parity.
  - Include labelled input/output ordering, unused inputs, multiple outputs, and nested calls.
- [ ] Add artifact lifecycle tests.
  - `CompiledArtifact` remains valid through repeated calls.
  - Execution handles do not outlive their aligned/mapped owner.
  - Persisted artifact reload uses mapping and rejects bad alignment without copying.
- [ ] Benchmark the desktop baseline.
  - Report phase count, workspace, gather cost, nnz, archive bytes, compile time, and execution time.
  - Do not profile STM32F7 or Raspberry Pi 4B in this rewrite; that work belongs to the later SIMD/optimization effort.
- [ ] Remove obsolete ordinary-path code only after all tests pass.
  - Python `SparseNet` lowering, `BilinearWeights` compiler metadata, ordinary JSON export/compiler path, and old layer/row-op/sparse-entry execution paths.
  - Keep Python reference evaluation solely as test oracle.

## Exit criterion

The complete existing ordinary Python test suite passes against the compact mapped Rust runtime. The production ordinary backend no longer uses Python lowering logic.

## Do not do

- Do not start QP migration before this phase is accepted.
- Do not keep legacy readers/writers as compatibility shims.
