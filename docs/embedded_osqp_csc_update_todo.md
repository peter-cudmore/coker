# Embedded OSQP CSC update TODO

## Goal

Replace archived `u32` CSC arrays that are cast to `*mut i32` at the embedded
OSQP boundary with an explicitly versioned, OSQP-compatible binary CSC ABI.
The mapped runtime must continue to borrow archive data and execute without
allocations after binding.

## Current status

All CSC ABI work is complete as of 2026-08-25. The mapped artifact now carries
signed OSQP indices and explicit `nnz` in embedded-plan ABI version 3; mapped
runtime binding validates that count, terminal `indptr`, live OSQP descriptors,
and submitted numeric slice lengths agree before `update_data_mat`.

Verification passed:

```text
cargo test --workspace                         # 43 tests
uv run python -m pytest tests/backends/coker/test_coker_qp.py -q  # 21 tests
uv run --directory C:/projects/coker --extra casadi python -m pytest \
  C:/projects/hexapod/hexapod_py/tests/test_phase5_fusion_qp.py \
  -k "coker_fusion_qp_matches_casadi and nominal_support" -q  # 1 test
```

The fusion reproducer reaches solve and matches CasADi without
`update_data_mat` status 1. Do not restore dense assembly, basis probing, or an
application-level workaround.

## Archive ABI

- [x] Define `EmbeddedCscPattern` with fixed-width signed OSQP indices
  (`i32`), explicit `nnz`, and CSC dimensions.
  - `indptr` has `ncols + 1` entries, starts at zero, is nondecreasing, and
    ends at `nnz`.
  - `indices` has exactly `nnz` entries, has sorted in-range rows per column,
    and uses the same byte order and alignment contract as the OSQP C ABI.
  - The archive header records the required alignment for direct mapped use.
  - No runtime conversion into owned vectors or scratch arrays.

- [x] Bump the QP archive/embedded-plan ABI version.
  - Producers emit only ABI V3 CSC patterns.
  - Consumers reject older mapped QP archives before binding OSQP.
  - Preserve the existing whole-archive alignment and payload-offset checks.

- [x] Migrate Python payload export and Rust compiler lowering.
  - Emit P, A, KKT, and symbolic-L patterns in the V3 representation.
  - Keep P upper-triangular, A general CSC, and all coefficient output order
    canonical column-major order.
  - Derive `nnz`, arena sizes, and output slices from the same pattern instance.

## Embedded runtime

- [x] Bind direct mapped V3 CSC views to OSQP without `u32`/`i32` pointer casts.
  - Enforce little-endian target and archive alignment before exposing pointers.
  - Keep archive data immutable; only P/A numerical value regions in the
    caller-owned arena are mutable.

- [x] Validate the C-facing descriptor before `osqp_update_data_mat`.
  - Compare `nzmax`, `indptr[ncols]`, archive `nnz`, and supplied coefficient
    slice length for both P and A.
  - Validate every C-facing pointer, dimension, and index range.
  - Reject mismatches locally; do not submit a partial update with null indices.

- [x] Make matrix-update failures diagnosable.
  - Runtime errors report P/A `nzmax`, terminal `indptr`, and submitted lengths.
  - Descriptor mismatch remains distinct from an OSQP numerical-update failure.

## Verification

- [x] Add V3 archive tests.
  - Valid aligned V3 archives map directly without copying.
  - Older and malformed/misaligned archives are rejected.
  - Invalid terminal `indptr`, nonmonotonic columns, unsorted rows, and index
    range violations are rejected.

- [x] Add embedded sparse-update integration coverage.
  - Bind a non-dense upper-triangular P and sparse A to a caller-owned arena.
  - Execute coefficient updates and solves, asserting exact update lengths,
    deterministic output, and no mapped-bytecode copy.

- [x] Run the external Hexapod fusion reproducer.
  - The editable Coker checkout reaches solve and agrees with CasADi.

## Non-goals

- No dense P/A fallback.
- No basis-probing fallback.
- No Hexapod-specific dimensions or ABI workaround.
- No owned runtime decoding, allocation, or copied CSC tables.
