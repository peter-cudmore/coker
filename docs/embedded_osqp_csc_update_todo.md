# Embedded OSQP CSC update TODO

## Goal

Replace archived `u32` CSC arrays that are cast to `*mut i32` at the embedded
OSQP boundary with an explicitly versioned, OSQP-compatible binary CSC ABI.
The mapped runtime must continue to borrow archive data and execute without
allocations after binding.

## Archive ABI

- [ ] Define `EmbeddedOsqpCscPatternV2` with fixed-width signed OSQP indices
  (`i32`), explicit `nnz`, and CSC dimensions.
  - `indptr` has `ncols + 1` entries, starts at zero, is nondecreasing, and
    ends at `nnz`.
  - `indices` has exactly `nnz` entries, has sorted in-range rows per column,
    and uses the same byte order and alignment contract as the OSQP C ABI.
  - The archive header records the required alignment for direct mapped use.
  - No runtime conversion into owned vectors or scratch arrays.

- [ ] Bump the QP archive/embedded-plan ABI version.
  - Producers emit only V2 CSC patterns.
  - Consumers reject V1 mapped QP archives before binding OSQP.
  - Preserve the existing whole-archive alignment and payload-offset checks.

- [ ] Migrate Python payload export and Rust compiler lowering.
  - Emit P, A, KKT, and symbolic-L patterns in the V2 representation.
  - Keep P upper-triangular, A general CSC, and all coefficient output order
    canonical column-major order.
  - Derive `nnz`, arena sizes, and output slices from the same pattern instance.

## Embedded runtime

- [ ] Bind direct mapped V2 CSC views to OSQP without `u32`/`i32` pointer casts.
  - Enforce little-endian target and archive alignment before exposing pointers.
  - Keep archive data immutable; only P/A numerical value regions in the
    caller-owned arena are mutable.

- [ ] Validate the C-facing descriptor before `osqp_update_data_mat`.
  - Compare `nzmax`, `indptr[ncols]`, archive `nnz`, and supplied coefficient
    slice length for both P and A.
  - Validate every C-facing pointer, dimension, and index range.
  - Reject mismatches locally; do not submit a partial update with null indices.

- [ ] Make matrix-update failures diagnosable.
  - Extend the runtime error with P/A dimensions, archive `nnz`, terminal
    `indptr`, `nzmax`, and submitted update lengths.
  - Preserve the raw OSQP status, but distinguish a descriptor mismatch from an
    OSQP numerical-update failure.

## Verification

- [ ] Add V2 archive tests.
  - Valid aligned V2 archive maps directly without copying.
  - V1 and malformed/misaligned V2 archives are rejected.
  - Invalid terminal `indptr`, nonmonotonic columns, unsorted rows, and index
    range violations are rejected.

- [ ] Add embedded sparse-update integration coverage.
  - Bind a non-dense upper-triangular P and sparse A to a caller-owned arena.
  - Execute at least two coefficient updates and solves.
  - Assert exact P/A update lengths, deterministic output, and no runtime
    allocation path.

- [ ] Run the external Hexapod fusion reproducer.
  - Execute `coker_fusion_qp_matches_casadi and nominal_support` with the
    editable Coker checkout.
  - Confirm the embedded runtime reaches solve, returns a decision equivalent
    to CasADi, and does not report `update_data_mat` status 1.

## Non-goals

- No dense P/A fallback.
- No basis-probing fallback.
- No Hexapod-specific dimensions or ABI workaround.
- No owned runtime decoding, allocation, or copied CSC tables.
