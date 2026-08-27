# Phase 07 — Bilinear reduction and conservative residual layers

## Goal

Add pluggable B reduction and the separate in-place residual B layer. Preserve numerical behavior and use residual execution only when its proof and cost model justify it.

## Tasks

- [ ] Implement sparse canonical matrix construction for each candidate B phase.
  - Rows are outputs; columns are canonical homogeneous monomials.
  - Keep sparse data; never form dense `S` merely to reduce it.
- [ ] Define the compile-time `BilinearReducer` interface.
  - Input: sparse `f64` matrix and `LoweringPolicy`.
  - Output: direct map or factorization `S = R*T`, rank, nnz, residual certificate, and cost estimate.
- [ ] Implement direct sparse reducer first.
  - Canonicalize/merge only.
  - This is the mandatory fallback for every later reducer.
- [ ] Implement deterministic rank-factor reducer second.
  - Fixed pivot order and tolerance.
  - Verify pre-quantization and post-`f32` residual bounds.
  - Propagate reconstruction through later B phases where possible; materialize only at N/call/output boundaries.
- [ ] Implement residual candidate analysis for same-span B maps.
  - Canonicalize `S = I + R` and store `R = S - I`, including diagonal residual coefficients.
  - Reject if residual density exceeds `LoweringPolicy.residual_max_terms_per_row`.
  - Build directed dependency edges `row -> externally read row`; remove self edges.
  - Use iterative topological sort to derive `update_order`; reject cycles.
  - Compare overwrite and residual candidates under phase count, workspace, archive bytes, and calibrated cycle policy.
- [ ] Implement separate mapped residual primal/tangent kernels.
  - Process rows in `update_order`.
  - Accumulate `R` into the existing primal/tangent row.
  - Do not support gather or unproven overlap for residual layers.

## Required tests

- [ ] `1.01*x` emits/executes a safe residual candidate when policy permits.
- [ ] Cyclic off-diagonal dependency is rejected as residual and falls back to overwrite.
- [ ] Residual and overwrite candidates produce matching primal/tangent outputs.
- [ ] Invalid update order, duplicate/missing rows, and out-of-range residual terms fail validation.
- [ ] Direct reducer is selected when factorization or residual fails its numerical/cost certificate.

## Exit criterion

Reduction and residual selection are deterministic, validated, benchmark-visible, and never required for correctness.
