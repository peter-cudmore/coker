# Coker sparse-performance audit

## Scope

Reviewed all Coker uses of DOK contraction, bilinear graph construction, and
DOK-to-dense conversion after optimizing compact QP construction.

## Findings

### 1. `tensor_ndarray_product` repeats the retired dense scan

**Severity:** performance risk.

**Evidence:** `src/coker/backends/coker/sparse_tensor.py`,
`tensor_ndarray_product` at lines 452–488. For every entry of a dense right
operand it scans `tensor.keys` to find entries on the matching contracted axis.
`dok_ndarray.__matmul__` selects this routine for `dok_ndarray @ ndarray`.

**Consequence:** this has the same $O(\text{dense entries} \times \text{tensor
nonzeros})$ shape as the QP bottleneck that was removed from `__rmatmul__`.
Large dense-right contractions can again exhibit the old construction-time and
allocation behavior.

**Recommendation:** apply the same sparse contraction scheme used by
`__rmatmul__`: collect only nonzero dense-array entries by contracted index,
then visit source DOK entries once. Preserve `tensor_ndarray_product`'s output
axis order: uncontracted tensor axes first, then uncontracted array axes.

### 2. `tensor_sum` is quadratic in sparse nonzeros

**Severity:** performance risk.

**Evidence:** `src/coker/backends/coker/sparse_tensor.py`, `tensor_sum` at
lines 491–516 cross-products every left and right DOK key before discarding
keys whose contracted coordinates differ. `weights.BilinearWeights.__matmul__`
uses this for sparse-by-sparse contraction.

**Consequence:** dot products of sparse bilinear maps scale as
$O(\mathrm{nnz}_{left} \times \mathrm{nnz}_{right})$, rather than matching only
keys that share the contraction coordinate. The optimized QP profile still
shows 1.113 s in `weights.dot`; larger sparse maps will expose this more
strongly.

**Recommendation:** index the smaller operand's keys by its contracted
coordinate, then iterate the other operand and only its matching bucket. This
is the highest-value remaining general sparse-algebra improvement.

### 3. Generic QP extraction intentionally remains dense and probe-based

**Severity:** performance and archive-size risk; documented fallback.

**Evidence:** `src/coker/backends/coker/optimisation.py`,
`_generic_coefficient_function` lines 1049–1200 creates `np.eye(n)` and
executes the cost for zero, positive/negative basis vectors, and every pair of
decision basis vectors. It emits full upper-triangular `P` and rectangular `A`
patterns regardless of actual support.

**Consequence:** opaque QPs have $O(n^2)$ cost evaluations and dense matrix
metadata. This is not used by QPs with raw bilinear provenance, including the
weighted-norm path just optimized; it remains a major scalability limit for
unsupported/opaque QP graphs.

**Recommendation:** keep it as an explicit compatibility path. Do not attempt
a local micro-optimization. A future replacement needs a symbolic sparse
provenance contract for opaque operators; otherwise it would silently infer an
incorrect sparse pattern.

### 4. Cross-product lowering intentionally densifies parameter coefficients

**Severity:** conditional performance/memory risk.

**Evidence:** `src/coker/backends/coker/op_impl.py`, `cross` lines 31–50 calls
`toarray()` for constant/linear tensors, computes `Q_result` with dense
`einsum`, then converts back to DOK.

**Consequence:** a cross product has fixed three-vector output, but its dense
quadratic temporary is shaped `(3, parameter_count, parameter_count)`. Sparse
parameter dependencies can therefore become quadratic in parameter count.

**Recommendation:** defer until profiling identifies cross-product-heavy
models. A sparse outer-product implementation is straightforward but would add
complexity to a fixed-size operation that is not on the compact-QP profile.

### 5. Other DOK-to-dense conversions are execution or ingress boundaries

**Severity:** observation.

**Evidence:**

- `weights.BilinearWeights.__call__` and `push_forwards` convert evaluated
  results to NumPy arrays for host execution.
- `layers.GenericVectorLayer` materializes constants for NumPy execution.
- `core._as_numpy_value` and `_build_opaque_operand` densify SciPy sparse
  constants before Coker lowering.

**Consequence:** these are expected at a numerical host-execution boundary,
but large SciPy sparse constants lose sparsity on ingress.

**Recommendation:** do not change evaluator-result conversions for this work.
If large SciPy sparse constants become a workload, accept them directly as DOK
in `_constant_to_bw` and `_build_opaque_operand`; add an explicit import path
rather than extending the hot evaluator.

## Clean-up applied

Removed the now-unused single-tracer `graph_for` helper from
`_bilinear_coefficient_function` after its multi-output graph conversion.

## Priority

1. Optimize `tensor_sum` for contracted-index matching.
2. Optimize `tensor_ndarray_product` with nonzero dense entries.
3. Only profile and address cross-product densification when a real model
   demonstrates it.
4. Treat generic QP probing as a format/compiler redesign, not a local fix.
