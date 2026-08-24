# Coker sparse-performance audit

## Scope

Reviewed all Coker uses of DOK contraction, bilinear graph construction, and
DOK-to-dense conversion after optimizing compact QP construction.

## Resolved findings

### Sparse DOK contractions

`tensor_ndarray_product` and `tensor_sum` now index the right operand by the
contracted coordinate. They visit only matching nonzero pairs; neither routine
cross-products unrelated DOK entries or dense-array zeros. Regression tests
compare both paths with `numpy.tensordot`.

### Generic QP probing

Removed `_generic_coefficient_function`. QP coefficient extraction now rejects
graphs without raw bilinear provenance with a clear `ValueError` rather than
probing $O(n^2)$ decision bases and emitting dense `P` and `A` patterns.

This is the required clean cutover for sparse mapped QP artifacts. Supporting
opaque QPs again requires an explicit symbolic sparse-provenance contract.

### Cross-product lowering

`op_impl.cross` now accumulates constant, linear, and quadratic DOK entries
directly from the Levi-Civita coefficients. It no longer creates dense
`(3, parameter_count, parameter_count)` intermediates.

### SciPy sparse constants

`dok_ndarray.from_scipy` imports COO entries directly. `_constant_to_bw` and
`_build_opaque_operand` use it, preserving sparse constants through lowering.

## Remaining intentional dense boundaries

- `BilinearWeights.__call__` and `push_forwards` return NumPy results for host
  execution.
- `GenericVectorLayer` materializes values for its NumPy execution path.

These are numerical execution boundaries, not symbolic-lowering paths. They
remain appropriate unless a caller needs a sparse host-evaluator API.

## Verification

- Sparse dense-right and sparse-right contractions match `numpy.tensordot`.
- Sparse SciPy import retains only nonzero COO entries and canonicalizes
  duplicate coordinates through SciPy's COO conversion.
- QP extraction rejects missing raw bilinear provenance.
