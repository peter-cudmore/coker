# Compact QP construction performance report

## Scope and method

Measured on 2026-08-24 on the configured Windows 11 workstation (Ryzen 7
7700), using:

```text
PYTHONPATH=src python -m cProfile -s cumulative \
  tests/benchmarks/reproduce_compact_qp_construction.py
```

The independent workload has 24 decisions, 78 weighted-objective rows, 66
affine constraint rows, five parameter spaces, and nine decision columns per
sparse row. It measures Python-side QP extraction and module serialization;
it does not solve the QP.

## Result

| Metric | Before | After |
| --- | ---: | ---: |
| Unprofiled construction | 39.298 s | **2.722 s** |
| Profiled construction | 43.375 s | **6.825 s** |
| `__rmatmul__` cumulative time | 29.743 s | **0.114 s** |
| `create_opgraph` calls | 6 | **2** |
| Serialized module | 99,888 bytes | **99,888 bytes** |

The optimized unprofiled result is a **14.4× speedup** for this workload.
Module bytes are unchanged.

## Applied changes

### Sparse left contraction

`dok_ndarray.__rmatmul__` formerly iterated every dense selector element and,
for each, scanned all source DOK entries. It now:

1. extracts nonzero selector entries;
2. groups those entries by their contracted index;
3. visits each source nonzero once and accumulates only matching selector
   entries.

It supports both NumPy and DOK selectors and preserves the established output
axis order and duplicate accumulation. Cancellation removes the resulting zero
entry, consistent with `dok_ndarray.__setitem__`.

### Shared coefficient graph

`_bilinear_coefficient_function` now collects cost/residual/weight/bound
tracers by tape index and lowers them through one multi-output graph. The graph
is mapped back to each coefficient component by its original tracer index.
Duplicate bound or residual references therefore share one symbolic result.

### Less DOK copy work

DOK addition now copies the larger operand once and merges the smaller operand.
The operation remains persistent: neither source dictionary is mutated.

## Current profile

The prior selector contraction is no longer material:

| Cumulative time | Calls | Location |
| ---: | ---: | --- |
| 4.045 s | 2 | `core.create_opgraph` |
| 2.065 s | 1 | `optimisation._bilinear_coefficient_function` |
| 1.824 s | 79,892 | `kernel._emit` |
| 1.701 s | 79,892 | `kernel.append` |
| 1.485 s | 1 | `kernel.function` |
| 1.267 s | 2,100 | `op_impl.dot` |
| 1.113 s | 1,506 | `weights.dot` |
| 0.114 s | 1,782 | `sparse_tensor.dok_ndarray.__rmatmul__` |
| 0.016 s | 1 | `ast_preprocessing.export_payload` |
| 0.014 s | 1 | `runtime.compile` |

The remaining construction cost is now trace creation and symbolic
`BilinearWeights` arithmetic. It is not an embedded runtime cost.

## Next opportunities

1. Profile a target-sized application with repeated construction, separating
   imports from QP construction. This run includes approximately 0.8 s of
   imports.
2. If construction remains material, reduce `kernel._emit`/`append` traffic by
   building coefficient expressions directly into the final output tape rather
   than creating intermediate symbolic scalars. This needs a dedicated
   equivalence benchmark; the present 2.467 s result does not justify a
   riskier rewrite.
3. Add an environment-specific performance target only after repeated
   measurements; retain the current 90-second guard solely as a regression
   detector.

## Non-recommendations

- Do not optimize payload export or `RuntimeQpProgram.compile`; together they
  are approximately 0.03 s in the profile.
- Do not reintroduce dense `P`, `A`, KKT, or LDL structures. That violates the
  sparse mapped-archive/runtime contract.
- Do not infer OSQP solve performance from this construction-only benchmark.
