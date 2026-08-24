# QP performance review

## Scope and execution model

The QP path has three distinct phases:

1. Python constructs an objective, constraints, and a coefficient evaluator in `src/coker/backends/coker/optimisation.py`.
2. Python serialises the exported graph as JSON and the Rust compiler turns it into mapped bytecode.
3. The embedded runtime evaluates coefficients into caller-provided storage, copies them into the bound OSQP numeric arrays, updates OSQP, and solves.

The embedded steady-state path borrows mapped bytecode and caller-owned evaluator, coefficient, OSQP-arena, and output buffers. It does not allocate Rust-owned execution buffers. Host Python wrappers are convenience APIs and do allocate temporary input lists/conversions and copy the returned solution.

## Measurement

Host measurement, Windows 11 / Ryzen 7 7700, CPython test process:

```text
python -m pytest tests/backends/coker/test_coker_qp.py -q --durations=3
```

The 48-decision / 49-row sparse weighted-norm code-generation regression took
37.20 s before the change below and 24.18 s after it. This is a construction/
code-generation result, not an embedded target latency claim. The same run's
next-slowest QP extraction took 0.21 s.

## Measured result

### Weighted-norm coefficient graph rebuilt identical symbolic terms

**Severity:** measured host code-generation cost.

The weighted-norm lowering rebuilt `W r`, and each derivative component of
`W r`, for every `P`, `q`, and constant expression. The fusion-scale regression
therefore spent 37.20 s in code generation.

**Implemented fix:** cache coefficient expressions for each CSC data value,
residual offset, residual Jacobian component, weighted offset, and weighted
Jacobian component within the generated evaluator. `P`, `q`, and the constant
reuse those tracers.

**Result:** 37.20 s to 24.18 s for the existing 48-decision / 49-row
regression: 35% less host code-generation time. Numerical coefficient and
runtime solve tests remain unchanged.

## Current-branch correctness fix: remove dense structural assumptions

Dense `P`, `A`, KKT, and LDL structures are an implementation error, not an
acceptable fallback. A QP program must contain the exact structural nonzeros
proven by its symbolic objective and affine constraints. If the compiler cannot
prove that structure, it must reject the QP rather than silently emit dense
storage.

### Contract

1. `BilinearWeights` is the source of symbolic structure. Its nonzero linear
   and quadratic keys determine the candidate `A` and upper-triangular `P`
   entries; sparse-matrix provenance supplies the structure for weighted norms.
2. A structural entry remains present when its runtime value can be zero. Only
   an entry proven identically zero is omitted. This keeps OSQP sparsity stable
   across parameter updates.
3. `P` is CSC upper triangular and `A` is CSC by decision column. Their
   `indptr`, `indices`, and coefficient-output order are produced together from
   the same canonical sorted entry lists.
4. Exact patterns flow through `ExtractedQpProgram`, the exported QP payload,
   Rust lowering, archive validation, arena layout, and embedded OSQP updates.
   Existing archived fields already carry CSC patterns; the producer now
   derives them rather than filling them densely.
5. The QDLDL plan is computed from the exact KKT pattern with deterministic
   natural-order symbolic elimination. A dense factor pattern is not permitted.

### Implementation

1. Coefficient extraction derives canonical `P` and `A` entry sets before it
   creates output slices. The CSC patterns and coefficient outputs use the same
   column-major order.
2. Ordinary QPs derive `P` and `A` support from raw `BilinearWeights`.
   Weighted norms derive Hessian support only between decision derivatives that
   share a sparse weight row; they do not materialise dense `WᵀW`.
3. The dense pattern helpers and dense symbolic-L generator are deleted.
   Unsupported opaque or relocated provenance continues to fail QP extraction
   rather than selecting a dense fallback.
4. The existing archive fields, Rust compiler, and runtime scatter/update path
   consume the smaller `p_nnz` and `a_nnz` patterns unchanged.

### Acceptance criteria

- A diagonal Hessian and selector constraints produce diagonal/selector CSC
  patterns, not dense `P`/`A`.
- A sparse weighted norm produces only the structural nonzeros of
  `WᵀW` after residual composition.
- Coefficients and solutions match the current dense reference for constant
  and parameterized QPs, including a parameter whose value zeros a structural
  entry.
- QDLDL plan validation and embedded solve pass for non-dense `P` and `A`.
- Archive coefficient lengths, arena bytes, and `update_data_mat` lengths equal
  actual CSC nnz.
- Opaque or unproven coefficient provenance is rejected with a clear error.

## Deferred later-task TODO

- [ ] Add target-side phase timing for evaluator, scatter/validation, vector
  update, matrix update, and OSQP solve. Report median and maximum cycles for
  warm and cold sparse-controller workloads.
- [ ] Classify coefficient streams as immutable or parameter-dependent so static
  `P`/`A` values are bound once instead of updated every solve.
- [ ] Replace JSON host compiler input with a compact typed export/direct rkyv
  production path; keep JSON out of the mapped runtime path.
- [ ] Evaluate pinning the self-referential embedded OSQP instance only after
  target timing proves `refresh_self_pointers` material.
