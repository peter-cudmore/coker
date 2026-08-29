# Geometry-aware rigid-body and spatial operations

## Problem

The current rigid-body forward-kinematics implementation represents rotations as
axis-angle values and composes them through a generic quaternion round trip:

```text
Rotation3.__mul__
  -> as_quaternion
  -> quaternion multiplication
  -> Rotation3.from_quaterion
       -> normalise(q.v)
       -> atan2
```

Later consumers convert the resulting axis-angle representation back to
quaternions and apply `sin`/`cos` again. This loses the closed-form geometry of
revolute-joint chains and introduces dependent nonlinear and bilinear subgraphs
that a general-purpose lowering pass cannot safely collapse.

The one-leg hexapod forward-kinematics trace demonstrates the consequence:

| Metric | Observed |
|---|---:|
| Joint inputs | 3 |
| Cartesian outputs | 3 |
| Workspace scalars | 2,380 |
| B phases | 346 |
| N phases | 123 |
| Gather phases | 113 |
| Case phases | 14 |
| Bilinear terms | 826 |
| N rows | 173 |
| Gather rows | 2,806 |

The original expectation of one or two bilinear layers and one nonlinear layer
is false for the implementation path above. The 14 Case phases still require
source attribution from the Python tape; aggregate counters cannot identify
whether they originate in `normalise`, quaternion conversion, tensor mapping,
or another helper.

## Decision

Defer Phase 12 lowering compaction and performance profiling to this follow-up.
Do not raise archive table limits and do not add lowering special cases to hide
the observed expansion. The source representation must preserve geometric
semantics before numerical lowering can achieve the expected compact graph.

## Scope

Implement geometrically aware operations in the rigid-body and spatial-algebra
libraries:

1. Represent and compose fixed-axis revolute transforms without repeatedly
   converting quaternion products to axis-angle and back.
2. Preserve concrete geometric operation identity through tracing, so lowering
   can see a rotation-chain operation rather than its incidental
   `normalise`/`atan2`/`sin`/`cos` expansion.
3. Provide direct forward-kinematics primitives for common rigid-body chains
   where their primal and forward-JVP semantics are explicit.
4. Trace every emitted `Case` in the isolated leg tape. Record its condition,
   branch producer chain, and whether the branch is a required geometric
   singularity policy or an avoidable representation artifact.
5. Keep numerical policy explicit at genuine geometric singularities. Do not
   replace required singularity handling with an unchecked algebraic rewrite.
6. Preserve mapped archives, deterministic lowering, caller-owned runtime
   buffers, and allocation-free no-std execution.

## Acceptance criteria

1. The isolated one-leg forward-kinematics trace records exact DAG dependency
   levels and B/N/Case phase sequences before and after the representation
   change.
2. Every previous Case phase is source-attributed; representation-induced Cases
   are eliminated or replaced by a documented geometric policy.
3. The new operation set preserves primal and forward-JVP parity for one leg,
   the complete hexapod forward kinematics, and the forward spatial Jacobian.
4. The resulting hexapod artifacts emit successfully without increasing archive
   limits.
5. Reproduce Phase 12 measurements only after successful emission: table sizes,
   compiler stage timings, first-call and repeated-call costs, workspace, and
   archive bytes.
6. Compare before/after phase counts, including B/N/gather/Case/call/residual,
   and report the source-level reason for each material change.

## Deferred Phase 12 work

The following tasks are moved here from the current pull request:

1. Emit forward-kinematics and forward-spatial-Jacobian production artifacts.
2. Measure compiler stages and first/repeated mapped runtime evaluation.
3. Apply only measured lowering compaction after the geometric representation
   change establishes a compact source graph.
4. Publish the hotspot report, archive/table attribution, and before/after
   evidence.

## Phase 13 evidence record

### Hexapod baseline

- Hexapod revision: `781d78cc07cfd6cfe81ccba0bdd80fde9d0a7a88`.
- Host: Windows 11 Pro 10.0.26200, AMD Ryzen 7 7700, Python 3.13.
- Command: `uv run python scripts/profile_phase12_hexapod.py`.
- Result: forward-kinematics archive finalization failed before artifact emission:
  `archive header count exceeds maximum`.
- Bounded table: `phase_headers`; actual count exceeds
  `ARCHIVE_MAX_PHASES = 4,096`. Therefore archive bytes, runtime timing, and
  runtime allocation measurements are not available for this workload.

### Phase 13 simplification record

| Change | Before | After | Evidence |
|---|---|---|---|
| Emitter organization | One ~2,300-line `emitter.rs`, including tests, function lowering, and module assembly | 22-line facade plus `function.rs`, `module.rs`, and `tests.rs` | Public emitter APIs unchanged; lowering behavior is covered by compiler tests. |
| Finalization failures | Archive-limit failures omitted lowering diagnostics | Typed errors retain diagnostics and table actual/limit | Rust regression `finalization_tests::archive_limit_error_retains_pre_finalization_diagnostics`. |
| Python diagnostics | Ordinary artifact diagnostics were discarded after finalization | Artifact retains and exposes compiler diagnostics | `test_compile_artifact_exposes_complete_success_diagnostics`. |

The module split adds no runtime-owned execution allocations or execution
branches: it is compiler-only organization. The finalization error payload is
compiler-only and boxes diagnostics to keep `CompileError` small. Exact
runtime allocation evidence remains unavailable because the representative
hexapod artifact still cannot be emitted; this is the recorded Phase 12
blocker, not a claimed optimization result.
