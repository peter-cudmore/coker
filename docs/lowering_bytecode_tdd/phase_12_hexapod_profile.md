# Phase 12 — Hexapod baseline

## Workload

- Source model: `C:\projects\hexapod\hexapod_py\hexapod\model.py`, six legs, a free body, and 24 joint coordinates.
- Entry points: independent 18-element forward-foot-position output and direct flattened forward spatial Jacobian output from `RigidBody.spatial_manipulator_jacobian`.
- Compiler path: Python tracing → `CokerBackend.lower` → `coker_compiler` release extension → mapped bytecode archive.
- Host: Windows 11 Pro 10.0.26200, AMD Ryzen 7 7700, Python 3.13, NumPy version reported by `scripts/profile_phase12_hexapod.py`.

## Result

The first artifact cannot be emitted. The production compiler returns:

```text
bytecode validation or encoding failed: failed to decode bytecode module:
archive header count exceeds maximum
```

`ARCHIVE_MAX_PHASES` is 4,096. Thus the current lowering produces more than 4,096 phase headers for this representative workload. No archive bytes, workspace size, first-call timing, repeated-call timing, or runtime hotspot attribution is valid until the compiler emits an artifact.

This is an evidence-backed lower bound against the Phase 12 targets: low tens of phases for forward kinematics and low hundreds at most for the forward spatial Jacobian.

## Validation correction

The initial attempt instead failed on the unrelated `ARCHIVE_MAX_FRAME_DEPTH` check. A function's sequential phase count is not nesting depth; only call phases consume execution frames. The invalid `function.phase_count > ARCHIVE_MAX_FRAME_DEPTH` check was removed from owned and archived validation. The archive remains bounded by `ARCHIVE_MAX_PHASES`; the corrected run reaches that actual limit.

## Next measurement gate

Do not raise `ARCHIVE_MAX_PHASES` to hide this defect. Add compiler-stage and emitted-table counters, then reduce the B/N/gather phase partitioning until this exact workload archives successfully. Only then collect the required first-call and steady-state timing and table-range attribution.
