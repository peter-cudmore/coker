# Phase 04 — SoA artifact, aligned ownership, and one-time validation

## Goal

Introduce the versioned rkyv bytecode artifact and Python `CompiledArtifact` owner without replacing numerical execution yet.

## Tasks

- [ ] Define the new bytecode version and archive header.
  - Keep magic, version, payload alignment, and payload offset explicit.
  - Document every maximum: function count, phase count, frame depth, local indices, terms, and workspace.
  - Legacy artifacts must fail with a clear version error.
- [ ] Define archived SoA tables for program/function headers, overwrite B, residual B, gather, N, and call phases.
  - Use contiguous arrays for hot data; do not retain AoS sparse entries in the new format.
  - Keep immutable metadata separated from caller-owned execution state.
- [ ] Implement owned-model validation.
  - Validate every table span, row range, index, output range, function target, and workspace range before serialization.
- [ ] Implement archive finalization validation.
  - Serialize the owned model into compiler-created aligned storage.
  - Validate the aligned archived view before returning it.
  - A validation failure returns no artifact.
- [ ] Implement Python `CompiledArtifact`.
  - `Builder.build()` returns this owner, not Python bytes.
  - `to_bytes()` is persistence only.
  - `load_path()` maps a persisted artifact read-only and rejects an unaligned payload rather than copying it.
  - Execution handles borrow the owner and cannot outlive it.
- [ ] Implement bind-time validation only.
  - Validate mapped header, alignment, lengths, and table bounds once when creating a runtime handle.
  - Numerical calls must not redo archive validation.

## Acceptance criteria

- [ ] An owned model round-trips through an aligned archive and exposes borrowed archived views.
- [ ] Corrupt header, offset, alignment, and table-range fixtures are rejected at finalization or bind time.
- [ ] `to_bytes()` cannot be passed back as an implicit execution backing store.
- [ ] No runtime execution path has been changed yet.

## Do not do

- Do not copy unaligned mapped bytecode into aligned vectors.
- Do not add a legacy bytecode reader.
