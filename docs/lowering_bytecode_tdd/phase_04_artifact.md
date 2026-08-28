# Phase 04 — Archived artifact, aligned ownership, and one-time validation

## Goal

Introduce the versioned rkyv bytecode artifact and aligned Rust-owned
`CompiledArtifact` storage without replacing numerical execution or switching
the production Python builder. Phase 04 exposes a construction/finalization
API for synthetic owned models; Phase 08 makes `Builder.build()` use it after
the ordinary executable subset exists.

## Tasks

- [ ] Define the new bytecode version and archive header.
  - Keep magic, version, payload alignment, and payload offset explicit.
  - Document every maximum: function count, phase count, frame depth, local indices, terms, and workspace.
  - Legacy artifacts must fail with a clear version error.
- [ ] Define archived tables for program/function headers, overwrite B, residual B, gather, N, and call phases.
  - Use contiguous arrays for hot data; do not retain AoS sparse entries in the new format.
  - Keep immutable metadata separated from caller-owned execution state.
- [ ] Implement owned-model validation.
  - Validate every table span, row range, index, output range, function target, and workspace range before serialization.
- [ ] Implement archive finalization validation.
  - Serialize the owned model into compiler-created aligned storage.
  - Validate the aligned archived view before returning it.
  - A validation failure returns no artifact.
- [ ] Implement aligned `CompiledArtifact` ownership.
  - Finalization returns allocation-base-aligned storage, not a padded `Vec<u8>`.
  - `to_bytes()` is persistence only.
  - `load_path()` maps a persisted artifact read-only and rejects an unaligned payload rather than copying it.
  - Future execution handles borrow the owner and cannot outlive it.
- [ ] Implement bind-time validation only.
  - Validate mapped header, alignment, lengths, and table bounds once when creating a runtime handle.
  - Numerical calls must not redo archive validation.

## Acceptance criteria

- [ ] A synthetic owned model round-trips through an aligned archive and exposes borrowed archived views.
- [ ] Corrupt header, offset, alignment, and table-range fixtures are rejected at finalization or bind time.
- [ ] `to_bytes()` cannot be passed back as an implicit execution backing store.
- [ ] The legacy numerical execution path is unchanged.
- [ ] Production `Builder.build()` is unchanged; Phase 08 owns that cutover.

## Do not do

- Do not copy unaligned mapped bytecode into aligned vectors.
- Do not add a legacy bytecode reader.
