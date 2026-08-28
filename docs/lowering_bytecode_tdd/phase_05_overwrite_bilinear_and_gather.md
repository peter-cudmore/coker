# Phase 05 — Overwrite bilinear and gather execution

## Goal

Execute compact overwrite B phases and explicit gather phases from mapped SoA bytecode. Reach parity for constants, linear maps, degree-two polynomial operations, and static reindexing.

## Workspace-bank responsibility

This phase owns the physical ordinary-phase workspace contract. It must reserve
disjoint read/write banks for B and gather execution; Phase 06 extends those
same banks to N, Case, and Call. Do not repair a later overlap by writing to a
temporary span and gathering back: select the final disjoint destination during
bank placement.

## Tasks

- [ ] Implement compiler emission for `OverwriteBilinearPhase`.
  - Build canonical homogeneous monomials using ordered `(left, right)` pairs.
  - Merge duplicate terms deterministically in `f64` before `f32` emission.
  - Sort terms by output row.
  - Allocate output spans disjoint from source spans unless an explicit gather has already materialized a snapshot.
- [ ] Implement mapped overwrite kernel.
  - Process one output row at a time with a local accumulator initialized to zero.
  - Read the homogeneous-one sentinel without workspace access.
  - Write each output row once.
  - Keep no allocation, no archive copy, and no per-term dynamic dispatch.
- [ ] Implement mapped overwrite push-forward kernel.
  - Write primal and tangent output rows once.
  - Match the Phase 01 finite-input current desktop-runtime oracle within the fixture tolerance.
- [ ] Implement `GatherPhase` compiler selection.
  - First attempt to fold static mapping into the consuming overwrite B phase.
  - Emit gather only for an un-fusible static mapping: reshape, permutation, slice, concatenate, repeated index, or broadcast duplication.
  - Allocate a fresh contiguous output span and account for its liveness cost.
- [ ] Implement gather primal/tangent kernels.
  - Copy the same index mapping from primal/tangent source workspace to destination workspace.
  - Validate source indices, destination span, and mapping length at bind time.
- [ ] Add direct output-buffer paths only after ordinary workspace paths pass parity.

## Required tests

- [ ] Constant, identity, linear, multiplication, dot, and matrix-product fixtures match `f32` oracle primal and tangent results.
- [ ] Canonical swapped monomials merge to one numerical contribution.
- [ ] Every gather mapping class matches a folded B equivalent where one exists.
- [ ] Invalid gather source index and invalid B term index fail validation.
- [ ] Allocation instrumentation reports zero allocations during repeated execution.

## Exit criterion

A program consisting only of B and gather phases runs from `CompiledArtifact` with parity against all applicable Phase 01 fixtures.
