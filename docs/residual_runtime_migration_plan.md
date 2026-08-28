# Residual runtime migration checklist

## Goal

Replace compact-and-copy workspace lowering with residual slot scheduling for every ordinary runtime program. A residual is compiler-only state: each live value has a stable caller-workspace slot and a known final use. The mapped runtime executes a fixed instruction stream over caller-provided buffers; it does not allocate, construct a graph, or schedule dynamically.

The current FK artifact has 3,090 layers and 414,044 generic identity rows. The migration eliminates relocation as a runtime operation rather than merely merging its layers.

> **Status: superseded for ordinary-phase workspace placement.** The stable
> global-slot, first-fit reuse, and exact alias-scratch design below is not the
> active CI-recovery architecture. Ordinary B/N/Case/Call/gather phases now use
> compiler-assigned ping-pong live-frontier banks as specified in
> `lowering_bytecode_rewrite_plan.md` and the Phase 06 recovery checklist.
> Retain this document only for its residual certification, QP, no-std, and
> caller-owned-buffer requirements. A residual remains the only certified
> same-span execution phase; ordinary phases have disjoint read/write banks.

## Non-negotiable runtime contract

- [ ] Preserve `no_std` execution and require no global allocator.
- [ ] Retain borrowed, aligned mapped bytecode; do not deserialize it on the runtime path.
- [ ] Keep workspace, tangent workspace, outputs, nested-call scratch, QP buffers, and solver arenas caller-owned.
- [ ] Validate bytecode once while mapping; hot loops perform no dynamic validation, allocation, sorting, or graph repair.
- [ ] Keep scalar execution serial and deterministic. Design batches and layouts for future SIMD/MIMD, but do not implement parallel execution in this migration.

## Format cutover

- [ ] Bump `COKERB03` / bytecode version `5` to `COKERB04` / version `6`.
- [ ] Bump `COKERQ03` to `COKERQ04` and `COKERP03` to `COKERP04`.
- [ ] Reject every pre-v6 mapped artifact. Do not maintain a legacy executor, decoder alias, or compatibility shim.
- [ ] Regenerate all Coker and Hexapod artifacts in the same cutover.

## Compiler-only residual schedule

- [ ] Add an IR between graph lowering and archive serialization:

  ```rust
  struct ValueLifetime {
      value: ValueId,
      width: u16,
      first_definition: u32,
      final_use: u32,
      slot: SlotRange,
  }

  struct SlotRange {
      start: u32,
      length: u16,
  }
  ```

- [ ] Compute final use from graph consumers, graph outputs, nested-function bindings, and push-forward dependencies.
- [ ] Allocate stable slots with deterministic lowest-address first-fit reuse.
  - Inputs receive fixed ABI slots.
  - Vectors and matrices receive contiguous ranges.
  - Free a range immediately after final use.
  - Reuse only an adequately sized, alignment-compatible free range.
  - Record peak live slots, maximum nested-call scratch, and exact alias scratch.
- [ ] Represent reshape and shape-only transpose as compiler views, not runtime rows.
- [ ] Preserve concatenate as a logical slice sequence until a consumer actually requires a contiguous materialization.
- [ ] Emit only genuine copies required by aliasing or ABI materialization.

## Scheduled bytecode

- [ ] Replace layer-relative input/output ranges with stable slot operands.

  ```rust
  struct ScheduledGenericOp {
      output: u16,
      first: u16,
      second: u16,
      third: u16,
      op: ScalarOp,
  }
  ```

- [ ] Add a tagged wide-index form only when peak live workspace exceeds `u16::MAX`; default to compact `u16` fields for bytecode density and cache locality.
- [ ] Replace sparse bilinear scatter entries with output-row groups:

  ```rust
  struct BilinearRow {
      output: u16,
      term_start: u32,
      term_count: u16,
  }

  struct BilinearTerm {
      left: u16,
      right: u16,
      value: f32,
  }
  ```

- [ ] Sort rows by output slot and terms by `(left_slot, right_slot)`.
- [ ] Encode homogeneous constants explicitly; do not materialize workspace slots just for multiplication by one.
- [ ] Batch only dependency-safe homogeneous operations: unary generic, binary generic, select/comparison, bilinear rows, and nested-call boundaries.
- [ ] Preserve serialized execution order; do not reorder floating-point reductions at runtime.

## Mutation and alias contract

- [ ] A destination may overwrite a source only when that source has its final use in the current operation/layer.
- [ ] Read every aliased source before writing its destination.
- [ ] Accumulate each bilinear row into a local scalar and write its destination once.
- [ ] Allocate only the exact temporary scalar/range required for a real alias hazard. Never copy the entire live workspace.
- [ ] Keep final-output slots live through output materialization.

## Python graph lowering

- [ ] Refactor `src/coker/backends/coker/core.py::_create_opgraph`.
- [ ] Remove retained-value prefix compaction in `flush_bilinear`.
- [ ] Remove retained-value identity emission in `lower_generic`.
- [ ] Replace `current_memory` / `next_memory` compaction with stable slot assignments.
- [ ] Lower pending bilinear expressions into stable destination-row groups.
- [ ] Lower generic scalar operations directly to scheduled stable-slot operations.
- [ ] Keep tape traversal, free-slot selection, row ordering, and archive output deterministic.

## Rust compiler and runtime

- [ ] Replace compact layer structures in `coker-bytecode` with scheduled v6 structures.
- [ ] Replace scratch accumulation in `coker-compiler/src/lower.rs` and `context.rs`.
- [ ] Define required workspace as peak live slots plus maximum nested-call and exact alias scratch.
- [ ] Replace `input_output_slices`, ordinary `prepare_input_range`, and whole-range scratch copying in `coker-runtime/src/execute.rs`.
- [ ] Implement allocation-free scheduled generic and row-grouped bilinear executors.
- [ ] Remove output-range clearing for sparse bilinear layers.
- [ ] Migrate push-forward with matching primal/tangent slot lifetimes and local bilinear primal/tangent accumulation.
- [ ] Reserve one reusable nested-call region sized to maximum callee requirement; pack only declared inputs and copy only declared outputs.
- [ ] Delete old compacting layer types, validators, executors, and tests after v6 migration.

## QP migration

- [ ] Compile coefficient evaluators as v6 scheduled ordinary programs.
- [ ] Preserve sparse CSC `P` and `A` patterns, embedded QDLDL metadata, direct coefficient scatter, and prepared caller-owned OSQP arenas.
- [ ] Revalidate `coefficient_function_id`, coefficient-output partitions, and CSC numeric lengths against v6 evaluator metadata.
- [ ] Do not introduce dense QP assembly, basis probing, solver reconstruction, or runtime allocation.
- [ ] Keep explicit accepted-primal warm starts. Never warm-start from rejected or infeasible sentinel output.

## Validation

- [ ] Reject out-of-bounds slots, malformed row term ranges, invalid wide-index tags, illegal aliases, invalid nested bindings, and old versions.
- [ ] Test deterministic slot allocation, scalar/vector range reuse, output retention, nested-call scratch reuse, and exact alias-copy insertion.
- [ ] Test primal equivalence for arithmetic, reshape, concatenate, mixed constants/workspace values, sparse bilinear expressions, nested functions, and output paths.
- [ ] Test push-forward equivalence for every operation category, including slot reuse.
- [ ] Test QP coefficient equivalence, CSC ordering, prepared solver reuse, nominal warm-started fusion solves, infeasibility classification, and foothold behavior.

## Artifact and performance gates

- [ ] Add a `kinematics.coker` inspection regression requiring:

  ```text
  identity relocation operations == 0
  explicit copies == actual alias hazards only
  required workspace == peak live + bounded scratch
  required workspace << 784,211 f32
  ```

- [ ] Record artifact bytes, logical/required primal and tangent workspace, layer/batch count, generic arithmetic rows, explicit-copy rows, bilinear rows/terms, output-clear bytes, and workspace bytes touched.
- [ ] Verify FK and Jacobian outputs plus push-forward equivalence against the Python graph.
- [ ] Verify no allocations after mapped-program construction.
- [ ] Measure release host median/p95/max only as a development signal.
- [ ] Accept performance only after target or faithful-target measurements of cycles, p95/worst case, workspace/cache behavior, and complete controller tick time.

## Completion condition

The migration is complete when all runtime programs execute v6 residual scheduled bytecode; compacting layers and accumulated overlap scratch are deleted; FK relocation identities are eliminated; required workspace reflects peak live state; Coker and Hexapod primal, tangent, and QP regressions pass; and all mapped-bytecode, no-std, caller-owned-buffer, allocation-free invariants remain intact.
