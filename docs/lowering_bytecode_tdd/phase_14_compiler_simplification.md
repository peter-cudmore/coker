# Phase 14 — Compiler simplification and compile-time efficiency

## Purpose

Remove compiler-only machinery that has no current semantic effect, make the
remaining lowering pipeline own each analysis fact exactly once, and hoist
node-invariant work out of scalar-row emission. This phase implements the
Phase 13 compiler review in dependency order. It is not a Phase 12 compaction
experiment: it must preserve emitted archive semantics before any new lowering
policy is considered.

## Invariants

- `TypedDag` remains the immutable compiler input. Compiler construction may
  allocate; runtime execution remains mapped, borrowed, and allocation-free
  after caller-owned buffers are constructed.
- Archive format, archive alignment, numerical results, deterministic lowering
  order, sparse CSC patterns, QP coefficient ordering, and solver settings do
  not change unless a focused behavioral regression explicitly establishes the
  intended contract.
- Ordinary and QP compilation keep the existing mapped-artifact ownership
  model: `OwnedModel -> CompiledArtifact -> ArchivedModel<'_>`.
- Phase 14 does not raise `ARCHIVE_MAX_PHASES`, coalesce phases, remove
  required gathers, alter QP tolerances, or claim a Phase 12 timing result.
- Prefer deletion and direct concrete functions. Do not introduce a pass trait,
  generic operation-lowerer, compatibility alias, or secondary DAG builder.
- Every potentially behavior-changing simplification starts with the smallest
  durable regression covering its archive tables and runtime observable result.

## Ordered implementation checklist

| Review finding | Implementation step |
|---|---|
| 1. Unused rank-factor reduction | 14.1 |
| 2. False scheduler policy | 14.2 |
| 3. Unused graph/workspace metadata | 14.3 |
| 4. Duplicate graph analysis | 14.4 |
| 5. Per-row shape allocation and validation | 14.6 |
| 6. Quadratic term-start construction | 14.5 |
| 7. Retained QP scenarios | 14.9 |
| 8. Duplicate Python QP conversion | 14.10 |
| 9. Repeated QP output lookup | 14.11 |
| 10. Redundant emitter setup branches | 14.7 |
| 11. Raw phase-kind protocol | 14.8 |


### 14.1 — Remove unused rank-factor reduction

- [ ] Add a lowering regression for an overwrite phase considered for residual
  conversion. Assert emitted overwrite/residual tables and ordinary execution
  values before changing reduction code.
- [ ] Replace `RankFactorReducer::reduce(...).direct` in
  `emitter::function` with direct canonical sparse construction.
- [ ] Delete `BilinearReducer`, `DirectSparseReducer`, `RankFactorReducer`,
  `ReductionResult`, factor reconstruction helpers, and tests that test only
  the removed factorization representation.
- [ ] Retain `SparseMatrix`, `analyze_residual`, and `select_residual` only if
  the residual certificate and cost decision remain required by emission.
- [ ] Verify the focused regression plus ordinary, tangent, and QP archive
  parity; record removed types, branches, and compile-time work.

### 14.2 — Simplify deterministic scheduler policy

- [ ] Add a scheduler regression that fixes phase class/order for a DAG with a
  mixed ready frontier.
- [ ] Replace `schedule(dag, analysis, SchedulingPolicy::default())` with a
  parameter-free deterministic scheduler.
- [ ] Delete `SchedulingPolicy`, `max_exact_ready_frontier`,
  `max_exact_schedules`, and the conditional path that claims to select an
  exact schedule but only sorts classes.
- [ ] Keep the current deterministic class preference and critical-path/node
  tie-break ordering unless the regression proves a deliberate replacement.
- [ ] Update scheduler rustdoc to describe the actual algorithm, not an
  unimplemented exact search.

### 14.3 — Prune unused graph and workspace metadata

- [ ] Establish a regression covering reachable-node pruning, schedule order,
  workspace span offsets, and explicit emitted gathers.
- [ ] Remove `WorkspacePlan.gathers`, `GatherRequirement`, and
  `WorkspaceSpan.last_phase`; gather ownership remains solely in emitter
  lowering.
- [ ] Remove unused `NodeMetadata` fields (`scalar`, `constant_known`,
  `consumer_count`, `last_use`, `earliest_phase`, and `latest_phase`) and their
  calculations.
- [ ] Make analysis internals crate-private where no Rust consumer requires a
  public API. If an exported field is removed, perform a clean caller migration
  rather than retain an alias.
- [ ] Reduce `plan_workspace` arguments to only the facts it consumes.

### 14.4 — Unify emission analysis ownership

- [ ] Add a regression proving dead unsupported branches are pruned while a
  reachable function emits identical tables and executes identically.
- [ ] Remove the caller-supplied `GraphAnalysis` argument from
  `emit_bilinear_model` and its tests/callers.
- [ ] Make emission own reachability pruning and perform one complete graph
  analysis only after pruning. Do not retain an original-DAG full analysis
  merely to supply an invalidated graph to emission.
- [ ] Remove the duplicate pre-emission analysis from module and QP compile
  paths; preserve diagnostics timing with explicit stage boundaries.
- [ ] Verify diagnostics still distinguish analysis, workspace planning,
  emission, and finalization without counting an eliminated pass.

### 14.5 — Linearize overwrite term-start construction

- [ ] Add a regression with multiple overwrite rows and sparse, empty, and
  trailing term ranges. Assert exact `overwrite_term_start` contents and
  mapped execution output.
- [ ] Introduce one local CSR-style helper that builds row-term starts from
  canonical row-ordered terms with a single cursor.
- [ ] Use it before and after residual-table compaction; delete both repeated
  `terms.iter().filter(...).count()` implementations.
- [ ] Preserve `u16` row bounds and table validation behavior.

### 14.6 — Hoist shape-dependent node layouts

- [ ] Add direct archive regressions for vector/matrix/batched `MatMul` and
  scalar/vector/matrix `Transpose` forms presently supported by lowering.
- [ ] Add a borrowed `TypedDag` shape accessor with existing checked shape
  bounds, then remove temporary shape-vector allocation from the hot emission
  loop.
- [ ] Validate MatMul and Transpose dimensions once per node and retain a local
  concrete layout/index formula for the per-output-row loop.
- [ ] Hoist constant-multiply source selection and other node-invariant arity
  checks when doing so removes repeated scalar-row branching.
- [ ] Do not add an operation dispatch trait or generic operation-lowerer.

### 14.7 — Simplify remaining emitter control flow

- [ ] Add regressions for `Evaluate`, `Case`, nonlinear unary/binary operations,
  constant multiplication, and unsupported-operation diagnostics.
- [ ] Route `Evaluate` and `Case` before constructing bilinear-only operand
  spans and row bookkeeping when their table layouts do not need them.
- [ ] Retain the concrete operation `match` for archive-table emission; move
  only invariant validation/layout calculation outside the scalar-row loop.
- [ ] Keep each phase-kind dispatch explicit and once per phase; do not add
  runtime dynamic dispatch.

### 14.8 — Name the archived phase-kind protocol

- [ ] Add archive-table regressions covering overwrite, residual, gather,
  nonlinear, call, and Case phase rebasing in a multi-function module.
- [ ] Replace raw phase-kind literals with named internal constants shared by
  function emission and module rebasing, without changing archived values.
- [ ] Keep phase-kind dispatch explicit and once per phase; do not add runtime
  dynamic dispatch or a runtime trait layer.


### 14.9 — Stream QP finite-difference scenarios

- [ ] Add a small source-QP regression that asserts emitted `P/A/q/l/u/r`
  coefficient values, pattern ordering, and the solved result.
- [ ] Precompute decision-coordinate to input-slot/index mapping once.
- [ ] Keep the zero specialization, then generate signed-basis and pairwise
  specializations one at a time as cloning consumes them; do not retain the
  complete quadratic scenario vector.
- [ ] Remove `QpScenario` and `finite_difference_scenarios` if no supported
  external Rust API requires them. Otherwise preserve only a documented,
  iterator-style boundary; no compatibility alias.
- [ ] Preserve scenario order because QP finite-difference index formulas rely
  on it; test those formulas directly.

### 14.10 — Consolidate Python QP declaration parsing

- [ ] Add Python binding regressions showing equivalent symbolic QP input gives
  identical lowering metadata and compiled artifact through every retained
  entry point.
- [ ] Extract private helpers for `SymbolicQpDeclaration` conversion and the
  default `EmbeddedOsqpSettings` literal.
- [ ] Migrate `compile_archive_qp` and `lower_symbolic_qp` to the shared
  declaration parser.
- [ ] Replace the nine-argument source-QP binding with one explicit Python
  declaration object or mapping. Migrate every Python caller in this pull
  request and remove the positional API; do not retain a compatibility alias.
- [ ] Preserve PyO3 error types and messages for invalid bounds, labels, and
  dimensional declarations unless a regression authorizes an improved contract.

### 14.11 — Index QP evaluator output alignment

- [ ] Add regressions for ordinary outputs plus six QP coefficient outputs,
  duplicate labels, missing labels, slice-length mismatch, and exact contiguous
  coefficient-region layout.
- [ ] Build one validated label-to-output-spec index from evaluator outputs.
- [ ] Derive ordinary output metadata and `P/A/q/l/u/r` slices from that index;
  remove repeated DAG-label scans and separate spec lookup paths.
- [ ] Retain explicit six-slice ordering and all compiler/runtime materialization
  contracts from Phase 13.

### 14.12 — Validate the complete cutover

- [ ] For each completed item, record before/after public types, compiler-only
  allocations, table-construction branches, and relevant source line count.
- [ ] Run focused compiler tests first, then `cargo test --workspace`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo check -p coker-runtime --no-default-features`.
- [ ] Rebuild the PyO3 compiler extension and run the non-performance Python
  suite, Black, and flake8.
- [ ] Re-run the Phase 12 hexapod baseline only to confirm whether artifact
  emission is unblocked. Do not profile or compact until its existing geometry
  follow-up permits it.

## Exit criteria

1. No unused rank-factor reducer, false scheduler-policy API, unused graph
   metadata, duplicate graph analysis, retained all-scenario QP store, or
   duplicate Python symbolic-QP conversion remains.
2. Shape-dependent MatMul/Transpose validation occurs once per source node, and
   overwrite row starts are built linearly from canonical terms.
3. Emitted ordinary and QP archives retain identical observable semantics:
   table validation, primal and forward-JVP results, coefficient ordering,
   sparse patterns, QP solutions, and error contracts all pass focused tests.
4. Runtime mapping, alignment, caller-buffer ownership, and no-std allocation
   constraints remain unchanged.
5. Full Rust and Python CI gates pass, and the implementation record separates
   measured compile-time simplification from deferred Phase 12 performance work.
