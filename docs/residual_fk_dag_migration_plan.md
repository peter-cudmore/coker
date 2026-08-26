# Residual DAG FK migration plan

## Decision

Use two sequential milestones:

```text
Tape ──semantic DAG, views, reverse uses──> Python residual SparseNet
                                                  │
                                                  ├─ Python/reference-backend correctness
                                                  └─ FK stage-count gate
                                                         │
                                                         ▼
                                              bytecode schema/ABI and mapped runtime
```

The tape is a tracing representation only. The first deliverable is a compiler-owned, deterministic Python residual `SparseNet`: `InputLayer` → ordered stable-slot bilinear/nonlinear/call stages → `OutputLayer`. A nonlinear stage may consume either materialized slots or retained epoch expressions; expression-backed consumption is an operand form, not a separate stage kind. The program owns value lifetimes, epoch roots and expressions, ordered frontier/boundary batches, nested-call bindings, output bindings, and exact Python workspace requirements.

Do not continue the current hybrid path as the destination. `unfused_lowering._create_opgraph` still uses `MemorySpec`, growing `current_memory`, dense `BilinearWeights`, constant-extension layers, and `GenericVectorLayer`. That is compact-layer lowering with residual metadata beside it, not the intended residual `SparseNet`. Evolve the Python `SparseNet` evaluator and stage records first. Confirm its primal and push-forward behavior against the existing reference backends and establish the FK stage count before designing or changing bytecode serialization, archive ABI, validators, or mapped execution.

Only after the Python gate passes, revisit the bytecode representation as a separate cutover. Encode the proven residual stages mechanically; bytecode must not discover dependencies, repair lifetimes, materialize views, compact ranges, or schedule rows. At that point decide the archive version/magic, AoS-versus-SoA physical layout, validators, and mapped executor together. Do not pre-commit `COKERB04` / version 6, `COKERQ04`, or `COKERP04` as the residual ABI before the Python execution model is settled. The later SoA work remains tracked in [the separate SoA todo](soa_bytecode_transform_todo.md).

## Non-negotiable runtime contract

- Runtime bytecode is an aligned borrowed `rkyv` archive mapped from embedding-owned bytes. The header records archive alignment and payload offset; mapping rejects unaligned or malformed artifacts.
- Construction validates once. Execution performs no allocation, bytecode decoding, sorting, graph scheduling, repair, or global-allocator access.
- The embedding owns primal/tangent workspace, output buffers, exact alias scratch, nested-call scratch, QP coefficient buffers, and solver arenas.
- Scalar execution is serial and deterministic. The compiler may deterministically reassociate only `f32` constant, affine, and degree-at-most-two algebra. Runtime executes serialized row and term order exactly.
- When Milestone 2 selects the scheduled-program ABI, QP evaluators use that same ABI; sparse CSC patterns, direct coefficient scatter, prepared caller-owned solver arenas, and accepted-primal-only warm starts remain intact.

## Lowered residual program

### Lifetime and slot allocation

The semantic-DAG pass builds deterministic direct execution edges and view-resolved materialized edges, then computes reverse uses. It produces compiler-only lifetimes:

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

Inputs occupy fixed ABI slots. Allocate contiguous ranges with deterministic lowest-address first-fit reuse only after final uses include every epoch root, boundary consumer, declared output, nested binding, and tangent consumer. Free immediately after final use only when size and alignment permit reuse. Record peak live slots, maximum nested-call scratch, and exact alias scratch; required workspace is their bounded composition, not accumulated overlap.

Reshape, transpose, and concatenate are lowered views. They resolve source indices for a consumer and produce neither a frontier boundary nor a bytecode row unless an ABI binding requires materialization.

### Epochs and boundary batches

An epoch starts from materialized root slots `x` and retains reachable values as canonical sparse degree-two polynomials:

```text
y_r = c_r + sum_i L[r, i] * x_i + sum_{i <= j} Q[r, i, j] * x_i * x_j
```

Canonicalize quadratic monomials as `(min(left, right), max(left, right))`; deterministically combine duplicates and delete exact cancellations. This policy never changes nonlinear scalar semantics, comparisons, control flow, or ABI layout.

Extend an epoch through affine expressions, views, and bilinear products that remain degree at most two. Independent branches sharing roots remain in the same epoch. Close only before a nonlinear scalar operation, nested call, degree-three-or-higher product, required ABI materialization, or proven source/destination alias hazard. Reverse uses select only expressions required at the boundary. A boundary emits one of:

- a row-grouped bilinear materialization batch for selected values;
- a nonlinear batch that either reads stable slots or evaluates a retained expression locally before applying its scalar operation and writing a stable output slot; or
- a nested-call boundary with direct stable-output binding and one reusable packed input region only for non-addressable arguments.

### Nonlinear-stage retained operands

A `NonlinearStage` is the only nonlinear stage kind. Each operation has stable
output slots and reads either:

- a materialized stable slot; or
- a retained degree-at-most-two expression over the stage's stable root slots.

For a retained operand, the stage evaluates

```text
p(x) = c + sum_i L_i*x_i + sum_{i <= j} Q_ij*x_i*x_j
```

into a local scalar, applies the ordered scalar operation, and writes the
output slot once. It does not allocate a temporary workspace row or create a
separate stage. Push-forward evaluates the matching local tangent

```text
dp(x, dx) = sum_i L_i*dx_i
          + sum_{i <= j} Q_ij*(dx_i*x_j + x_i*dx_j)
```

before applying the scalar operation's derivative rule. Operations in one
nonlinear stage must be independent: an operation that consumes another
operation's output starts the next nonlinear stage.

The scheduler uses reverse uses to choose this operand form. Materialize an
expression when it has another stable-slot consumer, is required by an ABI or
call boundary, or materialization is cheaper than repeated evaluation; keep it
retained only for direct nonlinear consumption. The later bytecode schema
encodes these as nonlinear operand records, not as a distinct layer variant.

A generic dependency chain is ordered batches, never a falsely unordered generic batch. A destination may overwrite a source only when that source reaches final use in that operation/batch; otherwise close the epoch or reserve the smallest exact alias scratch range. Never copy the live workspace.

### Deferred bytecode encoding

Do not choose the archive record layout or ABI until Milestone 1 has a correct, measured Python residual program. The later encoder must serialize the completed stable-slot stages directly. It must preserve canonical bilinear term order, local destination accumulation, deterministic scalar order, and the Python stage alias contract; it must not introduce a second scheduler.

The later mapped executor must read archived records directly. Any owned conversion in `coker-bytecode::convert` remains tooling-only and must not enter runtime execution. Decide compact/wide operands and AoS/SoA layout together during that cutover.

## FK target and proof rule

The first engineering target is **fewer than 50 Python residual stages** for `kinematics.coker`, alternating algebraic bilinear materializations with nonlinear/identity boundaries where dependencies require them. A correct Python program below 100 stages is interim evidence, not completion of the performance target.

Do not claim that more than 50 stages are unavoidable from the current 116-layer artifact: it is produced by hybrid compact lowering and therefore is not a lower bound. A valid exception to the <50 target requires a checked Python-program-specific proof:

1. Collapse the tape's direct dependency DAG into maximal degree-at-most-two epoch regions without crossing nonlinear, call, degree-three, output, or proven alias boundaries.
2. Emit the remaining scalar nonlinear/call boundary dependency DAG in deterministic order.
3. Record a longest dependency chain of more than 50 boundary operations, including tape node IDs, operation names, and predecessor links.
4. Verify the chain against the Python residual `SparseNet`. Independent ready operations must not appear as serial chain members.

`ColoredBarrierDag` is useful scheduler diagnostics, but its present algebraic-component model is not this proof: it does not yet enforce degree closure or reflect emitted epoch-aware batches.

## Implementation sequence

### Milestone 1 — Python residual program

1. Evolve `SparseNet` into the residual-program data model: stable workspace metadata, `InputLayer`, ordered bilinear/nonlinear/call stages, and `OutputLayer`. A nonlinear stage has materialized-slot and retained-expression operand forms. Remove compact-layer-only `MemorySpec`/`BilinearWeights` assumptions from the execution representation.
2. Complete semantic DAG construction, view resolution, deterministic topological order, reverse uses, output/nested/tangent consumers, stable slot allocation, and exact scratch accounting.
3. Implement canonical `EpochValue` algebra and epoch partitioning. Materialize only selected boundary values; retain roots until their complete final use.
4. Implement Python primal and push-forward execution for every residual stage, including ordered generic chains, alias handling, and nested calls.
5. Compare Python residual primal and push-forward results against the existing NumPy and other available reference backends across arithmetic, views, calls, FK, Jacobian, and QP coefficient cases.
6. Regenerate the Python FK lowering, measure stage/row/workspace counts, and establish <50 stages, interim <100 stages, or the required >50-chain proof.

### Milestone 2 — bytecode and runtime, only after Milestone 1 passes

7. Make the Rust bytecode crate the single source of truth for the selected archive schema, validation rules, and serialization. Add a typed Rust module/program builder that constructs archived bytecode from stage records; expose that builder through `coker-python` bindings so Python lowering supplies typed fields directly.
8. Replace the current Python-dict → `json.dumps` → `compile_exported_json`/`serde_json` path. No JSON payload, exported graph model, or JSON parser participates in normal program or QP compilation. The Python binding invokes the typed builder, which validates and serializes the module.
9. Select magic/version and SoA layout from the settled Python stage model, then encode the Python residual program mechanically and migrate mapped primal, push-forward, nested-call, and QP coefficient execution together.
10. Delete compact-layer schemas, JSON export/compiler paths, validators, executors, conversions used at runtime, fixtures, and generated artifacts after the clean cutover.

## Acceptance gates

### Milestone 1 — Python correctness and FK evidence

- Python residual primal and push-forward execution match the NumPy backend and each installed/reference backend for arithmetic, mixed constants/workspace values, sparse bilinear expressions, reshape, transpose, concatenate, nested functions, outputs, and every push-forward operation category.
- FK and Jacobian primal/push-forward results match their reference evaluations. CROSS-style alias and nested-call scratch regressions pass.
- Python QP coefficient values and CSC ordering match the existing QP extraction/reference path; no QP runtime or solver migration is required at this milestone.
- The regenerated Python FK program has <50 stages, or a correct <100-stage interim result, or the checked >50 dependency-chain proof. Record stage count, bilinear rows, nonlinear operations (including expression-backed operands), terms, explicit alias copies, views materialized, and logical workspace size.
- The old mapped artifact's 116-layer result is baseline evidence only; it is not a Milestone 1 acceptance artifact.

### Milestone 2 — bytecode and runtime

- The Rust bytecode crate owns the only program schema and serializer. Python constructs modules through its typed binding builder; ordinary graph and QP compilation pass no JSON bytes or exported JSON model across the Python/Rust boundary.
- Builder validation rejects incomplete programs and archive mapping rejects the selected old version(s), bad alignment/payload offsets, out-of-bounds slots, malformed row/expression ranges, invalid tags, noncanonical terms, illegal aliases, and invalid nested bindings.
- Mapped primal/push-forward, QP coefficient scatter, nested calls, prepared solver reuse, and all Python-stage equivalence cases pass against the settled Python residual program.
- Runtime and QP execution preserve borrowed aligned mapped archives, caller-owned buffers, no post-construction allocation, and `coker-runtime --no-default-features`. Record the final artifact bytes; required primal/tangent workspace; mapped batch and row counts; output-clear/copy rows; workspace bytes touched; and target or faithful-target timing.

## Outstanding implementation checklist

### Milestone 1 — Python residual `SparseNet`

- [x] Remove dynamic source-function relowering from `GenericVectorLayer._eval_opaque_program`; nested host execution resolves a retained `FunctionTable` entry by function ID.
- [x] Build compiler-only semantic DAG direct/materialized edge projections and deterministic waves.
- [x] Define the residual `SparseNet` stage model: `InputMap`, stable-slot bilinear/nonlinear/call stages, retained-expression operands, `OutputMap`, direct Python primal/push-forward stage evaluation, and strict intra-stage dependency rejection.
- [ ] Make stable residual slots and exact first-fit reuse authoritative after final uses include retained roots, all boundaries, outputs, nested bindings, and tangent consumers.
- [ ] Replace `current_memory`/`next_memory` growth, dense `BilinearWeights`, retained-prefix identities, and constant-extension layers with residual `SparseNet` stages.
- [ ] Complete root replacement, selective boundary materialization, degree closure, ordered generic chains, and alias handling.
- [ ] Implement residual-stage Python primal and push-forward evaluation.
- [ ] Add backend-comparison regressions for the Python residual program, including FK/Jacobian and QP coefficient values.
- [ ] Add the degree-aware Python boundary-chain diagnostic; it must distinguish independent ready nodes from true serial nonlinear/call dependencies.
- [ ] Regenerate Python FK/Jacobian lowering and establish <50 stages, an interim <100 result, or the required >50-chain proof.

### Milestone 2 — deferred bytecode and ABI cutover

- [ ] After Milestone 1 passes, make the Rust bytecode crate own typed stage records, archive validation, serialization, version/magic, and the selected SoA layout.
- [ ] Expose a typed module/program builder through `coker-python`; migrate Python lowering and QP construction to direct builder calls.
- [ ] Delete `export_payload`, `compile_exported_graph`, `compile_exported_qp`, `compile_exported_json`, `compile_exported_qp_json`, the exported JSON model, and normal-path `json.dumps`/`serde_json` parsing.
- [ ] Migrate mapped primal/push-forward, nested calls, QP evaluator metadata, direct CSC scatter, and prepared solver reuse to the selected ABI; verify them against the Python residual program.
- [ ] Delete compact-layer schemas, validators, executors, conversions used at runtime, fixtures, and generated artifacts after the clean cutover.

## Completion condition

Milestone 1 completes when the Python residual `SparseNet` is the only normal lowering representation, matches reference backend primal/push-forward behavior, and meets the FK stage-count gate or supplies the required dependency-chain proof. Only then does Milestone 2 select and implement the bytecode ABI and mapped executor. Final completion requires every runtime program to execute bytecode mechanically encoded from that proven Python program; compact layers, relocation identities, accumulated overlap scratch, legacy paths, and runtime owned decode are deleted, and Coker, Hexapod, push-forward, QP, mapped-bytecode, caller-owned-buffer, allocation-free, and no-std gates pass.
