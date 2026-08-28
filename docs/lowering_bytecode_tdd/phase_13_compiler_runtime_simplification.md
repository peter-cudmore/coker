# Phase 13 — Compiler and runtime simplification

## Purpose

Use profiling and hotspot analysis to reduce compiler/runtime code size, branching, allocation, and indirection without weakening the mapped-archive, caller-owned-buffer, or no-std contracts.

This phase is deliberately deferred. It is a list of candidate reductions, not an implementation authorization. Profile representative ordinary and QP workloads first; retain only candidates justified by measured cost or demonstrated maintenance burden.

## Required evidence before selecting work

For every selected candidate, record:

- representative input model and target profile;
- baseline wall time, allocation count, peak workspace, archive bytes, and solver preparation cost where applicable;
- hot symbols and their inclusive/exclusive samples;
- expected removed branches, allocations, types, and lines of code;
- a focused behavioral and allocation/ownership acceptance test.

Do not replace direct data access with a new abstraction unless it removes a measured cost or establishes an ownership invariant.

## Candidate simplifications

### 1. Delete the dual ordinary-execution stack

The columnar archive executor currently lives in `coker-bytecode::archive_execute*` and is called directly by `coker-compiler-python`, while `coker-runtime::static_module` still owns the older `BytecodeModule`/`MappedModule` layer runtime.

**Candidate cutover:** move archive execution functionality into `coker-runtime`, route PyO3 through that runtime API, then delete the legacy layer-based bytecode model and executor.

**Expected benefit:** one ordinary execution path, one mapped-program binding route, fewer execution wrappers and compatibility branches. This is the primary structural simplification candidate.

**Guard:** archive data remains data-only in `coker-bytecode`; runtime owns execution behavior. Preserve direct mapped access, no allocation after construction, and caller-owned workspaces.

### 2. Reduce archive ownership to three explicit types

Target a compiler-only `OwnedModel`, a borrowed runtime `ArchivedModel<'a>`, and a host-only aligned `CompiledArtifact`. Keep headers as archive data tables, not behavior-heavy wrappers.

Delete or internalize artifact/view/module/program wrappers that only forward a field or preserve no independent invariant.

### 3. Use explicit free-function runtime entry points

Prefer direct functions over execution-object hierarchies:

```rust
execute(model, function_id, inputs, workspace, outputs)
push_forward(model, function_id, inputs, tangents, workspace, tangent_workspace, outputs, tangent_outputs)
```

A small buffer aggregate is permitted only if it removes a repeated parameter group without hiding buffer ownership. Do not introduce execution traits or dynamic dispatch.

### 4. Split runtime modes before hot loops

Route ordinary, tangent, and QP execution once at the executable boundary. Each hot loop should have a fixed buffer contract and must not carry optional tangent, frame, QP, or parameter-mode state that it does not use.

Profile branch samples before restructuring. Preserve phase-kind dispatch once per phase.

### 5. Unify QP solver lifecycle

`BoundMappedQpProgram` and `PreparedQpProgram` have overlapping execution semantics. Evaluate replacing them with one prepared solver handle bound to a caller-provided arena, plus one internal execution path.

Keep a detached/foreign-function construction adapter only if its lifetime contract cannot be represented by the same handle.

### 6. Recast symbolic QP lowering as linear data passes

Keep QP lowering as concrete transformations:

```text
QpSourceDeclaration -> QpStructure -> QpSparsity -> QpModel
```

Each pass accepts immutable data and returns one concrete value. Avoid a shared mutable compiler context and avoid pass traits.

### 7. Compact decision-support propagation only if hot

Default to the current deterministic `BTreeSet` representation. Replace it only if profiling identifies support propagation as material and a compact implementation reduces the measured cost.

Any replacement must preserve both independently propagated relations:

```text
linear_offsets: Vec<u32>
linear_indices: Vec<u16>
quadratic_offsets: Vec<u32>
quadratic_pairs: Vec<(u16, u16)>
```

Residual sparsity derives from the linear relation; Hessian sparsity derives from the quadratic-pair relation. The additional merge machinery must outperform the existing ordered sets on representative QP models. Do not add a generic sparse-set abstraction.

### 8. Keep source DAG immutable; localize synthesis

Keep `TypedDag` as the immutable source representation. Place specialization, cloning, scalar append, output append, and shape synthesis in concrete compiler-only free functions such as `specialize(&TypedDag, ...)` and `append_output(&mut TypedDag, ...)`.

Do not add another builder lifecycle type: ingestion already owns `typed_dag::ingest::Builder`, and a second builder would increase abstraction count. Do not make the DAG transformation mechanism pluggable; the operation set is closed.

### 9. Normalize static mappings once

Measure archive-size impact before changing this. If later compiler passes repeatedly interpret reshape/permutation/slice/concatenate mappings, normalize ingestion output to `Identity` or a flat `Gather { indices }` relation.

The emitter and support analysis should then consume one simple mapping form rather than branch on tensor operation kinds.

### 10. Restrict graph-analysis outputs to emitter needs

`analysis` may retain reverse CSR, reachability scratch, and scheduling internals privately. Export only deterministic order, node classification, and workspace plan unless another pass has a demonstrated need for additional metadata.

Do not mechanically split cohesive graph analysis by line count.

### 11. Assign validation to one boundary each

Validation ownership should be:

| Boundary | Validation |
|---|---|
| Typed-DAG ingestion | capacities, indices, source order, mapping completeness |
| Compiler finalization | archive ranges, references, call graph, QP-plan consistency |
| Runtime binding | header, version, alignment, archive access, table ranges |
| QP preparation | arena size/alignment and executable identity |
| Execution | input/output/workspace sizes and numeric update validity |

Remove wrapper-local revalidation that duplicates a prior boundary without defending a distinct safety property.

## Non-goals

- No changes before profiling establishes a target.
- No owned deserialization or copies in runtime execution.
- No global allocator requirement on the no-std path.
- No new traits, plug-in systems, or compatibility aliases merely to support this cleanup.
- No behavior or numerical-tolerance changes as a simplification substitute.

## Exit criteria

For every implemented candidate:

1. measured target cost or documented maintenance reduction;
2. no regression in mapped alignment, buffer ownership, or no-std allocation constraints;
3. focused behavioral tests and full affected compiler/runtime gates pass;
4. before/after line count, branch/allocator evidence, and profile result are recorded in the implementation PR.
