# Phase 13 — Compiler and runtime simplification

## Purpose

Make lowering and runtime ownership locally auditable, then unblock the
Phase 12 hexapod compaction experiment. This is an implementation phase, not
an unranked candidate list.

Phase 12 cannot provide a valid runtime profile: ordinary lowering exceeds
`ARCHIVE_MAX_PHASES` before an archive is emitted. The structural work below
is therefore justified by the demonstrated maintenance and correctness blocker.
After it restores artifact emission, Phase 12 measurements select any further
performance work. It does not authorize raising `ARCHIVE_MAX_PHASES`.

## Invariants

- Runtime execution borrows aligned mapped archive data; it never decodes into
  an owned model or copies archive bytes.
- Ordinary, tangent, and QP execution allocate nothing after their caller-owned
  buffers and QP arena have been constructed.
- `coker-bytecode` contains archive data, archive validation, and compiler-side
  serialization only. `coker-runtime` owns execution behavior.
- Every QP coefficient slice is materialized in the evaluator workspace before
  a solver reads it. The evaluator's declared outputs and the contiguous
  coefficient-output region are one explicit, tested contract.
- Deterministic lowering order, numerical semantics, sparse patterns, mapped
  alignment, and caller buffer ownership do not change.
- No execution traits, dynamic dispatch, compatibility aliases, second DAG
  builder, or generic sparse-set abstraction.

## Checklist

### 13.1 — Establish regressions and measurement boundaries

1. Add the smallest ordinary mapped-execution regression and the smallest QP
   coefficient-evaluator regression that reproduce the failed compaction
   condition: values needed by the QP coefficient slices may be workspace
   values beyond the ordinary declared-function outputs.
2. The QP regression must solve the known case that previously regressed from
   `[3.0, -1.0]` to `[0.27272728, -0.09090908]`; it must fail when the required
   mapping materialization is removed and pass only when the solver sees the
   intended contiguous coefficient values.
3. Add release-mode compiler counters for DAG nodes, scheduled phases by
   B/N/Case/Call/residual kind, B terms, N rows, gathers, workspace spans,
   evaluator output region, archive bytes, and per-stage compile time. The
   counters are compiler diagnostics, not runtime behavior.
4. Record the hexapod model revision, command, host, baseline compiler failure,
   and bounded-table diagnostics in
   `docs/geometrically_aware_rigid_body_operations.md`. Artifact emission,
   runtime timing, and compaction are deferred to that follow-up.
5. Isolate one hexapod-leg forward-kinematics model and record its aggregate
   B/N/Case phase counts, table sizes, and nonlinear-chain diagnosis. Derive
   enough of the implemented quaternion path to establish whether the
   one-to-two bilinear / one nonlinear-layer estimate is plausible. Exact DAG
   dependency-level tracing and Case-source attribution are deferred to
   `docs/geometrically_aware_rigid_body_operations.md`; do not change lowering
   policy in this pull request.

**Observed one-leg trace (2026-08-28):** three joint inputs and one
three-element foot translation emitted one program/function, 596 phase
headers (346 B, 123 N, 113 gather, 14 Case), 826 bilinear terms, 173 N rows,
2,806 gather rows, and a 2,380-scalar workspace. The one-to-two bilinear /
one-N-layer hypothesis is false for the implemented quaternion-to-axis-angle
path. Exact Case attribution is deferred to the geometry-aware follow-up.

### 13.2 — Make the compiler pipeline concrete and observable

1. Retain `TypedDag` as immutable input. Keep ingestion in
   `typed_dag::ingest::Builder`; use concrete compiler-only free functions for
   specialization, scalar append, output append, and shape synthesis.
2. Make the ordinary lowering sequence explicit and inspectable:
   `TypedDag -> analysis -> workspace plan -> emitted OwnedModel ->
   finalization -> CompiledArtifact`. Each stage has one input/output value,
   documented ownership, and its counters.
3. Make QP lowering equally explicit:
   `QpSourceDeclaration -> QpStructure -> QpSparsity -> QpModel`. Each pass
   receives immutable input and returns one concrete value; no mutable
   compiler context or pass trait.
4. Define a single evaluator-output layout produced by QP lowering. It must
   identify declared ordinary outputs, every `P/A/q/l/u/r` coefficient slice,
   their workspace spans, and the required contiguous solver-input region.
   Validate this at compiler finalization and use it as the sole source of
   truth for runtime coefficient scattering.
5. Only after the regressions in 13.1 pass, retry N-header coalescing and
   static-mapping elimination. Preserve mapping materialization whenever the
   evaluator-output layout proves the value is a QP coefficient dependency.

### 13.3 — Collapse archive ownership to data, artifact, and mapped view

1. Retain exactly these public archive ownership roles:
   `OwnedModel` for compiler-owned construction, `CompiledArtifact` for
   host-only aligned serialized storage, and `ArchivedModel<'a>` for a
   validated borrowed mapped view.
2. Delete or internalize wrappers that merely forward those types. Headers are
   archived data tables, not execution objects.
3. Remove the legacy owned `BytecodeModule`/`Program`/`Layer` runtime route,
   owned decode fallback, and layer executor once every caller has migrated.
   Compiler/tooling-only construction must end at `OwnedModel` finalization;
   it is never reachable from runtime execution.
4. Add ownership tests proving unaligned mappings are rejected, aligned mapped
   views borrow the caller bytes, and executing either ordinary or QP archives
   creates no owned decoded model and performs no allocation.

### 13.4 — Establish one mapped ordinary runtime API

1. Move archive phase execution from `coker-bytecode::archive_execute*` into
   `coker-runtime`. Keep archive layout traversal direct over
   `ArchivedModel<'_>`; `coker-bytecode` remains data-only.
2. Expose concrete free-function entry points in `coker-runtime` for ordinary
   execution and push-forward. They take a mapped model/function identifier and
   caller-owned input, output, workspace, tangent-workspace, and frame buffers.
   A compact buffer aggregate is allowed only when it names this repeated
   caller-owned group without obscuring individual ownership.
3. Validate mapped binding once. Route ordinary and tangent mode once before
   their phase loops; do not carry optional tangent, QP, frame, or
   parameter-mode state through unrelated hot loops. Retain phase-kind dispatch
   once per phase.
4. Migrate both PyO3 bindings: `coker-compiler-python` must stop calling
   `coker-bytecode::archive_execute*`, and `coker-python` must stop constructing
   the legacy owned `Module` stack. Delete both old ordinary routes, their
   wrappers, and compatibility branches in the same cutover.
5. Verify primal and forward-JVP parity through each PyO3 surface, mapped
   alignment rejection, repeated no-allocation execution, and nested-call
   frame behavior.

### 13.5 — Unify QP preparation and execution in host and embedded builds

1. Replace the overlapping `BoundMappedQpProgram` and `PreparedQpProgram`
   public lifecycles with one prepared mapped-QP handle that binds exactly one
   validated QP executable and one caller-provided arena for its lifetime.
2. The handle owns no arena storage and exposes one execution path: evaluate
   coefficients into caller buffers, scatter directly into caller-provided
   solver update buffers, update the prebuilt solver, solve, and write
   caller-provided outputs. Backend-specific host/embedded mechanics remain
   private concrete implementations, not public traits.
3. Preserve transactional recovery after a failed numeric update: a rejected
   update returns its defined error without leaving stale solver pointers or
   partially committed solver coefficients. A subsequent valid update on the
   same handle must solve successfully. Any permanent-invalidation policy is a
   numerical-behavior change and requires separate authorization and regression
   coverage.
4. Apply the same public lifetime and buffer contract under host and embedded
   configurations. Test arena size/alignment rejection, executable identity,
   warm start, repeated solve reuse, update failure invalidation, and
   allocation-free execution in each available configuration.
5. Delete detached/foreign-function adapters unless a real call site proves
   that the unified handle cannot express its lifetime. If retained, make it a
   private adapter over the same handle and test its lifetime invariant.

### 13.6 — Remove repeated interpretation and duplicate validation

1. Measure archive-size and compile-time effects of normalizing
   reshape/permutation/slice/concatenate mappings at ingestion. If it wins,
   lower them once to `Identity` or flat `Gather { indices }`; otherwise retain
   the existing compact representation and record why.
2. Keep reverse CSR, reachability scratch, and scheduling internals private to
   graph analysis. Export only deterministic schedule, node classification, and
   workspace plan unless an identified downstream pass needs more.
3. Keep deterministic `BTreeSet` support propagation unless phase counters or
   samples show it is material. A replacement must separately preserve linear
   offsets/indices and quadratic offsets/pairs and outperform the ordered sets
   on representative QP models before landing.
4. Assign validation once per boundary: ingestion validates capacities,
   indices, source order, and mapping completeness; finalization validates
   archive references, ranges, call graph, and QP layout; binding validates
   header/version/alignment/archive access; QP preparation validates arena and
   executable identity; execution validates buffer sizes and numeric updates.
   Remove only duplicate wrapper checks that protect no distinct safety
   property.

### 13.7 — Verify cutover

1. For each deletion or simplification, record before/after public types,
   execution branches, owned runtime allocations, and relevant line count.
2. Run focused compiler, archive, ordinary runtime, push-forward, QP host, QP
   embedded, and both PyO3 behavioral tests affected by the cutover. Preserve
   the regressions from 13.1 as durable contract tests.
3. Verify compiler diagnostics on both a successful mapped artifact and an
   archive-finalization failure. The failure path must report bounded-table
   lengths and limits. Do not run Phase 12 profiling or compaction in this
   pull request.

## Exit criteria

1. There is one direct mapped ordinary executor in `coker-runtime`, used by
   both PyO3 crates; no legacy owned/layer execution path remains.
2. Archive ownership is the explicit three-role model, and runtime execution
   cannot deserialize or copy mapped bytecode.
3. QP has one prepared-handle lifecycle with matching host and embedded
   caller-arena semantics; coefficient-output ownership and materialization are
   explicit and regression-tested.
4. Lowering stages, counters, output layout, and validation boundaries make
   bytecode growth and buffer lifetimes directly attributable.
5. The hexapod geometry/lowering expansion, its bounded-table diagnostics, and
   the deferred Phase 12 measurement plan are recorded in
   `docs/geometrically_aware_rigid_body_operations.md`. The geometry-aware
   follow-up, not this Phase 13 cleanup, owns artifact emission and compaction.
