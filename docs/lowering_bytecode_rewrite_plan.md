# Lowering and bytecode rewrite plan

## Status and scope

This plan replaces the Coker ordinary-tape lowering path and bytecode interpreter. The working branch is `lowering-bytecode-rewrite-plan`, created from `main`.

The rewrite is a clean bytecode-format cutover. Legacy bytecode artifacts require recompilation; the runtime will not retain a dual-format reader or writer.

QP compilation is required work, but is deliberately the final migration phase. Ordinary tape lowering, bytecode, runtime execution, Python bindings, and their verification must be complete before QP extraction and lowering move from Python into Rust.

### Deferred residual execution

Residual B execution and workspace-span reuse are deferred until Phase 12 is complete. Until then, ordinary lowering MUST use append-only logical workspace spans and ordinary overwrite B phases; it MUST NOT introduce residual ownership transfer.

## Goals

1. Reduce numerical bytecode execution to two phase kinds:

   \[
   x_{\mathrm{out}} \mathrel{+}= B(h(x_{\mathrm{in}}), h(x_{\mathrm{in}}))
   \]

   and

   \[
   x_{\mathrm{out}} = N(x_{\mathrm{in}}, x_{\mathrm{in}}),
   \]

   where \(h(x) = [x, 1]\), \(B\) is sparse bilinear, and \(N\) is an explicitly encoded nonlinear operator.

2. Canonicalize bilinear maps as sparse output-by-monomial matrices and support compile-time-selectable reduction/factorization algorithms.

3. Partition tape DAGs into dependency-aware B and N phases while minimizing B phases without moving polynomial work into N.

4. Move all ordinary compilation logic from Python to Rust. Python remains the symbolic front end and invokes Rust through PyO3 to compile a live tape directly.

5. Use compact struct-of-arrays bytecode and compiler storage suitable for direct mapped execution, cache locality, and later SIMD kernels.

6. Preserve primal and forward-mode directional-derivative execution.

## Non-negotiable runtime constraints

- Runtime execution is allocation-free after construction and does not require a global allocator.
- Bytecode remains a self-contained, rkyv-archived mapped artifact.
- Runtime APIs borrow validated archived views from aligned mapped bytes; they do not deserialize or copy bytecode into owned buffers.
- The embedding application owns workspace, tangent workspace, outputs, solver state, and all scratch memory.
- Runtime bytecode scalar storage and execution use `f32`.
- Compiler algebra and reduction use deterministic `f64` until final `f32` quantization.

### Numerical behavior

Runtime primal and tangent kernels initially preserve the current runtime
floating-point implementation. The compiler records deterministic source/term
order and performs reduction in deterministic `f64`; this controls artifact
construction, but does not make Python/NumPy and target math libraries
bitwise-equivalent.

Phase 01 records current desktop finite-input primal and tangent behavior
against the Python semantic oracle using explicitly registered tolerances.
Transcendental results and special floating-point values are recorded as
desktop observations rather than assumed cross-platform bitwise parity.
Fast-math, FMA, flush-to-zero, SIMD numerical semantics, and target-device
profiling are deliberately deferred until the later SIMD/optimization work.

All tuning values live in one Rust `LoweringPolicy` constructor, rather than
being distributed across passes. Initial defaults are deliberately
conservative and benchmark-visible:

```text
max_exact_scheduler_states = 50_000
max_exact_ready_frontier = 16
residual_max_terms_per_row = 1
reduction_f64_relative_tolerance = 256 * f64::EPSILON
reduction_f64_absolute_tolerance = 64 * f64::EPSILON
quantized_f32_relative_tolerance = 128 * f32::EPSILON
quantized_f32_absolute_tolerance = 32 * f32::EPSILON
```

The build API accepts an explicit policy for experiments. Defaults are used
only when no policy is supplied.

### Archive finalization

The compiler validates before emitting or returning an archive. It validates
the owned bytecode model, serializes it, validates the resulting aligned
archived view, and only then finalizes the artifact. A failed validation returns
no archive.

Mapped consumers validate header, alignment, length, and table bounds once
when binding an artifact. Prepared runtime execution trusts that bound view and
does not repeat structural archive validation on numerical calls.

## Compiler boundary

Rust traverses the live Python tape through PyO3. Python constructs the symbolic graph but does not lower it into `SparseNet`, `BilinearWeights`, generic layers, JSON, or bytecode.

The PyO3 adapter extracts a compact Rust `TypedDag`:

- operation kind per node;
- flattened operand spans;
- constant-table references;
- flat shape/dimension metadata;
- labelled input and output node indices;
- nested-function identities for retained calls.

The rest of the compiler is pure Rust and operates only on `TypedDag`. This keeps PyO3 as a narrow front-end adapter, makes compiler passes independently testable, and removes Python from the compilation implementation.

### Maturin compiler builder API

The Python backend imports a dedicated `coker_compiler` maturin package. It
constructs a builder with capacities known from the tape, then pushes the tape
in stable node order:

```python
builder = coker_compiler.Builder(
    node_capacity=len(tape.nodes),
    operand_capacity=estimated_operand_count,
    constant_capacity=estimated_constant_count,
    input_capacity=len(tape.input_indicies),
    output_capacity=len(function.output),
)
builder.push_constant(...)
builder.push_node(...)
artifact = builder.build(policy=...)  # Phase 08: CompiledArtifact owner, not bytes
```

The builder owns Rust vectors sized from those capacities. Python injects
constants and node records but performs no lowering or graph optimization.
`push_node` enforces the declared sequence and operand bounds immediately.
Phase 04 first provides the owned-model finalizer and aligned archive owner
independently of production lowering. Until the executable ordinary subset is
complete, `build` continues to produce the existing artifact. Phase 08
switches it to the pure-Rust graph/compiler pipeline and returns a
`CompiledArtifact` backed by compiler-created aligned Rust storage. Runtime
programs borrow that owner; the Python artifact object retains the storage for
at least as long as any execution handle.

`CompiledArtifact.to_bytes()` exists only for persistence or transport. Bytes
returned to Python are not an execution backing store. Loading a persisted
artifact for execution uses `CompiledArtifact.load_path()`, backed by a
page-aligned read-only mapping, or an explicitly aligned owner supplied by a
host integration. The loader validates the header and payload alignment and
rejects an unaligned mapping rather than copying it to align it. No Python
references survive past a `push_*` call.

## Canonical execution algebra

### Homogeneous coordinates

The homogeneous coordinate is implicit. It is encoded by a reserved phase-local index and is never stored in workspace.

A bilinear phase represents a sparse degree-at-most-two polynomial map:

\[
y_i = \sum_{(p,q) \in M} S_{i,(p,q)} h_p h_q.
\]

Monomials use canonical ordered pairs:

\[
(p,q) = (\min(p,q), \max(p,q)).
\]

Constants use \((1,1)\), and linear terms use \((i,1)\).

### B phase eligibility

B phases absorb operations whose result remains a degree-at-most-two polynomial in that phase's input coordinates:

- constants;
- identity, reshape, and concatenation;
- addition, subtraction, and negation;
- multiplication, dot product, and matrix multiplication when degree permits;
- nonnegative compile-time integer powers, decomposed into B multiplication stages;
- multiplication or division by compile-time constants;
- linear transforms.

A polynomial degree overflow is **not** an N operation. It flushes the active B frame and starts a new B phase, in which the previously materialized operands are input coordinates again.

For example, `(a*b)*(c*d)` lowers to:

1. B phase: `u = a*b`, `v = c*d`;
2. B phase: `w = u*v`;
3. zero N operations.

### N phase eligibility

N phases contain only operations that cannot be represented by the active degree-two B coordinate system:

- transcendentals;
- variable division;
- variable, fractional, or negative powers;
- comparisons;
- `case`;
- retained nested-function calls.

The initial N opcode domain preserves existing scalar-operation semantics. It does not support opaque Python callbacks or host-only operations in an embedded artifact.

## Sparse bilinear reduction

### Matrix representation

For each candidate B phase, form a sparse matrix \(S\) whose rows are outputs and whose columns are canonical homogeneous monomials.

Do not materialize a dense matrix. Store and transform sparse rows/columns with deterministic ordering.

### Meaning of minimal equivalent B

Row reduction changes output coordinates unless the original outputs can be reconstructed. The factorized form is:

\[
S = R T,
\qquad z = Tm(h(x)),
\qquad y = Rz.
\]

`z` is a reduced bilinear basis and `R` reconstructs original coordinates.

The compiler must not eagerly materialize `Rz`:

- propagate coordinate maps through later B lowering;
- compose reconstruction into a later B matrix where possible;
- materialize it only at an N boundary, a retained call boundary, or a labelled external output.

This prevents a rank factorization from mechanically adding a B phase. If a standalone reconstruction B phase is unavoidable, compare it with the direct sparse form under the cost model.

### Pluggable reducers

Reduction is a compile-time strategy, not a bytecode feature. The compiler provides a common reducer interface with factor matrices, rank, structural nnz, deterministic residual evidence, and estimated cost.

Initial implementations:

1. Direct sparse reduction: canonical monomial merging only.
2. Deterministic rank factorization: sparse-aware elimination with a fixed pivot order and tolerance.

Future strategies may include unique-monomial compaction, fill-aware factorization, and target-cost-guided optimization. They compete under the same validation and cost rules.

### Numerical contract

- Use fixed monomial order, pivot tie-breaks, tolerance, and zero-pruning rules.
- Verify pre-serialization residual:

  \[
  \lVert S - RT \rVert_\infty
  \leq \epsilon_{\mathrm{reduce}} \max(1, \lVert S \rVert_\infty).
  \]

- Quantize factors to `f32`, then repeat the residual check on the quantized factors.
- Reject failed candidates and fall back to a valid less-reduced representation.

## DAG metadata and scheduling

### Forward iterative graph passes

The compiler prefers forward iterative algorithms. It does not use recursive graph traversal; tail recursion is permitted only where it is demonstrably the most efficient compiled form.

The tape is guaranteed to be a DAG with labelled inputs and outputs. The extraction pass and subsequent iterative graph algorithms produce:

1. Stable Kahn topological ranks.
2. Reverse adjacency in CSR form: consumer offsets and consumer-node arrays.
3. Backward output-reachability markings.
4. Forward degree/constant/shape propagation.
5. Consumer counts, critical-path metadata, last-use bounds, and materialization-boundary metadata.
6. A validated, iterative nested-call graph. Recursive call cycles are rejected for the initial embedded runtime.

Stable source order is retained only as a deterministic tie-breaker.

### Dependency-aware minimum-phase scheduling

Do not greedily schedule in source order. Independent B work may be deferred until it can share a later B phase unlocked by an N result.

For example:

```text
b1 = B(inputs)  # independent
n1 = N(input)
b2 = B(n1)
```

The scheduler chooses `n1`, then `{b1, b2}` in one B phase, rather than `b1`, `n1`, `b2` in two B phases.

Optimization is lexicographic:

1. B/N phase boundaries, particularly B phase count;
2. peak live caller workspace;
3. stored/evaluated B nnz and N operation count;
4. stable source order.

Implementation:

- build candidates from dependency-ready frontiers;
- execute ready N closures before committing a B phase when they unlock joinable B work;
- pack all dependency-valid B work into a B phase, including deferred independent work;
- use bounded exact search or dynamic programming for small/medium DAGs;
- use deterministic look-ahead list scheduling after the configured state bound.

The fallback is valid and deterministic but must not claim global optimality.

## Pruning

Prune before scheduling and after factorization.

### Graph pruning

- Remove nodes not backward-reachable from labelled outputs.
- Remove functions not reachable from the entry function through retained calls.
- Constant-fold reachable all-constant subgraphs.
- Eliminate identity/reshape/concatenation nodes when represented by direct bindings or views.
- Preserve labelled input/output ABI order. An unused input remains declared but causes no computation or workspace traffic.

### Phase and factorization pruning

- Remove dead B rows and N destinations.
- Merge duplicate monomials and remove zero terms.
- Remove unused factor-basis rows, zero factor rows, and zero reconstruction columns.
- Reclaim workspace spans after their scheduled last use.
- Prune empty phases.
- Coalesce adjacent compatible phases only if B-frame semantics and the cost model permit it.

## Archived bytecode

Replace `Layer`, `BilinearLayer`, `GenericLayer`, `RowOp`, `SparseTensor`, and AoS `SparseEntry` execution storage with directly mapped columnar tables.

```text
Program
  workspace_size: u32
  input_specs: [InputSpec]
  output_specs: [OutputSpec]
  phase_headers: [PhaseHeader]
OverwriteBilinearPhase
  input_base: u32
  output_base: u32
  input_len: u16
  output_len: u16
  output_term_start: [u32; output_len + 1]
  term_left:  [u16]
  term_right: [u16]
  term_coeff: [f32]

ResidualBilinearPhase
  workspace_base: u32
  length: u16
  update_order: [u16; length]
  output_term_start: [u32; length + 1]
  term_left:  [u16]
  term_right: [u16]
  term_coeff: [f32]

GatherPhase
  source_workspace_indices: [u32; length]
  output_base: u32
  length: u16

NonlinearPhase
  input_base: u32
  input_len: u16
  output_base: u32
  output_len: u16
  frame_len: u16
  output_frame_start: u16
  opcode: [u8]
  dst:    [u16]
  arg0:   [u16]
  arg1:   [u16]
  arg2:   [u16]

CallPhase
  callee_id: u16
  input_binding tables
  output_binding tables
```

Phase-local `u16` indices bound a phase width; the compiler splits larger phases. Global workspace bases and archive table offsets use `u32`.

B terms are sorted by output row for sequential output sweeps. N operations remain topological, while contiguous same-op runs provide a future SIMD dispatch boundary without changing the archive format.

### Phase-boundary gathers and ping-pong banks

Ordinary execution uses two disjoint absolute caller-workspace bank ranges.
For every scheduled phase, the compiler gathers live-through values needed by
the next phase into its read bank, then emits B, N, Case, Call, and ordinary
gather writes into the opposite bank. The following phase swaps bank roles.

The compiler resolves static mappings before final bank placement. It tracks
the live frontier at each phase boundary and allocates compact offsets within
each bank; static reshape, permutation, slice, concatenate, repeated index,
and broadcast mappings become source-to-destination gathers when they cannot
be folded into their consuming operation. A phase's source and destination
ranges are concrete `u32` workspace offsets in the existing archive format.
No new runtime frame fields are required.

Ordinary phases never write their read bank. A gather copies only into the
opposite bank; it is not a post-execution restore of a temporary output.
Primal and tangent execution use identical bank offsets and mappings in their
corresponding caller-provided workspaces.

The scheduler's ready-frontier property means nodes in an N phase have no
same-phase producer dependency. N rows therefore read only their phase input
bank and write only the opposite output bank; no N-local frame, temporary
slot stream, or output copy is required.

### Separate overwrite and residual layers

Overwrite and residual bilinear execution are separate bytecode layer types and
separate runtime kernels. They are not a per-layer mode flag.

An `OverwriteBilinearPhase` computes a complete output span from a disjoint
input span. It starts each output-row accumulator at zero and writes the row
once. This remains the default B representation.

A `ResidualBilinearPhase` is a true in-place update:

\[
x_i \mathrel{+}= r_i(h(x)).
\]

For every same-span candidate, the compiler canonicalizes the complete map
as \(S = I + R\) and stores \(R = S - I\). The identity contribution is
implicit; `term_*` tables encode every residual coefficient, including a
diagonal residual such as \(0.01x_i\) for \(S_{ii} = 1.01\).

The compiler may emit it only when all of the following hold:

1. Input and output are exactly the same workspace span.
2. The residual \(R\) is sparse: initially, no more than one residual term per
   row on average, counting diagonal residual terms.
3. The row-dependency graph of \(R\) is acyclic after self-dependencies are
   removed. A directed edge `i -> j` means row `i` reads the old value of row
   `j`; the compiler stores a topological `update_order` so every read occurs
   before its source row is updated.
4. The residual form wins the established cost comparison against the
   overwrite form, including workspace, archive bytes, and target-cycle
   estimates.

This is the conservative initial meaning of \"close to identity and reducible
to triangular form\": \(I\) is the fixed implicit baseline, and a permutation
of the non-self dependency graph of \(R\) is triangular. A map such as
\(y = 1.01x\) is eligible because \(R = 0.01x\) is sparse and has only a safe
self-dependency. Cyclic off-diagonal dependencies, dense residuals, and any
ambiguity in the dependency proof remain overwrite layers.

Residual push-forward follows the same order and accumulates the product-rule
derivative into the tangent span. No residual kernel may allocate or take an
unproved overlapping source/destination span.

## Runtime

Runtime kernels borrow archived table slices and execute through direct iterative loops:

- `execute_overwrite_bilinear_phase`;
- `push_forward_overwrite_bilinear_phase`;
- `execute_residual_bilinear_phase`;
- `push_forward_residual_bilinear_phase`;
- `execute_nonlinear_phase`;
- `push_forward_nonlinear_phase`;
- `execute_call_phase`;
- `push_forward_call_phase`.

The B tangent rule is:

\[
\dot y_i = \sum_{p,q} S_{i,pq}(\dot h_p h_q + h_p \dot h_q).
\]

No execute or push-forward path allocates, copies mapped bytecode, constructs solver state, or uses per-operation dynamic dispatch.

### Iterative retained-call execution

The compiler computes each function's maximum acyclic call depth and aggregate
workspace requirements over the retained call graph. It rejects recursive
cycles, call depth above the configured artifact limit, and a call stack that
does not fit caller-provided storage.

Runtime calls use an explicit caller-provided `CallFrame` stack, never Rust or
C recursion. A frame records the callee function id, next phase index, input
binding progress, output bindings, and its assigned primal/tangent workspace
subranges. `CallPhase` pushes a prepared callee frame; function completion
writes declared outputs, pops the frame, and resumes the caller's next phase.

The archive declares required frame-stack capacity and cumulative workspace.
Binding validates supplied frame storage once. Primal and push-forward use the
same iterative state machine and frame ordering; the latter reserves matching
tangent workspace subranges.

## Migration order

### 1. Rust ordinary-tape compiler core

Create pure-Rust modules for:

- typed DAG representation and validation;
- graph metadata and pruning;
- polynomial analysis;
- phase partitioning;
- canonical monomials;
- reducers;
- workspace scheduling;
- archived bytecode emission;
- structural/algebraic validation.

### 2. PyO3 ordinary compilation entry point

Replace Python lowering and JSON export with tape traversal into the Rust compiler. Return compiled archive bytes to Python and preserve runtime loading/execution from Python.

### 3. Bytecode and runtime cutover

Version-bump the bytecode. Implement mapped archive validation and execution. Remove ordinary runtime dependence on legacy layer structures.

### 4. Ordinary-path cleanup

After parity and mapped-runtime verification:

- remove Python `SparseNet` lowering;
- remove `BilinearWeights` compile-time lowering metadata;
- remove ordinary JSON graph compilation;
- remove old layer/row-op/sparse-entry bytecode paths;
- retain Python reference evaluation only as a test oracle.

### 5. QP migration — final phase

Only after all ordinary-tape work above is complete:

- move QP coefficient extraction from Python to Rust;
- make it consume the Rust typed/lowered representation;
- derive coefficient evaluators, CSC patterns, output slices, and embedded-plan inputs in Rust;
- preserve mapped QP artifacts and caller-provided OSQP buffers/workspaces;
- remove the temporary isolated Python QP extraction adapter.

QP is not optional; it is deferred so the ordinary lowering architecture is stable before it becomes the basis for the QP path.

## Verification gates

1. Python tape to Rust DAG extraction preserves operations, dimensions, constants, labels, sharing, and calls.
2. Direct and factorized B forms match within the post-quantization `f32` residual contract.
3. Polynomial degree overflow creates B-to-B boundaries, never N operations.
4. Dependency-aware scheduling eliminates avoidable B phases in fork/join DAGs.
5. Graph and factorization pruning preserve labelled input/output behavior.
6. Folded and explicit gather paths produce identical primal and tangent outputs for branched DAG frontiers.
7. Nonlinear frames reject an input-prefix destination, a forward temporary reference, an undefined final output slot, and an out-of-frame argument.
8. Residual phases reject invalid permutations, cyclic off-diagonal dependencies, invalid same-span ranges, and residual candidates that lose the cost comparison.
9. Mapped archive validation rejects bad alignment, offsets, table spans, indices, calls, gather sources, nonlinear frames, and workspace bounds.
10. Allocation instrumentation proves execute/push-forward paths allocate nothing.
11. Desktop benchmarks report B/N phase count, peak workspace, gather cost, archive bytes, B nnz, N-op count, compile time, and execution time.
12. QP verification is added only in the final QP migration phase and covers coefficient parity, CSC correctness, mapped plans, and caller-owned solver buffers.

## Phase 10 — Legacy Python lowering removal

Phase 09 does not by itself permit deletion of the old Python lowering stack.
The production ordinary `CokerFunction` path uses the Rust mapped artifact,
but QP extraction, graph-runtime tests, and benchmarks still import the old
graph representation and its bilinear metadata.

### Blocking references

- `src/coker/backends/coker/qp_extract.py` still calls
  `create_function_table` and `create_opgraph` to build coefficient
  evaluators and patterns.
- `src/coker/backends/coker/layers.py` and `op_impl.py` still use
  `BilinearWeights` for the legacy graph/layer execution model.
- `tests/backends/coker/test_coker_runtime.py` directly exercises
  `create_function_table` and `CompiledGraph`.
- `benchmarks/benchmark_backends.py` constructs a legacy graph with
  `create_opgraph`.
- `lowering.py`, `unfused_lowering.py`, and `weights.py` form the legacy
  Python lowering implementation and cannot be removed independently.
### Phase 10.1 — QP producer cutover

1. Add a PyO3 `compile_archive_qp` bridge returning the aligned unified QP artifact
   owner and persistence bytes. Wire Python Tape-to-`TypedDag` and symbolic QP
   declaration construction to that bridge.
2. Route every production QP builder through the Rust symbolic-QP declaration
   API and unified `QpModule` artifact.
3. Make Python supply only labelled symbolic declarations and evaluator tape
   references. Rust derives coefficient output ordering, CSC patterns, maps,
   QDLDL metadata, arena requirements, and the embedded plan.
4. Execute unified QP artifacts through the mapped runtime using
   caller-provided evaluator, update, arena, prepared-solver, and output
   buffers.
5. Remove production use of `qp_extract.py`; retain a Python reference
   evaluator only where a test needs an oracle.

Gate:

- Rust/Python coefficient and CSC ordering parity over representative
  parameters.
- A single aligned archive binds evaluator, patterns, and plan together.
- Repeated prepared solve writes caller outputs and allocates zero times.
- Corrupt coefficient slices, CSC tables, plan ranges, and alignment fail
  before evaluation or scatter.

### Phase 10.2 — Legacy graph consumer removal

1. Replace `CompiledGraph`/`FunctionTable` runtime tests with mapped-artifact
   primal and JVP tests through `CokerFunction` or the Rust test API.
2. Replace the legacy graph benchmark with the mapped Rust artifact benchmark;
   retain phase count, workspace, gather cost, nnz, archive bytes, compile
   time, and execution time reporting.
3. Delete Python tests whose sole assertion is the internal `SparseNet`,
   `FunctionTable`, layer, or `BilinearWeights` structure. Do not delete
   behavioural tests; port them to the mapped artifact path first.
4. Verify no production source, test, or benchmark imports
   `create_opgraph`, `create_function_table`, `CompiledGraph`, or
   `BilinearWeights`.

Gate:

- The complete Coker backend suite and mapped benchmark run without a legacy
  graph import.
- Python reference evaluation remains only an oracle, never a compiler or
  executor.

### Phase 10.3 — Atomic legacy-lowerer deletion

After Phases 10.1 and 10.2 pass in the same change:

1. Delete `lowering.py`, `unfused_lowering.py`, `weights.py`, and Python-only
   `SparseNet`/layer execution support that has no non-lowering use.
2. Delete the ordinary JSON graph export/compiler and old row-op/sparse-entry
   execution paths.
3. Remove compatibility aliases, dead imports, and obsolete internal tests;
   do not leave shims.
4. Re-run the complete Coker backend suite, Rust workspace tests, no-std
   checks, and mapped artifact lifecycle/corruption tests.

Exit criterion: no production Coker compilation or execution path contains
Python lowering logic. The only retained Python numerical code is an explicit
test oracle.

## Phase 11 — Code quality, ownership, and module organization

Run a deliberate cleanup after the functional migration is accepted. This phase
changes organization and local implementation shape without broadening public
behaviour.

### Phase 11.1 — Module boundary audit

1. Map each Rust crate and Python package by owned state, public entry points,
   hot paths, and imports.
2. Organize modules as a directed tree: a branch may call its descendants and
   its narrow parent API, but unrelated branches must not call sideways.
3. Replace cross-branch helpers with a small parent-owned boundary or move the
   owned operation and state to one coherent branch. Do not introduce service
   locators, registries, or bidirectional imports.
4. Shrink public API surface to real ownership, archive, compiler, and runtime
   boundaries. Remove forwarding wrappers and aliases that do not enforce an
   invariant.

### Phase 11.2 — File, flow, and ownership cleanup

1. Treat 1,000 source lines as a review threshold, not an automatic split.
   Split only where a file contains independently owned concepts; retain a
   longer cohesive procedural implementation when linear flow is clearer.
2. Prefer direct, phase-ordered functions with guard clauses over webs of tiny
   helpers. Extract only a block with a distinct invariant or ownership
   contract.
3. Remove avoidable branch fan-out in compiler and interpreter hot paths by
   separating phase setup from steady-state loops and by using validated table
   kinds rather than repeated defensive mode checks.
4. Make ownership explicit:
   - Rust APIs identify mapped immutable input, caller-owned mutable buffers,
     and compiler-owned construction storage.
   - Python objects have one owner for each artifact, workspace, solver, and
     lifecycle; remove ownership cycles and callback/reference retention that
     prevents deterministic release.
5. Preallocate when an upper bound is known. Construction may reserve exact
   capacities; runtime paths must reuse caller-owned fixed buffers.
6. Replace magic numbers with named constants at their semantic owner. Each
   archive version, alignment, capacity, opcode sentinel, tolerance, and solver
   layout value has one authoritative definition.
7. Rename verbose layout-repeating identifiers to short domain names. The
   module establishes storage layout; cross-layout boundaries use concrete
   archive terminology instead.
### Phase 11.2.1 — Cohesion audit

- `coker-bytecode/src/artifact.rs`: **deferred to Phase 13; not a cohesion
  exception.** It contains three independently owned regions: owned-model
  validation, ordinary artifact binding, and unified-QP artifact binding.
  Phase 13 must profile and then split those regions or record measured
  evidence that their shared private archive validation makes colocation the
  lower-complexity result. Phase 11 does not claim this file is complete.
- `coker-runtime/src/qp/mod.rs`: recorded cohesion exception. It owns the QP
  runtime boundary: archived metadata views, evaluator selection, workspace
  requirements, and the host/embedded feature boundary. Execution is already
  isolated in `qp/embedded.rs`; splitting the remaining declarations would add
  re-exports and borrow-threading without separating independent behavior.


### Phase 11.3 — Quality gates

- All public Rust boundaries have rustdoc stating ownership, allocation, mapped
  alignment, and error contracts where relevant.
- Rust crate files over 1,000 lines have a recorded cohesion rationale or a
  localized split.
- No production Python ownership cycle remains in artifact/QP runtime paths.
- `cargo clippy --workspace --all-targets -- -D warnings`, no-std checks, and
  focused Python lifecycle tests pass.
- Benchmark-sensitive changes retain Phase 12 before/after measurements.

## Phase 12 — Hexapod kinematics lowering and repeated-evaluation performance

Use the hexapod forward kinematics and forward spatial Jacobian as the
representative production workload. Locate the test-suite model first; if it
is absent, use `C:\projects\hexapod` without copying the model into the
repository.

### Phase 12.1 — Reproducible workload and baseline

1. Record the exact model revision, input/output dimensions, parameter values,
   backend settings, and release-build machine details.
2. Compile independent forward-kinematics and forward-spatial-Jacobian
   artifacts through the production Rust path.
3. Record: total phase count by B/N/Case/Call/residual kind, B term count,
   N row count, gather count, peak workspace/tangent/frame/solver storage,
   archive bytes, compile time, first evaluation time, and repeated evaluation
   time.
4. Treat the historical layer count above 1,000 as the baseline defect. The
   acceptance targets are low tens of phases for Forward Kinematics and low
   hundreds at most for the Forward Spatial Jacobian, without changing
   numerical semantics.

### Phase 12.2 — Hotspot analysis

1. Profile compiler stages separately: tape extraction, DAG analysis, workspace
   planning, B construction/reduction, N/Case/call partitioning, archive
   validation/finalization.
2. Profile repeated bytecode calls separately from first call: phase dispatch,
   sparse B term traversal, gathers, N rows, tangent propagation, bounds
   checks, caller-buffer clearing, and Python/Rust boundary costs.
3. Attribute archive bytes and evaluation time to concrete phase/table ranges;
   report the largest contributors rather than only aggregate timings.
4. Inspect upstream tape construction for avoidable scalarization, repeated
   subexpressions, unnecessary reshape/concatenate/gather nodes, missed common
   transforms, or output materialization that prevents DAG sharing.

### Phase 12.3 — Measured minor fixes

1. Apply only obvious non-structural improvements backed by the baseline:
   canonical term merging, dead-value pruning, stable gather elimination,
   capacity reservation, constant folding, or tape-construction sharing.
2. Re-run primal and forward-JVP parity after every change. Do not trade away
   mapped ownership, caller-owned execution buffers, deterministic ordering, or
   numerical certificates for a phase-count result.
3. Report before/after artifact bytes, phase counts, compile time, repeated
   evaluation cost, and allocation count.

### Phase 12.4 — Runtime optimization recommendations

Publish a hotspot report containing:

- the steady-state critical path and working-set estimate;
- phase/table categories dominating repeated calls;
- branch, bounds-check, cache locality, copy, and Python boundary costs;
- safe next optimizations ranked by expected measured benefit;
- optimizations explicitly deferred because they require structural changes,
  target-specific SIMD, or altered numerical policy.

Exit criterion: Forward Kinematics reaches the low-tens phase target and the
Forward Spatial Jacobian reaches the low-hundreds target, or each exception has
an evidence-backed lower bound. The report gives a reproducible baseline plus
prioritized runtime optimization actions.

### Deferred optimization profiles

STM32F7 and Raspberry Pi 4B profiling, target-specific policy calibration,
cycle measurements, and SIMD optimization are outside this rewrite. The
ordinary-path implementation is validated on the desktop and must retain its
embedded structural constraints: mapped archives, caller-provided buffers, and
allocation-free runtime execution. A later SIMD/optimization plan will define
the device profiles and may add measured target-specific policy overrides.
