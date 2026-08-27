# Phase 03 — Iterative graph metadata, pruning, and scheduling proof

## Goal

Turn `TypedDag` into a pruned, metadata-rich DAG using iterative classic graph algorithms. Produce scheduling decisions and workspace requirements, but keep the legacy executor as the behavior oracle.

## Tasks

- [ ] Build reverse adjacency in CSR form.
  - First count consumers per node.
  - Prefix-sum counts into offsets.
  - Fill one contiguous consumer-index array.
  - Keep node IDs stable after pruning through an old-to-new mapping.
- [ ] Run iterative Kahn topological sorting.
  - The source tape is guaranteed a DAG, but verify operands precede users.
  - Preserve source node order only as a final tie-breaker.
  - Reject malformed ingestion if the queue does not consume every node.
- [ ] Mark output reachability with a backward worklist.
  - Start from labelled outputs only.
  - Retain labels even when values are aliases or constants.
  - Remove unreachable nodes and unreachable nested functions.
- [ ] Add forward/reverse metadata passes.
  - flat width and shape class;
  - constant-known status;
  - consumer count and last-use bound;
  - operation class: B-eligible, N-required, call, or compile-time-only;
  - earliest/latest legal phase metadata and critical-path rank.
- [ ] Implement a dependency-aware phase scheduler prototype.
  - Use ready frontiers, never source-order greedy scheduling.
  - For small frontiers, use bounded exact search under `LoweringPolicy` limits.
  - For larger frontiers, use deterministic look-ahead list scheduling.
  - Compare candidates lexicographically: B phase count, peak workspace, nnz/op estimate, source-order tie-break.
- [ ] Add liveness/workspace planning over scheduled values.
  - Keep live values in allocated spans; do not add identity copies merely for compaction.
  - Record when a future phase needs an explicit gather.

## Required tests

- [ ] Unused branch is removed without changing labelled outputs.
- [ ] `(a*b)*(c*d)` produces B→B and no N phase.
- [ ] Independent `b1`, `n1`, and `b2(n1)` schedules `n1` before one joinable B phase.
- [ ] A B-dependent N followed by B requires B→N→B.
- [ ] Exact-search and fallback schedules are deterministic.

## Do not do

- Do not implement residual conversion yet.
- Do not emit a new bytecode artifact yet.
