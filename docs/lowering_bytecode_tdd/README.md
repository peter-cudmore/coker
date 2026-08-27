# Test-driven lowering and bytecode rewrite phases

Read `../lowering_bytecode_rewrite_plan.md` first. These files turn that architecture into small test-driven deliverables. Complete phases in order; a later phase may add tests for earlier behavior but must not skip its acceptance gate.

| Phase | Deliverable | Gate |
|---|---|---|
| [01](phase_01_contracts_baseline.md) | Python semantic and current-runtime tolerance oracle | Deterministic finite-domain fixtures cover current operation surface. |
| [02](phase_02_maturin_builder.md) | PyO3/maturin builder producing Rust `TypedDag` | Python node ingestion equals Rust fixture DAG. |
| [03](phase_03_graph_analysis_pruning.md) | Iterative graph metadata, pruning, and scheduler | Required DAG scheduling cases pass deterministically. |
| [04](phase_04_artifact_and_soa.md) | Versioned aligned SoA artifact and owner | Emit/bind validation works without runtime execution changes. |
| [05](phase_05_overwrite_bilinear_and_gather.md) | Overwrite B and gather runtime path | B/gather primal and tangent parity. |
| [06](phase_06_nonlinear_calls_and_tangents.md) | N frames, iterative calls, and tangents | Scalar-op and acyclic-call parity. |
| [07](phase_07_reduction_and_residuals.md) | Reducers and conservative residual B layers | Residual/factorization choice is validated and deterministic. |
| [08](phase_08_ordinary_parity_and_cleanup.md) | Production ordinary Rust backend | Existing ordinary Python suite passes; legacy ordinary lowering removed. |
| [09](phase_09_qp_migration.md) | Rust QP compiler and embedded runtime path | QP parity and caller-owned no-std execution pass. |

## Working rules

- Write the failing behavioral test before implementation changes.
- Run the phase’s focused tests after every increment; run the phase gate before advancing.
- Do not weaken tolerances or skip edge cases to pass a test. Correct classification, lowering, bytecode, or runtime behavior instead.
- Preserve caller-provided workspace ownership and allocation-free mapped runtime execution in every phase.
- QP is mandatory but starts only after Phase 08 accepts ordinary-path parity.
