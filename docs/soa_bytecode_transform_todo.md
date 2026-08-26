# SoA bytecode transform — deferred todo

## Decision

Defer the struct-of-arrays transform until the Python residual `SparseNet` is correct against reference backends and has met its FK stage-count gate. SoA is the physical archive layout selected during Milestone 2; it must not determine DAG scheduling, epoch closure, slot lifetime, or FK layer count.

The Milestone 2 bytecode cutover introduces the selected magic/version and SoA archive in one clean change. The Rust bytecode crate owns the typed builder, schema, validation, and serialization; `coker-python` invokes that builder directly. Migrate every producer, validator, mapped primal executor, mapped push-forward executor, QP coefficient evaluator, test, and generated artifact together. Do not retain an AoS compatibility decoder or executor.

## Preconditions

- The Python tape → residual `SparseNet` lowering is the only ordinary-program path and passes its reference-backend correctness and FK stage-count gates.
- Stable-slot lifetime reuse, epoch boundaries, wide operands, and nested-call scratch accounting are complete and measured in Python.
- The selected SoA ABI preserves borrowed-archive, caller-buffered, allocation-free, and `no_std` runtime execution.

## Todo

- [ ] Specify the SoA archived record families: generic output/first/second/third/opcode arrays; bilinear row-output/term-start/term-count arrays; and bilinear term-left/right/value arrays. Specify compact and wide layouts together.
- [ ] Define archive invariants for equal field lengths, row/term range bounds, canonical row/term ordering, homogeneous operand tags, slot bounds, alias safety, and required alignment.
- [ ] Implement the Rust bytecode crate's typed builder so `coker-python` constructs SoA modules directly from residual stages. Do not route normal graph or QP compilation through Python dictionaries, JSON bytes, an exported JSON model, or a second scheduler.
- [ ] Implement mapped archived primal execution by indexed parallel slices, with local scalar accumulators and no allocations or decoded `Vec`s.
- [ ] Implement the matching mapped push-forward execution and retain local primal/tangent bilinear accumulation.
- [ ] Migrate nonlinear retained-expression operand records, nested-call bindings, QP coefficient evaluation, and every compact/wide validator to the same SoA archive contract.
- [ ] Delete AoS scheduled record encoding, validation, executor paths, conversions used by execution, fixtures, and generated artifacts in the same cutover.
- [ ] Regenerate Coker and Hexapod artifacts; verify mapped primal/push-forward equivalence, QP CSC scatter and prepared-solver reuse, archive rejection cases, and `coker-runtime --no-default-features`.
- [ ] Compare AoS-versus-SoA artifact size, workspace bytes touched, and target/faithful-target controller-tick timing. Treat this as an optimization measurement, not a reason to relax scheduler correctness or FK layer gates.

## Non-goals

- No SoA work may change algebraic reassociation rules, scalar operation order, output ABI, or caller-owned workspace ownership.
- No runtime execution path may decode the archive into owned records to recover an AoS-like representation.
