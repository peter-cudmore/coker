# Coker Rust runtime design

## Runtime constraints

The Rust runtime is intended for embedded and other `no_std` targets. Its execution path MUST be allocation-free after construction and MUST NOT require a global allocator.

Runtime bytecode is a mapped artifact, not an owned deserialized model. The bytecode format MUST permit a function and all of its referenced tables to be accessed directly from an aligned `&[u8]` supplied by the embedding application. The runtime MUST retain borrowed archived views into that mapping; it MUST NOT copy bytecode into `Vec`, `AlignedVec`, or equivalent owned buffers in order to decode or execute it.

The embedding application owns every mutable execution buffer. Workspace, tangent workspace, outputs, solver workspaces, coefficient buffers, warm-start vectors, and any scratch storage MUST be caller-provided or statically provisioned. Constructors MAY validate buffer sizes but MUST NOT allocate them.

## Bytecode design

- Archives MUST be self-contained, alignment-aware, and directly readable through `rkyv` archived types.
- A bytecode header MUST record the archive alignment and payload offset. Producers MUST pad the payload to that alignment; consumers MUST reject unaligned mappings rather than copying them to align them.
- Runtime-facing APIs SHOULD accept borrowed mapped bytes and return borrowed archived program/function views.
- Function lookup, input/output specifications, layers, sparse-pattern tables, and nested-function references MUST remain in the mapped archive and be traversed by offset/index.
- Owned decode APIs are compiler/tooling conveniences only; they MUST NOT be used by the runtime execution path.

## QP runtime implications

QP bytecode MUST embed its coefficient evaluator and sparse CSC patterns in the same mapped artifact. The QP runtime MUST borrow those archived tables, execute the evaluator into caller-provided coefficient buffers, scatter directly into caller-provided OSQP update buffers, and reuse a caller-provided/prebuilt solver workspace. It MUST NOT deserialize a QP into vectors, rebuild dense matrices, construct a solver per solve, or allocate solution/status values during `solve`.

The currently selected `osqp` Rust crate is `std`/allocator-oriented and is not acceptable on the no-std execution path without a compatible lower-level integration. Use a no-std-capable OSQP C build or a thin FFI wrapper configured with caller-supplied allocator/workspace hooks; gate host conveniences behind a `std` feature.

## Bug-fix workflow

For every hypothesized bug fix, first add the smallest behavioral regression that reproduces the suspected cause. Implement only after that test demonstrates the diagnosis; remove the test afterward only when it provides no durable contract coverage.

