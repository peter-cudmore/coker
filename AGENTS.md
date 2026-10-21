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

Tests SHOULD NOT use `monkeypatch`. Prefer public behavior or explicit,
constructor-provided test doubles; test seams MUST be part of the implementation
contract rather than patched internals.

## Python style

Prefer explicit interfaces and direct attribute access. Do not use `getattr` or
`hasattr` for normal control flow; add or use a defined protocol instead.
Avoid string arguments except for names; use typed values, enums, or dedicated
objects for behavioral choices.
Do not pass strings through functions solely to improve diagnostics; catch the
relevant exception and attach context when re-raising.
Do not use `typing.Protocol`. Prefer a local abstract base class or metaclass
when an interface is required. Keep inheritance shallow: one interface or base
class and one child class unless additional depth is necessary.
Prefer polymorphism to define interfaces, with only simple shared helpers.
Do not use inheritance to share model behavior or state. Share behavior through
free functions that explicitly dispatch on the interface implementation.

Function and method names MUST use an active verb, omit context apparent from
their module or owning type, and remain concise. Abbreviations are allowed when
they are established, consistent conventions in the existing codebase.

## Public API documentation

Keep the public-facing API small and well documented. Every public class and
method MUST have a Google-style docstring with basic usage. Documentation MUST
provide detailed examples, and CI MUST execute those examples.

## Backend policy

The CasADi backend MUST remain at least as feature complete as the NumPy
backend. Prefer CasADi for optimisation, simulation, and high-performance
numerical workloads; retain NumPy as the reference and small-system fallback.

## Pre-commit review

Before committing, review the changed code for dead code, duplicated behavior,
and helpers or branches that can be inlined without reducing clarity.

## Pull requests

Normal pull requests MUST target `develop`, never `main`. Verify the base branch
before creating or updating a pull request. If host tooling cannot select
`develop`, escalate instead of using `main`. Maintainer release promotion is the
exception documented in `CONTRIBUTING.md`.

Normal pull requests MUST be squash-merged into `develop`. Preserve the PR title
with its `#N` reference and the complete body in the squash commit. The body MUST
contain a single `Changelog` section with `Added`, `Changed`, and `Fixed`
categories in that order, listing only this PR's user-visible changes. Unused
categories MUST stay empty without filler; if all are empty, `Description` MUST
explain why the work has no user-visible changes. Every post-cutover commit,
including internal and release-preparation work, MUST retain this PR metadata.

Commits and PR titles with user-facing or functional changes MUST use the
headline format `(type): Description`, where `type` is `fix` for a bug fix, `add`
for a new feature, `mod` for modified functionality, or `rem` for a removed
feature. Internal documentation, test, tooling, and refactoring titles MUST omit
the format. Headlines are NOT the source of release-note entries.

Normal PRs MUST NOT edit cumulative `CHANGELOG.md` or copy its entries into their
PR bodies. Maintainer release preparation aggregates the preserved PR-local
changelog metadata into `CHANGELOG.md` and the reviewed versioned release-notes
file. Follow the release procedure in `CONTRIBUTING.md`; publication MUST reuse
that file rather than regenerate notes from `main`.

