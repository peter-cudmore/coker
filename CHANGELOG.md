# Changelog

## Unreleased

### Added
- `CasadiVariationalOptions` for CasADi solver policy and adaptive mesh refinement controls.
- Local defect estimation with p-refinement before h-refinement.
- Reuse of the previous path as the refined NLP initial guess.
- Function-valued variational parameters with monotone piecewise-linear, perceptron, and radial-basis realizations.
- `UnboundedVariable` declarations for scalar decisions without user-supplied bounds.
- SymPy symbolic lowering preserves named function parameters and external calls.
- Symbolic local accessibility and structural identifiability analysis for supported ODE and semi-explicit index-one DAE systems.

### Changed
- Bump Coker to version 0.4.3.
- Collocation reference-operator caching is scoped to a transcription factory rather than process-global state.
- CasADi transcription compilation reuses bounded cached mesh-specific solvers during parameter sweeps.
- Existing `TranscriptionOptions` CasADi fields remain supported and are translated when no explicit backend policy is provided.
- Exclude CasADi 3.8.x from the optional dependency because the failure reproduces with a CasADi-only replay, indicating a regression in CasADi or its bundled solver stack.

### Fixed
- Preserve configured adaptive transcription interval counts.
- Preserve legacy CasADi `TranscriptionOptions` compatibility.
