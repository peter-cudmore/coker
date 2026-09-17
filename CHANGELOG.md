# Changelog

## Unreleased

### Added
- `CasadiVariationalOptions` for CasADi solver policy and adaptive mesh refinement controls.
- Local defect estimation with p-refinement before h-refinement.
- Reuse of the previous path as the refined NLP initial guess.

### Changed
- Bump Coker to version 0.4.2.
- Collocation reference-operator caching is scoped to a transcription factory rather than process-global state.
- CasADi transcription compilation reuses bounded cached mesh-specific solvers during parameter sweeps.
- Existing `TranscriptionOptions` CasADi fields remain supported and are translated when no explicit backend policy is provided.

### Fixed
- Preserve configured adaptive transcription interval counts.
- Preserve legacy CasADi `TranscriptionOptions` compatibility.
