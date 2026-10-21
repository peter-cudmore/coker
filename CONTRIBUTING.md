# Contributing to Coker

## Contributions we encourage

We encourage focused contributions that need little additional work to review
and integrate: documentation improvements, reproducible bug reports, bug fixes,
and targeted tests. Maintainer time for mentoring, redesign, and repeated review
is limited. Please submit changes that are ready to integrate, rather than drafts
that need substantial maintainer development.

Significant contributions will rarely be accepted unless they come from existing
contributors familiar with Coker's architecture and maintenance requirements.
This includes major features, new backends, public API redesigns, and broad
refactoring. Discuss such work in an issue and obtain maintainer agreement on its
scope before implementing it. An issue discussion is not a promise of acceptance.
Small documentation corrections and focused bug fixes do not need prior approval.

## Prepare a contribution

1. Check existing issues and pull requests to avoid duplicating work. For a bug,
   include a minimal reproducer, expected and actual results, and relevant Python,
   Coker, and backend versions.
2. Create a branch from the current `develop` branch. Keep each pull request
   focused on one problem; avoid unrelated cleanup or formatting changes.
3. Follow the repository's implementation conventions. For a bug fix, first add
   the smallest behavioral regression that demonstrates the problem, then fix it.
   Update affected documentation and examples. Public classes and methods need
   Google-style docstrings with basic usage; documented examples must run in CI.
4. Follow the [development instructions](#development) to install the
   tooling and run the relevant tests, formatting, lint, and documentation checks.
   For documentation changes, also follow the
   [documentation authoring guide](docs/AUTHORING.md).
5. Add your name to [AUTHORS.md](AUTHORS.md) if it is not already listed. Identify
   everyone who contributed to the work in the pull request, including relevant
   prior work, following the [credit policy](CREDITS.md).

Commits with functional or user-facing changes use `(type): Description`, where
`type` is `fix`, `add`, `mod`, or `rem`. Internal documentation, test, tooling, and
refactoring commits omit this prefix.

## Development

The repository uses `uv` for environment management. Use Python 3.11 or newer
for development with all optional backends; the base package supports Python
3.10, but JAX requires Python 3.11 or newer.

Install the development, CI, and documentation tools and optional backends:

```bash
uv sync --frozen --group dev --group ci --group docs --extra casadi --extra jax --extra pytorch
```

Run the default test suite used in PR CI:

```bash
uv run pytest -m "not perf"
```

Performance assertions run separately in the manual CI workflow; they are not
part of the default PR suite.

Check formatting and lint:

```bash
uv run black --check --diff src
uv run flake8 src tests docs examples scripts
```

Run the documentation examples and build HTML with warnings treated as errors:

```bash
uv run sphinx-build -b doctest docs/ docs/_build/doctest -W --keep-going
uv run sphinx-build docs/ docs/_build/html -W --keep-going
```

See the [documentation authoring guide](docs/AUTHORING.md) for section goals,
writing conventions, and executable examples. Source-only `Author notes`
record page-specific directions and planned additions.

## Submit a pull request

- Target **`develop`**, never `main`.
- Use the pull request template. Explain the problem, the change, and the checks
  you ran, including any checks you could not run locally.
- Update `CHANGELOG.md` under `Unreleased` for user-visible changes. The PR's
  `Changelog` section must exactly match the current `Unreleased` entries under
  `Added`, `Changed`, and `Fixed`; the metadata CI check enforces this even when
  your change does not need a new changelog entry.
- Include accurate contributor attribution and your `AUTHORS.md` update.

## CI and review

**All applicable PR CI checks must pass before maintainer review will begin.**
This applies to every contribution, including documentation-only changes. Resolve
failing checks and merge conflicts before requesting review. Jobs intentionally
skipped by the workflow, such as the manual performance suite, do not count as
failures. If a workflow needs maintainer approval to run, ask for that approval;
it is not a substitute for passing CI.

Passing CI makes a contribution eligible for review; it does not guarantee an
immediate review or acceptance. Maintainers assess correctness, scope,
architectural fit, and ongoing maintenance cost. We may request focused changes,
make small integration edits ourselves, or decline work that requires substantial
additional effort. We normally squash-merge accepted pull requests and preserve
credit according to [CREDITS.md](CREDITS.md).
