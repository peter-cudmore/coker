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

## Contribution licensing

By submitting code, documentation, examples, tests, or other changes for inclusion
in Coker, you agree to license your contributions under the project's
[Mozilla Public License 2.0](LICENSE.TXT).

You or the relevant rights holder retain copyright in your contributions.
Submitting a contribution does not transfer that copyright to Pneumatica Bio.
We do not require a separate contributor licence agreement (CLA) or copyright
assignment.

Only submit work that you have the right to contribute under MPL-2.0. If your
employer or another party owns the work, obtain their authorization before
submitting it.

Identify any third-party material included or adapted in your contribution,
provide its source and licence, and preserve all required notices. Do not submit
material whose terms or permissions are unclear.

For named recognition, follow the [credit policy](CREDITS.md).

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

Commits and pull request titles with functional or user-facing changes use
`(type): Description`, where `type` is `fix`, `add`, `mod`, or `rem`. Internal
documentation, test, tooling, and refactoring titles omit this prefix. This title
convention describes the work; titles are not the source of release-note entries.

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
- In the template's single `Changelog` section, list only this PR's user-visible
  changes as bullets under `Added`, `Changed`, and `Fixed`, in that order. Leave
  unused categories empty, without filler such as "None" or "N/A". If the work
  has no user-visible changes, leave all three empty and explain why in
  `Description`; otherwise `Description` is optional.
- Do **not** edit cumulative `CHANGELOG.md` in normal PRs or copy its entries
  into your PR body. Release preparation aggregates PR-local changelog entries.
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
additional effort. Maintainers squash-merge accepted pull requests into
`develop`, retaining the PR title with its `#N` reference and the complete PR
body, including `Summary`, any `Description`, and `Changelog`. Preserve credit
according to [CREDITS.md](CREDITS.md). Every post-cutover commit on `develop`,
including documentation, tooling, and release-preparation work, must come through
a squash PR with this metadata; direct commits or missing metadata break release
preparation. User-visible entries require the squash title's PR number so the
release notes can link back to the PR.

## Prepare a release (maintainers)

Release preparation is the exception to the normal rule against editing
`CHANGELOG.md`. The preparation script reads the explicit, ancestral
`--since` (exclusive) to `--until` (inclusive) range of first-parent squash
commits. It rejects missing or malformed PR metadata rather than silently
dropping changes. Do not substitute formatted commit headlines for the PR-local
`Added`, `Changed`, and `Fixed` entries.

1. Select the source snapshot on `develop` and the release version and date.
   Prepare a branch from that snapshot, not from `main`. The following commands
   use Bash, Git, the GitHub CLI authenticated for this repository, and Python:

   ```bash
   git fetch origin
   REPOSITORY="$(gh repo view --json nameWithOwner --jq '.nameWithOwner')"
   read -r -p "Release version (for example, 0.4.5): " VERSION
   read -r -p "Release date (YYYY-MM-DD): " RELEASE_DATE
   git switch --create "release/$VERSION" origin/develop
   SOURCE="$(git rev-parse HEAD)"
   ```

2. Set the exclusive baseline. **For the first release under this policy**,
   wait until this policy PR has actually been squash-merged into `develop`.
   Replace the prompt with its real PR number; do not use this branch's current
   commit, a guessed SHA, or a release tag:

   ```bash
   read -r -p "Merged changelog-policy PR number: " POLICY_PR_NUMBER
   BASE="$(gh pr view "$POLICY_PR_NUMBER" --repo "$REPOSITORY" --json mergeCommit --jq '.mergeCommit.oid')"
   ```

   That actual merge SHA is the cutover boundary. Do not scan older PR bodies:
   they contain cumulative changelogs. The first preparation preserves all
   existing `Unreleased` entries as the legacy baseline and adds the selected
   post-cutover PR entries.

   **For subsequent releases**, instead read the immutable source-range HTML
   comment immediately below the version heading in the latest published
   `release-notes/VERSION.md`. Its exact format is
   `<!-- release-source: since=<40-hex SHA> until=<40-hex SHA> -->`.
   Use that recorded `until` source endpoint as the next `--since`:

   ```bash
   read -r -p "Previous release notes' until source SHA: " BASE
   ```

   Do not use the release tag or its `main` commit as `--since`: promotion to
   `main` may itself be squash-merged and need not preserve the `develop`
   ancestry. Keep the previous notes' source-range comment unchanged.

3. Generate the cumulative changelog and the versioned release-notes file from
   the explicit source SHA captured above:

   ```bash
   python scripts/prepare_release.py --since "$BASE" --until "$SOURCE" --version "$VERSION" --date "$RELEASE_DATE" --repository "$REPOSITORY"
   ```

   The script updates `CHANGELOG.md` and writes `release-notes/$VERSION.md`.
   The file contains exactly the same `## VERSION - DATE` section as the
   cumulative changelog, with `Added`, `Changed`, and `Fixed` categories,
   PR-linked entries, and the immutable source range. Review both files,
   including the range and the first release's retained legacy baseline.
   Resolve invalid metadata or missing PR linkage at its source; do not omit
   commits from the range to make preparation pass.

4. Bump `[project].version` in `pyproject.toml` to the selected version and
   refresh the lockfile with `uv lock`. Commit the version, lockfile,
   `CHANGELOG.md`, and the generated `release-notes/$VERSION.md` together.
   Open a release-preparation PR into `develop` for review. Its PR-local
   `Changelog` categories are all empty: explain in `Description` that it only
   records already documented changes and prepares the release. Preserve this
   complete metadata when squash-merging it, too.

5. After the preparation PR is reviewed and merged, promote the prepared
   `develop` snapshot to `main` through the maintainer release PR (the exception
   to the contributor rule to target `develop`). Do not include unrecorded
   user-visible changes after the selected source snapshot; if the release
   scope changes, prepare and review notes for the new source range first.
   The production publication workflow consumes the reviewed
   `release-notes/$VERSION.md` unchanged for the GitHub release and tags the
   published `main` SHA. It does not regenerate notes from `main` history.
   Keep the generated file committed: the next release uses its recorded source
   endpoint, not that publication tag.
