import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_release.py"
HISTORY = """## 0.9.0 - 2025-01-02

### Added
- A historical item with  deliberate spacing.

### Changed

### Fixed
- An older fix.
"""
EMPTY_UNRELEASED = """# Changelog

## Unreleased

### Added

### Changed

### Fixed

"""
BASELINE = (
    EMPTY_UNRELEASED.replace(
        "### Added\n", "### Added\n- Existing cumulative feature.\n"
    )
    .replace("### Changed\n", "### Changed\n- Existing cumulative change.\n")
    .replace("### Fixed\n", "### Fixed\n- Existing cumulative fix.\n")
    + HISTORY
)


def git(root, *arguments, input=None):
    environment = os.environ | {
        "GIT_AUTHOR_NAME": "Release Tests",
        "GIT_AUTHOR_EMAIL": "release-tests@example.com",
        "GIT_COMMITTER_NAME": "Release Tests",
        "GIT_COMMITTER_EMAIL": "release-tests@example.com",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-C", str(root), *arguments],
        input=input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def metadata(
    added=(),
    changed=(),
    fixed=(),
    description="Implementation details stay private to the PR.",
):
    lines = [
        "## Summary",
        "- A summary that must not appear in release notes.",
        "",
        "## Description",
        description,
        "",
        "## Changelog",
    ]
    for category, entries in (("Added", added), ("Changed", changed), ("Fixed", fixed)):
        lines.append(f"### {category}")
        lines.extend(f"- {entry}" for entry in entries)
        lines.append("")
    return "\n".join(lines)


def commit(root, title, body=""):
    git(root, "add", "--all")
    git(root, "commit", "--allow-empty", "-F", "-", input=f"{title}\n\n{body}\n")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "core.autocrlf", "false")
    (tmp_path / "CHANGELOG.md").write_bytes(BASELINE.encode("utf-8"))
    cutover = commit(tmp_path, "The policy cutover has a legacy body")
    return tmp_path, cutover


def prepare(
    root,
    since,
    until,
    version="1.0.0",
    release_date="2026-10-22",
    repository="owner/project",
):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            f"--since={since}",
            f"--until={until}",
            "--version",
            version,
            "--date",
            release_date,
            "--repository",
            repository,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def snapshot(root):
    paths = [root / "CHANGELOG.md"]
    notes = root / "release-notes"
    if notes.is_dir():
        paths.extend(sorted(notes.rglob("*")))
    elif notes.exists():
        paths.append(notes)
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in paths
        if path.is_file()
    }


def assert_rejected_unchanged(root, since, until, **arguments):
    before = snapshot(root)
    directory_existed = (root / "release-notes").exists()
    result = prepare(root, since, until, **arguments)
    assert result.returncode == 1, result.stdout
    assert result.stderr.startswith("Release preparation failed:")
    assert snapshot(root) == before
    assert (root / "release-notes").exists() == directory_existed
    return result


def release_section(changelog, version):
    start = changelog.index(f"## {version} - ")
    end = changelog.find("\n## ", start + 1)
    return changelog[start:] if end == -1 else changelog[start : end + 1]


def test_aggregates_baseline_and_structured_squashes_in_order(repository):
    root, cutover = repository
    first = commit(
        root,
        "feat: Do not extract this title (#12)",
        metadata(added=("First feature.", "Second feature."), fixed=("First repair.",)),
    )
    commit(root, "Document contribution expectations", metadata())
    endpoint = commit(
        root,
        "fix: This title is not a release entry (#34)",
        metadata(
            added=("Third feature.",),
            changed=("A changed behavior.",),
            fixed=("Second repair.",),
        ),
    )
    git(root, "tag", "cutover", cutover)
    git(root, "tag", "source", endpoint)

    result = prepare(root, "cutover", "source")

    assert result.returncode == 0, result.stderr
    notes = (root / "release-notes" / "1.0.0.md").read_bytes()
    changelog = (root / "CHANGELOG.md").read_bytes()
    text = notes.decode("utf-8")
    assert notes == release_section(changelog.decode("utf-8"), "1.0.0").encode("utf-8")
    assert notes.endswith(b"\n") and not notes.endswith(b"\n\n")
    assert f"<!-- release-source: since={cutover} until={endpoint} -->" in text
    assert first not in text
    assert changelog.endswith(HISTORY.encode("utf-8"))
    assert changelog.startswith(EMPTY_UNRELEASED.encode("utf-8"))
    assert text.index("Existing cumulative feature.") < text.index("First feature.")
    assert (
        text.index("First feature.")
        < text.index("Second feature.")
        < text.index("Third feature.")
    )
    assert text.index("### Added") < text.index("### Changed") < text.index("### Fixed")
    assert text.index("First repair.") < text.index("Second repair.")
    assert "- First feature. ([#12](https://github.com/owner/project/pull/12))" in text
    assert (
        "- A changed behavior. ([#34](https://github.com/owner/project/pull/34))"
        in text
    )
    assert "- Existing cumulative feature.\n" in text
    for excluded in (
        "## Summary",
        "## Description",
        "summary that",
        "Implementation details",
        "feat:",
        "fix:",
        "Do not extract",
    ):
        assert excluded not in text


def test_first_preparation_can_release_only_the_cutover_baseline(repository):
    root, cutover = repository

    result = prepare(root, cutover, cutover)

    assert result.returncode == 0, result.stderr
    notes = (root / "release-notes" / "1.0.0.md").read_text(encoding="utf-8")
    assert notes.count("Existing cumulative feature.") == 1
    assert f"since={cutover} until={cutover}" in notes


def test_rerun_rejects_duplicate_without_mutation(repository):
    root, cutover = repository
    endpoint = commit(root, "A feature (#2)", metadata(added=("A release feature.",)))
    result = prepare(root, cutover, endpoint)
    assert result.returncode == 0, result.stderr

    duplicate = assert_rejected_unchanged(root, cutover, endpoint)

    assert "already exist" in duplicate.stderr
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.count("A release feature.") == 1
    assert text.count("Existing cumulative feature.") == 1


def test_duplicate_version_in_changelog_rejected_without_notes_file(repository):
    root, cutover = repository
    endpoint = commit(root, "A feature (#2)", metadata(added=("A release feature.",)))
    result = prepare(root, cutover, endpoint)
    assert result.returncode == 0, result.stderr
    (root / "release-notes" / "1.0.0.md").unlink()

    result = assert_rejected_unchanged(root, cutover, endpoint)

    assert "already exists in CHANGELOG.md" in result.stderr


@pytest.mark.parametrize(
    "body",
    [
        "",
        "A plain squash body with no PR metadata.",
        metadata(added=("A feature.",)).replace("## Summary", "## Overview"),
        metadata(added=("A feature.",)).replace("### Fixed", "### Removed"),
        metadata(added=("A feature.",)).replace("- A feature.", "Unstructured prose."),
        metadata(description="<!-- No substantive explanation. -->"),
        metadata(added=("A feature.",))
        + "\n## Changelog\n### Added\n### Changed\n### Fixed\n",
    ],
)
def test_invalid_commit_metadata_leaves_both_outputs_unchanged(repository, body):
    root, cutover = repository
    endpoint = commit(root, "A change (#3)", body)

    result = assert_rejected_unchanged(root, cutover, endpoint)

    assert "invalid PR metadata" in result.stderr


@pytest.mark.parametrize(
    "title", ["feat: missing number", "A change (#0)", "A change (#3) extra"]
)
def test_nonempty_entries_require_pr_linkage(repository, title):
    root, cutover = repository
    endpoint = commit(root, title, metadata(added=("A feature.",)))

    result = assert_rejected_unchanged(root, cutover, endpoint)

    assert "squash title ending in (#N)" in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        {"version": "../escape"},
        {"version": "1.0.0/../../escape"},
        {"version": "v1.0.0"},
        {"version": "01.0.0"},
        {"version": "1.0.0-01"},
        {"release_date": "2026-02-30"},
        {"release_date": "2026-2-03"},
        {"repository": "https://github.com/owner/project"},
        {"repository": "owner/../project"},
        {"repository": "owner/.."},
        {"repository": "owner/project?query"},
    ],
)
def test_invalid_arguments_are_rejected_before_writes(repository, arguments):
    root, cutover = repository
    endpoint = commit(root, "A feature (#4)", metadata(added=("A feature.",)))

    assert_rejected_unchanged(root, cutover, endpoint, **arguments)


def test_semver_prerelease_and_build_metadata_are_safe_filenames(repository):
    root, cutover = repository

    result = prepare(root, cutover, cutover, version="1.2.3-rc.1+build.2")

    assert result.returncode == 0, result.stderr
    assert (root / "release-notes" / "1.2.3-rc.1+build.2.md").is_file()


def test_missing_references_and_reversed_range_are_rejected(repository):
    root, cutover = repository
    endpoint = commit(root, "A feature (#4)", metadata(added=("A feature.",)))

    assert_rejected_unchanged(root, "does-not-exist", endpoint)
    assert_rejected_unchanged(root, cutover, "does-not-exist")
    assert_rejected_unchanged(root, endpoint, cutover)
    assert_rejected_unchanged(root, "--all", endpoint)


def test_unrelated_history_is_rejected(repository):
    root, cutover = repository
    git(root, "checkout", "--orphan", "unrelated")
    endpoint = commit(root, "Unrelated root (#4)", metadata(added=("A feature.",)))

    assert_rejected_unchanged(root, cutover, endpoint)


def test_ancestor_on_side_branch_is_not_a_valid_source_baseline(repository):
    root, cutover = repository
    git(root, "checkout", "-b", "side")
    side = commit(root, "A side feature (#4)", metadata(added=("A side feature.",)))
    git(root, "checkout", "main")
    commit(root, "A main feature (#5)", metadata(added=("A main feature.",)))
    git(root, "merge", "--no-ff", "side", "-m", "A merge")
    endpoint = git(root, "rev-parse", "HEAD")

    result = assert_rejected_unchanged(root, side, endpoint)

    assert "first-parent lineage" in result.stderr
    result = assert_rejected_unchanged(root, cutover, endpoint)
    assert "nonmerge squash history" in result.stderr


@pytest.mark.parametrize("empty_commits", [False, True])
def test_noop_release_without_baseline_is_rejected(repository, empty_commits):
    root, cutover = repository
    (root / "CHANGELOG.md").write_text(EMPTY_UNRELEASED + HISTORY, encoding="utf-8")
    if empty_commits:
        endpoint = commit(root, "Document workflows", metadata())
    else:
        endpoint = cutover

    result = assert_rejected_unchanged(root, cutover, endpoint)

    assert "no release entries" in result.stderr


def test_existing_release_directory_file_is_rejected(repository):
    root, cutover = repository
    (root / "release-notes").write_text("Do not overwrite this file.", encoding="utf-8")

    result = assert_rejected_unchanged(root, cutover, cutover)

    assert "must be a directory" in result.stderr


def test_legacy_history_line_endings_are_preserved_verbatim(repository):
    root, cutover = repository
    history = HISTORY.replace("\n", "\r\n").encode("utf-8")
    (root / "CHANGELOG.md").write_bytes(
        BASELINE.removesuffix(HISTORY).encode("utf-8") + history
    )

    result = prepare(root, cutover, cutover)

    assert result.returncode == 0, result.stderr
    assert (root / "CHANGELOG.md").read_bytes().endswith(history)


def test_next_release_uses_source_endpoint_after_release_pr_is_squashed(repository):
    root, cutover = repository
    first_source = commit(
        root, "Add first feature (#10)", metadata(added=("First shipped feature.",))
    )
    git(root, "checkout", "-b", "release-first")
    first = prepare(root, cutover, first_source)
    assert first.returncode == 0, first.stderr
    first_notes = (root / "release-notes" / "1.0.0.md").read_bytes()
    release_commit = commit(root, "Prepare first release (#11)", metadata())
    git(root, "tag", "v1.0.0", release_commit)
    git(root, "checkout", "main")
    git(root, "merge", "--squash", "release-first")
    main_release = commit(root, "Prepare first release on main (#11)", metadata())
    assert main_release != release_commit
    second_source = commit(
        root, "Add second feature (#12)", metadata(fixed=("Second shipped repair.",))
    )

    wrong_baseline = assert_rejected_unchanged(
        root, "v1.0.0", second_source, version="1.1.0"
    )
    assert wrong_baseline.returncode == 1
    wrong_baseline = assert_rejected_unchanged(
        root, main_release, second_source, version="1.1.0"
    )
    assert "previous release's recorded source endpoint" in wrong_baseline.stderr
    second = prepare(root, first_source, second_source, version="1.1.0")

    assert second.returncode == 0, second.stderr
    second_notes = (root / "release-notes" / "1.1.0.md").read_bytes()
    changelog = (root / "CHANGELOG.md").read_bytes()
    assert second_notes == release_section(changelog.decode("utf-8"), "1.1.0").encode(
        "utf-8"
    )
    assert b"Second shipped repair." in second_notes
    assert b"First shipped feature." not in second_notes
    assert b"Existing cumulative feature." not in second_notes
    assert b"Prepare first release" not in second_notes
    assert f"since={first_source} until={second_source}".encode("ascii") in second_notes
    assert (root / "release-notes" / "1.0.0.md").read_bytes() == first_notes
    assert (
        release_section(changelog.decode("utf-8"), "1.0.0").encode("utf-8")
        == first_notes
    )
    assert changelog.endswith(HISTORY.encode("utf-8"))
    assert changelog.count(b"Existing cumulative feature.") == 1
    assert_rejected_unchanged(root, first_source, second_source, version="1.1.0")


def test_later_nonempty_unreleased_is_rejected(repository):
    root, cutover = repository
    first_source = commit(root, "A feature (#10)", metadata(added=("First feature.",)))
    first = prepare(root, cutover, first_source)
    assert first.returncode == 0, first.stderr
    path = root / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "### Added\n", "### Added\n- A stray cumulative entry.\n", 1
        ),
        encoding="utf-8",
    )
    endpoint = commit(
        root, "Another feature (#11)", metadata(added=("Another feature.",))
    )

    result = assert_rejected_unchanged(root, first_source, endpoint, version="1.1.0")

    assert "Unreleased must be empty" in result.stderr


def test_malformed_recorded_range_cannot_be_silently_treated_as_first_release(
    repository,
):
    root, cutover = repository
    path = root / "CHANGELOG.md"
    path.write_text(
        BASELINE.replace(
            "## 0.9.0 - 2025-01-02",
            "## 0.9.0 - 2025-01-02\n\n"
            "<!-- release-source: since=invalid until=invalid -->",
        ),
        encoding="utf-8",
    )

    result = assert_rejected_unchanged(root, cutover, cutover)

    assert "malformed recorded release source range" in result.stderr


def test_redirected_release_notes_directory_is_rejected(repository):
    root, cutover = repository
    destination = root / "external-notes"
    destination.mkdir()
    sentinel = destination / "retained.txt"
    sentinel.write_bytes(b"Do not change redirected output.")
    try:
        (root / "release-notes").symlink_to(destination, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symbolic links is not permitted on this system")

    result = assert_rejected_unchanged(root, cutover, cutover)

    assert "symbolic links" in result.stderr
    assert sentinel.read_bytes() == b"Do not change redirected output."
    assert not (destination / "1.0.0.md").exists()


def test_missing_latest_source_marker_is_rejected(repository):
    root, cutover = repository
    first = prepare(root, cutover, cutover)
    assert first.returncode == 0, first.stderr
    path = root / "CHANGELOG.md"
    original = path.read_bytes()
    latest = b"## 1.1.0 - 2026-10-23\n\n### Added\n\n### Changed\n\n### Fixed\n"
    path.write_bytes(original.replace(b"## 1.0.0 - ", latest + b"## 1.0.0 - ", 1))
    endpoint = commit(root, "A new feature (#13)", metadata(added=("New feature.",)))

    result = assert_rejected_unchanged(root, cutover, endpoint, version="1.2.0")

    assert "latest release is missing its recorded source range" in result.stderr


def test_duplicate_latest_source_markers_are_rejected(repository):
    root, cutover = repository
    first = prepare(root, cutover, cutover)
    assert first.returncode == 0, first.stderr
    path = root / "CHANGELOG.md"
    marker = f"<!-- release-source: since={cutover} until={cutover} -->".encode("ascii")
    path.write_bytes(path.read_bytes().replace(marker, marker + b"\n" + marker, 1))
    endpoint = commit(root, "A new feature (#13)", metadata(added=("New feature.",)))

    result = assert_rejected_unchanged(root, cutover, endpoint, version="1.1.0")

    assert "exactly one recorded source range" in result.stderr
