"""Prepare release notes from validated, first-parent squash-commit metadata.

Usage:
    python scripts/prepare_release.py --since BASE --until SOURCE \
        --version 1.2.3 --date 2026-10-22 --repository OWNER/REPO
"""

import argparse
from datetime import date
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

if __package__:
    from .check_pr_metadata import CHANGELOG_CATEGORIES, parse_changelog, validate
else:
    from check_pr_metadata import CHANGELOG_CATEGORIES, parse_changelog, validate


_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_SHA = r"(?:[0-9a-f]{40}|[0-9a-f]{64})"
_SOURCE = re.compile(
    rf"^<!-- release-source: since=({_SHA}) until=({_SHA}) -->\r?$", re.MULTILINE
)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Git command failed")
    return result.stdout.rstrip("\n")


def _resolve(root: Path, reference: str) -> str:
    if not reference or reference.startswith("-"):
        raise ValueError("source references must be explicit Git commit references")
    return _git(
        root, "rev-parse", "--verify", "--end-of-options", f"{reference}^{{commit}}"
    )


def _validate_arguments(version: str, release_date: str, repository: str) -> None:
    if _VERSION.fullmatch(version) is None:
        raise ValueError("version must be SemVer without a 'v' prefix")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", release_date) is None:
        raise ValueError("date must be YYYY-MM-DD")
    try:
        date.fromisoformat(release_date)
    except ValueError as error:
        raise ValueError("date must be a valid YYYY-MM-DD date") from error
    if (
        re.fullmatch(
            r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?/[A-Za-z0-9_.-]+", repository
        )
        is None
    ):
        raise ValueError("repository must be OWNER/REPO")
    if repository.split("/")[1] in {".", ".."}:
        raise ValueError("repository must be OWNER/REPO")


def _split_changelog(document: str) -> tuple[str, str, str]:
    headings = list(re.finditer(r"^## ([^\r\n]+)\r?$", document, re.MULTILINE))
    unreleased = [heading for heading in headings if heading.group(1) == "Unreleased"]
    if len(unreleased) != 1 or not headings or headings[0] != unreleased[0]:
        raise ValueError(
            "CHANGELOG.md must start with exactly one ## Unreleased section"
        )
    heading = unreleased[0]
    end = headings[1].start() if len(headings) > 1 else len(document)
    return document[: heading.start()], document[heading.end() : end], document[end:]


def _source_endpoint(history: str) -> str | None:
    markers = list(_SOURCE.finditer(history))
    if history.count("<!-- release-source:") != len(markers):
        raise ValueError("malformed recorded release source range")
    if not markers:
        return None
    sections = list(re.finditer(r"^## [^\r\n]+\r?$", history, re.MULTILINE))
    if not sections or not (sections[0].end() < markers[0].start()):
        raise ValueError("recorded release source must belong to the latest release")
    if len(sections) > 1 and markers[0].start() >= sections[1].start():
        raise ValueError("latest release is missing its recorded source range")
    latest_end = sections[1].start() if len(sections) > 1 else len(history)
    if sum(marker.start() < latest_end for marker in markers) != 1:
        raise ValueError("latest release must have exactly one recorded source range")
    return markers[0].group(2)


def _entries(root: Path, since: str, until: str) -> dict[str, list[tuple[str, str]]]:
    commits = _git(
        root, "rev-list", "--first-parent", "--reverse", f"{since}..{until}"
    ).splitlines()
    entries = {category: [] for category in CHANGELOG_CATEGORIES}
    for commit in commits:
        parents = _git(root, "show", "-s", "--format=%P", commit).split()
        if len(parents) != 1:
            raise ValueError(f"{commit}: expected nonmerge squash history")
        message = _git(root, "show", "-s", "--format=%B", commit)
        title, _, body = message.partition("\n")
        try:
            validate(body)
            changes = parse_changelog(body)
        except ValueError as error:
            raise ValueError(f"{commit}: invalid PR metadata: {error}") from error
        if not any(changes.values()):
            continue
        match = re.search(r" \(#([1-9][0-9]*)\)$", title)
        if match is None:
            raise ValueError(
                f"{commit}: nonempty changelog requires a squash title ending in (#N)"
            )
        number = match.group(1)
        for category in CHANGELOG_CATEGORIES:
            entries[category].extend((entry, number) for entry in changes[category])
    return entries


def _write_release(
    root: Path, changelog: Path, notes: Path, document: str, section: str
) -> None:
    temporary_paths = []
    try:
        for content in (document, section):
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=root, delete=False
            ) as temporary:
                temporary_paths.append(Path(temporary.name))
                temporary.write(content)
        notes.parent.mkdir(exist_ok=True)
        os.replace(temporary_paths[1], notes)
        os.replace(temporary_paths[0], changelog)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def prepare_release(
    since: str,
    until: str,
    version: str,
    release_date: str,
    repository: str,
    root: Path | None = None,
) -> Path:
    """Aggregate one explicit source range and write its versioned release notes.

    Args:
        since: Exclusive source commit; use the cutover PR merge SHA initially,
            then the last release's recorded source endpoint, not its release tag.
        until: Inclusive source commit on the same first-parent lineage.
        version: SemVer release number without a ``v`` prefix.
        release_date: Release date in YYYY-MM-DD format.
        repository: GitHub OWNER/REPO used for pull-request links.
        root: Git working directory; defaults to the current directory.

    Returns:
        The path to ``release-notes/VERSION.md``. For example, call
        ``prepare_release(base, source, "1.2.3", "2026-10-22", "owner/repo")``.

    Raises:
        ValueError: Inputs, history, or metadata are invalid. Validation completes
            before either output is written.
        OSError: Git cannot be started or release files cannot be read or written.
    """
    _validate_arguments(version, release_date, repository)
    root = Path(_git(root or Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    changelog = root / "CHANGELOG.md"
    notes = root / "release-notes" / f"{version}.md"
    if (
        changelog.is_symlink()
        or notes.parent.is_symlink()
        or notes.is_symlink()
        or notes.parent.resolve() != notes.parent
    ):
        raise ValueError(
            "release output paths must not be symbolic links or redirected directories"
        )
    if not changelog.is_file():
        raise ValueError("CHANGELOG.md must be a regular file")
    if notes.exists():
        raise ValueError(f"release notes already exist for {version}")
    if notes.parent.exists() and not notes.parent.is_dir():
        raise ValueError("release-notes must be a directory")
    with changelog.open(encoding="utf-8", newline="") as source:
        document = source.read()
    prefix, unreleased, history = _split_changelog(document)
    for heading in re.finditer(r"^## ([^\r\n]+)\r?$", history, re.MULTILINE):
        if re.match(rf"(?:\[)?{re.escape(version)}(?:\])?(?:\s|$)", heading.group(1)):
            raise ValueError(f"version {version} already exists in CHANGELOG.md")
    baseline = parse_changelog("## Changelog\n" + unreleased)
    source_since = _resolve(root, since)
    source_until = _resolve(root, until)
    _git(root, "merge-base", "--is-ancestor", source_since, source_until)
    lineage = _git(root, "rev-list", "--first-parent", source_until).splitlines()
    if source_since not in lineage:
        raise ValueError("--since must be on --until's first-parent lineage")
    previous_endpoint = _source_endpoint(history)
    if previous_endpoint is not None:
        if source_since != previous_endpoint:
            raise ValueError(
                "--since must equal the previous release's recorded source endpoint"
            )
        if any(baseline.values()):
            raise ValueError(
                "Unreleased must be empty after the first release preparation"
            )
    additions = _entries(root, source_since, source_until)
    if not any(baseline.values()) and not any(additions.values()):
        raise ValueError("source range and Unreleased contain no release entries")
    lines = [
        f"## {version} - {release_date}",
        "",
        f"<!-- release-source: since={source_since} until={source_until} -->",
        "",
    ]
    for category in CHANGELOG_CATEGORIES:
        lines.append(f"### {category}")
        lines.extend(f"- {entry}" for entry in baseline[category])
        lines.extend(
            f"- {entry} ([#{number}](https://github.com/{repository}/pull/{number}))"
            for entry, number in additions[category]
        )
        lines.append("")
    section = "\n".join(lines).rstrip("\n") + "\n"
    empty_unreleased = "## Unreleased\n\n" + "".join(
        f"### {category}\n\n" for category in CHANGELOG_CATEGORIES
    )
    updated = prefix + empty_unreleased + section + history
    _write_release(root, changelog, notes, updated, section)
    return notes


def main() -> int:
    """Run the CLI; use ``python scripts/prepare_release.py --help`` for arguments.

    Returns:
        Zero after preparing both files, or one on invalid input or an I/O error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since", required=True, help="exclusive immutable source baseline"
    )
    parser.add_argument("--until", required=True, help="inclusive source endpoint")
    parser.add_argument("--version", required=True, help="SemVer without a v prefix")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--repository", required=True, help="GitHub OWNER/REPO")
    arguments = parser.parse_args()
    try:
        prepare_release(
            arguments.since,
            arguments.until,
            arguments.version,
            arguments.date,
            arguments.repository,
        )
    except (OSError, ValueError) as error:
        print(f"Release preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
