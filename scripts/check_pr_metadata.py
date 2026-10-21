"""Validate PR-local changelog metadata preserved in squash commit bodies."""

import os
import re
import sys


CHANGELOG_CATEGORIES = ("Added", "Changed", "Fixed")


def _without_comments(document: str) -> str:
    return re.sub(r"<!--.*?(?:-->|\Z)", "", document, flags=re.DOTALL)


def _second_level_sections(document: str, title: str) -> list[re.Match[str]]:
    pattern = re.compile(rf"^## {re.escape(title)}[ \t]*\r?$", re.MULTILINE)
    return list(pattern.finditer(document))


def _second_level_section(document: str, title: str) -> str:
    matches = _second_level_sections(document, title)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '## {title}' section")

    start = matches[0].end()
    following = re.search(r"^##(?:[ \t]+|\r?$)", document[start:], re.MULTILINE)
    end = start + following.start() if following else len(document)
    return document[start:end]


def _is_placeholder(content: str) -> bool:
    normalized = content.strip().rstrip(".!:;").strip().casefold()
    return normalized in {"none", "n/a", "no changes"}


def _has_content(section: str) -> bool:
    for line in section.splitlines():
        content = line.strip(" \t\r\n-*")
        if any(character.isalnum() for character in content) and not _is_placeholder(
            content
        ):
            return True
    return False


def parse_changelog(document: str) -> dict[str, tuple[str, ...]]:
    """Extract the ordered Added, Changed, and Fixed entries from a PR body.

    HTML comments are ignored. Each category must appear exactly once, in order,
    and contain only single-line ``- entry`` bullets or blank lines. Empty
    categories are allowed; placeholder bullets are not.

    Args:
        document: Markdown containing exactly one ``## Changelog`` section.

    Returns:
        A category-to-entry mapping in Added, Changed, Fixed order. Entry
        whitespace is normalized and empty categories contain empty tuples.

    Raises:
        ValueError: If the changelog section, categories, or entries are invalid.
    """
    section = _second_level_section(_without_comments(document), "Changelog")
    headings = list(re.finditer(r"^### ([^\r\n]+?)[ \t]*\r?$", section, re.MULTILINE))
    titles = [heading.group(1) for heading in headings]
    if titles != list(CHANGELOG_CATEGORIES):
        expected = ", ".join(CHANGELOG_CATEGORIES)
        actual = ", ".join(titles) or "none"
        raise ValueError(f"expected changelog categories {expected}; found {actual}")
    if section[: headings[0].start()].strip():
        raise ValueError("'## Changelog' contains content before the first category")

    entries = {}
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(section)
        category = heading.group(1)
        category_entries = []
        for line in section[start:end].splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.fullmatch(r"-\s+(.+?)\s*", stripped)
            if match is None:
                raise ValueError(
                    f"'{category}' contains non-bullet content: {stripped}"
                )
            entry = " ".join(match.group(1).split())
            if _is_placeholder(entry) or not _has_content(entry):
                raise ValueError(f"'{category}' contains a placeholder entry: {entry}")
            category_entries.append(entry)
        entries[category] = tuple(category_entries)
    return entries


def validate(pr_body: str) -> None:
    """Validate a PR body without consulting the cumulative changelog.

    Args:
        pr_body: Markdown with a substantive Summary, an optional single
            Description, and a Changelog parsed by :func:`parse_changelog`.
            An entirely empty changelog requires a substantive Description
            explaining why no release note is needed.

    Raises:
        ValueError: If required metadata is absent, duplicated, or malformed.
    """
    document = _without_comments(pr_body)
    summary = _second_level_section(document, "Summary")
    if not _has_content(summary):
        raise ValueError("'## Summary' must contain content")

    descriptions = _second_level_sections(document, "Description")
    if len(descriptions) > 1:
        raise ValueError("expected at most one optional '## Description' section")

    entries = parse_changelog(document)
    if not any(entries.values()):
        description = (
            _second_level_section(document, "Description") if descriptions else ""
        )
        if not _has_content(description):
            raise ValueError(
                "an empty '## Changelog' requires a substantive '## Description' "
                "explaining why no release note is needed"
            )


def main() -> int:
    """Validate the PR_BODY environment variable for command-line use.

    Returns:
        Zero for valid PR metadata, or one with a diagnostic on standard error.
    """
    pr_body = os.environ.get("PR_BODY")
    if pr_body is None:
        print("PR_BODY is required", file=sys.stderr)
        return 1

    try:
        validate(pr_body)
    except ValueError as error:
        print(f"PR metadata validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
