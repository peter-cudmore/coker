"""Validate pull-request metadata against the unreleased changelog."""

import os
import re
import sys
from pathlib import Path


CHANGELOG_CATEGORIES = ("Added", "Changed", "Fixed")


def _second_level_sections(document: str, title: str) -> list[re.Match[str]]:
    pattern = re.compile(rf"^## {re.escape(title)}[ \t]*\r?$", re.MULTILINE)
    return list(pattern.finditer(document))


def _second_level_section(document: str, title: str) -> str:
    matches = _second_level_sections(document, title)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '## {title}' section")

    start = matches[0].end()
    following = re.search(r"^## [^#].*$", document[start:], re.MULTILINE)
    end = start + following.start() if following else len(document)
    return document[start:end]


def _has_content(section: str) -> bool:
    without_comments = re.sub(r"<!--.*?-->", "", section, flags=re.DOTALL)
    return bool(without_comments.strip(" \t\r\n-"))


def _changelog_entries(section: str) -> dict[str, tuple[str, ...]]:
    headings = list(
        re.finditer(r"^### ([^\r\n]+)[ \t]*\r?$", section, re.MULTILINE)
    )
    titles = [heading.group(1) for heading in headings]
    if titles != list(CHANGELOG_CATEGORIES):
        expected = ", ".join(CHANGELOG_CATEGORIES)
        actual = ", ".join(titles) or "none"
        raise ValueError(
            f"expected changelog categories {expected}; found {actual}"
        )

    entries = {}
    for index, heading in enumerate(headings):
        start = heading.end()
        end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(section)
        )
        content = section[start:end]
        category_entries = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.fullmatch(r"-\s+(.+?)\s*", stripped)
            if match is None:
                category = heading.group(1)
                raise ValueError(
                    f"'{category}' contains non-bullet content: {stripped}"
                )
            category_entries.append(" ".join(match.group(1).split()))
        entries[heading.group(1)] = tuple(category_entries)
    return entries


def validate(pr_body: str, changelog: str) -> None:
    summary = _second_level_section(pr_body, "Summary")
    if not _has_content(summary):
        raise ValueError("'## Summary' must contain content")

    descriptions = _second_level_sections(pr_body, "Description")
    if len(descriptions) > 1:
        raise ValueError(
            "expected at most one optional '## Description' section"
        )

    pr_entries = _changelog_entries(
        _second_level_section(pr_body, "Changelog")
    )
    changelog_entries = _changelog_entries(
        _second_level_section(changelog, "Unreleased")
    )
    if not any(pr_entries.values()):
        raise ValueError("'## Changelog' must contain at least one entry")
    if pr_entries != changelog_entries:
        message = (
            "PR changelog must exactly match CHANGELOG.md's Unreleased "
            "section\n"
            f"PR: {pr_entries}\nCHANGELOG.md: {changelog_entries}"
        )
        raise ValueError(message)


def main() -> int:
    pr_body = os.environ.get("PR_BODY")
    if pr_body is None:
        print("PR_BODY is required", file=sys.stderr)
        return 1

    try:
        validate(pr_body, Path("CHANGELOG.md").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"PR metadata validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
