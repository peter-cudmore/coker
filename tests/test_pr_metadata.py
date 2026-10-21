import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_pr_metadata.py"
SPEC = importlib.util.spec_from_file_location("check_pr_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)

PR_BODY = """## Summary
- Add adaptive refinement.

## Description
<!-- Optional implementation context. -->

## Changelog
### Added
- adaptive mesh refinement

### Changed
- cache operators per formulation

### Fixed
"""
EMPTY_CHANGELOG = """## Changelog
### Added

### Changed

### Fixed
"""


def run_validator(tmp_path, pr_body: str | None = PR_BODY):
    environment = os.environ.copy()
    environment.pop("PR_BODY", None)
    if pr_body is not None:
        environment["PR_BODY"] = pr_body
    return subprocess.run(
        [sys.executable, SCRIPT],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_accepts_pr_local_changelog_without_cumulative_file(tmp_path):
    result = run_validator(tmp_path)

    assert not (tmp_path / "CHANGELOG.md").exists()
    assert result.returncode == 0
    assert result.stderr == ""


def test_accepts_changed_pr_local_entries_independently_of_cumulative_file(tmp_path):
    # A directory makes any attempted cumulative-changelog file read fail.
    (tmp_path / "CHANGELOG.md").mkdir()
    body = PR_BODY.replace("adaptive mesh refinement", "a different local feature")

    result = run_validator(tmp_path, body)

    assert result.returncode == 0
    assert result.stderr == ""


def test_exports_ordered_normalized_changelog_entries():
    body = PR_BODY.replace(
        "- adaptive mesh refinement", "-   adaptive   mesh\trefinement  "
    ).replace("### Fixed\n", "### Fixed\n- correct boundary conditions\n")

    entries = metadata.parse_changelog(body)

    assert metadata.CHANGELOG_CATEGORIES == ("Added", "Changed", "Fixed")
    assert tuple(entries) == metadata.CHANGELOG_CATEGORIES
    assert entries == {
        "Added": ("adaptive mesh refinement",),
        "Changed": ("cache operators per formulation",),
        "Fixed": ("correct boundary conditions",),
    }
    assert metadata.validate(body) is None


def test_parser_allows_empty_categories_without_validating_description():
    assert metadata.parse_changelog(EMPTY_CHANGELOG) == {
        "Added": (),
        "Changed": (),
        "Fixed": (),
    }


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_ignores_comments_before_detecting_sections_and_headings(tmp_path, newline):
    body = (
        "<!--\n## Summary\n## Description\n## Changelog\n"
        "### Added\n- hidden entry\n### Invalid\n-->\n"
        + PR_BODY.replace("### Added", "<!-- ### Invalid -->\n### Added")
        .replace("### Changed", "<!--\n### Added\n-->\n### Changed")
        .replace(
            "- adaptive mesh refinement", "- adaptive <!-- detail -->mesh refinement"
        )
    ).replace("\n", newline)

    result = run_validator(tmp_path, body)

    assert result.returncode == 0
    assert metadata.parse_changelog(body)["Added"] == ("adaptive mesh refinement",)


def test_accepts_optional_absent_description(tmp_path):
    body = PR_BODY.replace(
        "## Description\n<!-- Optional implementation context. -->\n\n", ""
    )

    assert run_validator(tmp_path, body).returncode == 0


def test_accepts_empty_changelog_with_meaningful_explanation(tmp_path):
    body = (
        "## Summary\n- Correct contributor instructions.\n\n"
        "## Description\n"
        "Only contributor documentation changes; shipped behavior is unchanged.\n\n"
        + EMPTY_CHANGELOG
    )

    assert run_validator(tmp_path, body).returncode == 0
    assert metadata.validate(body) is None


@pytest.mark.parametrize(
    "description",
    [
        None,
        "",
        "<!-- Documentation-only explanation goes here. -->",
        "-",
        "...",
        "None",
        "N/A",
        "- no changes.",
        "- N/A\n- None.",
    ],
)
def test_rejects_empty_changelog_without_meaningful_explanation(tmp_path, description):
    body = "## Summary\n- Update documentation.\n\n"
    if description is not None:
        body += f"## Description\n{description}\n\n"
    body += EMPTY_CHANGELOG

    result = run_validator(tmp_path, body)

    assert result.returncode == 1
    assert "substantive '## Description'" in result.stderr


@pytest.mark.parametrize(
    ("pr_body", "message"),
    [
        (PR_BODY.replace("## Summary\n- Add adaptive refinement.\n\n", ""), "Summary"),
        (PR_BODY.replace("- Add adaptive refinement.", "<!-- hidden -->"), "Summary"),
        (PR_BODY.replace("- Add adaptive refinement.", "-"), "Summary"),
        (PR_BODY.replace("## Summary", "## Summary\n- First.\n## Summary"), "Summary"),
        (PR_BODY.replace("## Changelog", "## Notes"), "Changelog"),
        (PR_BODY + "\n## Changelog\n### Added\n### Changed\n### Fixed\n", "Changelog"),
        (PR_BODY.replace("### Fixed", "### Removed"), "categories"),
        (PR_BODY.replace("### Fixed", "### Changed\n### Fixed"), "categories"),
        (PR_BODY.replace("### Added", "### added"), "categories"),
        (PR_BODY.replace("### Added", "#### Added"), "categories"),
        (PR_BODY.replace("### Added", "###Added"), "categories"),
        (PR_BODY.replace("### Added", "### Added ###"), "categories"),
        (PR_BODY.replace("### Fixed", "### Fixed\n### Other"), "categories"),
        (
            PR_BODY.replace("### Added", "Unexpected introduction\n### Added"),
            "before the first category",
        ),
        (
            PR_BODY.replace("### Added", "- misplaced entry\n### Added"),
            "before the first category",
        ),
        (
            PR_BODY.replace("- adaptive mesh refinement", "adaptive mesh refinement"),
            "non-bullet",
        ),
        (
            PR_BODY.replace("- adaptive mesh refinement", "* adaptive mesh refinement"),
            "non-bullet",
        ),
        (PR_BODY.replace("- adaptive mesh refinement", "-"), "non-bullet"),
        (
            PR_BODY.replace(
                "- adaptive mesh refinement", "- first line\n  continuation"
            ),
            "non-bullet",
        ),
        (
            PR_BODY.replace("### Fixed", "### Fixed\n#### Incorrect heading"),
            "non-bullet",
        ),
        (
            PR_BODY.replace(
                "## Changelog",
                "## Description\n- Duplicate description.\n\n## Changelog",
            ),
            "optional",
        ),
    ],
)
def test_rejects_malformed_metadata_in_cli_and_shared_validator(
    tmp_path, pr_body, message
):
    result = run_validator(tmp_path, pr_body)

    assert result.returncode == 1
    assert message in result.stderr
    with pytest.raises(ValueError, match=message):
        metadata.validate(pr_body)


@pytest.mark.parametrize(
    "entry", ["None", "none.", "N/A", "n/a", "No changes", "no changes.", "..."]
)
def test_rejects_placeholder_changelog_bullets(tmp_path, entry):
    body = PR_BODY.replace("- adaptive mesh refinement", f"- {entry}")

    result = run_validator(tmp_path, body)

    assert result.returncode == 1
    assert "placeholder" in result.stderr
    with pytest.raises(ValueError, match="placeholder"):
        metadata.parse_changelog(body)


def test_missing_environment_body_reports_clear_error(tmp_path):
    result = run_validator(tmp_path, None)

    assert result.returncode == 1
    assert "PR_BODY is required" in result.stderr


@pytest.mark.parametrize(
    "changelog",
    [
        "## Changelog\n### Changed\n### Added\n### Fixed\n",
        "## Changelog\n### Added\n### Fixed\n",
        "## Changelog\n### Added\n### Changed\n",
        "<!--\n## Changelog\n### Added\n### Changed\n### Fixed\n-->\n",
    ],
)
def test_parser_requires_all_categories_in_order_in_visible_section(changelog):
    with pytest.raises(ValueError):
        metadata.parse_changelog(changelog)
