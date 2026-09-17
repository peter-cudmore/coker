import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_pr_metadata.py"
CHANGELOG = """# Changelog

## Unreleased

### Added
- adaptive mesh refinement

### Changed
- cache operators per formulation

### Fixed
"""
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


def run_validator(tmp_path, pr_body=PR_BODY, changelog=CHANGELOG):
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    environment = os.environ | {"PR_BODY": pr_body}
    return subprocess.run(
        [sys.executable, SCRIPT],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_accepts_matching_pr_changelog(tmp_path):
    result = run_validator(tmp_path)

    assert result.returncode == 0


@pytest.mark.parametrize(
    ("pr_body", "message"),
    [
        (
            PR_BODY.replace("## Summary\n- Add adaptive refinement.\n\n", ""),
            "Summary",
        ),
        (
            PR_BODY.replace("- adaptive mesh refinement", "- another feature"),
            "exactly match",
        ),
        (PR_BODY.replace("### Fixed", "### Removed"), "categories"),
        (
            PR_BODY.replace(
                "## Changelog",
                "## Description\n- Duplicate description.\n\n## Changelog",
            ),
            "optional",
        ),
    ],
)
def test_rejects_invalid_pr_metadata(tmp_path, pr_body, message):
    result = run_validator(tmp_path, pr_body=pr_body)

    assert result.returncode == 1
    assert message in result.stderr
