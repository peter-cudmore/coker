import io
import subprocess
import sys
import tarfile
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest


# The release tool runs on Python 3.13; runtime CI also covers Python 3.10.
pytest.importorskip("tomllib")

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_release.py"
VERSION = "0.4.4"
METADATA = (
    f"Metadata-Version: 2.4\nName: coker\nVersion: {VERSION}\n\n".encode()
)
WHEEL_METADATA = f"coker-{VERSION}.dist-info/METADATA"
SDIST_METADATA = f"coker-{VERSION}/PKG-INFO"


def write_wheel(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def write_sdist(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


@pytest.fixture
def release(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "coker"\nversion = "{VERSION}"\n',
        encoding="utf-8",
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    write_wheel(
        dist / f"coker-{VERSION}-py3-none-any.whl",
        {WHEEL_METADATA: METADATA},
    )
    write_sdist(
        dist / f"coker-{VERSION}.tar.gz", {SDIST_METADATA: METADATA}
    )
    return tmp_path


def run_validator(root, tag=None):
    command = [sys.executable, str(SCRIPT)]
    if tag is not None:
        command.extend(["--tag", tag])
    return subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def write_changelog(root, heading):
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## Unreleased\n\n{heading}\n\n- Released.\n",
        encoding="utf-8",
    )


def test_accepts_artifacts_without_release_changelog(release):
    result = run_validator(release)

    assert result.returncode == 0, result.stderr
    assert f"coker {VERSION}" in result.stdout


def test_accepts_matching_tag_with_todays_changelog(release):
    write_changelog(release, f"## {VERSION} - {date.today().isoformat()}")

    result = run_validator(release, f"v{VERSION}")

    assert result.returncode == 0, result.stderr
    assert f"tag v{VERSION}" in result.stdout


def test_rejects_mismatched_tag(release):
    write_changelog(release, f"## {VERSION} - 2024-02-29")

    result = run_validator(release, "v0.0.1")

    assert result.returncode == 1
    assert f"tag must be v{VERSION}" in result.stderr


@pytest.mark.parametrize(
    "heading",
    [
        "## 0.0.1 - 2024-02-29",
        f"## {VERSION}",
        f"## {VERSION} - 20240229",
        f"## {VERSION} - 2024-02-30",
        f"## {VERSION} - {(date.today() + timedelta(days=1)).isoformat()}",
        f"## {VERSION} - 2024-02-29\n\n## {VERSION} - 2024-03-01",
    ],
    ids=[
        "wrong-version",
        "missing-date",
        "non-iso-date",
        "impossible-date",
        "future-date",
        "duplicate-release",
    ],
)
def test_rejects_invalid_release_changelog(release, heading):
    write_changelog(release, heading)

    result = run_validator(release, f"v{VERSION}")

    assert result.returncode == 1
    assert "Release validation failed" in result.stderr


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (b"Name: coker", b"Name: another-package", "Name must be coker"),
        (
            f"Version: {VERSION}".encode(),
            b"Version: 0.0.1",
            f"Version must be {VERSION}",
        ),
    ],
    ids=["wrong-name", "wrong-version"],
)
def test_rejects_inconsistent_artifact_metadata(
    release, artifact, original, replacement, message
):
    metadata = METADATA.replace(original, replacement)
    if artifact == "wheel":
        write_wheel(
            next((release / "dist").glob("*.whl")),
            {WHEEL_METADATA: metadata},
        )
    else:
        write_sdist(
            next((release / "dist").glob("*.tar.gz")),
            {SDIST_METADATA: metadata},
        )

    result = run_validator(release)

    assert result.returncode == 1
    assert message in result.stderr


@pytest.mark.parametrize("artifact", ["wheel", "sdist"])
@pytest.mark.parametrize("count", [0, 2], ids=["missing", "ambiguous"])
def test_rejects_missing_or_ambiguous_metadata(release, artifact, count):
    if artifact == "wheel":
        write_wheel(
            next((release / "dist").glob("*.whl")),
            {
                f"package-{index}.dist-info/METADATA": METADATA
                for index in range(count)
            },
        )
    else:
        members = {
            f"package-{index}/PKG-INFO": METADATA for index in range(count)
        }
        # Nested egg-info is not the source distribution's own metadata.
        members[f"coker-{VERSION}/src/coker.egg-info/PKG-INFO"] = METADATA
        write_sdist(next((release / "dist").glob("*.tar.gz")), members)

    result = run_validator(release)

    assert result.returncode == 1
    assert "expected exactly one" in result.stderr


@pytest.mark.parametrize("suffix", ["whl", "tar.gz"])
@pytest.mark.parametrize("count", [0, 2], ids=["missing", "extra"])
def test_requires_one_artifact_of_each_kind(release, suffix, count):
    artifact = next((release / "dist").glob(f"*.{suffix}"))
    if count == 0:
        artifact.unlink()
    else:
        (artifact.parent / f"extra.{suffix}").write_bytes(
            artifact.read_bytes()
        )

    result = run_validator(release)

    assert result.returncode == 1
    assert "exactly one wheel and one sdist" in result.stderr


@pytest.mark.parametrize("suffix", ["whl", "tar.gz"])
def test_rejects_unreadable_archives(release, suffix):
    next((release / "dist").glob(f"*.{suffix}")).write_bytes(b"not an archive")

    result = run_validator(release)

    assert result.returncode == 1
    assert "Release validation failed" in result.stderr
