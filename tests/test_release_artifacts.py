"""Artifact metadata tests are offline; installed numeric smoke runs in release CI."""

import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_release_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)
VERSION = "0.4.4"


def metadata(name="coker", version=VERSION):
    return (
        f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n"
        "An offline artifact-metadata fixture.\n"
    ).encode()


def wheel(root, version=VERSION, members=None):
    path = root / f"coker-{version}-py3-none-any.whl"
    if members is None:
        members = [(f"coker-{version}.dist-info/METADATA", metadata(version=version))]
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)
    return path


def sdist(root, version=VERSION, members=None):
    path = root / f"coker-{version}.tar.gz"
    if members is None:
        members = [(f"coker-{version}/PKG-INFO", metadata(version=version))]
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


@pytest.mark.parametrize(
    "filename",
    [
        f"coker-{VERSION}-py3-none-any.whl",
        f"coker-{VERSION}-cp310-cp310-manylinux_2_17_x86_64.whl",
        f"coker-{VERSION}-1build-py2.py3-none-any.whl",
    ],
)
def test_accepts_unique_pair_and_returns_absolute_paths(tmp_path, filename):
    built_wheel = wheel(tmp_path)
    expected = built_wheel.rename(tmp_path / filename), sdist(tmp_path)
    assert artifacts.validate_artifacts(tmp_path, VERSION) == expected
    assert all(path.is_absolute() for path in expected)


def test_selects_release_pair_without_unpublished_hidden_build_files(tmp_path):
    expected = wheel(tmp_path), sdist(tmp_path)
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")

    assert artifacts.validate_artifacts(tmp_path, VERSION) == expected


@pytest.mark.parametrize("present", [(), ("wheel",), ("sdist",)])
def test_rejects_missing_artifacts(tmp_path, present):
    if "wheel" in present:
        wheel(tmp_path)
    if "sdist" in present:
        sdist(tmp_path)
    with pytest.raises(ValueError, match="exactly one wheel and one sdist"):
        artifacts.validate_artifacts(tmp_path, VERSION)


@pytest.mark.parametrize("extra", ["wheel", "sdist"])
def test_rejects_ambiguous_release_pairs(tmp_path, extra):
    wheel(tmp_path)
    sdist(tmp_path)
    if extra == "wheel":
        wheel(tmp_path, version="0.4.3")
    else:
        sdist(tmp_path, version="0.4.3")
    with pytest.raises(ValueError, match="exactly one wheel and one sdist"):
        artifacts.validate_artifacts(tmp_path, VERSION)


def test_rejects_stray_zip_alongside_release_pair(tmp_path):
    wheel(tmp_path)
    sdist(tmp_path)
    (tmp_path / f"coker-{VERSION}.zip").write_bytes(b"unsupported sdist")
    with pytest.raises(ValueError, match="unexpected release artifact"):
        artifacts.validate_artifacts(tmp_path, VERSION)


@pytest.mark.parametrize(
    "kind, document, message",
    [
        ("wheel", metadata(name="another-package"), "expected package coker"),
        ("wheel", metadata(version="0.4.3"), "expected version 0.4.4"),
        ("sdist", metadata(version="0.4.3"), "expected version 0.4.4"),
        ("wheel", b"Name: coker\n\n", "exactly one Version header"),
        ("wheel", b"Version: 0.4.4\n\n", "exactly one Name header"),
        (
            "wheel",
            b"Name: coker\nName: another-package\nVersion: 0.4.4\n\n",
            "exactly one Name header",
        ),
        (
            "wheel",
            b"Name: coker\nVersion: 0.4.4\nVersion: 0.4.3\n\n",
            "exactly one Version header",
        ),
        ("wheel", b"Name: coker\nVersion: \n\n", "exactly one Version header"),
    ],
)
def test_rejects_inconsistent_embedded_identity(tmp_path, kind, document, message):
    wheel_members = None
    sdist_members = None
    if kind == "wheel":
        wheel_members = [(f"coker-{VERSION}.dist-info/METADATA", document)]
    else:
        sdist_members = [(f"coker-{VERSION}/PKG-INFO", document)]
    wheel(tmp_path, members=wheel_members)
    sdist(tmp_path, members=sdist_members)
    with pytest.raises(ValueError, match=message):
        artifacts.validate_artifacts(tmp_path, VERSION)


@pytest.mark.parametrize(
    "members",
    [
        [],
        [("coker-0.4.3.dist-info/METADATA", metadata())],
        [
            ("coker-0.4.4.dist-info/METADATA", metadata()),
            ("other-0.4.4.dist-info/METADATA", metadata(name="other")),
        ],
    ],
)
def test_rejects_missing_or_ambiguous_wheel_metadata(tmp_path, members):
    wheel(tmp_path, members=members)
    sdist(tmp_path)
    with pytest.raises(ValueError, match="expected exactly one.*METADATA"):
        artifacts.validate_artifacts(tmp_path, VERSION)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_rejects_duplicate_metadata_archive_entries(tmp_path, kind):
    wheel_members = None
    sdist_members = None
    if kind == "wheel":
        wheel_members = [(f"coker-{VERSION}.dist-info/METADATA", metadata())] * 2
    else:
        sdist_members = [(f"coker-{VERSION}/PKG-INFO", metadata())] * 2
    if kind == "wheel":
        with pytest.warns(UserWarning, match="Duplicate name"):
            wheel(tmp_path, members=wheel_members)
    else:
        wheel(tmp_path)
    sdist(tmp_path, members=sdist_members)
    with pytest.raises(ValueError, match="expected exactly one"):
        artifacts.validate_artifacts(tmp_path, VERSION)


@pytest.mark.parametrize(
    "members, message",
    [
        ([], "expected exactly one.*PKG-INFO"),
        (
            [("coker-0.4.4/PKG-INFO", metadata()), ("other/file", b"content")],
            "member outside package root",
        ),
        (
            [("coker-0.4.4/PKG-INFO", metadata()), ("coker-0.4.4/../file", b"")],
            "member outside package root",
        ),
    ],
)
def test_rejects_invalid_sdist_layout(tmp_path, members, message):
    wheel(tmp_path)
    sdist(tmp_path, members=members)
    with pytest.raises(ValueError, match=message):
        artifacts.validate_artifacts(tmp_path, VERSION)


def test_rejects_sdist_metadata_symlink(tmp_path):
    wheel(tmp_path)
    source = tmp_path / f"coker-{VERSION}.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        member = tarfile.TarInfo(f"coker-{VERSION}/PKG-INFO")
        member.type = tarfile.SYMTYPE
        member.linkname = "somewhere-else"
        archive.addfile(member)
    with pytest.raises(ValueError, match="PKG-INFO must be a regular file"):
        artifacts.validate_artifacts(tmp_path, VERSION)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_rejects_filename_and_metadata_version_disagreement(tmp_path, kind):
    artifact = wheel(tmp_path) if kind == "wheel" else sdist(tmp_path)
    artifact.rename(artifact.with_name(artifact.name.replace(VERSION, "0.4.3")))
    if kind == "wheel":
        sdist(tmp_path)
    else:
        wheel(tmp_path)
    with pytest.raises(ValueError, match="filename must identify coker 0.4.4"):
        artifacts.validate_artifacts(tmp_path, VERSION)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_cli_rejects_corrupt_archives_before_installing(tmp_path, kind):
    artifacts_paths = {"wheel": wheel(tmp_path), "sdist": sdist(tmp_path)}
    artifacts_paths[kind].write_bytes(b"not an archive")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dist", str(tmp_path), "--version", VERSION],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stderr.startswith("Release artifact verification failed:")
    assert "Validated release pair" not in result.stdout


def test_cli_rejects_wrong_release_version_even_with_optimization(tmp_path):
    wheel(tmp_path)
    sdist(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(SCRIPT),
            "--dist",
            str(tmp_path),
            "--version",
            "0.4.5",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "filename must identify coker 0.4.5" in result.stderr
    assert "Validated release pair" not in result.stdout
