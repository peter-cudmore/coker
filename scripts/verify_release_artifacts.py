"""Validate and exercise the exact wheel and sdist selected for publication.

Usage:
    python scripts/verify_release_artifacts.py --dist dist --version 0.4.4

Only uv's wheel and .tar.gz sdist pair is supported. Only the supplied wheel is
installed; the smoke runs in disposable environments outside the checkout.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile


_CHECKOUT = Path(__file__).resolve().parents[1]
_SMOKE = _CHECKOUT / "scripts" / "smoke_installation.py"


def _identity(document: bytes, version: str) -> None:
    metadata = BytesParser(policy=default).parsebytes(document)
    for key in ("Name", "Version"):
        values = metadata.get_all(key, [])
        if len(values) != 1 or not str(values[0]).strip():
            raise ValueError(f"expected exactly one {key} header")
    name = re.sub(r"[-_.]+", "-", str(metadata["Name"])).lower()
    if name != "coker":
        raise ValueError(f"expected package coker, found {metadata['Name']}")
    if str(metadata["Version"]) != version:
        raise ValueError(f"expected version {version}, found {metadata['Version']}")


def _wheel_metadata(wheel: Path, version: str) -> None:
    parts = wheel.name.removesuffix(".whl").split("-")
    if (
        len(parts) not in (5, 6)
        or parts[0] != "coker"
        or parts[1] != version
        or any(not part for part in parts)
        or (len(parts) == 6 and re.fullmatch(r"[0-9][A-Za-z0-9_]*", parts[2]) is None)
    ):
        raise ValueError(f"{wheel.name}: filename must identify coker {version}")
    expected = f"coker-{version}.dist-info/METADATA"
    with zipfile.ZipFile(wheel) as archive:
        candidates = [
            member
            for member in archive.infolist()
            if member.filename.endswith(".dist-info/METADATA")
        ]
        if len(candidates) != 1 or candidates[0].filename != expected:
            raise ValueError(f"{wheel.name}: expected exactly one {expected}")
        _identity(archive.read(candidates[0]), version)


def _sdist_metadata(sdist: Path, version: str) -> None:
    root = f"coker-{version}"
    if sdist.name != f"{root}.tar.gz":
        raise ValueError(f"{sdist.name}: filename must identify coker {version}")
    expected = f"{root}/PKG-INFO"
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        if sum(member.name == expected for member in members) != 1:
            raise ValueError(f"{sdist.name}: expected exactly one {expected}")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not path.parts
                or path.parts[0] != root
                or ".." in path.parts
                or "\\" in member.name
            ):
                raise ValueError(
                    f"{sdist.name}: member outside package root: {member.name}"
                )
        member = archive.getmember(expected)
        if not member.isfile():
            raise ValueError(f"{sdist.name}: PKG-INFO must be a regular file")
        document = archive.extractfile(member)
        if document is None:
            raise ValueError(f"{sdist.name}: unreadable PKG-INFO")
        with document:
            _identity(document.read(), version)


def validate_artifacts(dist: Path, version: str) -> tuple[Path, Path]:
    """Check the unique wheel/.tar.gz pair's filenames and embedded metadata.

    Args:
        dist: Directory containing the wheel and sdist intended for publication.
        version: Exact expected package version, without a tag's ``v`` prefix.

    Returns:
        Absolute wheel and sdist paths, in that order.

    Raises:
        ValueError: A package is missing, ambiguous, or has inconsistent metadata.
        OSError: The artifact directory or an archive cannot be read.

    Example:
        wheel, sdist = validate_artifacts(Path("dist"), "0.4.4")
    """
    if not version or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_]*", version) is None:
        raise ValueError("version must be an explicit package version")
    dist = dist.resolve(strict=True)
    # Match upload-artifact's publication boundary: hidden build files are excluded.
    entries = sorted(path for path in dist.iterdir() if not path.name.startswith("."))
    wheels = [path for path in entries if path.name.endswith(".whl")]
    sdists = [path for path in entries if path.name.endswith(".tar.gz")]
    unexpected = [
        path.name for path in entries if not path.name.endswith((".whl", ".tar.gz"))
    ]
    if unexpected:
        raise ValueError(f"unexpected release artifact(s): {', '.join(unexpected)}")
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            "expected exactly one wheel and one sdist; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )
    wheel, sdist = wheels[0], sdists[0]
    if not wheel.is_file() or not sdist.is_file():
        raise ValueError("release artifacts must be regular files")
    _wheel_metadata(wheel, version)
    _sdist_metadata(sdist, version)
    return wheel, sdist


def _verify_installed(wheel: Path) -> None:
    temporary_parent = Path(tempfile.gettempdir()).resolve()
    if temporary_parent.is_relative_to(_CHECKOUT):
        temporary_parent = _CHECKOUT.parent
    with tempfile.TemporaryDirectory(
        prefix="coker-release-", dir=temporary_parent
    ) as temporary:
        work = Path(temporary).resolve()
        for profile in ("base", "casadi"):
            environment = work / profile
            subprocess.run(
                [
                    "uv",
                    "--no-config",
                    "venv",
                    "--python",
                    sys.executable,
                    str(environment),
                ],
                cwd=work,
                check=True,
            )
            python = environment / (
                "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
            )
            requirement = str(wheel) + ("[casadi]" if profile == "casadi" else "")
            subprocess.run(
                [
                    "uv",
                    "--no-config",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--only-binary",
                    ":all:",
                    requirement,
                ],
                cwd=work,
                check=True,
            )
            subprocess.run(
                [str(python), "-I", str(_SMOKE), profile], cwd=work, check=True
            )


def main() -> int:
    """Validate the release pair and run its installed-wheel smoke profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args()
    try:
        wheel, sdist = validate_artifacts(arguments.dist, arguments.version)
        print(f"Validated release pair: {wheel.name}, {sdist.name}", flush=True)
        _verify_installed(wheel)
    except (
        ValueError,
        OSError,
        tarfile.TarError,
        zipfile.BadZipFile,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Release artifact verification failed: {error}", file=sys.stderr)
        return 1
    print(f"Release artifact verification passed: {arguments.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
