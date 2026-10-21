"""Validate built distribution metadata and, optionally, release identity."""

import argparse
import re
import sys
import tarfile
import tomllib
import zipfile
from datetime import date
from email.parser import BytesParser
from pathlib import Path


def _read_wheel(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(members) != 1:
            raise ValueError("expected exactly one .dist-info/METADATA file")
        return archive.read(members[0])


def _read_sdist(path: Path) -> bytes:
    with tarfile.open(path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and member.name.count("/") == 1
            and member.name.endswith("/PKG-INFO")
        ]
        if len(members) != 1:
            raise ValueError("expected exactly one top-level PKG-INFO file")
        stream = archive.extractfile(members[0])
        assert stream is not None
        with stream:
            return stream.read()


def _validate_metadata(content: bytes, version: str) -> None:
    metadata = BytesParser().parsebytes(content)
    if metadata.get_all("Name") != ["coker"]:
        raise ValueError("metadata Name must be coker")
    if metadata.get_all("Version") != [version]:
        raise ValueError(f"metadata Version must be {version}")


def _validate_tag(tag: str, version: str, changelog: str) -> None:
    if tag != f"v{version}":
        raise ValueError(f"tag must be v{version}; got {tag}")
    headings = re.findall(
        rf"^## {re.escape(version)} - (\d{{4}}-\d{{2}}-\d{{2}})[ \t]*\r?$",
        changelog,
        flags=re.MULTILINE,
    )
    if len(headings) != 1:
        raise ValueError(
            f"expected exactly one dated changelog section: "
            f"## {version} - YYYY-MM-DD"
        )
    released = date.fromisoformat(headings[0])
    if released > date.today():
        raise ValueError("changelog release date must not be in the future")


def validate(root: Path, tag: str | None = None) -> str:
    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    wheels = list((root / "dist").glob("*.whl"))
    sdists = list((root / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("dist/ must contain exactly one wheel and one sdist")
    try:
        _validate_metadata(_read_wheel(wheels[0]), version)
    except ValueError as error:
        raise ValueError(f"{wheels[0].name}: {error}") from error
    try:
        _validate_metadata(_read_sdist(sdists[0]), version)
    except ValueError as error:
        raise ValueError(f"{sdists[0].name}: {error}") from error
    if tag is not None:
        _validate_tag(
            tag,
            version,
            (root / "CHANGELOG.md").read_text(encoding="utf-8"),
        )
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", help="release tag, exactly v<project.version>"
    )
    arguments = parser.parse_args()
    try:
        version = validate(Path.cwd(), arguments.tag)
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as error:
        print(f"Release validation failed: {error}", file=sys.stderr)
        return 1
    identity = f", tag {arguments.tag}" if arguments.tag is not None else ""
    print(f"Validated coker {version}: wheel and sdist metadata{identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
