"""Bind built documentation to a release and verify the publicly served bytes.

Usage:
    python scripts/release_docs.py prepare --html docs/_build/html \
        --project pyproject.toml --readme README.md --commit FULL_SHA
    python scripts/release_docs.py verify --manifest docs/_build/html/release.json
"""

import argparse
from collections.abc import Mapping
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.error import URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_NAVIGATION = (
    "basics/index.html",
    "tutorials/index.html",
    "how_to/index.html",
    "architecture/index.html",
    "api/index.html",
)
REQUIRED_ROUTES = (
    "index.html",
    *_NAVIGATION,
    "basics/getting_started.html",
    "how_to/backend_interoperability.html",
    "api/functions.html",
    "api/optimisation.html",
    "api/dynamics.html",
)


class _Page(HTMLParser):
    """Extract real anchors, keeping labels to identify README's Documentation link."""

    def __init__(self, content: bytes):
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.is_html = False
        self._href: str | None = None
        self._label: list[str] = []
        self.feed(content.decode("utf-8"))
        self.close()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "html":
            self.is_html = True
        if tag == "a" and attributes.get("href") is not None:
            self._href = attributes["href"]
            self._label = []

    def handle_data(self, data):
        if self._href is not None:
            self._label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append(("".join(self._label), self._href))
            self._href = None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ValueError(f"documentation URLs must not redirect: {request.full_url}")


def _root_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("project Documentation URL must be a string")
    parsed = urlsplit(value)
    local_http = parsed.scheme == "http" and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if (
        not (parsed.scheme == "https" or local_http)
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/")
        or re.fullmatch(r"/[A-Za-z0-9_./-]*", parsed.path) is None
        or "//" in parsed.path
        or any(part in {".", ".."} for part in parsed.path.split("/"))
        or any(character.isspace() for character in value)
    ):
        raise ValueError("Documentation must be an absolute HTTPS root URL ending in /")
    # Force urllib's port validation before any request is made.
    parsed.port
    return value


def _route(value: object) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_./-]+\.html", value) is None
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"unsafe or non-HTML documentation route: {value!r}")
    return value


def _url_route(url: str, root: str) -> str:
    parsed = urlsplit(url)
    base = urlsplit(root)
    if (
        parsed.scheme != base.scheme
        or parsed.netloc != base.netloc
        or not parsed.path.startswith(base.path)
        or parsed.query
    ):
        raise ValueError(f"documentation link is outside the advertised root: {url}")
    path = parsed.path[len(base.path) :]
    if not path or path.endswith("/"):
        path += "index.html"
    return _route(path)


def _readme_routes(readme: str, root: str) -> set[str]:
    # README's inline/reference Markdown links and literal HTML anchors are links,
    # unlike bare example strings that merely resemble URLs.
    links = re.findall(r"\[([^\]\n]+)\]\(\s*<?([^\s)>]+)>?(?:\s+[^)]*)?\)", readme)
    definitions = dict(
        (label.casefold(), target)
        for label, target in re.findall(
            r"^\s*\[([^\]\n]+)\]:\s*<?([^\s>]+)>?", readme, re.MULTILINE
        )
    )
    for label, reference in re.findall(r"\[([^\]\n]+)\]\[([^\]\n]*)\]", readme):
        target = definitions.get((reference or label).casefold())
        if target is not None:
            links.append((label, target))
    links.extend(_Page(readme.encode("utf-8")).links)
    roots = [
        target for label, target in links if label.strip().casefold() == "documentation"
    ]
    if not roots or any(target != root for target in roots):
        raise ValueError("README must advertise the exact project Documentation root")
    routes = set()
    base = urlsplit(root)
    for _, target in links:
        parsed = urlsplit(target)
        if target.startswith("docs/") or parsed.path.endswith(".rst"):
            raise ValueError(
                f"README documentation links must use live HTML URLs: {target}"
            )
        if parsed.scheme == base.scheme and parsed.netloc == base.netloc:
            routes.add(_url_route(target, root))
    return routes


def _validate_page(content: bytes, route: str, root: str) -> None:
    page = _Page(content)
    if not page.is_html:
        raise ValueError(f"documentation page is not HTML: {route}")
    if route == "index.html":
        linked = set()
        for _, href in page.links:
            target = urljoin(root + route, href)
            try:
                linked.add(_url_route(target, root))
            except ValueError:
                continue
        missing = set(_NAVIGATION) - linked
        if missing:
            raise ValueError(
                f"documentation navigation is missing: {', '.join(sorted(missing))}"
            )


def _validate_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict) or set(manifest) != {
        "version",
        "commit",
        "documentation_url",
        "pages",
    }:
        raise ValueError("invalid release documentation manifest fields")
    version = manifest["version"]
    if not isinstance(version, str) or not version or version.strip() != version:
        raise ValueError("release documentation version must be nonempty")
    if (
        not isinstance(manifest["commit"], str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", manifest["commit"]) is None
    ):
        raise ValueError("release documentation commit must be an exact full Git SHA")
    _root_url(manifest["documentation_url"])
    pages = manifest["pages"]
    if not isinstance(pages, dict) or not set(REQUIRED_ROUTES).issubset(pages):
        raise ValueError("release documentation manifest is missing required pages")
    for route, digest in pages.items():
        _route(route)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid SHA-256 digest for {route}")
    return manifest


def prepare_release_docs(
    html: Path, project: Mapping, readme: str, commit: str
) -> Path:
    """Write a manifest only after checking built pages and advertised links.

    Args:
        html: Sphinx HTML output directory.
        project: The parsed ``[project]`` metadata mapping, including URLs.
        readme: README Markdown text.
        commit: Exact full release commit SHA.

    Returns:
        Path to the deterministic ``release.json`` manifest.

    Raises:
        ValueError: Metadata, advertised routes, or HTML navigation are invalid.
        OSError: A required built page cannot be read.

    Example:
        prepare_release_docs(Path("html"), project, readme, "a" * 40)
    """
    urls = project.get("urls", {})
    if not isinstance(urls, Mapping):
        raise ValueError("project URLs must be a metadata mapping")
    root = _root_url(urls.get("Documentation"))
    routes = set(REQUIRED_ROUTES) | _readme_routes(readme, root)
    html = html.resolve(strict=True)
    pages = {}
    for route in sorted(routes):
        path = (html / route).resolve(strict=True)
        if not path.is_relative_to(html):
            raise ValueError(f"documentation page escapes HTML directory: {route}")
        content = path.read_bytes()
        _validate_page(content, route, root)
        pages[route] = hashlib.sha256(content).hexdigest()
    manifest = _validate_manifest(
        {
            "version": project.get("version"),
            "commit": commit,
            "documentation_url": root,
            "pages": pages,
        }
    )
    destination = html / "release.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _fetch(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "coker-release-verification",
            "Cache-Control": "no-cache",
        },
    )
    with build_opener(_NoRedirect()).open(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError(f"documentation request did not return HTTP 200: {url}")
        return response.read()


def verify_release_docs(manifest_path: Path) -> None:
    """Verify live release metadata, HTML hashes, and required navigation.

    Args:
        manifest_path: Local trusted manifest from the tested HTML artifact.

    Raises:
        ValueError: Live metadata, HTML bytes, or navigation differ.
        OSError: The trusted manifest or a public page cannot be read.

    Example:
        verify_release_docs(Path("html/release.json"))
    """
    trusted = _validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    root = trusted["documentation_url"]
    remote = json.loads(_fetch(root + "release.json"))
    if remote != trusted:
        raise ValueError("live documentation manifest differs from the trusted release")
    for route, digest in sorted(trusted["pages"].items()):
        content = _fetch(root + route)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(
                f"live documentation page differs from the trusted release: {route}"
            )
        _validate_page(content, route, root)


def main() -> int:
    """Run the prepare or verify command; print failures and return an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--html", type=Path, required=True)
    prepare.add_argument("--project", type=Path, required=True)
    prepare.add_argument("--readme", type=Path, required=True)
    prepare.add_argument("--commit", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "prepare":
            # Tool-only TOML parsing stays out of imports on supported Python 3.10.
            import tomllib

            with arguments.project.open("rb") as source:
                metadata = tomllib.load(source)
            project = metadata.get("project")
            if not isinstance(project, dict):
                raise ValueError("pyproject.toml must contain [project] metadata")
            prepare_release_docs(
                arguments.html,
                project,
                arguments.readme.read_text(encoding="utf-8"),
                arguments.commit,
            )
        else:
            verify_release_docs(arguments.manifest)
    except (ValueError, OSError, URLError, ImportError) as error:
        print(f"Release documentation verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
