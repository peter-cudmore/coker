import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.error import HTTPError

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "release_docs.py"
SPEC = importlib.util.spec_from_file_location("release_docs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release_docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_docs)

COMMIT = "a" * 40
VERSION = "1.2.3"
ADVERTISED_ROUTE = "how_to/casadi_solver_tuning.html"
NAVIGATION = (
    ("Basics", "basics/index.html"),
    ("Tutorials", "tutorials/index.html"),
    ("How-to", "how_to/index.html"),
    ("Architecture", "architecture/index.html"),
    ("API", "api/index.html"),
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@pytest.fixture
def publication(tmp_path):
    public = tmp_path / "public"
    html = public / "coker"
    html.mkdir(parents=True)
    handler = functools.partial(QuietHandler, directory=str(public))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}/coker/"
    navigation = "".join(
        f'<a href="{route}">{label}</a>' for label, route in NAVIGATION
    )
    for route in (*release_docs.REQUIRED_ROUTES, ADVERTISED_ROUTE):
        path = html / route
        path.parent.mkdir(parents=True, exist_ok=True)
        nav = (
            f'<nav aria-label="Documentation">{navigation}</nav>'
            if route == "index.html"
            else ""
        )
        path.write_text(
            '<!doctype html><html lang="en"><head>'
            f"<title>{route} — Coker {VERSION}</title></head>"
            f"<body>{nav}<main><h1>{route}</h1><p>Release {VERSION}.</p>"
            "</main></body></html>",
            encoding="utf-8",
        )
    project = {"version": VERSION, "urls": {"Documentation": root}}
    readme = (
        f"[Documentation]({root})\n\n"
        f"[Solver tuning]({root}{ADVERTISED_ROUTE}#options)\n"
    )
    try:
        yield html, project, readme
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def prepare(publication):
    html, project, readme = publication
    return release_docs.prepare_release_docs(html, project, readme, COMMIT)


def trusted_copy(manifest):
    trusted = manifest.parent.parent.parent / "trusted-release.json"
    shutil.copyfile(manifest, trusted)
    return trusted


def test_prepared_release_verifies_public_bytes_and_advertised_pages(publication):
    manifest = prepare(publication)
    content = manifest.read_bytes()
    data = json.loads(content)
    assert data["commit"] == COMMIT
    assert data["version"] == VERSION
    assert data["documentation_url"] == publication[1]["urls"]["Documentation"]
    assert set(data["pages"]) == set(release_docs.REQUIRED_ROUTES) | {ADVERTISED_ROUTE}
    release_docs.verify_release_docs(trusted_copy(manifest))
    assert prepare(publication).read_bytes() == content


@pytest.mark.parametrize("field,value", [("commit", "b" * 40), ("version", "1.2.2")])
def test_stale_remote_release_metadata_fails(publication, field, value):
    manifest = prepare(publication)
    trusted = trusted_copy(manifest)
    remote = json.loads(manifest.read_text(encoding="utf-8"))
    remote[field] = value
    manifest.write_text(json.dumps(remote), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        release_docs.verify_release_docs(trusted)


@pytest.mark.parametrize("route", ["api/functions.html", ADVERTISED_ROUTE])
def test_stale_remote_page_bytes_fail_even_with_current_manifest(publication, route):
    manifest = prepare(publication)
    trusted = trusted_copy(manifest)
    page = manifest.parent / route
    page.write_bytes(page.read_bytes().replace(b"Release 1.2.3", b"Release 1.2.2"))
    with pytest.raises(ValueError, match="page differs"):
        release_docs.verify_release_docs(trusted)


@pytest.mark.parametrize(
    "route", ["release.json", "api/functions.html", ADVERTISED_ROUTE]
)
def test_missing_remote_manifest_or_page_fails(publication, route):
    manifest = prepare(publication)
    trusted = trusted_copy(manifest)
    (manifest.parent / route).unlink()
    with pytest.raises(HTTPError, match="404"):
        release_docs.verify_release_docs(trusted)


@pytest.mark.parametrize("route", ["api/functions.html", ADVERTISED_ROUTE])
def test_prepare_requires_built_required_and_readme_pages(publication, route):
    html, _, _ = publication
    (html / route).unlink()
    with pytest.raises(FileNotFoundError):
        prepare(publication)
    assert not (html / "release.json").exists()


@pytest.mark.parametrize("label,route", NAVIGATION)
def test_prepare_requires_actual_navigation_links(publication, label, route):
    html, _, _ = publication
    index = html / "index.html"
    content = index.read_text(encoding="utf-8")
    # Visible labels and comment text cannot stand in for working links.
    content = content.replace(
        f'<a href="{route}">{label}</a>',
        f'<span>{label}</span><!-- <a href="{route}">{label}</a> -->',
    )
    index.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="navigation is missing"):
        prepare(publication)
    assert not (html / "release.json").exists()


@pytest.mark.parametrize(
    "advertisement",
    ["[Documentation](https://example.org/stale/)", "# Coker", "{root}"],
)
def test_readme_must_advertise_exact_metadata_documentation_root(
    publication, advertisement
):
    html, project, _ = publication
    readme = advertisement.format(root=project["urls"]["Documentation"])
    with pytest.raises(ValueError, match="exact project Documentation root"):
        release_docs.prepare_release_docs(html, project, readme, COMMIT)


@pytest.mark.parametrize(
    "target",
    ["docs/api/index.rst", "https://example.org/api/index.rst", "../api/index.html"],
)
def test_readme_source_or_outside_documentation_routes_fail(publication, target):
    html, project, readme = publication
    root = project["urls"]["Documentation"]
    if target.startswith("../"):
        target = root + target
    with pytest.raises(ValueError):
        release_docs.prepare_release_docs(
            html, project, readme + f"\n[API]({target})", COMMIT
        )


@pytest.mark.parametrize(
    "readme",
    [
        "[Documentation][docs]\n[Solver tuning][tuning]\n\n"
        "[docs]: {root}\n[tuning]: {root}{route}\n",
        "[Documentation][]\n[Solver tuning][]\n\n"
        "[Documentation]: {root}\n[Solver tuning]: {root}{route}\n",
        '<a href="{root}"><strong>Documentation</strong></a>\n'
        '<a href="{root}{route}">Solver tuning</a>\n',
    ],
)
def test_readme_reference_and_html_links_are_hashed(publication, readme):
    html, project, _ = publication
    readme = readme.format(
        root=project["urls"]["Documentation"], route=ADVERTISED_ROUTE
    )
    manifest = release_docs.prepare_release_docs(html, project, readme, COMMIT)
    assert ADVERTISED_ROUTE in json.loads(manifest.read_text(encoding="utf-8"))["pages"]
    release_docs.verify_release_docs(trusted_copy(manifest))


@pytest.mark.parametrize("commit", ["main", "a" * 7, "A" * 40, "a" * 41])
def test_prepare_requires_exact_commit_sha(publication, commit):
    html, project, readme = publication
    with pytest.raises(ValueError, match="exact full Git SHA"):
        release_docs.prepare_release_docs(html, project, readme, commit)


@pytest.mark.parametrize(
    "root",
    [
        "https://example.org/coker",
        "https://user:password@example.org/coker/",
        "https://example.org/coker/?old=1",
        "https://example.org/coker/#old",
        "http://example.org/coker/",
        "file:///tmp/coker/",
        "https://example.org/../coker/",
        "https://example.org//coker/",
        "https://example.org/%2e%2e/coker/",
    ],
)
def test_prepare_rejects_unsafe_documentation_roots(publication, root):
    html, _, _ = publication
    with pytest.raises(ValueError, match="absolute HTTPS root"):
        release_docs.prepare_release_docs(
            html,
            {"version": VERSION, "urls": {"Documentation": root}},
            f"[Documentation]({root})",
            COMMIT,
        )


def test_verify_rejects_unsafe_manifest_route_before_network(publication):
    manifest = prepare(publication)
    trusted = trusted_copy(manifest)
    data = json.loads(trusted.read_text(encoding="utf-8"))
    data["pages"]["../secrets.html"] = "0" * 64
    trusted.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe or non-HTML"):
        release_docs.verify_release_docs(trusted)


def test_verify_does_not_follow_redirects_to_other_pages(publication):
    manifest = prepare(publication)
    trusted = trusted_copy(manifest)
    # SimpleHTTPRequestHandler redirects directory requests to a trailing slash.
    manifest.unlink()
    manifest.mkdir()
    with pytest.raises(ValueError, match="must not redirect"):
        release_docs.verify_release_docs(trusted)


def test_verify_cli_reports_page_failure_from_external_working_directory(
    publication, tmp_path
):
    manifest = prepare(publication)
    trusted = trusted_copy(manifest)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--manifest", str(trusted)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    (manifest.parent / "api/index.html").write_text("not the release", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--manifest", str(trusted)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "api/index.html" in result.stderr
    assert "Release documentation verification failed:" in result.stderr


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="prepare CLI uses tool-only tomllib"
)
def test_prepare_cli_reads_project_metadata(publication, tmp_path):
    html, project, readme = publication
    project_path = tmp_path / "pyproject.toml"
    project_path.write_text(
        f'[project]\nversion = "{VERSION}"\n[project.urls]\n'
        f'Documentation = "{project["urls"]["Documentation"]}"\n',
        encoding="utf-8",
    )
    readme_path = tmp_path / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            "--html",
            str(html),
            "--project",
            str(project_path),
            "--readme",
            str(readme_path),
            "--commit",
            COMMIT,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    release_docs.verify_release_docs(trusted_copy(html / "release.json"))
