from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from urllib.parse import urlsplit

import pytest

from tests.test_prepare_release import git


SCRIPT = Path(__file__).parents[1] / "scripts" / "release_promotion.py"
REPOSITORY = "owner/coker"
SOURCE = "a" * 40
MERGE = "b" * 40


def promotion():
    return {
        "number": 17,
        "merged": True,
        "merged_at": "2026-10-22T12:00:00Z",
        "merge_commit_sha": MERGE,
        "base": {"ref": "main", "sha": "c" * 40, "repo": {"full_name": REPOSITORY}},
        "head": {"ref": "develop", "sha": SOURCE, "repo": {"full_name": REPOSITORY}},
    }


def evidence():
    return {
        "pulls": [promotion()],
        "check_runs": [
            {
                "name": "Release promotion gate",
                "head_sha": SOURCE,
                "status": "completed",
                "conclusion": "success",
                "app": {"slug": "github-actions"},
                "check_suite": {"id": 123},
            }
        ],
        "workflow_runs": [
            {
                "head_sha": SOURCE,
                "head_branch": "develop",
                "head_repository": {"full_name": REPOSITORY},
                "event": "pull_request",
                "path": ".github/workflows/pypi.yml",
                "check_suite_id": 123,
                "status": "completed",
                "conclusion": "success",
            }
        ],
    }


@contextmanager
def serve_api(fixtures, *, more_pages=False, status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            requests.append(request)
            expected = {
                f"/repos/{REPOSITORY}/commits/{MERGE}/pulls": fixtures["pulls"],
                f"/repos/{REPOSITORY}/commits/{SOURCE}/check-runs": {
                    "check_runs": fixtures["check_runs"]
                },
                f"/repos/{REPOSITORY}/actions/workflows/pypi.yml/runs": {
                    "workflow_runs": fixtures["workflow_runs"]
                },
            }
            if request.path not in expected:
                self.send_error(404)
                return
            assert self.headers["Authorization"] == "Bearer fixture-token"
            if status != 200:
                self.send_error(status)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if more_pages:
                self.send_header("Link", '<http://fixture.invalid/page2>; rel="next"')
            self.end_headers()
            self.wfile.write(json.dumps(expected[request.path]).encode())

        def log_message(self, *arguments):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.fixture
def versions(tmp_path):
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "core.autocrlf", "false")

    def write(version):
        (tmp_path / "pyproject.toml").write_text(
            f'[project]\nname = "coker"\nversion = "{version}"\n', encoding="utf-8"
        )

    write("1.0.0")
    git(tmp_path, "add", "pyproject.toml")
    git(tmp_path, "commit", "-m", "Base release")
    base = git(tmp_path, "rev-parse", "HEAD")
    return tmp_path, base, write


def invoke(versions, event_name="pull_request", event=None, api_url=None):
    root, base, _ = versions
    if event is None:
        event = {"pull_request": promotion()}
        event["pull_request"]["base"]["sha"] = base
    event_path = root / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    output = root / "outputs"
    output.unlink(missing_ok=True)
    arguments = [
        sys.executable,
        str(SCRIPT),
        "--event-name",
        event_name,
        "--event",
        str(event_path),
        "--repository",
        REPOSITORY,
        "--sha",
        MERGE,
        "--output",
        str(output),
    ]
    if api_url:
        arguments.extend(["--api-url", api_url])
    result = subprocess.run(
        arguments,
        cwd=root,
        capture_output=True,
        text=True,
        env=os.environ | {"GH_TOKEN": "fixture-token"},
        check=False,
    )
    return result, output.read_text() if output.exists() else ""


def push_event(base):
    return {"ref": "refs/heads/main", "before": base, "after": MERGE}


@pytest.mark.parametrize("version", ["1.0.1", "1.1.0rc1", "1.0.0.post1", "1!0.1"])
def test_allows_same_repository_develop_promotion(versions, version):
    versions[2](version)
    result, output = invoke(versions)
    assert result.returncode == 0, result.stderr
    assert "version_bumped=true\n" in output
    assert f"head_version={version}\n" in output


@pytest.mark.parametrize(
    ("part", "field", "value"),
    [
        ("head", "ref", "feature/release"),
        ("base", "ref", "develop"),
        ("head", "repo", {"full_name": "fork/coker"}),
        ("head", "repo", None),
        ("base", "repo", {"full_name": "other/coker"}),
    ],
)
def test_rejects_wrong_source_target_and_fork(versions, part, field, value):
    versions[2]("1.1.0")
    event = {"pull_request": promotion()}
    event["pull_request"][part][field] = value
    result, output = invoke(versions, event=event)
    assert result.returncode == 1
    assert not output


@pytest.mark.parametrize("version", ["1.0.0", "1.0", "0.9.9", "1.0.0rc1", "invalid"])
def test_rejects_absent_decreased_or_invalid_version(versions, version):
    versions[2](version)
    result, output = invoke(versions)
    assert result.returncode == 1
    assert not output


def test_accepts_merged_production_promotion_with_exact_source_gate(versions):
    versions[2]("1.1.0")
    with serve_api(evidence()) as (url, _):
        result, output = invoke(versions, "push", push_event(versions[1]), url)
    assert result.returncode == 0, result.stderr
    assert "version_bumped=true\n" in output


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("pulls",), []),
        (("pulls", 0, "merged_at"), None),
        (("pulls", 0, "merge_commit_sha"), "d" * 40),
        (("pulls", 0, "head", "ref"), "feature/release"),
        (("pulls", 0, "head", "repo", "full_name"), "fork/coker"),
        (("check_runs",), []),
        (("check_runs", 0, "conclusion"), "failure"),
        (("check_runs", 0, "status"), "in_progress"),
        (("check_runs", 0, "head_sha"), "d" * 40),
        (("check_runs", 0, "app", "slug"), "custom-status-publisher"),
        (("workflow_runs", 0, "path"), ".github/workflows/other.yml"),
        (("workflow_runs", 0, "check_suite_id"), 456),
        (("workflow_runs", 0, "head_sha"), "d" * 40),
        (("workflow_runs", 0, "status"), "in_progress"),
        (("workflow_runs", 0, "head_branch"), "feature/release"),
        (("workflow_runs", 0, "event"), "push"),
        (("workflow_runs", 0, "head_repository", "full_name"), "fork/coker"),
    ],
)
def test_rejects_missing_or_untrusted_production_evidence(versions, path, value):
    versions[2]("1.1.0")
    fixtures = evidence()
    target = fixtures
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with serve_api(fixtures) as (url, _):
        result, output = invoke(versions, "push", push_event(versions[1]), url)
    assert result.returncode == 1, result.stdout
    assert not output


def test_bootstrap_push_without_version_change_never_requests_publication(versions):
    result, output = invoke(
        versions, "push", push_event(versions[1]), "http://127.0.0.1:1"
    )
    assert result.returncode == 0, result.stderr
    assert "version_bumped=false\n" in output


def test_initial_push_never_requests_publication(versions):
    versions[2]("1.1.0")
    result, output = invoke(
        versions, "push", push_event("0" * 40), "http://127.0.0.1:1"
    )
    assert result.returncode == 0, result.stderr
    assert "version_bumped=false\n" in output


@pytest.mark.parametrize(
    ("base_version", "head_version", "allowed"),
    [
        ("1.1.0rc2", "1.1.0rc10", True),
        ("1.1.0rc10", "1.1.0rc2", False),
        ("1.1.0", "1.1.0rc1", False),
        ("1.1.0rc1", "1.1.0", True),
        ("1.1.0.dev2", "1.1.0.dev10", True),
        ("1.0.0.post1", "1.0.0", False),
        ("1!1.0", "2.0", False),
    ],
)
def test_compares_pep440_release_order(versions, base_version, head_version, allowed):
    root, _, write = versions
    write(base_version)
    git(root, "add", "pyproject.toml")
    git(root, "commit", "-m", "Prepare baseline")
    base = git(root, "rev-parse", "HEAD")
    write(head_version)
    result, output = invoke((root, base, write))
    assert (result.returncode == 0) == allowed, result.stderr
    assert ("version_bumped=true\n" in output) == allowed


@pytest.mark.parametrize("status", [403, 404, 500])
def test_rejects_unavailable_production_api(versions, status):
    versions[2]("1.1.0")
    with serve_api(evidence(), status=status) as (url, requests):
        result, output = invoke(versions, "push", push_event(versions[1]), url)
    assert result.returncode == 1
    assert not output
    assert len(requests) == 1


def test_rejects_partial_provenance_without_unbounded_pagination(versions):
    versions[2]("1.1.0")
    with serve_api(evidence(), more_pages=True) as (url, requests):
        result, output = invoke(versions, "push", push_event(versions[1]), url)
    assert result.returncode == 1
    assert not output
    assert len(requests) == 1
