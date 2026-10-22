"""Validate release events and the successful source-head trial publication.

Offline PR smoke (run in a checkout containing both version commits):
    python scripts/release_promotion.py --event-name pull_request \
        --event event.json --repository OWNER/REPO --output outputs

For a production fixture, use --event-name push --sha MERGE_SHA and point
--api-url at a local HTTP fixture server. GH_TOKEN supplies the API credential;
no package upload is performed by this validator.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from packaging.version import Version


_GATE = "Release promotion gate"


def _validate_source(pull: dict, repository: str) -> None:
    head = pull.get("head") or {}
    base = pull.get("base") or {}
    if (
        head.get("ref") != "develop"
        or base.get("ref") != "main"
        or (head.get("repo") or {}).get("full_name") != repository
        or (base.get("repo") or {}).get("full_name") != repository
    ):
        raise ValueError(
            "Release candidates must be same-repository develop -> main PRs"
        )


def _read_version(document: str) -> str:
    version = tomllib.loads(document)["project"]["version"]
    if not isinstance(version, str) or any(
        character in version for character in "\r\n"
    ):
        raise ValueError("Package version must be a single-line PEP 440 version")
    Version(version)
    return version


def _read_base(root: Path, sha: str) -> str:
    if not sha or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError("Missing or invalid base commit SHA")
    if len(sha) not in {40, 64}:
        raise ValueError("Missing or invalid base commit SHA")
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{sha}:pyproject.toml"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return _read_version(result.stdout)


def _request(api_url: str, token: str, path: str, **query):
    request = Request(
        f"{api_url.rstrip('/')}/{path}?{urlencode(query | {'per_page': 100})}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        # Reject incomplete provenance instead of following unbounded result pages.
        if 'rel="next"' in response.headers.get("Link", ""):
            raise ValueError("Release provenance exceeds the bounded API result page")
        result = json.load(response)
    if isinstance(result, dict) and result.get("total_count", 0) > 100:
        raise ValueError("Release provenance exceeds the bounded API result page")
    return result


def _validate_provenance(repository: str, sha: str, api_url: str, token: str) -> None:
    if not token:
        raise ValueError("GH_TOKEN is required to verify production provenance")
    prefix = f"repos/{repository}"
    pulls = _request(api_url, token, f"{prefix}/commits/{sha}/pulls")
    candidates = []
    for pull in pulls:
        if not pull.get("merged_at") or pull.get("merge_commit_sha") != sha:
            continue
        try:
            _validate_source(pull, repository)
        except ValueError:
            continue
        candidates.append(pull)
    if len(candidates) != 1:
        raise ValueError(
            "Production must be the merge of one develop -> main promotion PR"
        )
    source = candidates[0]["head"]["sha"]
    checks = _request(
        api_url,
        token,
        f"{prefix}/commits/{source}/check-runs",
        check_name=_GATE,
        filter="latest",
    )["check_runs"]
    runs = _request(
        api_url,
        token,
        f"{prefix}/actions/workflows/pypi.yml/runs",
        event="pull_request",
        head_sha=source,
    )["workflow_runs"]
    trusted_suites = {
        run["check_suite_id"]
        for run in runs
        if run.get("head_sha") == source
        and run.get("head_branch") == "develop"
        and (run.get("head_repository") or {}).get("full_name") == repository
        and run.get("event") == "pull_request"
        and run.get("path") == ".github/workflows/pypi.yml"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
    }
    gates = [
        check
        for check in checks
        if check.get("name") == _GATE
        and check.get("head_sha") == source
        and (check.get("app") or {}).get("slug") == "github-actions"
        and (check.get("check_suite") or {}).get("id") in trusted_suites
    ]
    if not gates or any(
        check.get("status") != "completed" or check.get("conclusion") != "success"
        for check in gates
    ):
        raise ValueError(
            "Exact promotion source head requires a successful Release promotion gate "
            "from the pull_request pypi.yml workflow"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-name",
        default=os.environ.get("GITHUB_EVENT_NAME"),
        choices=("pull_request", "push"),
    )
    parser.add_argument(
        "--event", type=Path, default=os.environ.get("GITHUB_EVENT_PATH")
    )
    parser.add_argument("--repository", default=os.environ.get("GH_REPO"))
    parser.add_argument("--sha", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=os.environ.get("GITHUB_OUTPUT"))
    parser.add_argument(
        "--api-url",
        default=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    arguments = parser.parse_args()
    try:
        if not arguments.event or not arguments.repository or not arguments.event_name:
            raise ValueError("An event, event name, and repository are required")
        event = json.loads(arguments.event.read_text(encoding="utf-8"))
        if arguments.event_name == "pull_request":
            pull = event["pull_request"]
            _validate_source(pull, arguments.repository)
            base_sha = pull["base"]["sha"]
        else:
            if (
                event.get("ref") != "refs/heads/main"
                or event.get("deleted")
                or not arguments.sha
                or event.get("after") != arguments.sha
            ):
                raise ValueError("Production requires the exact main push commit")
            base_sha = event["before"]
        head_version = _read_version(
            (arguments.root / "pyproject.toml").read_text(encoding="utf-8")
        )
        base_version = ""
        bumped = False
        if base_sha and set(base_sha) != {"0"}:
            base_version = _read_base(arguments.root, base_sha)
            bumped = Version(head_version) > Version(base_version)
        if arguments.event_name == "pull_request" and not bumped:
            raise ValueError(
                "Promotion requires a strictly increased prepared package version"
            )
        if arguments.event_name == "push" and bumped:
            _validate_provenance(
                arguments.repository,
                arguments.sha,
                arguments.api_url,
                os.environ.get("GH_TOKEN", ""),
            )
        outputs = (
            f"head_version={head_version}\nbase_version={base_version}\n"
            f"version_bumped={'true' if bumped else 'false'}\n"
        )
        if arguments.output:
            with arguments.output.open("a", encoding="utf-8") as output:
                output.write(outputs)
        print(outputs, end="")
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Release promotion failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
