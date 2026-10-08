"""GitHub writes the transformers-ci dashboard asks serge to make.

The dashboard's trace exporter is public-facing, so it holds no GitHub write
credential. When a signed-in maintainer clicks "Re-run failed tests" on a PR
page, the exporter keeps the action record and asks serge, in-cluster, to make
the two writes with serge's App: cancel the CI runs the person confirmed, then
dispatch the targeted rerun caller.

Three layers keep that narrow:

* the caller is the dashboard: ``Authorization: Bearer <DASHBOARD_API_TOKEN>``,
  compared in constant time (no token configured -> every route is 404);
* the acting user named in the request must have write access to the
  repository, read with serge's App, whatever the dashboard believes;
* every operation is allowlisted by configuration: the repositories, the
  workflow *paths* whose runs may be cancelled, and the workflow *files* that
  may be dispatched (on the default branch only).

Nothing here touches ``/tasks``. Agent work requested from the dashboard goes
through ``/tasks``; this module is for the GitHub writes that are not tasks.
"""

from __future__ import annotations

import hmac
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

WRITE_ROLES = frozenset({"admin", "maintain", "write"})
ROLE_CACHE_SECONDS = 300
MAX_INPUTS = 10
# GitHub caps one workflow_dispatch input at 65,535 characters.
MAX_INPUT_CHARS = 65_000
MAX_TOTAL_INPUT_CHARS = 100_000

_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_WORKFLOW_FILE = re.compile(r"^[A-Za-z0-9._-]+\.ya?ml$")
_INPUT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,99}$")


class DashboardError(Exception):
    """A refused request; ``status`` and ``detail`` become the HTTP reply."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class GitHubWrites(Protocol):
    def collaborator_role(self, owner: str, repo: str, login: str) -> str: ...

    def get_workflow_run(self, owner: str, repo: str, run_id: int) -> dict: ...

    def cancel_workflow_run(self, owner: str, repo: str, run_id: int) -> int: ...

    def get_repo(self, owner: str, repo: str) -> dict: ...

    def dispatch_workflow(
        self,
        owner: str,
        repo: str,
        workflow_file: str,
        *,
        ref: str,
        inputs: dict[str, Any],
    ) -> None: ...


@dataclass(frozen=True)
class Target:
    owner: str
    repo: str
    actor: str

    @property
    def repository(self) -> str:
        return f"{self.owner}/{self.repo}"


def authorize_service(configured_token: str, authorization: str) -> None:
    """The caller must be the dashboard. 404 when the API is off, so an
    unconfigured serge looks exactly like one without these routes."""
    if not configured_token:
        raise DashboardError(404, "dashboard_api_disabled")
    if not authorization.lower().startswith("bearer "):
        raise DashboardError(401, "missing_bearer_token")
    if not hmac.compare_digest(
        authorization[7:].strip().encode(), configured_token.encode()
    ):
        raise DashboardError(401, "invalid_dashboard_token")


def parse_target(body: object, repositories: tuple[str, ...]) -> Target:
    if not isinstance(body, dict):
        raise DashboardError(400, "invalid_json_body")
    repository = body.get("repository")
    actor = body.get("actor")
    if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
        raise DashboardError(400, "bad_repository")
    if repository.lower() not in {r.lower() for r in repositories}:
        raise DashboardError(403, "repository_not_allowed")
    if not isinstance(actor, str) or not _LOGIN.fullmatch(actor):
        raise DashboardError(400, "bad_actor")
    owner, repo = repository.split("/", 1)
    return Target(owner, repo, actor)


class RoleCache:
    """Short-lived cache of a user's role per repository: a re-run makes up to
    three writes in a few minutes, and a role change taking five minutes to
    show is acceptable for a write gate that GitHub also enforces on serge."""

    def __init__(self, ttl: float = ROLE_CACHE_SECONDS) -> None:
        self.ttl = ttl
        self._lock = threading.Lock()
        self._roles: dict[tuple[str, str], tuple[float, str]] = {}

    def role(self, client: GitHubWrites, target: Target) -> str:
        key = (target.repository.lower(), target.actor.lower())
        now = time.monotonic()
        with self._lock:
            hit = self._roles.get(key)
            if hit and now - hit[0] < self.ttl:
                return hit[1]
        role = client.collaborator_role(target.owner, target.repo, target.actor)
        with self._lock:
            self._roles[key] = (now, role)
        return role


def permission(client: GitHubWrites, target: Target, cache: RoleCache) -> dict:
    role = cache.role(client, target)
    return {"actor": target.actor, "role": role, "can_write": role in WRITE_ROLES}


def _require_write(client: GitHubWrites, target: Target, cache: RoleCache) -> None:
    if not permission(client, target, cache)["can_write"]:
        raise DashboardError(403, "no_write_access")


def cancel_run(
    client: GitHubWrites,
    target: Target,
    body: dict,
    *,
    cancel_workflows: tuple[str, ...],
    cache: RoleCache,
) -> dict:
    """Cancel one run of an allowlisted workflow. Returns GitHub's answer as
    data, not as an error: 202 is "accepted" (the run is not cancelled yet;
    the caller polls), 409 means it is already finishing."""
    run_id = body.get("run_id")
    if isinstance(run_id, str) and run_id.isdigit():
        run_id = int(run_id)
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise DashboardError(400, "bad_run_id")
    _require_write(client, target, cache)
    run = client.get_workflow_run(target.owner, target.repo, run_id)
    path = str(run.get("path") or "").split("@", 1)[0]
    if path not in cancel_workflows:
        raise DashboardError(403, "workflow_not_cancellable")
    status = str(run.get("status") or "")
    if status == "completed":
        return {"run_id": run_id, "github_status": None, "run_status": status}
    github_status = client.cancel_workflow_run(target.owner, target.repo, run_id)
    if github_status not in (202, 409):
        raise DashboardError(502, f"github_cancel_returned_{github_status}")
    return {"run_id": run_id, "github_status": github_status, "run_status": status}


def dispatch_workflow(
    client: GitHubWrites,
    target: Target,
    body: dict,
    *,
    dispatch_workflows: tuple[str, ...],
    cache: RoleCache,
) -> dict:
    """Dispatch an allowlisted workflow file on the repository's default
    branch, with string inputs only. The run is found later by the caller,
    through a correlation id its run-name echoes."""
    workflow = body.get("workflow")
    inputs = body.get("inputs")
    if not isinstance(workflow, str) or not _WORKFLOW_FILE.fullmatch(workflow):
        raise DashboardError(400, "bad_workflow")
    if workflow not in dispatch_workflows:
        raise DashboardError(403, "workflow_not_dispatchable")
    if (
        not isinstance(inputs, dict)
        or len(inputs) > MAX_INPUTS
        or not all(
            isinstance(k, str)
            and _INPUT_NAME.fullmatch(k)
            and isinstance(v, str)
            and len(v) <= MAX_INPUT_CHARS
            for k, v in inputs.items()
        )
        or sum(len(v) for v in inputs.values()) > MAX_TOTAL_INPUT_CHARS
    ):
        raise DashboardError(400, "bad_inputs")
    _require_write(client, target, cache)
    ref = str(client.get_repo(target.owner, target.repo).get("default_branch") or "")
    if not ref:
        raise DashboardError(502, "no_default_branch")
    client.dispatch_workflow(
        target.owner, target.repo, workflow, ref=ref, inputs=dict(inputs)
    )
    return {"workflow": workflow, "ref": ref, "dispatched": True}
