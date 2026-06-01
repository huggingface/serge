"""Sandboxed repo tools exposed to the reviewer LLM.

Each tool operates against a single repository checkout root. All paths
are resolved via realpath and rejected if they escape the root, and a
small denylist of noisy directories (``.git``, ``node_modules``, etc.) is
hidden from the model so it can't accidentally pull in junk. Repos may
also expose a small set of trusted helper commands via a default-branch
JSON config; those run without a shell and are still rooted inside the
checked-out repo.

The module is decoupled from the LLM client: it just exposes
``TOOL_SPECS`` (the OpenAI-style schemas) and ``run_tool``, which takes a
function name + JSON-decoded arguments and returns a string the model
will see as the ``tool`` message content.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from .sandbox import AUTO, SandboxUnavailable, wrap_command

log = logging.getLogger(__name__)


# Per-call output cap: tool messages re-bill the prompt on every loop
# iteration, so unbounded output silently inflates cost and risks blowing
# the context window. 8 KB is enough for a couple hundred lines of code or
# a meaningful grep result; the model can always re-call with a narrower
# range if it wants more.
MAX_TOOL_OUTPUT_CHARS = 8000

# File reads are line-bounded as well; this is the upper bound enforced
# even if the model asks for more.
MAX_READ_LINES = 400

# Directories we never expose to the model. ``.git`` would let it read
# refs / object data; the rest are caches that bloat output without
# adding signal.
DENY_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)

# Hosts the model is allowed to fetch via ``fetch_url``. Kept narrow on
# purpose: the goal is letting it verify Hugging Face paper/model/dataset
# links cited in PR docs, not arbitrary outbound HTTP from the runner.
ALLOWED_FETCH_HOSTS = frozenset({"huggingface.co"})

# Cap on the response body returned to the model. Same rationale as
# MAX_TOOL_OUTPUT_CHARS — and a model card / paper page is huge.
MAX_FETCH_BODY_CHARS = 4000

# Network timeout for fetch_url. Shorter than the LLM streaming budget
# so a slow URL doesn't delay an iteration much.
FETCH_TIMEOUT_SECONDS = 10
MAX_HELPER_ARGS = 32
MAX_HELPER_ARG_CHARS = 200
MAX_HELPER_TIMEOUT_SECONDS = 120
_HELPER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

# Allowlist of installers a repo can declare in `install`. We map each one
# to a python-API invocation in `_run_helper_install` so the reviewer
# doesn't depend on the installer being on PATH and so the install lands
# in this process's interpreter.
_ALLOWED_INSTALLERS = frozenset({"pip"})
MAX_HELPER_INSTALL_ARGS = 16
INSTALL_TIMEOUT_SECONDS = 300
# Pip install args are restricted to plain package specifiers and version
# pins. We deliberately reject characters that enable VCS / URL installs
# (`@`, `/`, `:`) and flag forms that point pip at attacker-controlled
# indexes — anyone with default-branch write access to a configured repo
# would otherwise be able to land arbitrary code in the reviewer's
# Python environment.
_INSTALL_ARG_RE = re.compile(r"^[A-Za-z0-9._\-+=,\[\]]+$")
_INSTALL_ALLOWED_FLAGS = frozenset({"install", "--upgrade", "-U", "--user"})
_INSTALL_DENY_FLAG_PREFIXES = (
    "--index-url",
    "--extra-index-url",
    "-i",
    "--find-links",
    "-f",
    "--trusted-host",
    "--config-settings",
    "--editable",
    "-e",
    "--target",
    "-t",
    "--prefix",
    "--root",
    "--no-deps",
)


# Helper subprocesses run binaries pulled from a repo-controlled config
# (.ai/review-tools.json on the default branch). Even though that config
# requires default-branch write access to land, the helper binary itself
# is opaque to us — so we hand it a minimal env that omits every secret
# the reviewbot process holds (GitHub App private key, LLM API key, OAuth
# client secret, session secret, webhook secret, etc.).
_HELPER_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "TMPDIR",
    "PYTHONHASHSEED",
    "SHELL",
)


def _helper_subprocess_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in _HELPER_ENV_PASSTHROUGH if k in os.environ}
    # Guard rails against the running git config / interactive prompts
    # leaking into helper subprocesses that shell out to git.
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    # Helpers installed via the pip-install hook land in the running
    # interpreter's bin directory. Under systemd that directory is NOT
    # on PATH (venvs are normally surfaced by `activate`, which we
    # don't run). Prepend it so a freshly-installed `mlinter` is
    # actually reachable by name from the helper subprocess.
    venv_bin = os.path.dirname(sys.executable)
    if venv_bin:
        existing = env.get("PATH", "")
        parts = existing.split(os.pathsep) if existing else []
        if venv_bin not in parts:
            env["PATH"] = venv_bin + (os.pathsep + existing if existing else "")
    return env


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a slice of a file from the checked-out PR head. Use this "
                "to gather context that the diff alone doesn't show (callers, "
                "callees, surrounding class, related constants). Lines are 1-indexed."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the repository root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to return (1-indexed). Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": (
                            f"Last line to return, inclusive. Capped at "
                            f"{MAX_READ_LINES} lines per call."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List the entries (files and subdirectories) of a directory in "
                "the checked-out PR head. Use this to discover where related "
                "code lives before reading specific files."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory relative to repo root. Defaults to the root.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a URL on huggingface.co to verify it resolves and read "
                "its content. Only https://huggingface.co/... URLs are allowed; "
                "other hosts return an error. Use this to check that paper, "
                "model, or dataset links cited in the diff actually exist "
                "before flagging them as typos. Returns the HTTP status line, "
                "content-type, and the first ~4KB of body."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL, e.g. https://huggingface.co/papers/2403.09611",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a regex pattern across the checked-out PR head "
                "using git grep. Returns matches as `path:line:text`. Use this "
                "to find call sites, definitions, or references not visible in the diff."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern (extended POSIX, as accepted by git grep -E).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path-spec to scope the search. Defaults to the whole repo.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Cap on returned matches. Defaults to 50, hard max 200.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]

_BUILTIN_TOOL_NAMES = frozenset(spec["function"]["name"] for spec in TOOL_SPECS)


@dataclass(frozen=True)
class RepoHelperTool:
    name: str
    description: str
    command: tuple[str, ...]
    cwd: str = "."
    allow_args: bool = False
    max_args: int = 0
    timeout_seconds: int = 20
    # Optional installer hook. First element is one of _ALLOWED_INSTALLERS
    # (e.g. "pip"); the rest are args passed to that installer. Empty
    # tuple means the helper is expected to already be on PATH.
    install: tuple[str, ...] = ()


def load_repo_helper_tools(raw: str | None) -> list[RepoHelperTool]:
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"helper tools config is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("helper tools config must be a JSON object")
    raw_helpers = parsed.get("helpers")
    if raw_helpers is None:
        return []
    if not isinstance(raw_helpers, list):
        raise ValueError("helper tools config field 'helpers' must be a list")

    helpers: list[RepoHelperTool] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(raw_helpers, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"helper #{idx} must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not _HELPER_NAME_RE.match(name):
            raise ValueError(
                f"helper #{idx} has invalid name {name!r}; use [A-Za-z][A-Za-z0-9_-]*"
            )
        if name in _BUILTIN_TOOL_NAMES:
            raise ValueError(f"helper name {name!r} conflicts with a built-in tool")
        if name in seen_names:
            raise ValueError(f"helper name {name!r} is duplicated")
        seen_names.add(name)

        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"helper {name!r} is missing a non-empty description")

        command = item.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part.strip() for part in command)
        ):
            raise ValueError(
                f"helper {name!r} must declare a non-empty string array 'command'"
            )

        cwd = item.get("cwd", ".")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ValueError(f"helper {name!r} has invalid cwd")

        allow_args = bool(item.get("allow_args", False))
        default_max_args = 8 if allow_args else 0
        max_args = int(item.get("max_args", default_max_args))
        if max_args < 0 or max_args > MAX_HELPER_ARGS:
            raise ValueError(
                f"helper {name!r} max_args must be between 0 and {MAX_HELPER_ARGS}"
            )
        if not allow_args and max_args:
            raise ValueError(
                f"helper {name!r} cannot set max_args when allow_args=false"
            )

        timeout_seconds = int(item.get("timeout_seconds", 20))
        if timeout_seconds < 1 or timeout_seconds > MAX_HELPER_TIMEOUT_SECONDS:
            raise ValueError(
                f"helper {name!r} timeout_seconds must be between 1 and "
                f"{MAX_HELPER_TIMEOUT_SECONDS}"
            )

        install = _parse_install_spec(name, item.get("install"))

        helpers.append(
            RepoHelperTool(
                name=name,
                description=description.strip(),
                command=tuple(part.strip() for part in command),
                cwd=cwd.strip(),
                allow_args=allow_args,
                max_args=max_args,
                timeout_seconds=timeout_seconds,
                install=install,
            )
        )
    return helpers


def _parse_install_spec(name: str, raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(part, str) and part.strip() for part in raw)
    ):
        raise ValueError(
            f"helper {name!r} 'install' must be a non-empty array of non-empty strings"
        )
    parts = [part.strip() for part in raw]
    installer = parts[0]
    if installer not in _ALLOWED_INSTALLERS:
        raise ValueError(
            f"helper {name!r} install command must start with one of "
            f"{sorted(_ALLOWED_INSTALLERS)}, got {installer!r}"
        )
    args = parts[1:]
    if len(args) > MAX_HELPER_INSTALL_ARGS:
        raise ValueError(
            f"helper {name!r} install accepts at most "
            f"{MAX_HELPER_INSTALL_ARGS} arguments"
        )
    for arg in args:
        if len(arg) > MAX_HELPER_ARG_CHARS:
            raise ValueError(
                f"helper {name!r} install args are capped at "
                f"{MAX_HELPER_ARG_CHARS} chars each"
            )
        if arg.startswith("-"):
            # Pip flags: explicit allowlist. Reject anything that lets the
            # repo redirect pip at attacker-controlled indexes or VCS URLs.
            if arg in _INSTALL_ALLOWED_FLAGS:
                continue
            for bad in _INSTALL_DENY_FLAG_PREFIXES:
                if arg == bad or arg.startswith(bad + "="):
                    raise ValueError(
                        f"helper {name!r} install flag {arg!r} is not allowed"
                    )
            raise ValueError(
                f"helper {name!r} install flag {arg!r} is not in the allowlist"
            )
        if not _INSTALL_ARG_RE.match(arg):
            raise ValueError(
                f"helper {name!r} install arg {arg!r} contains disallowed "
                "characters (VCS / URL installs are not permitted)"
            )
    return tuple(parts)


@dataclass(frozen=True)
class HelperInstallResult:
    name: str
    ok: bool
    message: str


# Process-local cache: each unique install spec runs at most once per
# worker. pip is idempotent, but we re-bill the install time on every
# review otherwise — and on a long-lived web worker that adds up.
_INSTALL_CACHE_LOCK = threading.Lock()
_INSTALL_CACHE: dict[tuple[str, ...], HelperInstallResult] = {}


def install_helper_tools(
    helpers: list[RepoHelperTool],
) -> list[HelperInstallResult]:
    """Run the install hook for each helper that declares one.

    Returns one result per helper that had an install spec; helpers
    without `install` are skipped silently. Failures are logged but not
    raised — if the package isn't actually installed the helper's first
    tool call will surface the error to the model, which is more
    informative than aborting the whole review.
    """
    results: list[HelperInstallResult] = []
    for helper in helpers:
        if not helper.install:
            continue
        with _INSTALL_CACHE_LOCK:
            cached = _INSTALL_CACHE.get(helper.install)
        if cached is not None:
            results.append(
                HelperInstallResult(
                    name=helper.name,
                    ok=cached.ok,
                    message=f"{helper.name}: {cached.message} (cached)",
                )
            )
            continue
        result = _run_helper_install(helper)
        with _INSTALL_CACHE_LOCK:
            # Only cache successes — failed installs may be transient
            # (network blip, pypi flake) and we want a retry to refresh.
            if result.ok:
                _INSTALL_CACHE[helper.install] = result
        results.append(result)
    return results


def _run_helper_install(helper: RepoHelperTool) -> HelperInstallResult:
    installer = helper.install[0]
    args = list(helper.install[1:])
    if installer == "pip":
        cmd = [sys.executable, "-m", "pip", *args]
    else:  # pragma: no cover — validated in _parse_install_spec
        return HelperInstallResult(
            name=helper.name,
            ok=False,
            message=f"unsupported installer {installer!r}",
        )
    log.info("installing helper %s via: %s", helper.name, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
            check=False,
            env=_helper_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        return HelperInstallResult(
            name=helper.name,
            ok=False,
            message=f"install timed out after {INSTALL_TIMEOUT_SECONDS}s",
        )
    except FileNotFoundError as exc:
        return HelperInstallResult(
            name=helper.name,
            ok=False,
            message=f"install failed: {exc}",
        )
    if proc.returncode != 0:
        tail = (proc.stderr.strip() or proc.stdout.strip())[-300:]
        return HelperInstallResult(
            name=helper.name,
            ok=False,
            message=f"install exited {proc.returncode}: {tail}",
        )
    return HelperInstallResult(
        name=helper.name,
        ok=True,
        message=f"installed via {installer} ({' '.join(args)})",
    )


def build_tool_specs(env: "ToolEnv") -> list[dict[str, Any]]:
    specs = list(TOOL_SPECS)
    for helper in env.helper_tools.values():
        properties: dict[str, Any] = {}
        if helper.allow_args:
            properties["args"] = {
                "type": "array",
                "description": (
                    "Additional CLI arguments appended after the helper's "
                    f"configured command. Max {helper.max_args} items."
                ),
                "items": {"type": "string"},
            }
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": helper.name,
                    "description": helper.description,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": properties,
                    },
                },
            }
        )
    return specs


@dataclass
class ToolEnv:
    """The repo root the tools operate against. Constructed once per
    review and threaded into ``run_tool``."""

    repo_root: str
    helper_tools: dict[str, RepoHelperTool] = field(default_factory=dict)
    # Isolation policy for helper subprocesses; see reviewbot/sandbox.py.
    sandbox_mode: str = AUTO

    def __post_init__(self) -> None:
        self.repo_root = os.path.realpath(self.repo_root)
        if not os.path.isdir(self.repo_root):
            raise ValueError(f"repo_root is not a directory: {self.repo_root}")


def run_tool(env: ToolEnv, name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call. Always returns a string; on failure, returns
    a short error message rather than raising — the model sees it as the
    ``tool`` message content and can decide what to do next."""
    try:
        if name == "read_file":
            return _read_file(env, arguments)
        if name == "list_dir":
            return _list_dir(env, arguments)
        if name == "grep":
            return _grep(env, arguments)
        if name == "fetch_url":
            return _fetch_url(arguments)
        helper = env.helper_tools.get(name)
        if helper is not None:
            return _run_repo_helper(env, helper, arguments)
        return f"error: unknown tool {name!r}"
    except _ToolError as exc:
        return f"error: {exc}"
    except Exception:  # pragma: no cover — defensive
        log.exception("tool %s crashed", name)
        return f"error: tool {name} crashed; see action log"


class _ToolError(Exception):
    pass


def _resolve_path(env: ToolEnv, raw: Any, *, default: str = ".") -> str:
    if raw is None or raw == "":
        raw = default
    if not isinstance(raw, str):
        raise _ToolError(f"path must be a string, got {type(raw).__name__}")
    # Reject obviously-hostile inputs early so the error message is clearer
    # than a realpath mismatch.
    if raw.startswith("/") or raw.startswith("~"):
        raise _ToolError("absolute or home-relative paths are not allowed")
    candidate = os.path.realpath(os.path.join(env.repo_root, raw))
    root_with_sep = env.repo_root + os.sep
    if candidate != env.repo_root and not candidate.startswith(root_with_sep):
        raise _ToolError(f"path {raw!r} escapes the repository root")
    parts = os.path.relpath(candidate, env.repo_root).split(os.sep)
    if any(p in DENY_DIR_NAMES for p in parts):
        raise _ToolError(f"path {raw!r} is inside a denylisted directory")
    return candidate


def _truncate(text: str, *, suffix_note: str = "") -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    head = text[:MAX_TOOL_OUTPUT_CHARS]
    note = (
        f"\n[... truncated; output exceeded {MAX_TOOL_OUTPUT_CHARS} chars"
        + (f"; {suffix_note}" if suffix_note else "")
        + " ...]"
    )
    return head + note


def _read_file(env: ToolEnv, args: dict[str, Any]) -> str:
    path = _resolve_path(env, args.get("path"), default="")
    if not path or path == env.repo_root:
        raise _ToolError("read_file requires a file path, not the repo root")
    if not os.path.isfile(path):
        raise _ToolError(f"not a file: {os.path.relpath(path, env.repo_root)}")
    start = max(1, int(args.get("start_line") or 1))
    end_arg = args.get("end_line")
    end = int(end_arg) if end_arg is not None else start + MAX_READ_LINES - 1
    if end < start:
        raise _ToolError(f"end_line ({end}) must be >= start_line ({start})")
    end = min(end, start + MAX_READ_LINES - 1)
    rel = os.path.relpath(path, env.repo_root)
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if idx < start:
                    continue
                if idx > end:
                    break
                lines.append(f"{idx:>6}\t{line.rstrip(chr(10))}")
    except OSError as exc:
        raise _ToolError(f"could not read {rel}: {exc}") from exc
    if not lines:
        return f"{rel}: no lines in range {start}-{end}"
    body = "\n".join(lines)
    return _truncate(
        f"{rel} (lines {start}-{start + len(lines) - 1}):\n{body}",
        suffix_note="re-call with a narrower line range",
    )


def _list_dir(env: ToolEnv, args: dict[str, Any]) -> str:
    path = _resolve_path(env, args.get("path"), default=".")
    if not os.path.isdir(path):
        raise _ToolError(
            f"not a directory: {os.path.relpath(path, env.repo_root) or '.'}"
        )
    rel = os.path.relpath(path, env.repo_root) or "."
    entries: list[str] = []
    try:
        for name in sorted(os.listdir(path)):
            if name in DENY_DIR_NAMES:
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                entries.append(f"{name}/")
            else:
                entries.append(name)
    except OSError as exc:
        raise _ToolError(f"could not list {rel}: {exc}") from exc
    if not entries:
        return f"{rel}: (empty)"
    return _truncate(f"{rel}:\n" + "\n".join(entries))


_SAFE_PATTERN_RE = re.compile(r"^[\x20-\x7E]+$")


def _grep(env: ToolEnv, args: dict[str, Any]) -> str:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise _ToolError("pattern must be a non-empty string")
    if len(pattern) > 200:
        raise _ToolError("pattern is too long (max 200 chars)")
    if not _SAFE_PATTERN_RE.match(pattern):
        raise _ToolError("pattern contains non-printable or non-ASCII characters")
    path_arg = args.get("path")
    pathspec = _resolve_path(env, path_arg, default=".") if path_arg else env.repo_root
    rel_pathspec = os.path.relpath(pathspec, env.repo_root) or "."
    max_results = int(args.get("max_results") or 50)
    max_results = max(1, min(max_results, 200))
    cmd = [
        "git",
        "-C",
        env.repo_root,
        "grep",
        "-n",
        "-I",
        "-E",
        "--max-count=10",
        "--",
        pattern,
    ]
    if path_arg:
        cmd.append(rel_pathspec)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _ToolError("grep timed out (20s)") from exc
    except FileNotFoundError as exc:
        raise _ToolError("git is not installed in the runner") from exc
    if proc.returncode == 1 and not proc.stdout:
        return f"no matches for {pattern!r} in {rel_pathspec}"
    if proc.returncode > 1:
        return f"error: git grep exited {proc.returncode}: {proc.stderr.strip()[:300]}"
    lines = proc.stdout.splitlines()
    truncated = len(lines) > max_results
    lines = lines[:max_results]
    body = "\n".join(lines) if lines else "(no output)"
    suffix = f"showing {len(lines)} of {len(lines)}+ matches" if truncated else ""
    header = f"git grep -E {pattern!r} -- {rel_pathspec}:"
    return _truncate(f"{header}\n{body}", suffix_note=suffix)


def _fetch_url(args: dict[str, Any]) -> str:
    raw = args.get("url")
    if not isinstance(raw, str) or not raw:
        raise _ToolError("url must be a non-empty string")
    parsed = urlparse(raw)
    if parsed.scheme != "https":
        raise _ToolError("only https URLs are allowed")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_FETCH_HOSTS:
        raise _ToolError(
            f"host {host!r} is not in the allowlist; only "
            f"{sorted(ALLOWED_FETCH_HOSTS)} are permitted"
        )
    # Disable redirects so a 30x doesn't smuggle us off-host. The model
    # can re-call with the redirect target if it's still on the allowlist.
    try:
        r = requests.get(
            raw,
            timeout=FETCH_TIMEOUT_SECONDS,
            allow_redirects=False,
            headers={"User-Agent": "serge/1.0 (link verification)"},
        )
    except requests.Timeout as exc:
        raise _ToolError(f"fetch timed out after {FETCH_TIMEOUT_SECONDS}s") from exc
    except requests.RequestException as exc:
        raise _ToolError(f"fetch failed: {exc}") from exc

    header_lines = [f"HTTP {r.status_code} {r.reason}", f"URL: {raw}"]
    ct = r.headers.get("Content-Type")
    if ct:
        header_lines.append(f"Content-Type: {ct}")
    if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
        loc = r.headers.get("Location", "")
        header_lines.append(f"Location: {loc}")
    # Skip body for non-text responses; the model only needs to know
    # the link resolves.
    if ct and not (ct.startswith("text/") or "json" in ct or "xml" in ct):
        body = f"(non-text body, {len(r.content)} bytes; not shown)"
    else:
        text = r.text or ""
        body = text[:MAX_FETCH_BODY_CHARS]
        if len(text) > MAX_FETCH_BODY_CHARS:
            body += f"\n[... truncated; full body was {len(text)} chars ...]"
    return _truncate("\n".join(header_lines) + "\n\n" + body)


def _resolve_helper_command(env: ToolEnv, raw: str) -> str:
    # A bare name (no slash) is looked up on PATH — a system or
    # pip-installed binary, never fork-tree content.
    if "/" not in raw:
        return raw
    # A path-form command may only resolve under .ai/, which the clone
    # cache forces to the repo's default branch (upstream). This stops a
    # fork PR from pointing a helper at an executable it shipped elsewhere
    # in its tree (e.g. ./scripts/lint.sh).
    path = _resolve_path(env, raw, default="")
    ai_root = os.path.join(env.repo_root, ".ai")
    if path != ai_root and not path.startswith(ai_root + os.sep):
        raise _ToolError(
            f"helper command {raw!r} must be a PATH binary or live under .ai/"
        )
    if not os.path.isfile(path):
        raise _ToolError(f"helper command path is not a file: {raw}")
    return path


def _parse_helper_args(helper: RepoHelperTool, args: dict[str, Any]) -> list[str]:
    extra = args.get("args")
    if extra is None:
        return []
    if not helper.allow_args:
        raise _ToolError(f"helper {helper.name!r} does not accept additional args")
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise _ToolError("args must be an array of strings")
    if len(extra) > helper.max_args:
        raise _ToolError(
            f"helper {helper.name!r} accepts at most {helper.max_args} additional args"
        )
    cleaned: list[str] = []
    for item in extra:
        if not item:
            raise _ToolError("helper args may not be empty strings")
        if len(item) > MAX_HELPER_ARG_CHARS:
            raise _ToolError(
                f"helper args are capped at {MAX_HELPER_ARG_CHARS} chars each"
            )
        if "\x00" in item:
            raise _ToolError("helper args may not contain NUL bytes")
        cleaned.append(item)
    return cleaned


def _run_repo_helper(env: ToolEnv, helper: RepoHelperTool, args: dict[str, Any]) -> str:
    cwd = _resolve_path(env, helper.cwd, default=".")
    if not os.path.isdir(cwd):
        raise _ToolError(f"helper cwd is not a directory: {helper.cwd!r}")

    extra_args = _parse_helper_args(helper, args)
    command = [
        _resolve_helper_command(env, helper.command[0]),
        *helper.command[1:],
        *extra_args,
    ]
    rel_cwd = os.path.relpath(cwd, env.repo_root) or "."
    # Isolate the subprocess: no network, no host secrets, only the
    # worktree is writable. See reviewbot/sandbox.py.
    try:
        argv = wrap_command(
            command, workdir=cwd, write_root=env.repo_root, mode=env.sandbox_mode
        )
    except SandboxUnavailable as exc:
        raise _ToolError(str(exc)) from exc
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=helper.timeout_seconds,
            check=False,
            env=_helper_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise _ToolError(
            f"helper {helper.name!r} timed out after {helper.timeout_seconds}s"
        ) from exc
    except FileNotFoundError as exc:
        raise _ToolError(f"helper command not found: {helper.command[0]!r}") from exc

    # Helper output is attacker-influenced (the helper runs against the
    # PR's tree). Wrap stdout/stderr in explicit untrusted markers so the
    # model has a clear boundary; the system prompt already tells it not
    # to follow instructions found inside tool output.
    stdout = _scrub_helper_delimiters(proc.stdout.strip())
    stderr = _scrub_helper_delimiters(proc.stderr.strip())
    sections = [
        f"{helper.name} (cwd {rel_cwd})",
        f"exit_code: {proc.returncode}",
        "command: " + " ".join(command),
    ]
    if stdout:
        sections.append(
            "--- BEGIN UNTRUSTED HELPER STDOUT ---\n"
            + stdout
            + "\n--- END UNTRUSTED HELPER STDOUT ---"
        )
    if stderr:
        sections.append(
            "--- BEGIN UNTRUSTED HELPER STDERR ---\n"
            + stderr
            + "\n--- END UNTRUSTED HELPER STDERR ---"
        )
    if not stdout and not stderr:
        sections.append("(no output)")
    return _truncate("\n\n".join(sections))


def _scrub_helper_delimiters(text: str) -> str:
    """Defang the BEGIN/END markers if the helper happened to emit them
    in its own output."""
    if not text:
        return text
    for needle in (
        "--- BEGIN UNTRUSTED",
        "--- END UNTRUSTED",
    ):
        text = text.replace(needle, needle[:3] + "​" + needle[3:])
    return text
