"""Project-history tools: serge asking `relore` what this repo already decided.

serge's existing browse tools (:mod:`reviewbot.tools`) all answer *what the code
is right now*, from one checkout. They cannot answer the questions that actually
sink a review or a fix:

- **"Is this intentional?"** — the reviewer is about to flag something a
  maintainer already ruled correct, in a thread nobody is going to re-read.
- **"Has anyone hit this?"** — the ITF agent is about to diagnose a failure that
  has an issue open against it with the answer in the second comment.
- **"Is somebody already fixing this?"** — the most expensive mistake the task
  loop makes is writing a patch for a pull request that has been open for hours.
  ``relore inflight`` is one call and it prevents exactly that.

`relore` (https://github.com/huggingface/relore, deployed in-cluster as
``ghlore``) indexes issues, PR bodies, reviews and inline review comments, and
serves them ranked, trust-tiered and aged. This module exposes four of its verbs
to the LLM.

Why shell out to the ``relore`` client rather than call ``/api/v1`` directly
-------------------------------------------------------------------------
Three properties live in the client and would have to be re-implemented here,
where they would drift silently:

1. **The version handshake** (``relore.wire``). Client and daemon must be the
   exact same version; a mismatch is refused with a sentence naming which end is
   behind. A hand-rolled HTTP caller would have to declare a version it is not
   installed at, and the one workaround available to it — echoing back whatever
   the daemon reports — is precisely the failure the handshake exists to stop.
   Installing the client pins the contract in the image, where a drift surfaces
   as relore's own actionable error rather than as a quietly incomplete answer.
2. **The untrusted-content envelope** (``relore.security.untrusted``). Every
   indexed byte was written by whoever opened the issue. relore scrubs it
   server-side and wraps it in ``<<<RELORE-UNTRUSTED>>>`` with each quoted line
   prefixed ``>``, so an unmarked line is always relore's own assertion. We relay
   that verbatim — **we do not re-wrap or re-scrub it**, because serge's own
   ``--- BEGIN UNTRUSTED ---`` markers around the whole page would tell the model
   to discount ``[authoritative]``, which is the single most load-bearing thing
   in the output and relore's claim, not a quotation.
3. **The failure vocabulary.** A daemon that is down, an empty index, a repo out
   of scope and a stale client read identically to a naive caller and take
   opposite next actions. relore tells them apart in one sentence each; we relay
   the sentence.

The cost is a binary in the image (see ``docker/Dockerfile*``), pinned at build
time. That is deliberate: the September 2026 httpx incident is what taught us
that a runtime ``pip install`` is not available to a task pod anyway — its egress
allowlist has no PyPI.

Scope, and what the model may not do
------------------------------------
``--repo`` is **always supplied by serge**, from the pull request or task being
worked on, and is never model-supplied. The model cannot aim these tools at
another repository: relore's token scoping is a real perimeter, and a review of
repo A that quotes decisions from repo B is wrong even when both are in scope.

The tools are exposed only when the operator configured a base URL *and* listed
this repository in ``RELORE_REPOS``. relore indexes a handful of repositories; on
an unindexed one every call would come back empty and the model would spend
turns learning that. Config, not discovery, so the tool schema is deterministic.

There is no write verb here and there will not be one. relore has none by
design — an agent's conclusion must never become project memory that the next
agent retrieves as evidence.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

# The client's entry point. Installed from git at image build time; absent in a
# dev checkout that did not install it, which is reported once as a disabled
# feature rather than per call.
RELORE_BIN = "relore"

# Same budget as the other browse tools: a tool result is re-billed on every
# later turn of the session that read it, so its real cost is its size times the
# turns remaining. relore's own output is already compact-by-default when its
# stdout is a pipe (which it is here), so this cap is a backstop, not the usual
# path.
MAX_RELORE_OUTPUT_CHARS = 8000

# Per-call wall clock. The daemon's own HTTP timeout is 30s; this is that plus
# room for process start, so a hung call is attributed to the right layer.
DEFAULT_RELORE_TIMEOUT = 45

# `search --limit`. Ten is relore's default and the right default here: the
# question these tools answer is "did someone already settle this", which the
# top few hits answer or nothing does.
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 25

# What the model may put in a string argument. Deliberately permissive about
# content — an error message, a test id and a symbol are exactly the strings
# worth searching for — and strict about the two shapes that are not content:
# a leading `-` (which argparse would read as a flag, letting the model reach
# verbs and options this module does not expose) and control characters.
_LEADING_DASH = re.compile(r"^\s*-")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MAX_ARG_CHARS = 400
# Repeatable filters (`--error`, `--test`, `--file`, `--symbol`). More than a
# handful is a sign the model is pasting a traceback, which relore's own guidance
# warns against: every term ANDs, so a long list is an empty result.
MAX_FILTER_ITEMS = 5

_KINDS = ("failure", "precedent", "rationale")

# Environment the client subprocess gets. Minimal on purpose — it is a network
# client, not repo code, but it runs in the same process tree as the GitHub token
# and the LLM key and has no business seeing either.
_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    # The daemon's address and, where the deployment still uses one, its token.
    # `api.trustNetwork: true` means prod needs no token; the variable is passed
    # through anyway so a tokened deployment needs no code change.
    "RELORE_API",
    "RELORE_API_TOKEN",
    # A task pod has no route out except the allowlisting forward proxy, injected
    # as HTTPS_PROXY (reviewbot/k8s_sandbox.py). httpx honours these by default;
    # without them every call from a task pod would time out.
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
)


class ReloreToolError(Exception):
    """A call could not be made. The message goes to the model as the tool
    result, so it says what to do differently."""


@dataclass(frozen=True)
class ReloreEnv:
    """Where to ask, and about what.

    ``repo`` is the one being reviewed or patched. It is baked into every call
    and is not reachable from the tool schema.
    """

    repo: str
    api: Optional[str] = None
    timeout: int = DEFAULT_RELORE_TIMEOUT
    executable: str = RELORE_BIN


def make_relore_env(cfg: Any, repo_full_name: Optional[str]) -> Optional[ReloreEnv]:
    """Build the env for ``repo_full_name``, or ``None`` when the history tools
    should not be offered at all.

    Four ways to get ``None``, each logged at a level that matches how much the
    operator is likely to care: no base URL (feature off), no repo list (feature
    off), this repo not indexed (expected on most repos), client not installed
    (a deployment mistake — the image should carry it).
    """
    api = (getattr(cfg, "relore_api", None) or "").strip()
    if not api:
        return None
    indexed = tuple(getattr(cfg, "relore_repos", ()) or ())
    if not indexed:
        return None
    if not repo_full_name:
        return None
    # GitHub repo names are case-insensitive and the index stores one spelling;
    # match on the fold but send what the operator configured, so the daemon sees
    # the name it knows.
    wanted = repo_full_name.strip().lower()
    match = next((name for name in indexed if name.strip().lower() == wanted), None)
    if match is None:
        log.info(
            "relore history tools off for %s: not in RELORE_REPOS (%s)",
            repo_full_name,
            ", ".join(indexed),
        )
        return None
    executable = RELORE_BIN
    if shutil.which(executable) is None:
        log.warning(
            "RELORE_API is set and %s is indexed, but the %r client is not on PATH; "
            "history tools disabled. The image should install it "
            "(pip install 'relore @ git+https://github.com/huggingface/relore').",
            repo_full_name,
            executable,
        )
        return None
    env = ReloreEnv(
        repo=match,
        api=api,
        timeout=int(getattr(cfg, "relore_timeout", DEFAULT_RELORE_TIMEOUT)),
        executable=executable,
    )
    log.info("relore history tools enabled for %s against %s", env.repo, env.api)
    return env


# -- tool schema -----------------------------------------------------------
#
# Four verbs, not relore's fourteen. The code lens (`grep`, `symbol`, `copies`,
# `defs`, `refs`, `map`) is deliberately left out: it reads the daemon's clone of
# `main`, and serge already has `grep` and `read_file` rooted at the PR head,
# which is the tree that actually matters here and includes the change under
# review. Offering both would give the model two greps with different answers and
# no way to tell which it wanted.

RELORE_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "history_search",
            "description": (
                "Search this repository's issue and pull-request history — "
                "issue bodies, PR descriptions, reviews, inline review comments "
                "— for what people already decided or already hit. The query is "
                "an AND of every term: pass two or three distinctive ones, "
                "never a sentence. `query` may be empty if you pass a filter "
                "instead — searching on `test` alone is the natural shape when a "
                "test id is the question."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Two or three distinctive terms, ANDed.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": list(_KINDS),
                        "description": (
                            "Who may answer. 'rationale' = people entitled to "
                            "decide (use for 'is this intentional'). 'failure' = "
                            "reports welcome, a stranger's traceback counts. "
                            "'precedent' = judgements plus merged PRs. Omit for "
                            "everyone who commented."
                        ),
                    },
                    "error": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "This is where a traceback goes — a whole pasted "
                            "traceback, one error line, or a fragment all work: "
                            "the raised error is pulled out and normalized (paths "
                            "and numbers collapsed) to the form the index stored, "
                            "then matched by containment. It ANDs with `query`, "
                            "so an empty result may just mean nobody pasted this "
                            f"failure here. Max {MAX_FILTER_ITEMS}."
                        ),
                    },
                    "test": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "A test id as a runner spells it (path::Class::test), "
                            f"not a bare function name. Max {MAX_FILTER_ITEMS}."
                        ),
                    },
                    "file": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "A repository path as the repo spells it, not an "
                            f"absolute one from a traceback. Max {MAX_FILTER_ITEMS}."
                        ),
                    },
                    "symbol": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"A bare function or class name. Max {MAX_FILTER_ITEMS}."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Hits to return. Default {DEFAULT_SEARCH_LIMIT}, max "
                            f"{MAX_SEARCH_LIMIT}."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "history_thread",
            "description": (
                "Read one issue or pull request by number: the opening post "
                "and its comments. Use it after `history_search` returns a hit "
                "worth reading in full, or when a diff or commit message "
                "references '#1234'."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue or pull-request number.",
                    },
                    "focus": {
                        "type": "string",
                        "description": (
                            "Rank the comments by these terms, best first; with "
                            "`outline`, keep only the lines carrying one."
                        ),
                    },
                    "outline": {
                        "type": "boolean",
                        "description": (
                            "One line per comment for the WHOLE thread instead of "
                            "a page of ten. Ask for the shape first on a long one."
                        ),
                    },
                    "files": {
                        "type": "boolean",
                        "description": (
                            "Also list a pull request's changed-file paths. The "
                            "count and any truncation notice are always shown."
                        ),
                    },
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "history_why",
            "description": (
                "For one line of one file: the pull request that last changed "
                "it, and the review comments left on or near that line. `git "
                "blame` gives you the commit; this gives you the argument. Reach "
                "for it the moment you are looking at a line you do not "
                "understand."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the repository root.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number, 1-indexed.",
                    },
                },
                "required": ["path", "line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "history_copies",
            "description": (
                "READS THE REPOSITORY'S DEFAULT BRANCH, not this pull request's "
                "head — use `grep` and `read_file` for the code under review. "
                "Given a function or class name, lists every definition of it in "
                "the repo, GROUPED by what the body does (type annotations, "
                "docstrings and comments normalized away), largest group first. "
                "Use it where code is duplicated on purpose — transformers' "
                "per-model `modeling_*.py` files are the case it was built for. "
                "There the question is never 'where is it' but 'which one "
                "diverged', and the answer is a small group at the END of the "
                "output, not the majority shape at the top. A copy that differs "
                "from 180 identical siblings is either the bug or the fix that "
                "everything else is missing."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "A bare function or class name.",
                    },
                    "exact": {
                        "type": "boolean",
                        "description": (
                            "Group by the body's exact text instead, annotations "
                            "and docstrings included. Use to tell a real "
                            "divergence from a formatting one."
                        ),
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "history_inflight",
            "description": (
                "Given an ISSUE number, list the open pull requests that "
                "already claim to close it ('Fixes #N'). ASK THIS FIRST, before "
                "diagnosing and long before writing a patch: building a fix for "
                "something already in review is the most expensive mistake "
                "available here, and this is one cheap call."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number.",
                    },
                },
                "required": ["number"],
            },
        },
    },
]

RELORE_TOOL_NAMES = frozenset(spec["function"]["name"] for spec in RELORE_TOOL_SPECS)


# -- argument validation ---------------------------------------------------


def _text(raw: Any, field: str, *, required: bool = False) -> str:
    if raw is None:
        if required:
            raise ReloreToolError(f"{field} is required")
        return ""
    if not isinstance(raw, str):
        raise ReloreToolError(f"{field} must be a string, got {type(raw).__name__}")
    value = raw.strip()
    if not value:
        if required:
            raise ReloreToolError(f"{field} is required")
        return ""
    if len(value) > MAX_ARG_CHARS:
        raise ReloreToolError(
            f"{field} is {len(value)} chars; cap is {MAX_ARG_CHARS}. "
            "Cut it to the distinctive part — a query is an AND of every term, "
            "so a long one matches nothing anyway."
        )
    if _CONTROL_CHARS.search(value):
        raise ReloreToolError(f"{field} contains control characters")
    if _LEADING_DASH.match(value):
        # Not a safety net around argparse so much as a clear message: a value
        # starting with `-` would be parsed as a flag and the failure would read
        # as "unknown option", which tells the model nothing.
        raise ReloreToolError(
            f"{field} may not start with '-' (it would be read as a command-line flag)"
        )
    return value


def _items(raw: Any, field: str) -> list[str]:
    if raw is None:
        return []
    # A model that has one filter value often sends the bare string rather than a
    # one-element list. Accepting it costs nothing and saves a wasted turn.
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ReloreToolError(f"{field} must be an array of strings")
    if len(raw) > MAX_FILTER_ITEMS:
        raise ReloreToolError(
            f"{field} has {len(raw)} values; cap is {MAX_FILTER_ITEMS}. "
            "Filters AND with each other and with the query, so more of them "
            "narrows to nothing."
        )
    return [_text(item, field, required=True) for item in raw]


def _number(raw: Any, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        # A model that read "#1234" out of a diff sometimes sends the string.
        if isinstance(raw, str) and raw.strip().lstrip("#").isdigit():
            raw = int(raw.strip().lstrip("#"))
        else:
            raise ReloreToolError(f"{field} must be an integer")
    if raw < 1:
        raise ReloreToolError(f"{field} must be positive, got {raw}")
    return int(raw)


def _flag(raw: Any) -> bool:
    return bool(raw)


def _build_argv(env: ReloreEnv, name: str, args: dict[str, Any]) -> list[str]:
    """The exact command line, verb by verb.

    ``--repo`` is appended by us on every verb that takes one and is never read
    from ``args``: a daemon serving several repositories refuses a bare number
    rather than guess, and which repository serge is working on is serge's fact,
    not the model's.

    ``--plain`` forces the piped output form regardless of whether stdout looks
    like a terminal, so the bytes the model sees do not depend on how serge was
    started. Compact rendering is already the default for a pipe.
    """
    argv = [env.executable, "--plain"]

    if name == "history_search":
        # relore takes the query as an optional positional: a search on `--test`
        # or `--error` alone is a legitimate shape, and it is the natural one on
        # an integration failure, where the test id IS the question. So require
        # *something* rather than requiring the text.
        query = _text(args.get("query"), "query")
        filters = {
            field: _items(args.get(field), field)
            for field in ("error", "test", "file", "symbol")
        }
        if not query and not any(filters.values()):
            raise ReloreToolError(
                "give a query, or at least one of error/test/file/symbol"
            )
        argv += ["search", query]
        kind = _text(args.get("kind"), "kind")
        if kind:
            if kind not in _KINDS:
                raise ReloreToolError(
                    f"kind must be one of {', '.join(_KINDS)}, got {kind!r}"
                )
            argv += ["--kind", kind]
        for field in ("error", "test", "file", "symbol"):
            for value in filters[field]:
                argv += [f"--{field}", value]
        limit = args.get("limit")
        if limit is not None:
            argv += ["--limit", str(min(_number(limit, "limit"), MAX_SEARCH_LIMIT))]
        else:
            argv += ["--limit", str(DEFAULT_SEARCH_LIMIT)]
        argv += ["--repo", env.repo]
        return argv

    if name == "history_thread":
        argv += ["thread", str(_number(args.get("number"), "number"))]
        focus = _text(args.get("focus"), "focus")
        if focus:
            argv += ["--focus", focus]
        if _flag(args.get("outline")):
            argv.append("--outline")
        if _flag(args.get("files")):
            argv.append("--files")
        argv += ["--repo", env.repo]
        return argv

    if name == "history_why":
        path = _text(args.get("path"), "path", required=True)
        line = _number(args.get("line"), "line")
        if ":" in path:
            # relore takes PATH:LINE as one token, so a colon in the path would
            # split it somewhere else entirely and answer about a line the model
            # never asked about. Refuse rather than guess.
            raise ReloreToolError(f"path may not contain ':', got {path!r}")
        argv += ["why", f"{path}:{line}", "--repo", env.repo]
        return argv

    if name == "history_copies":
        argv += ["copies", _text(args.get("symbol"), "symbol", required=True)]
        if _flag(args.get("exact")):
            argv.append("--exact")
        argv += ["--repo", env.repo]
        return argv

    if name == "history_inflight":
        argv += [
            "inflight",
            str(_number(args.get("number"), "number")),
            "--repo",
            env.repo,
        ]
        return argv

    raise ReloreToolError(f"unknown history tool {name!r}")


# -- execution -------------------------------------------------------------


def _subprocess_env(env: ReloreEnv) -> dict[str, str]:
    out = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    if env.api:
        out["RELORE_API"] = env.api
    # RELORE_REPO would supply a default for --repo. We pass --repo explicitly on
    # every call, so unsetting it removes the only way a future verb could be
    # scoped by something other than serge's own fact.
    out.pop("RELORE_REPO", None)
    return out


def _truncate(text: str) -> str:
    """Cap the result by dropping the MIDDLE, not the tail.

    Two reasons, and both are the same mistake in different clothes — cutting
    off the part that carries the answer:

    * `history_copies` orders its groups largest-first, so on a repository that
      duplicates code on purpose the interesting shape (the one that diverged)
      is the last thing printed. `compute_default_rope_parameters` on
      transformers is 12KB of which the first 114 lines are the majority shape;
      head-truncation keeps the boring bulk and drops the finding.
    * relore closes every page with ``<<<RELORE-UNTRUSTED-END>>>``. Cutting the
      tail leaves the envelope unterminated, so the model cannot tell where
      quoted text stops — the one thing the envelope exists to mark.

    So keep both ends and say what went missing in between. No parsing of
    relore's format, so nothing here drifts when that format changes.
    """
    if len(text) <= MAX_RELORE_OUTPUT_CHARS:
        return text
    marker_budget = 200
    head_chars = (MAX_RELORE_OUTPUT_CHARS - marker_budget) * 3 // 5
    tail_chars = MAX_RELORE_OUTPUT_CHARS - marker_budget - head_chars
    dropped = len(text) - head_chars - tail_chars
    marker = (
        f"\n\n[... {dropped} chars dropped from the MIDDLE to fit the {MAX_RELORE_OUTPUT_CHARS}-char "
        "budget; the start and the end are both intact. Narrow the query, or use "
        "history_thread with outline=true for the shape first ...]\n\n"
    )
    return text[:head_chars] + marker + text[-tail_chars:]


def run_relore_tool(env: ReloreEnv, name: str, arguments: dict[str, Any]) -> str:
    """Dispatch one history tool. Always returns a string the model can act on.

    stdout is relayed **verbatim**. relore has already scrubbed the retrieved
    text and wrapped it in its own ``<<<RELORE-UNTRUSTED>>>`` envelope with each
    quoted line prefixed ``>``; re-wrapping the page in serge's markers would put
    relore's own trust labels inside a "do not trust the text below" region,
    which is the opposite of what they mean.

    A non-zero exit is relayed too, for the same reason: relore's failure
    messages tell a daemon that is down apart from an empty index apart from a
    stale client, and each takes a different next action. Swallowing them into
    "history unavailable" is how an agent concludes the project has no history.
    """
    try:
        argv = _build_argv(env, name, arguments)
    except ReloreToolError as exc:
        return f"error: {exc}"

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=env.timeout,
            check=False,
            env=_subprocess_env(env),
        )
    except subprocess.TimeoutExpired:
        return (
            f"error: {name} timed out after {env.timeout}s. The relore daemon may be "
            "slow or unreachable; carry on without the history rather than retrying."
        )
    except FileNotFoundError:
        return (
            f"error: the {env.executable!r} client is not installed in this runner, so "
            "the history tools cannot be used in this run."
        )
    except Exception:  # pragma: no cover — defensive
        log.exception("relore tool %s crashed", name)
        return f"error: {name} crashed; see action log"

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"exit {proc.returncode} with no output"
        return _truncate(f"error: {detail}")
    if not stdout:
        # relore exits 0 on an empty result and says so in prose; a genuinely
        # silent success would be a new shape, so name it rather than returning
        # nothing and letting it read as "no history".
        return f"{name}: no output (exit 0). Try broader terms, or drop a filter."
    return _truncate(stdout)


# -- the competing-PR check (serge asks; the model is not consulted) ---------
#
# `history_inflight` is in the model's schema, but a REVIEW has nothing useful
# to pass it: `inflight <the PR under review>` asks "what claims to close this
# pull request", and nothing closes a pull request, so it is well-formed and
# always empty. The useful question is one hop further out — take the issue this
# PR claims to close, and ask who ELSE claims to close it. The other claimants
# are competing pull requests, and "this duplicates #48758, also open" is a
# finding a human reviewer wants and serge could not previously see.
#
# Done here rather than left to the model, because measured over six review runs
# the model reached for the history tools 0, 0, 1, 3, 0, 0 times. A check worth
# having on every review cannot depend on that.

#: GitHub's closing keywords, as GitHub itself accepts them: `Fixes #123`,
#: `closed: #123`, and the full-URL form. Deliberately not `related to` or
#: `see also` — only a claim to CLOSE makes another PR a competitor.
_CLOSING = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s*"
    r"(?:https?://github\.com/[\w.-]+/[\w.-]+/issues/(?P<url>\d+)|#(?P<hash>\d+))",
    re.IGNORECASE,
)

#: Issues to follow per review. A PR closing more than a couple is unusual, and
#: each one costs a daemon call before the loop starts.
MAX_CLOSING_ISSUES = 3


@dataclass(frozen=True)
class CompetingPR:
    """Another open pull request claiming to close the same issue."""

    number: int
    title: str
    author: str
    url: str
    draft: bool
    issue: int


def closing_issue_numbers(body: str, *, limit: int = MAX_CLOSING_ISSUES) -> list[int]:
    """Issue numbers a PR body claims to close, in order, deduped."""
    seen: dict[int, None] = {}
    for match in _CLOSING.finditer(body or ""):
        raw = match.group("url") or match.group("hash")
        try:
            seen.setdefault(int(raw), None)
        except (TypeError, ValueError):
            continue
    return list(seen)[:limit]


def _relore_json(env: ReloreEnv, args: list[str]) -> dict[str, Any] | None:
    """One `relore --json` call for serge's own use. ``None`` on any failure."""
    try:
        proc = subprocess.run(
            [env.executable, "--json", *args],
            capture_output=True,
            text=True,
            timeout=env.timeout,
            check=False,
            env=_subprocess_env(env),
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def competing_open_prs(env: ReloreEnv, *, body: str, exclude: int) -> list[CompetingPR]:
    """Open pull requests claiming to close the same issues as this one.

    ``exclude`` is the PR under review, which is itself a claimant and must not
    be reported as its own duplicate. Empty on every failure path — this is
    extra context for a review, never a gate.
    """
    issues = closing_issue_numbers(body)
    if not issues:
        return []
    found: list[CompetingPR] = []
    seen: set[int] = set()
    for issue in issues:
        payload = _relore_json(env, ["inflight", str(issue), "--repo", env.repo])
        if not payload:
            continue
        for raw in payload.get("claims") or []:
            if not isinstance(raw, dict) or raw.get("type") != "pr":
                continue
            number = raw.get("number")
            if not isinstance(number, int) or number == exclude or number in seen:
                continue
            # Open only. A merged or closed claimant is history, not competition.
            if raw.get("state") != "open" or raw.get("merged"):
                continue
            seen.add(number)
            found.append(
                CompetingPR(
                    number=number,
                    title=str(raw.get("title") or ""),
                    author=str(raw.get("author") or ""),
                    url=str(raw.get("url") or ""),
                    draft=bool(raw.get("draft")),
                    issue=issue,
                )
            )
    return found


def competing_pr_note(competing: list[CompetingPR]) -> str:
    """The reviewer-side note, or ``""``. Trusted context: these are facts serge
    looked up, not text anyone wrote, so they carry no untrusted envelope."""
    if not competing:
        return ""
    lines = [
        "Another open pull request claims to close the same issue as this one. "
        "That may be a duplicate effort worth pointing out, or the two may be "
        "complementary — say which, and link it, rather than assuming:",
    ]
    for pr in competing:
        draft = " (draft)" if pr.draft else ""
        who = f" by @{pr.author}" if pr.author else ""
        lines.append(f"- #{pr.number}{draft}{who} — {pr.title} — {pr.url}")
        lines.append(f"  (both claim to close #{pr.issue})")
    return "\n".join(lines)


# -- deterministic prior-art lookup ----------------------------------------
#
# Why serge runs these searches itself instead of telling the model to.
#
# The task system prompt already says, in order: `history_inflight` BEFORE
# diagnosing, then `history_search` the failure. Measured over the 20 jobs in
# the store that ran on a relore-indexed repository (2026-09-16..18), 6 called a
# history tool at all, and the earliest any of them did was the **14th** tool
# call — median 22nd, worst 38th. The two tasks that ended in a published PR
# called none. 29 history calls against 877 `grep`/`read_file` calls, 3.2%.
#
# So the model does reach for history — but only once it is already lost, which
# is after it has committed to a reading of the failure. "Before diagnosing" is
# the one thing the instruction cannot buy, because by the time the model is
# choosing tools it is already diagnosing. The review path reached the same
# conclusion for competing PRs (see :func:`competing_open_prs`); this is the
# same move for the failure a task is asked to fix.
#
# It is deliberately a small, fixed set of queries, and it does NOT replace the
# tools: the note it produces names the searches that ran so the model does not
# repeat them, and points at `history_thread` for anything it wants to read.

# Node-ids to derive queries from. A failure group is usually 1-3 tests and they
# are near-duplicates of each other; past that the queries stop being distinct.
MAX_PRIOR_ART_QUERIES = 3
# Hits kept across all queries. The block is a pointer list in a prompt that is
# re-sent every turn, so it is capped hard.
MAX_PRIOR_ART = 6
PRIOR_ART_SEARCH_LIMIT = 4

# `tests/models/<model>/test_modeling_x.py::Class::test_name[param]`
_TEST_MODEL_RE = re.compile(r"tests/models/([^/]+)/")
_PARAM_RE = re.compile(r"\[.*\]$")


@dataclass(frozen=True)
class PriorThread:
    number: int
    kind: str
    title: str
    url: str
    author: str
    trust: str
    age: str
    query: str


@dataclass(frozen=True)
class PriorArtResult:
    """What the lookup did, not just what it found.

    The note tells the model "these searches have already run — spend your own
    `history_search` calls on different terms", so a query listed here that did
    not actually execute steers the model AWAY from a search nobody made. Three
    outcomes, kept apart on purpose:

    * ``ran``     — relore answered. Nothing found means nothing is there.
    * ``failed``  — relore was asked and did not answer (down, slow, 426). Says
      nothing about the history, so the model should still try it.
    * ``skipped`` — never sent, because the hit budget filled first.

    The first version of this collapsed all three into one list, which is the
    failure shape relore's own build plan §13.3 is about: confident, well-formed,
    silently incomplete output.
    """

    threads: list[PriorThread]
    ran: list[str]
    failed: list[str]
    skipped: list[str]

    @property
    def searched_anything(self) -> bool:
        return bool(self.ran or self.failed or self.skipped)


def failure_search_queries(node_ids: Iterable[str]) -> list[str]:
    """Search queries for a failure group's node-ids, best-first, deduped.

    The shape is `<model> <test function>` and it was chosen by measuring, not
    by taste. On the real nemotron failure of task 824a0f5c, against the
    production index:

    * `--test <exact node-id>`      -> 0 hits (the extraction indexes what
      threads mention, and nobody quotes a full node-id)
    * `--error AssertionError`      -> 3 hits, all the same unrelated PR
    * `nemotron test_model_8b_generation` -> #37665 "[tests] fix
      `test_nemotron_8b_generation_sdpa`" — the previous fix for that very test

    which is also what relore's own tool description asks for: two or three
    distinctive terms, never a sentence.
    """
    queries: list[str] = []
    for node_id in node_ids:
        if not node_id or "::" not in node_id:
            continue
        func = _PARAM_RE.sub("", node_id.rsplit("::", 1)[1]).strip()
        if not func:
            continue
        model = _TEST_MODEL_RE.search(node_id)
        query = f"{model.group(1)} {func}" if model else func
        if query not in queries:
            queries.append(query)
        if len(queries) >= MAX_PRIOR_ART_QUERIES:
            break
    return queries


def prior_art(env: ReloreEnv, *, node_ids: Iterable[str]) -> PriorArtResult:
    """Threads this repository already has about the failing tests.

    ``kind="failure"`` throughout: a task is by definition asking about a
    failure, and that is the slice §10's benchmark scores highest on. Never
    raises and never gates — this is context for a task, and a relore that is
    down costs it nothing.
    """
    queries = failure_search_queries(node_ids)
    found: list[PriorThread] = []
    ran: list[str] = []
    failed: list[str] = []
    seen: set[int] = set()

    for index, query in enumerate(queries):
        if len(found) >= MAX_PRIOR_ART:
            # Budget filled by an earlier query: the rest were never sent, and
            # must not be reported as searches that came back empty.
            return PriorArtResult(found, ran, failed, list(queries[index:]))
        payload = _relore_json(
            env,
            [
                "search",
                query,
                "--kind",
                "failure",
                "--limit",
                str(PRIOR_ART_SEARCH_LIMIT),
                "--repo",
                env.repo,
            ],
        )
        if payload is None:
            failed.append(query)
            continue
        ran.append(query)
        for raw in payload.get("hits") or []:
            if not isinstance(raw, dict):
                continue
            number = raw.get("number")
            if not isinstance(number, int) or number in seen:
                continue
            seen.add(number)
            found.append(
                PriorThread(
                    number=number,
                    kind=str(raw.get("type") or "thread"),
                    title=str(raw.get("title") or ""),
                    url=str(raw.get("url") or ""),
                    author=str(raw.get("author") or ""),
                    trust=str(raw.get("trust") or ""),
                    age=str(raw.get("age") or ""),
                    query=query,
                )
            )
            if len(found) >= MAX_PRIOR_ART:
                break
    return PriorArtResult(found, ran, failed, [])


def prior_art_note(result: PriorArtResult) -> str:
    """The task-side note, or ``""`` when no search ran at all.

    Carries thread *metadata* only — number, kind, title, author, age, url — and
    no snippet. That is what keeps this block trusted context like
    :func:`competing_pr_note`: a snippet is text a GitHub user wrote, and relore
    wraps those in an untrusted-content envelope that must be relayed verbatim
    (see :func:`run_relore_tool`). Re-wrapping it here to fit a prompt block is
    exactly what that rule forbids, so the note points at `history_thread`
    instead and lets the model fetch the envelope intact.
    """
    if not result.searched_anything:
        return ""
    lines = [
        "\n\u2500\u2500 PROJECT HISTORY (serge already searched \u2014 trusted) \u2500\u2500"
    ]

    if result.threads:
        lines.append(
            f"relore searched this repository's issue and pull-request history "
            f"for these failing tests before you were asked anything, and "
            f"{_count(len(result.threads), 'earlier thread')} matched:"
        )
        for t in result.threads:
            who = f" by @{t.author}" if t.author else ""
            meta = ", ".join(x for x in (t.trust, t.age) if x)
            lines.append(f"- #{t.number} {t.kind}{who} ({meta}) — {t.title}")
            lines.append(f"  {t.url}")
        lines.append(
            "These are pointers, not evidence: a title is not a decision. Read "
            "one with `history_thread <number>` before you rely on it, and cite "
            "it in `body` if it settles anything."
        )
    elif result.ran:
        lines.append(
            "relore searched this repository's issue and pull-request history "
            "for these failing tests before you were asked anything, and found "
            "nothing."
        )

    if result.ran:
        lines.append(
            f"Already run, do NOT repeat: {_queries(result.ran)} (kind=failure). "
            "Spend your `history_search` calls on different terms — the "
            "exception text, a symbol from the traceback."
        )
    if result.failed:
        # NOT "found nothing": relore did not answer, so the history is unread.
        lines.append(
            f"Could not be reached, so these are UNANSWERED rather than empty — "
            f"worth running yourself: {_queries(result.failed)}."
        )
    if result.skipped:
        lines.append(
            f"Not run (the list above filled first): {_queries(result.skipped)}."
        )
    return "\n".join(lines) + "\n"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _queries(queries: list[str]) -> str:
    return "; ".join(f"`{q}`" for q in queries)
