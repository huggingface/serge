"""Keep pre-existing repo drift out of serge's commit.

serge applies the model's patch and then runs the repo normalizer with its fixers
on, so files the patch *requires* to be regenerated ride along in the commit —
most consequentially transformers' modular coupling, where editing a
``modular_X.py`` must regenerate ``modeling_X.py``.

The trouble is that the normalizer fixes the whole worktree, not just the patch.
When ``main`` is not itself normalizer-clean, every unrelated stale file gets
regenerated too and lands in the PR. Prod task 433e8274 patched exactly one file,
``tests/models/glm_ocr/test_modeling_glm_ocr.py``, and committed **32**: the other
31 were regenerated ``modeling_*``/``processing_*`` files for bamba, colpali,
zamba2, modernbert and 27 more models that the fix never touched. No maintainer
can review that, and it cost 12 minutes of normalizer time per attempt.

So: validate against the whole tree (unchanged — the normalizer must still pass
repo-wide), but commit only what the patch is responsible for.

"Responsible for" is deliberately generous, because dropping a genuinely needed
generated file is the worse failure (serge #58 shipped a PR with a
``modular_*.py`` and no regenerated ``modeling_*.py``, which is unmergeable). A
normalizer-changed file is kept when it is
:func:`related to <_is_related>` a patched path — same directory, or the same
model-directory name, which is how transformers couples
``{src/transformers,tests}/models/<model>/``. Anything else is drift and is
dropped, loudly.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re

_GIT_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)", re.MULTILINE)
_PLUS_HEADER = re.compile(r"^\+\+\+ (?:b/)?(?P<path>\S+)", re.MULTILINE)


def patch_paths(patch: str) -> set[str]:
    """Every path the unified diff ``patch`` claims to touch.

    Reads ``diff --git`` headers, falling back to ``+++`` headers for diffs
    written without the git prefix. Both sides of a rename are returned — a
    rename makes the old path the patch's business too."""
    paths: set[str] = set()
    for match in _GIT_HEADER.finditer(patch or ""):
        paths.add(match.group("a"))
        paths.add(match.group("b"))
    if not paths:
        for match in _PLUS_HEADER.finditer(patch or ""):
            path = match.group("path")
            if path != "/dev/null":
                paths.add(path)
    return {p for p in paths if p and p != "/dev/null"}


def _scope_keys(paths: set[str]) -> tuple[set[str], set[str]]:
    """``(directories, directory names)`` the patch reaches into.

    The directory *name* is what carries the coupling in transformers: a patch to
    ``tests/models/glm_ocr/`` legitimately regenerates
    ``src/transformers/models/glm_ocr/`` — a different directory, same leaf.

    A patch to a **root-level** file scopes to the repo root, so root-level
    normalizer output rides along with it. That is intentional and slightly
    generous: it is how a generated file next to its source is kept (serge #58),
    and real generated files in the repos serge targets live in a model
    directory, never at the root — so it costs nothing against the drift this
    module exists to drop."""
    dirs = {posixpath.dirname(p) for p in paths}
    return dirs, {posixpath.basename(d) for d in dirs if d}


def _is_related(path: str, dirs: set[str], dir_names: set[str]) -> bool:
    directory = posixpath.dirname(path)
    if directory in dirs:
        return True
    if posixpath.basename(directory) in dir_names:
        return True
    # A patched file's subtree (e.g. a patch to a package's __init__ that makes
    # the normalizer rewrite files beneath it).
    return any(d and directory.startswith(f"{d}/") for d in dirs)


def scope_paths(
    changed: list[str],
    patch: str,
    *,
    always_include: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split ``changed`` into ``(keep, dropped)`` relative to ``patch``.

    Paths the patch names itself are always kept, as is anything matching an
    ``always_include`` glob (the operator's escape hatch for a normalizer that
    legitimately rewrites cross-cutting generated files, e.g.
    ``src/transformers/utils/dummy_*.py``).

    An empty or unparseable patch scopes to nothing, so everything is kept —
    with no patch we cannot tell drift from the change, and committing too much
    beats committing nothing."""
    touched = patch_paths(patch)
    if not touched:
        return list(changed), []
    dirs, dir_names = _scope_keys(touched)
    globs = always_include or []
    keep: list[str] = []
    dropped: list[str] = []
    for path in changed:
        if (
            path in touched
            or _is_related(path, dirs, dir_names)
            or any(fnmatch.fnmatch(path, pattern) for pattern in globs)
        ):
            keep.append(path)
        else:
            dropped.append(path)
    # Never scope the commit down to nothing: if the rule somehow rejected every
    # change, the rule is wrong about this repo and dropping the fix is worse.
    if not keep:
        return list(changed), []
    return keep, dropped


def describe_dropped(dropped: list[str], *, limit: int = 8) -> str:
    """Operator-facing one-liner naming what was left out, and how much."""
    shown = ", ".join(sorted(dropped)[:limit])
    more = len(dropped) - limit
    return shown + (f", +{more} more" if more > 0 else "")
