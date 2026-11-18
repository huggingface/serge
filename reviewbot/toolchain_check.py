"""Does the installed toolchain satisfy the target repo's own pins?

The task-runner image freezes the target repo's quality toolchain at build time,
and the normalize gate cannot refresh it: it installs the worktree with
``--no-deps`` because a task pod's egress allowlist has no PyPI. So whenever the
repo's ``main`` moves a pin, every task fails the gate on a checker unrelated to
its patch until the image is rebuilt (serge#126).

The preflight's ``import transformers`` catches a drift only when it breaks the
import. 2026-11-16 did not: transformers#50071 moved ``transformers-mlinter`` to
0.1.6, the image (built 2026-10-30) held 0.1.5, and only one checker died —
``modeling_rules_doc``, on ``Rule TRF001 is marked deprecated but trf001.py
still exists``. The import passed, so every group ran its full agent loop
(~2M input tokens each) before the gate rejected the patch.

This compares the installed distributions against the requirements the
*worktree's* metadata declares, which is exact: run it after the gate's own
editable install (``uv pip install -e . --no-deps``), so the metadata read here
is the checkout's, not the copy baked into the image. Core requirements plus the
requested extras (``quality`` — the toolchain ``make fix-repo`` runs) are
checked; a missing distribution counts as a mismatch.

Usage (exit 0 = consistent, 1 = drift, 2 = cannot tell)::

    python -m reviewbot.toolchain_check --dist transformers --extra quality
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class Mismatch:
    name: str
    required: str  # the requirement string as declared
    installed: Optional[str]  # None = not installed

    def describe(self) -> str:
        have = self.installed if self.installed is not None else "not installed"
        return f"{self.name}: requires {self.required!r}, image has {have}"


def find_mismatches(
    requirements: Iterable[str],
    *,
    extras: Iterable[str] = (),
    version_of: Callable[[str], Optional[str]],
) -> list[Mismatch]:
    """Requirements (core + ``extras``) the installed versions do not satisfy.

    ``requirements`` are PEP 508 strings as ``importlib.metadata.requires``
    returns them. A requirement whose marker excludes this interpreter, or that
    belongs to an extra not asked for, is skipped. Prereleases count as
    satisfying (a dev toolchain legitimately runs them)."""
    from packaging.requirements import Requirement

    wanted_extras = set(extras)
    out: list[Mismatch] = []
    seen: set[str] = set()
    for raw in requirements:
        req = Requirement(raw)
        if req.marker is not None:
            # Core requirement: marker evaluated with no extra. Extra-gated:
            # applies if any requested extra turns it on.
            envs = [{"extra": ""}] + [{"extra": e} for e in sorted(wanted_extras)]
            if not any(req.marker.evaluate(env) for env in envs):
                continue
        key = req.name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        installed = version_of(req.name)
        if installed is None or (
            req.specifier and not req.specifier.contains(installed, prereleases=True)
        ):
            out.append(Mismatch(req.name, raw.split(";")[0].strip(), installed))
    return out


def _installed_version(name: str) -> Optional[str]:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dist", default="transformers", help="distribution whose pins to check"
    )
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        help="extra whose requirements to include",
    )
    args = ap.parse_args(argv)

    try:
        requirements = md.requires(args.dist)
    except md.PackageNotFoundError:
        print(
            f"toolchain check: {args.dist} is not installed; cannot tell",
            file=sys.stderr,
        )
        return 2
    mismatches = find_mismatches(
        requirements or [], extras=args.extra, version_of=_installed_version
    )
    if not mismatches:
        print(f"toolchain check: installed versions satisfy {args.dist}'s pins")
        return 0
    print(
        f"toolchain check: the image does not satisfy {args.dist}'s own pins "
        "(rebuild the task-runner image; the gate installs with --no-deps and "
        "cannot fix this):"
    )
    for m in mismatches:
        print(f"  - {m.describe()}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
