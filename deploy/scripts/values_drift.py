#!/usr/bin/env python3
"""Report configuration a new values set would DROP from a live Helm release.

Deploying with a values file that is *almost* right is silent: helm reports
"Upgrade complete", every probe stays green, and the only evidence is a feature
that quietly stopped happening. On 2026-09-04 serge was upgraded with the
chart's own example values instead of the tracked production ones; that unset
``VERIFY_ON_GPU``, ``VERIFY_REPRODUCE_FIRST``, both ``*_COMMENT_BREVITY`` keys
and the whole ``backups`` block. For three nightlies serge opened fix PRs with
no GPU verification at all (and no "not verified" note either, because the
footer is only written when a run exists), and the SQLite backup CronJob
disappeared from the namespace. Nothing alerted.

So: compare the leaf paths of the live release's user-supplied values against
the ones the deploy is about to send, and report every path the new set no
longer has. Changing a value is what a deploy is for; *losing* one is almost
always the wrong file.

Both inputs are JSON, which is what helm already speaks:

    helm get values <release> -n <ns> -o json                       # live
    helm upgrade --install ... --dry-run=client -o json | .config   # new

Exit status: 0 = nothing dropped, 3 = paths dropped (listed on stdout),
2 = bad usage. Deliberately stdlib-only — it runs on a deploy box, not in a
virtualenv.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Iterator

# A list is compared as a whole, not element by element: helm replaces lists
# wholesale rather than merging them, and an intentional shorter list is an
# ordinary edit. What matters here is that the KEY still exists.
_LEAF = (str, int, float, bool, type(None), list)


def leaf_paths(value: Any, prefix: str = "") -> Iterator[str]:
    """Yield the dotted path of every leaf in a values mapping.

    A leaf is a scalar, a list, or an empty mapping — anything that carries a
    setting rather than more structure."""
    if isinstance(value, dict) and value:
        for key in value:
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from leaf_paths(value[key], child)
        return
    if prefix:
        yield prefix


def removed_paths(live: Any, new: Any) -> list[str]:
    """Paths present in ``live`` and absent from ``new``, in file order.

    A path also counts as removed when it survives only as a *branch* — e.g.
    live ``backups.enabled: true`` against new ``backups: {}`` — because the
    setting itself is gone either way."""
    new_paths = set(leaf_paths(new or {}))
    return [p for p in leaf_paths(live or {}) if p not in new_paths]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} LIVE_VALUES_JSON NEW_VALUES_JSON", file=sys.stderr)
        return 2
    with open(argv[1]) as fh:
        live = json.load(fh)
    with open(argv[2]) as fh:
        new = json.load(fh)
    dropped = removed_paths(live, new)
    for path in dropped:
        print(path)
    return 3 if dropped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
