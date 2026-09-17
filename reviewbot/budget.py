"""The runner pod's wall-clock budget, shared by every step that can wait.

serge puts ``TASK_RUNNER_TIMEOUT`` on the task Job as ``activeDeadlineSeconds``
(``k8s_sandbox.build_task_job_manifest``), so it is not advisory: when it
expires Kubernetes kills the pod wherever it happens to be, and whatever the
runner was holding is gone. Job ``d2c24049`` (2026-09-16) is the whole argument
for this module — it had a finished patch, a title, a body and a clean apply,
entered the normalize step at 00:06:03, and was killed at 00:11:22 with 5m19s of
normalize still to run. The transformers triage issue recorded it as ``⚠️ task
failed``; the two tests stayed unfixed; nothing was published.

So every step that can block for minutes has to ask how much budget is left
rather than assume its own configured timeout fits, and the agent loop has to
stop on its own terms while enough is left to land what it already has. That is
what :func:`remaining`, :func:`clamp` and :func:`exhausted` are for.

**Arming is explicit.** :func:`arm` is called once, from
``task_runner.build_runner_config`` — the single choke point both the task and
the review runner go through, and nothing else does. An unarmed budget reads as
unbounded, which is what the legacy in-process worker
(``TASK_EXECUTION=inprocess``, still the default) needs: there the process is
serge's own web app, up for days, and a budget measured from *its* start would
read as exhausted on every job forever.
"""

from __future__ import annotations

import time
from typing import Optional

# Import time. In a runner pod that is process start, which is what the Job's
# activeDeadlineSeconds is measured from (give or take the image pull, which is
# outside both). See the module docstring for why we do not measure from serge.
PROCESS_START = time.monotonic()

# What to keep back for the unconditional tail: commit the tree, push the
# branch, open the PR, POST the terminal callback. All GitHub API calls, all
# fast — this is slack, not a work budget.
WINDDOWN_SECONDS = 180

_deadline: Optional[float] = None


def arm(runner_timeout: Optional[int], *, start: Optional[float] = None) -> None:
    """Arm the budget for a runner process whose Job dies at ``runner_timeout``
    seconds. A falsy timeout disarms — the caller is unbounded."""
    global _deadline
    if not runner_timeout:
        _deadline = None
        return
    _deadline = (PROCESS_START if start is None else start) + runner_timeout


def disarm() -> None:
    """Drop the budget (unbounded). Used by tests and by any caller that is not
    a runner pod."""
    global _deadline
    _deadline = None


def armed() -> bool:
    return _deadline is not None


def remaining(*, now: Optional[float] = None, reserve: float = 0.0) -> Optional[float]:
    """Seconds left before the pod is killed, minus ``reserve``. ``None`` when
    the budget is not armed, which every caller must read as "unbounded" rather
    than "no time left"."""
    if _deadline is None:
        return None
    return _deadline - (time.monotonic() if now is None else now) - reserve


def clamp(
    configured: int,
    *,
    now: Optional[float] = None,
    reserve: float = WINDDOWN_SECONDS,
) -> int:
    """``configured``, reduced to what the budget can actually cover.

    Returns ``configured`` unchanged when the budget is not armed. Never returns
    less than 0; a caller that gets 0 has no room to run the step at all and
    should skip it rather than start something it cannot finish.
    """
    left = remaining(now=now, reserve=reserve)
    if left is None:
        return configured
    return max(0, int(min(configured, left)))


def tail_reserve(configured: int, normalize_timeout: int) -> int:
    """How much budget the agent loop must leave behind for the tail.

    ``configured`` (``TASK_TAIL_RESERVE``) wins when set. The default derives it
    from the step that dominates the tail — the repo normalizer — plus the
    wind-down, so raising ``TASK_NORMALIZE_TIMEOUT`` cannot silently leave the
    loop running past the point where its own patch can still be normalized and
    pushed.
    """
    if configured > 0:
        return configured
    return max(0, normalize_timeout) + int(WINDDOWN_SECONDS)


def reserve_for(cfg: object) -> int:
    """:func:`tail_reserve` read off a resolved ``Config``. Duck-typed so this
    module keeps no imports — ``reviewer`` cannot import ``tasks`` (cycle), and
    both need the same number."""
    return tail_reserve(
        getattr(cfg, "task_tail_reserve", 0) or 0,
        getattr(cfg, "task_normalize_timeout", 0) or 0,
    )


def exhausted(
    *, now: Optional[float] = None, reserve: float = WINDDOWN_SECONDS
) -> bool:
    """True when less than ``reserve`` of the budget is left. False when the
    budget is not armed."""
    left = remaining(now=now, reserve=reserve)
    return left is not None and left <= 0
