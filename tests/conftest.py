import pytest

from reviewbot import budget


@pytest.fixture(autouse=True)
def _unarmed_budget():
    """Every test starts with the runner budget disarmed.

    :mod:`reviewbot.budget` keeps one module-level deadline, armed once per
    runner process. A test that arms it and does not clean up would make every
    later test in the session look like a pod with no time left — the agent loop
    would bail on iteration 1 and the normalizer would be skipped — so the reset
    lives here rather than in each test that touches it.
    """
    budget.disarm()
    yield
    budget.disarm()
