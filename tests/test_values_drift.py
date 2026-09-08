"""The deploy preflight that refuses a values file which drops live settings.

Pinned to the 2026-09-04 incident: serge was upgraded with the chart's own
example values instead of the tracked production ones, which silently unset the
GPU verify loop, the comment-brevity passes and the backups block. helm
reported success and nothing else changed, so the only possible alarm is a
comparison of what the new values still carry against what the release already
had."""

import importlib.util
import json
import pathlib

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "deploy"
    / "scripts"
    / "values_drift.py"
)
_spec = importlib.util.spec_from_file_location("values_drift", _MODULE_PATH)
values_drift = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(values_drift)

leaf_paths = values_drift.leaf_paths
removed_paths = values_drift.removed_paths


# The shape of the incident, trimmed to what matters.
LIVE = {
    "image": {"tag": "sha-f8971d6"},
    "envVars": {
        "PORT": "8080",
        "VERIFY_ON_GPU": "1",
        "VERIFY_REPRODUCE_FIRST": "1",
        "TASK_COMMENT_BREVITY": "1",
    },
    "taskExecution": {"kubernetes": {"timeout": 7200, "enabled": True}},
    "backups": {"enabled": True, "repository": "s3:example/restic/serge"},
}
STALE = {
    "image": {"tag": "sha-a0a75bc"},
    "envVars": {"PORT": "8080"},
    "taskExecution": {"kubernetes": {"timeout": 3600, "enabled": True}},
}


def test_leaf_paths_walks_to_scalars():
    assert sorted(leaf_paths({"a": {"b": 1, "c": {"d": 2}}, "e": 3})) == [
        "a.b",
        "a.c.d",
        "e",
    ]


def test_the_incident_is_reported_in_full():
    assert removed_paths(LIVE, STALE) == [
        "envVars.VERIFY_ON_GPU",
        "envVars.VERIFY_REPRODUCE_FIRST",
        "envVars.TASK_COMMENT_BREVITY",
        "backups.enabled",
        "backups.repository",
    ]


def test_a_changed_value_is_not_a_removal():
    """What a deploy is *for*. Both the image tag and the halved task timeout
    changed in the incident; neither is what the check exists to catch."""
    kept = {**STALE, "envVars": LIVE["envVars"], "backups": LIVE["backups"]}
    assert removed_paths(LIVE, kept) == []


def test_added_settings_are_not_a_removal():
    new = json.loads(json.dumps(LIVE))
    new["envVars"]["VERIFY_MAX_ROUNDS"] = "2"
    assert removed_paths(LIVE, new) == []


def test_a_branch_that_survives_empty_still_lost_its_settings():
    """`backups: {}` keeps the key and none of the configuration."""
    new = {**LIVE, "backups": {}}
    assert removed_paths(LIVE, new) == ["backups.enabled", "backups.repository"]


def test_a_list_is_compared_as_a_whole():
    """helm replaces lists rather than merging them, so a shorter allowlist is
    an ordinary edit — only losing the key itself is drift."""
    live = {"egress": {"allowDomains": ["a", "b", "c"]}}
    assert removed_paths(live, {"egress": {"allowDomains": ["a"]}}) == []
    assert removed_paths(live, {"egress": {"image": "x"}}) == ["egress.allowDomains"]


def test_a_first_install_has_nothing_to_drop():
    assert removed_paths(None, STALE) == []
    assert removed_paths({}, STALE) == []


def test_dropping_everything_reports_every_path_in_live_order():
    """Reported in the order the live values declare them, so the operator reads
    the list grouped the way the file is written rather than alphabetised."""
    assert removed_paths(LIVE, {}) == list(leaf_paths(LIVE))


@pytest.mark.parametrize(
    "new,expected_status",
    [(LIVE, 0), (STALE, 3)],
)
def test_cli_exit_status_is_what_deploy_sh_branches_on(tmp_path, new, expected_status):
    live_file = tmp_path / "live.json"
    new_file = tmp_path / "new.json"
    live_file.write_text(json.dumps(LIVE))
    new_file.write_text(json.dumps(new))
    argv = ["values_drift.py", str(live_file), str(new_file)]
    assert values_drift.main(argv) == expected_status


def test_cli_rejects_bad_usage(tmp_path):
    assert values_drift.main(["values_drift.py"]) == 2


def test_a_live_empty_mapping_that_disappears_is_still_reported():
    """An empty mapping is a leaf, not "no configuration": `backups: {}` live
    means the block was declared, and a values file without it removes it."""
    assert removed_paths({"backups": {}}, {}) == ["backups"]
