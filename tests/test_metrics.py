"""Tests for the Prometheus exposition of finished jobs.

This endpoint is the only thing that carries a session's counters past the
25-job store retention, so its failure mode matters: a scrape that raises, or a
body Prometheus refuses to parse, silently loses the history rather than
reporting an error anyone sees.
"""

import unittest

from reviewbot.metrics import render_job_metrics

try:
    from prometheus_client.parser import text_string_to_metric_families
except ModuleNotFoundError:  # pragma: no cover
    text_string_to_metric_families = None


def _row(job_id: str, **overrides) -> dict:
    row = {
        "id": job_id,
        "target_owner": "huggingface",
        "target_repo": "transformers",
        "target_number": 48322,
        "status": "published",
        "kind": "task",
        "llm_model": "kimi-k2",
        "created_at": 1_756_000_000.0,
        "updated_at": 1_756_003_600.0,
        "pr_number": 48322,
        "verify_verdict": "fixed",
        "session": {
            "turns": 46,
            "tool_calls": 48,
            "prompt_tokens": 2_111_885,
            "completion_tokens": 9_000,
            "seconds": 1234.5,
            "stop_reason": "input_token_cap",
            "repeats": 2,
            "distinct_paths": 12,
            "path_revisits": 14,
            "validation_retries": 1,
            "truncation_retries": 0,
            "rounds": 1,
        },
    }
    row.update(overrides)
    return row


def _samples(body: str, name: str) -> list[str]:
    return [
        line
        for line in body.splitlines()
        if line.startswith(name + "{") or line == name or line.startswith(name + " ")
    ]


class ExpositionShapeTests(unittest.TestCase):
    def test_every_metric_declares_help_and_type(self) -> None:
        body = render_job_metrics([_row("a")], retention=25)
        named = {
            line.split()[0]
            for line in body.splitlines()
            if line and not line.startswith("#")
        }
        metric_names = {name.split("{", 1)[0] for name in named}
        for metric in metric_names:
            self.assertIn(f"# HELP {metric} ", body, metric)
            self.assertIn(f"# TYPE {metric} ", body, metric)

    def test_identity_lives_on_the_info_series_only(self) -> None:
        """Labels on the value series would fork a job's numbers into a new
        series every time one of them changed."""
        body = render_job_metrics([_row("a")])
        info = _samples(body, "serge_job_info")[0]
        self.assertIn('stop_reason="input_token_cap"', info)
        self.assertIn('repo="huggingface/transformers"', info)
        self.assertIn('pr="48322"', info)
        self.assertIn('verify_verdict="fixed"', info)
        self.assertTrue(info.endswith(" 1"))
        turns = _samples(body, "serge_job_turns")[0]
        self.assertEqual(turns, 'serge_job_turns{job_id="a"} 46')

    def test_counters_are_exported(self) -> None:
        body = render_job_metrics([_row("a")])
        self.assertIn('serge_job_input_tokens{job_id="a"} 2111885', body)
        self.assertIn('serge_job_path_revisits{job_id="a"} 14', body)
        self.assertIn('serge_job_distinct_paths{job_id="a"} 12', body)
        self.assertIn('serge_job_rounds{job_id="a"} 1', body)
        self.assertIn('serge_job_llm_seconds{job_id="a"} 1234.5', body)

    def test_finished_timestamp_orders_the_window(self) -> None:
        body = render_job_metrics([_row("a")])
        self.assertIn(
            'serge_job_finished_timestamp_seconds{job_id="a"} 1756003600', body
        )

    def test_retention_is_exported_so_a_gap_is_readable(self) -> None:
        body = render_job_metrics([], retention=25)
        self.assertIn("serge_job_retention 25", body)

    def test_build_info(self) -> None:
        body = render_job_metrics([], version="1.2.3", commit="abc1234")
        self.assertIn('serge_build_info{version="1.2.3",commit="abc1234"} 1', body)


class RobustnessTests(unittest.TestCase):
    """A scrape reads whatever the store happens to hold, including rows written
    by an older build."""

    def test_no_jobs_still_renders_a_valid_body(self) -> None:
        body = render_job_metrics([])
        self.assertTrue(body.endswith("\n"))
        self.assertIn("# TYPE serge_job_info gauge", body)
        self.assertEqual(_samples(body, "serge_job_turns"), [])

    def test_a_session_missing_a_counter_skips_that_sample(self) -> None:
        body = render_job_metrics([_row("a", session={"turns": 3})])
        self.assertIn('serge_job_turns{job_id="a"} 3', body)
        self.assertEqual(_samples(body, "serge_job_path_revisits"), [])
        # …but the job is still described.
        self.assertEqual(len(_samples(body, "serge_job_info")), 1)

    def test_a_junk_counter_is_dropped_not_rendered(self) -> None:
        body = render_job_metrics(
            [_row("a", session={"turns": "many", "tool_calls": None, "repeats": True})]
        )
        self.assertEqual(_samples(body, "serge_job_turns"), [])
        self.assertEqual(_samples(body, "serge_job_tool_calls"), [])
        # bool is an int in Python but not a meaningful count here.
        self.assertEqual(_samples(body, "serge_job_repeat_calls"), [])

    def test_missing_identity_renders_empty_labels_not_none(self) -> None:
        body = render_job_metrics(
            [_row("a", llm_model=None, verify_verdict=None, pr_number=None)]
        )
        info = _samples(body, "serge_job_info")[0]
        self.assertIn('model="none"', info)
        self.assertIn('verify_verdict="none"', info)
        self.assertIn('pr=""', info)
        self.assertNotIn("None", info)

    def test_label_values_are_escaped(self) -> None:
        body = render_job_metrics([_row("a", llm_model='we"ird\\model\nname')])
        info = _samples(body, "serge_job_info")[0]
        self.assertIn(r'model="we\"ird\\model\nname"', info)
        # One line per series, whatever the payload contained.
        self.assertEqual(len(body.strip().splitlines()), len(body.strip().split("\n")))
        self.assertEqual(len(_samples(body, "serge_job_info")), 1)

    def test_unknown_stop_reason_is_labelled_not_dropped(self) -> None:
        body = render_job_metrics([_row("a", session={"turns": 1})])
        self.assertIn('stop_reason="unknown"', _samples(body, "serge_job_info")[0])


class PrometheusParserTests(unittest.TestCase):
    """Check the body against Prometheus' own parser.

    Everything else here asserts on substrings, which cannot catch a body the
    scraper rejects outright — and a rejected scrape is invisible from this side.
    """

    def setUp(self) -> None:
        if text_string_to_metric_families is None:
            self.skipTest("prometheus_client not installed")

    def _parse(self, body: str):
        return list(text_string_to_metric_families(body))

    def test_a_populated_export_parses(self) -> None:
        rows = [_row(f"job{i}") for i in range(7)]
        families = self._parse(
            render_job_metrics(rows, retention=25, version="0.1.0", commit="abc")
        )
        by_name = {f.name: f for f in families}
        self.assertEqual(len(by_name["serge_job_info"].samples), 7)
        self.assertEqual(len(by_name["serge_job_turns"].samples), 7)
        for family in families:
            self.assertEqual(family.type, "gauge", family.name)

    def test_an_empty_export_parses(self) -> None:
        self._parse(render_job_metrics([], retention=25))

    def test_hostile_label_values_parse_back_unchanged(self) -> None:
        model = 'we"ird\\model\nname'
        families = self._parse(render_job_metrics([_row("a", llm_model=model)]))
        info = {f.name: f for f in families}["serge_job_info"]
        self.assertEqual(info.samples[0].labels["model"], model)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
