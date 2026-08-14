import types
import unittest

from reviewbot import pr_links, tasks

GEMMA = "tests/models/gemma3/test_modeling_gemma3.py"
NODE = f"{GEMMA}::Gemma3IntegrationTest::test_dynamic_sliding_window_is_default"
OTHER = "tests/models/foo/test_modeling_foo.py::FooTest::test_a"
DASH = "https://transformers-ci.lor-e.huggingface.cool/d/pytest-test/test"
# What a dispatcher sends: finished links, keyed by node-id. serge renders them
# without knowing what they point at.
LINKS = {NODE: [{"label": "Dashboard", "url": f"{DASH}?var-test_nodeid=x"}]}

# A failure report as transformers-ci's triage renders it: bullet, then the CI
# traceback in a 2-space-indented fence.
CONTEXT = (
    "## Serge candidate failure group 1/1: gemma3 · output_mismatch\n"
    "\n"
    "Failing tests (with the actual-vs-expected detail from the CI trace):\n"
    f"- `{OTHER}` [single-gpu] (output_mismatch, seen 5/7)\n"
    "  - AssertionError: ordinary mismatch\n"
    f"- `{NODE}` [single-gpu] (output_mismatch, seen 5/7)\n"
    "  ```\n"
    "  self = <tests.models.gemma3.test_modeling_gemma3.Gemma3IntegrationTest>\n"
    "  E   AssertionError: 'DynamicSlidingWindowLayer' unexpectedly found in "
    "'DynamicCache(...)'\n"
    "  ```\n"
)


class FailureTracebackTests(unittest.TestCase):
    def test_traceback_is_keyed_by_node_id(self):
        tracebacks = pr_links.failure_tracebacks(CONTEXT)
        self.assertEqual(list(tracebacks), [NODE])
        self.assertIn("DynamicSlidingWindowLayer", tracebacks[NODE])
        # The report's 2-space indent is stripped so the fence renders as code.
        self.assertTrue(tracebacks[NODE].startswith("self = <tests.models"))

    def test_no_fence_means_no_entry(self):
        self.assertEqual(pr_links.failure_tracebacks(f"- `{OTHER}` [single-gpu]"), {})

    def test_trim_keeps_the_tail(self):
        trimmed = pr_links.trim_traceback("x" * 50 + "the assertion", max_chars=20)
        self.assertTrue(trimmed.endswith("the assertion"))
        self.assertIn("truncated", trimmed)

    def test_error_section_only_covers_requested_tests(self):
        section = pr_links.error_section(CONTEXT, [NODE])
        self.assertIn("<details>", section)
        self.assertIn("DynamicSlidingWindowLayer", section)
        self.assertEqual(pr_links.error_section(CONTEXT, [OTHER]), "")


class TestLinkTests(unittest.TestCase):
    def test_section_renders_only_the_requested_tests(self):
        links = {**LINKS, OTHER: [{"label": "Dashboard", "url": f"{DASH}?other"}]}
        section = pr_links.test_links_section(links, [NODE])
        self.assertIn("test_dynamic_sliding_window_is_default — Dashboard", section)
        self.assertIn("var-test_nodeid=x", section)
        # The other candidate group's link is not this PR's business.
        self.assertNotIn("?other", section)

    def test_no_links_no_section(self):
        self.assertEqual(pr_links.test_links_section({}, [NODE]), "")
        self.assertEqual(pr_links.test_links_section(LINKS, [OTHER]), "")

    def test_sanitize_keeps_well_formed_links(self):
        clean = pr_links.sanitize_test_links(LINKS)
        self.assertEqual(clean, LINKS)

    def test_sanitize_drops_non_http_urls(self):
        for url in ("javascript:alert(1)", "/d/pytest-test/test", "ftp://x/y"):
            self.assertEqual(
                pr_links.sanitize_test_links({NODE: [{"url": url}]}), {}, url
            )

    def test_sanitize_drops_urls_that_break_the_markdown_link(self):
        bad = {NODE: [{"url": f"{DASH}?a=b) [x](javascript:0"}]}
        self.assertEqual(pr_links.sanitize_test_links(bad), {})

    def test_sanitize_flattens_the_label_and_defaults_it(self):
        raw = {NODE: [{"label": "two\nlines", "url": DASH}, {"url": DASH}]}
        labels = [link["label"] for link in pr_links.sanitize_test_links(raw)[NODE]]
        self.assertEqual(labels, ["two lines", "Dashboard"])

    def test_sanitize_tolerates_junk(self):
        for raw in (None, [], "x", {NODE: "x"}, {"": [{"url": DASH}]}, {NODE: [1]}):
            self.assertEqual(pr_links.sanitize_test_links(raw), {}, repr(raw))

    def test_sanitize_caps_links_per_test(self):
        many = {NODE: [{"url": f"{DASH}?i={i}"} for i in range(20)]}
        clean = pr_links.sanitize_test_links(many)
        self.assertEqual(len(clean[NODE]), pr_links.MAX_LINKS_PER_TEST)


class _FakeSearchGH:
    def __init__(self, items=None, exc=None):
        self.items = items or []
        self.exc = exc
        self.queries: list[str] = []

    def search_issues(self, query, **kwargs):
        self.queries.append(query)
        if self.exc is not None:
            raise self.exc
        return self.items


def _item(number, **over):
    item = {
        "number": number,
        "title": f"issue {number}",
        "html_url": f"https://github.com/huggingface/transformers/issues/{number}",
        "state": "open",
        "updated_at": "2026-08-01T10:00:00Z",
        "user": {"login": "someone"},
        "body": "",
    }
    item.update(over)
    return item


class RelatedIssueTests(unittest.TestCase):
    def test_query_matches_the_test_function(self):
        query = pr_links.related_search_query(
            "huggingface", "transformers", [NODE], "gemma3"
        )
        self.assertEqual(
            query,
            'repo:huggingface/transformers "test_dynamic_sliding_window_is_default"',
        )

    def test_parametrized_ids_collapse_to_one_term(self):
        query = pr_links.related_search_query(
            "o", "r", [f"{NODE}[bf16]", f"{NODE}[fp32]"], ""
        )
        self.assertEqual(query.count("OR"), 0)

    def test_falls_back_to_the_model_name(self):
        self.assertEqual(
            pr_links.related_search_query("o", "r", [], "gemma3"),
            'repo:o/r "gemma3" in:title',
        )
        self.assertEqual(pr_links.related_search_query("o", "r", [], ""), "")

    def test_serge_own_prs_are_dropped(self):
        gh = _FakeSearchGH(
            [
                _item(1, user={"login": "sergereview[bot]"}),
                _item(2, body="<!-- serge-task:integration-failure-triage -->"),
                _item(3),
            ]
        )
        related = pr_links.find_related_issues(
            gh, "o", "r", node_ids=[NODE], model="gemma3"
        )
        self.assertEqual([r["number"] for r in related], [3])

    def test_search_failure_is_not_fatal(self):
        gh = _FakeSearchGH(exc=RuntimeError("rate limited"))
        self.assertEqual(
            pr_links.find_related_issues(gh, "o", "r", node_ids=[NODE]), []
        )

    def test_results_are_capped(self):
        gh = _FakeSearchGH([_item(n) for n in range(10)])
        related = pr_links.find_related_issues(gh, "o", "r", node_ids=[NODE], limit=3)
        self.assertEqual(len(related), 3)

    def test_section_flags_the_match_as_unverified(self):
        gh = _FakeSearchGH([_item(42, pull_request={"url": "..."}, state="closed")])
        section = pr_links.related_section(
            pr_links.find_related_issues(gh, "o", "r", node_ids=[NODE]), [NODE]
        )
        self.assertIn("Possibly related", section)
        self.assertIn("#42", section)
        self.assertIn("PR, closed, updated 2026-08-01", section)
        self.assertIn("not verified", section)

    def test_no_hits_no_section(self):
        self.assertEqual(pr_links.related_section([], [NODE]), "")


class BodyDecorationTests(unittest.TestCase):
    def setUp(self):
        self.req = types.SimpleNamespace(
            context=CONTEXT,
            owner="huggingface",
            repo="transformers",
            test_links=LINKS,
        )
        self.plan = tasks.TaskPlan(
            title="Keep the explicit cache_implementation",
            body="Preserve cache_implementation when it is explicit.",
            patch="DynamicSlidingWindowLayer",
        )
        self.cfg = types.SimpleNamespace(is_staging=False)

    def test_failure_header_carries_error_and_links(self):
        context = tasks._selected_failure_context(self.req, self.plan)
        self.assertIn("Original CI failure", context)
        # Picked over the other bullet because the patch echoes an identifier
        # from this one's traceback — the bullet itself shares nothing.
        self.assertNotIn("FooTest", context)
        self.assertIn("DynamicSlidingWindowLayer' unexpectedly found", context)
        self.assertIn("/d/pytest-test/test?", context)

    def test_a_task_without_links_keeps_the_error(self):
        req = types.SimpleNamespace(
            context=CONTEXT, owner="huggingface", repo="transformers", test_links={}
        )
        context = tasks._selected_failure_context(req, self.plan)
        self.assertIn("CI traceback", context)
        self.assertNotIn("Where to watch it", context)

    def test_related_section_sits_above_the_disclaimer(self):
        gh = _FakeSearchGH([_item(47281)])
        body = tasks._decorate_body(self.cfg, self.plan, self.req, gh=gh)
        self.assertIn("Possibly related", body)
        self.assertLess(
            body.index("Possibly related"),
            body.index("This change was produced automatically"),
        )
        self.assertEqual(
            gh.queries,
            ['repo:huggingface/transformers "test_dynamic_sliding_window_is_default"'],
        )

    def test_no_github_client_no_search(self):
        body = tasks._decorate_body(self.cfg, self.plan, self.req)
        self.assertNotIn("Possibly related", body)
        # The error + dashboard link do not depend on the search.
        self.assertIn("CI traceback", body)


if __name__ == "__main__":
    unittest.main()
