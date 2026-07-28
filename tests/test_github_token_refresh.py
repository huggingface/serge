"""Tests for surviving the two mid-task failures that cost prod finished work.

**Expired token.** Installation tokens last one hour. A ``/tasks`` run longer
than that reached the publish step with a dead token and lost everything to
``401 Bad credentials`` (three prod runs of 65-67 minutes). The client session
replays a 401 with a re-minted token, which the runner fetches from serge over
the existing authenticated callback channel. Confirmed firing in prod at 60m01s.

**Transient gateway error.** A single ``502 Bad Gateway`` on a *read*
(``GET /git/commits/<sha>``) killed a finished, GPU-verified qwen3_omni_moe fix at
its commit step (transformers#47605). Retried with backoff — but only where a
replay cannot duplicate a side effect.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

from reviewbot.github_client import (
    GitHubClient,
    _AuthRetrySession,
    _is_replay_safe,
)
from reviewbot.task_runner import TokenRefresher


def _resp(status: int, payload=None, text="", headers=None):
    r = MagicMock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.json.return_value = payload if payload is not None else {}
    r.text = text
    r.headers = headers if headers is not None else {}
    return r


class AuthRetrySessionTests(unittest.TestCase):
    """The 401-replay contract, asserted at the session level so it holds for
    every GitHubClient method rather than the ones we remembered to plumb."""

    def test_replays_once_with_the_refreshed_token(self):
        session = _AuthRetrySession(lambda: "fresh")
        session.headers["Authorization"] = "Bearer stale"
        with patch.object(
            requests.Session, "request", side_effect=[_resp(401), _resp(201)]
        ) as inner:
            out = session.post("https://api.github.com/repos/o/r/git/blobs", json={})
        self.assertEqual(out.status_code, 201)
        self.assertEqual(inner.call_count, 2)
        self.assertEqual(session.headers["Authorization"], "Bearer fresh")

    def test_replay_carries_the_original_body(self):
        session = _AuthRetrySession(lambda: "fresh")
        session.headers["Authorization"] = "Bearer stale"
        with patch.object(
            requests.Session, "request", side_effect=[_resp(401), _resp(201)]
        ) as inner:
            session.post("https://api.github.com/x", json={"content": "abc"})
        first, second = inner.call_args_list
        self.assertEqual(first.kwargs["json"], {"content": "abc"})
        self.assertEqual(second.kwargs["json"], {"content": "abc"})

    def test_success_is_not_retried(self):
        provider = MagicMock(return_value="fresh")
        session = _AuthRetrySession(provider)
        with patch.object(requests.Session, "request", return_value=_resp(200)):
            session.get("https://api.github.com/x")
        provider.assert_not_called()

    def test_non_401_failure_is_not_retried(self):
        """A 403 (rate limit / missing permission) is not an expiry — surface it
        rather than burning a mint and doubling the request."""
        provider = MagicMock(return_value="fresh")
        session = _AuthRetrySession(provider)
        with patch.object(
            requests.Session, "request", return_value=_resp(403)
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 403)
        self.assertEqual(inner.call_count, 1)
        provider.assert_not_called()

    def test_without_a_provider_the_401_is_returned(self):
        session = _AuthRetrySession(None)
        with patch.object(
            requests.Session, "request", return_value=_resp(401)
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 401)
        self.assertEqual(inner.call_count, 1)

    def test_unchanged_token_is_not_replayed(self):
        """A 401 from a genuinely bad token (revoked install, wrong App) must
        stay a 401 instead of doubling every request forever."""
        session = _AuthRetrySession(lambda: "same")
        session.headers["Authorization"] = "Bearer same"
        with patch.object(
            requests.Session, "request", return_value=_resp(401)
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 401)
        self.assertEqual(inner.call_count, 1)

    def test_empty_token_is_not_replayed(self):
        session = _AuthRetrySession(lambda: "")
        session.headers["Authorization"] = "Bearer stale"
        with patch.object(
            requests.Session, "request", return_value=_resp(401)
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 401)
        self.assertEqual(inner.call_count, 1)

    def test_a_failing_provider_surfaces_the_original_401(self):
        def boom() -> str:
            raise RuntimeError("serge unreachable")

        session = _AuthRetrySession(boom)
        with patch.object(
            requests.Session, "request", return_value=_resp(401)
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 401)
        self.assertEqual(inner.call_count, 1)

    def test_retries_only_once(self):
        """Two consecutive 401s end after one replay — no unbounded loop."""
        session = _AuthRetrySession(lambda: "fresh")
        session.headers["Authorization"] = "Bearer stale"
        with patch.object(
            requests.Session, "request", side_effect=[_resp(401), _resp(401)]
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 401)
        self.assertEqual(inner.call_count, 2)


class GitHubClientRefreshTests(unittest.TestCase):
    """The regression this whole change exists for: the exact call that failed
    in prod (``create_blob`` on the first write of the publish step) now
    survives an expired token."""

    def test_create_blob_survives_an_expired_token(self):
        gh = GitHubClient("stale", token_provider=lambda: "fresh")
        with patch.object(
            requests.Session,
            "request",
            side_effect=[_resp(401, text="Bad credentials"), _resp(201, {"sha": "b1"})],
        ):
            sha = gh.create_blob("huggingface", "transformers", b"print()")
        self.assertEqual(sha, "b1")
        self.assertEqual(gh.session.headers["Authorization"], "Bearer fresh")

    def test_create_blob_still_raises_when_refresh_cannot_help(self):
        gh = GitHubClient("stale", token_provider=lambda: "stale")
        with (
            patch.object(
                requests.Session,
                "request",
                return_value=_resp(401, text="Bad credentials"),
            ),
            self.assertRaises(requests.HTTPError) as ctx,
        ):
            gh.create_blob("huggingface", "transformers", b"print()")
        self.assertIn("401 creating blob", str(ctx.exception))

    def test_no_provider_keeps_the_previous_behaviour(self):
        gh = GitHubClient("tok")
        self.assertIsInstance(gh.session, _AuthRetrySession)
        with patch.object(
            requests.Session, "request", return_value=_resp(200)
        ) as inner:
            gh.get_pr("o", "r", 1)
        self.assertEqual(inner.call_count, 1)


class ReplaySafetyTests(unittest.TestCase):
    """Which requests may be replayed. A 5xx is ambiguous — the write may have
    landed and only the response got lost — so writes are excluded unless
    replaying them is provably a no-op."""

    def test_reads_are_safe(self):
        self.assertTrue(_is_replay_safe("GET", "https://api.github.com/repos/o/r"))
        self.assertTrue(_is_replay_safe("get", "https://api.github.com/repos/o/r"))
        self.assertTrue(_is_replay_safe("HEAD", "https://api.github.com/repos/o/r"))

    def test_content_addressed_writes_are_safe(self):
        # Same bytes -> same SHA, so a replay creates nothing new.
        self.assertTrue(
            _is_replay_safe("POST", "https://api.github.com/repos/o/r/git/blobs")
        )
        self.assertTrue(
            _is_replay_safe("POST", "https://api.github.com/repos/o/r/git/trees")
        )

    def test_side_effecting_writes_are_not_safe(self):
        for method, url in (
            # A replay would 422 ("already exists") or move a branch twice.
            ("POST", "https://api.github.com/repos/o/r/git/refs"),
            ("PATCH", "https://api.github.com/repos/o/r/git/refs/heads/x"),
            # A replay would create a second commit object.
            ("POST", "https://api.github.com/repos/o/r/git/commits"),
            # A replay would double-post to the PR.
            ("POST", "https://api.github.com/repos/o/r/issues/1/comments"),
            ("POST", "https://api.github.com/repos/o/r/pulls/1/reviews"),
            ("POST", "https://api.github.com/repos/o/r/pulls"),
        ):
            with self.subTest(url=url):
                self.assertFalse(_is_replay_safe(method, url))


class TransientRetryTests(unittest.TestCase):
    def setUp(self):
        # Never actually sleep in tests.
        self._sleep = patch("reviewbot.github_client.time.sleep")
        self.sleep = self._sleep.start()
        self.addCleanup(self._sleep.stop)

    def test_502_on_a_read_is_retried(self):
        """The prod failure verbatim: GET /git/commits/<sha> -> 502."""
        session = _AuthRetrySession()
        url = (
            "https://api.github.com/repos/huggingface/transformers/git/commits/dff4572"
        )
        with patch.object(
            requests.Session, "request", side_effect=[_resp(502), _resp(200)]
        ) as inner:
            out = session.get(url)
        self.assertEqual(out.status_code, 200)
        self.assertEqual(inner.call_count, 2)

    def test_retries_are_bounded(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session, "request", return_value=_resp(503)
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 503)
        self.assertEqual(inner.call_count, 3)  # _MAX_ATTEMPTS
        self.assertEqual(self.sleep.call_count, 2)

    def test_backoff_grows(self):
        session = _AuthRetrySession()
        with patch.object(requests.Session, "request", return_value=_resp(500)):
            session.get("https://api.github.com/x")
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1.0, 3.0])

    def test_retry_after_header_is_honoured(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session,
            "request",
            side_effect=[_resp(503, headers={"Retry-After": "7"}), _resp(200)],
        ):
            session.get("https://api.github.com/x")
        self.sleep.assert_called_once_with(7.0)

    def test_absurd_retry_after_is_capped(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session,
            "request",
            side_effect=[_resp(503, headers={"Retry-After": "99999"}), _resp(200)],
        ):
            session.get("https://api.github.com/x")
        self.sleep.assert_called_once_with(30.0)

    def test_garbage_retry_after_falls_back_to_backoff(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session,
            "request",
            side_effect=[_resp(503, headers={"Retry-After": "soon"}), _resp(200)],
        ):
            session.get("https://api.github.com/x")
        self.sleep.assert_called_once_with(1.0)

    def test_502_on_a_side_effecting_write_is_not_retried(self):
        """create_ref must not be replayed — the ref may already exist."""
        session = _AuthRetrySession()
        with patch.object(
            requests.Session, "request", return_value=_resp(502)
        ) as inner:
            out = session.post("https://api.github.com/repos/o/r/git/refs", json={})
        self.assertEqual(out.status_code, 502)
        self.assertEqual(inner.call_count, 1)
        self.sleep.assert_not_called()

    def test_502_on_a_blob_upload_is_retried(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session, "request", side_effect=[_resp(502), _resp(201)]
        ) as inner:
            out = session.post("https://api.github.com/repos/o/r/git/blobs", json={})
        self.assertEqual(out.status_code, 201)
        self.assertEqual(inner.call_count, 2)

    def test_connection_error_on_a_read_is_retried(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session,
            "request",
            side_effect=[requests.ConnectionError("reset"), _resp(200)],
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 200)
        self.assertEqual(inner.call_count, 2)

    def test_timeout_on_a_read_is_retried(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session,
            "request",
            side_effect=[requests.Timeout("slow"), _resp(200)],
        ):
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 200)

    def test_persistent_connection_error_raises(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session, "request", side_effect=requests.ConnectionError("reset")
        ) as inner:
            with self.assertRaises(requests.ConnectionError):
                session.get("https://api.github.com/x")
        self.assertEqual(inner.call_count, 3)

    def test_connection_error_on_a_write_is_not_retried(self):
        session = _AuthRetrySession()
        with patch.object(
            requests.Session, "request", side_effect=requests.ConnectionError("reset")
        ) as inner:
            with self.assertRaises(requests.ConnectionError):
                session.post("https://api.github.com/repos/o/r/git/refs", json={})
        self.assertEqual(inner.call_count, 1)

    def test_404_is_not_retried(self):
        """Only 5xx is transient; a 404 is an answer."""
        session = _AuthRetrySession()
        with patch.object(
            requests.Session, "request", return_value=_resp(404)
        ) as inner:
            session.get("https://api.github.com/x")
        self.assertEqual(inner.call_count, 1)

    def test_expiry_and_a_gateway_blip_in_the_same_request(self):
        """Both repairs compose: re-mint on the 401, then ride out the 502."""
        session = _AuthRetrySession(lambda: "fresh")
        session.headers["Authorization"] = "Bearer stale"
        with patch.object(
            requests.Session,
            "request",
            side_effect=[_resp(401), _resp(502), _resp(200)],
        ) as inner:
            out = session.get("https://api.github.com/x")
        self.assertEqual(out.status_code, 200)
        self.assertEqual(inner.call_count, 3)
        self.assertEqual(session.headers["Authorization"], "Bearer fresh")

    def test_get_commit_tree_sha_survives_a_gateway_blip(self):
        """End to end through the client method that actually failed in prod."""
        gh = GitHubClient("tok")
        with patch.object(
            requests.Session,
            "request",
            side_effect=[_resp(502), _resp(200, {"tree": {"sha": "t1"}})],
        ):
            self.assertEqual(
                gh.get_commit_tree_sha("huggingface", "transformers", "dff4572"), "t1"
            )


class TokenRefresherTests(unittest.TestCase):
    def test_posts_to_serge_with_the_callback_token(self):
        refresher = TokenRefresher(
            "https://serge/internal/tasks/j1/github-token", "cb", "old"
        )
        refresher._session = MagicMock()
        refresher._session.post.return_value = _resp(200, {"token": "new"})
        self.assertEqual(refresher(), "new")
        self.assertEqual(refresher.current, "new")
        args, kwargs = refresher._session.post.call_args
        self.assertEqual(args[0], "https://serge/internal/tasks/j1/github-token")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer cb")

    def test_no_url_means_disabled_and_returns_the_initial_token(self):
        refresher = TokenRefresher(None, "cb", "old")
        self.assertFalse(refresher.enabled)
        self.assertEqual(refresher(), "old")

    def test_network_failure_returns_the_current_token(self):
        """Returning the *unchanged* token is what makes the session skip its
        replay and surface the real 401 — the failure mode must stay quiet."""
        refresher = TokenRefresher("https://serge/t", "cb", "old")
        refresher._session = MagicMock()
        refresher._session.post.side_effect = requests.ConnectionError("boom")
        self.assertEqual(refresher(), "old")
        self.assertEqual(refresher.current, "old")

    def test_http_error_returns_the_current_token(self):
        refresher = TokenRefresher("https://serge/t", "cb", "old")
        refresher._session = MagicMock()
        bad = _resp(502)
        bad.raise_for_status.side_effect = requests.HTTPError("502")
        refresher._session.post.return_value = bad
        self.assertEqual(refresher(), "old")

    def test_empty_token_in_the_response_keeps_the_current_one(self):
        refresher = TokenRefresher("https://serge/t", "cb", "old")
        refresher._session = MagicMock()
        refresher._session.post.return_value = _resp(200, {})
        self.assertEqual(refresher(), "old")

    def test_second_refresh_replaces_the_first(self):
        refresher = TokenRefresher("https://serge/t", "cb", "t0")
        refresher._session = MagicMock()
        refresher._session.post.side_effect = [
            _resp(200, {"token": "t1"}),
            _resp(200, {"token": "t2"}),
        ]
        self.assertEqual(refresher(), "t1")
        self.assertEqual(refresher(), "t2")
        self.assertEqual(refresher.current, "t2")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
