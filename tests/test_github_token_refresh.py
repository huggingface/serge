"""Tests for surviving GitHub installation-token expiry mid-task.

Installation tokens last one hour. A ``/tasks`` run longer than that reached the
publish step with a dead token and lost all its work to ``401 Bad credentials``
(observed on three prod runs of 65-67 minutes). Two pieces cover it: the client
session replays a 401 once with a re-minted token, and the runner asks serge for
that token over the existing authenticated callback channel.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

from reviewbot.github_client import GitHubClient, _AuthRetrySession
from reviewbot.task_runner import TokenRefresher


def _resp(status: int, payload=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.json.return_value = payload if payload is not None else {}
    r.text = text
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
