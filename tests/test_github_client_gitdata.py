"""Tests for the Git Data + Pulls write methods on GitHubClient. The HTTP
session is mocked — we assert on the URLs/payloads serge sends and how it
parses the responses."""

import unittest
from unittest.mock import MagicMock

from reviewbot.github_client import (
    SERGE_GIT_EMAIL,
    SERGE_GIT_NAME,
    GitHubClient,
)


def _resp(*, ok=True, status=200, payload=None):
    r = MagicMock()
    r.ok = ok
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = "" if ok else "boom"
    return r


class GitDataMethodsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gh = GitHubClient("tok")
        self.gh.session = MagicMock()

    def test_get_ref_sha(self):
        self.gh.session.get.return_value = _resp(payload={"object": {"sha": "abc123"}})
        sha = self.gh.get_ref_sha("o", "r", "heads/main")
        self.assertEqual(sha, "abc123")
        url = self.gh.session.get.call_args[0][0]
        self.assertTrue(url.endswith("/git/ref/heads/main"))

    def test_create_blob_base64(self):
        self.gh.session.post.return_value = _resp(payload={"sha": "blobsha"})
        sha = self.gh.create_blob("o", "r", b"hello\n")
        self.assertEqual(sha, "blobsha")
        body = self.gh.session.post.call_args.kwargs["json"]
        self.assertEqual(body["encoding"], "base64")
        # base64 of "hello\n"
        self.assertEqual(body["content"], "aGVsbG8K")

    def test_create_commit_stamps_serge_identity(self):
        self.gh.session.post.return_value = _resp(payload={"sha": "commitsha"})
        sha = self.gh.create_commit(
            "o", "r", message="msg", tree_sha="t", parents=["p"]
        )
        self.assertEqual(sha, "commitsha")
        body = self.gh.session.post.call_args.kwargs["json"]
        self.assertEqual(body["author"]["name"], SERGE_GIT_NAME)
        self.assertEqual(body["committer"]["email"], SERGE_GIT_EMAIL)
        self.assertEqual(body["parents"], ["p"])

    def test_create_tree_includes_base(self):
        self.gh.session.post.return_value = _resp(payload={"sha": "treesha"})
        entries = [{"path": "a", "mode": "100644", "type": "blob", "sha": "b"}]
        sha = self.gh.create_tree("o", "r", "basetree", entries)
        self.assertEqual(sha, "treesha")
        body = self.gh.session.post.call_args.kwargs["json"]
        self.assertEqual(body["base_tree"], "basetree")
        self.assertEqual(body["tree"], entries)

    def test_create_ref_and_update_ref(self):
        self.gh.session.post.return_value = _resp(payload={"ref": "refs/heads/x"})
        self.gh.create_ref("o", "r", "refs/heads/serge/fix-1", "sha")
        body = self.gh.session.post.call_args.kwargs["json"]
        self.assertEqual(body, {"ref": "refs/heads/serge/fix-1", "sha": "sha"})

        self.gh.session.patch.return_value = _resp(payload={})
        self.gh.update_ref("o", "r", "heads/serge/fix-1", "sha2", force=True)
        body = self.gh.session.patch.call_args.kwargs["json"]
        self.assertEqual(body, {"sha": "sha2", "force": True})

    def test_create_pull_request(self):
        self.gh.session.post.return_value = _resp(
            payload={"number": 7, "html_url": "u"}
        )
        pr = self.gh.create_pull_request(
            "o", "r", title="t", head="serge/fix-1", base="main", body="b"
        )
        self.assertEqual(pr["number"], 7)
        body = self.gh.session.post.call_args.kwargs["json"]
        self.assertEqual(body["head"], "serge/fix-1")
        self.assertEqual(body["base"], "main")

    def test_count_branch_commits_by_author(self):
        self.gh.session.get.return_value = _resp(
            payload=[
                {"commit": {"author": {"email": SERGE_GIT_EMAIL}}},
                {"commit": {"author": {"email": "someone@else.com"}}},
                {"commit": {"author": {"email": SERGE_GIT_EMAIL.upper()}}},
            ]
        )
        n = self.gh.count_branch_commits_by_author(
            "o", "r", "serge/fix-1", author_email=SERGE_GIT_EMAIL
        )
        self.assertEqual(n, 2)

    def test_error_raises_httperror(self):
        import requests

        self.gh.session.post.return_value = _resp(ok=False, status=422)
        with self.assertRaises(requests.HTTPError):
            self.gh.create_blob("o", "r", b"x")


if __name__ == "__main__":
    unittest.main()
