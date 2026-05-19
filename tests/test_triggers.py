import unittest

from reviewbot.triggers import build_review_request


class TriggerTests(unittest.TestCase):
    def test_build_review_request_accepts_serge_trigger(self) -> None:
        payload = {
            "action": "created",
            "comment": {
                "body": "@serge please review",
                "author_association": "MEMBER",
                "id": 123,
                "user": {"login": "reviewer"},
            },
            "issue": {
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/project/pulls/7"
                },
                "state": "open",
                "number": 7,
            },
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("issue_comment", payload, "@serge")

        self.assertIsNotNone(req)
        assert req is not None
        self.assertEqual(req.owner, "acme")
        self.assertEqual(req.repo, "project")
        self.assertEqual(req.number, 7)
        self.assertEqual(req.trigger_comment_id, 123)

    def test_build_review_request_rejects_non_matching_trigger(self) -> None:
        payload = {
            "action": "created",
            "comment": {
                "body": "@claude please review",
                "author_association": "MEMBER",
                "id": 123,
                "user": {"login": "reviewer"},
            },
            "issue": {
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/project/pulls/7"
                },
                "state": "open",
                "number": 7,
            },
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("issue_comment", payload, "@serge")

        self.assertIsNone(req)


if __name__ == "__main__":
    unittest.main()
