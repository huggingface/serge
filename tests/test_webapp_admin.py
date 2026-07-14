"""Tests for the admin view bypass: web-UI reviews are private to their
submitter, except users listed in WEB_ADMIN_USERS, who may follow any
user's review via a shared link. Webhook jobs stay viewable by anyone."""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None


class WebappAdminViewTests(unittest.TestCase):
    def _build(self, admin_users: str = ""):
        """Import a fresh webapp with DEV_NO_AUTH on (current user == "dev")
        and the given WEB_ADMIN_USERS list, returning a TestClient."""
        sys.modules.pop("reviewbot.webapp", None)
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
            "WEB_ADMIN_USERS": admin_users,
        }
        with patch.dict(os.environ, env, clear=True):
            webapp = importlib.import_module("reviewbot.webapp")
        return webapp, TestClient(webapp.app)

    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _insert_web_job(self, webapp, job_id="job-web", user="octocat"):
        webapp._store.insert_job(
            id=job_id,
            user=user,
            target_owner="acme",
            target_repo="widgets",
            target_number=7,
            trigger_comment="x",
            llm_provider="hf",
            llm_api_base=None,
            llm_model="m",
            created_at=1.0,
            status="done",
            source="web",
            kind="review",
            task_spec_json=None,
        )

    def test_non_admin_cannot_view_other_users_web_review(self):
        webapp, client = self._build(admin_users="")
        self._insert_web_job(webapp)
        r = client.get("/reviews/acme/widgets/7/job-web/info")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["detail"], "not_your_job")

    def test_admin_can_view_other_users_web_review(self):
        webapp, client = self._build(admin_users="dev")
        self._insert_web_job(webapp)
        r = client.get("/reviews/acme/widgets/7/job-web/info")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], "job-web")

    def test_admin_match_is_case_insensitive(self):
        webapp, client = self._build(admin_users="DEV")
        self._insert_web_job(webapp)
        r = client.get("/reviews/acme/widgets/7/job-web/info")
        self.assertEqual(r.status_code, 200)

    def test_owner_can_still_view_own_review(self):
        webapp, client = self._build(admin_users="")
        self._insert_web_job(webapp, user="dev")
        r = client.get("/reviews/acme/widgets/7/job-web/info")
        self.assertEqual(r.status_code, 200)


class ProviderTestEndpointTests(unittest.TestCase):
    """POST /admin/providers/{id}/test makes one tiny inference call with the
    stored key and reports the verdict, so an admin can catch a bad token
    before a review fails on it."""

    def _build(self):
        sys.modules.pop("reviewbot.webapp", None)
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "LLM_BILL_TO": "huggingface",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
        }
        with patch.dict(os.environ, env, clear=True):
            webapp = importlib.import_module("reviewbot.webapp")
        return webapp, TestClient(webapp.app)

    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _insert_config(self, webapp, config_id="cfg-1"):
        webapp._store.insert_provider_config(
            id=config_id,
            provider="hf",
            api_key="hf-token",
            api_base=None,
            default_model="zai-org/GLM-5.2",
            repo_pattern="huggingface/*",
            allowed_users=[],
            allowed_orgs=["huggingface"],
            created_by="dev",
        )
        return config_id

    def test_missing_config_is_404(self):
        webapp, client = self._build()
        r = client.post(
            "/admin/providers/nope/test", headers={"Origin": "http://testserver"}
        )
        self.assertEqual(r.status_code, 404)

    def test_valid_token_reports_ok(self):
        webapp, client = self._build()
        cfg_id = self._insert_config(webapp)

        class _FakeClient:
            def __init__(self, api_base, api_key, *, model=None, bill_to=None, **kw):
                # The billing org must be forwarded so the test reproduces the
                # exact permission a real review needs.
                assert bill_to == "huggingface"
                self.model = model

            def complete(self, messages, **kw):
                return object()

        with patch.object(webapp, "ChatCompletionClient", _FakeClient):
            r = client.post(
                f"/admin/providers/{cfg_id}/test",
                headers={"Origin": "http://testserver"},
            )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["model"], "zai-org/GLM-5.2")
        self.assertIn("huggingface", data["message"])

    def test_bad_token_surfaces_provider_error(self):
        from reviewbot.llm_client import LLMResponseError

        webapp, client = self._build()
        cfg_id = self._insert_config(webapp)
        exc = LLMResponseError(
            403,
            "Forbidden",
            "https://router.huggingface.co/v1/chat/completions",
            '{"error":"insufficient permissions to call Inference Providers '
            'on behalf of org huggingface"}',
        )

        class _FakeClient:
            def __init__(self, *a, **kw):
                self.model = kw.get("model")

            def complete(self, messages, **kw):
                raise exc

        with patch.object(webapp, "ChatCompletionClient", _FakeClient):
            r = client.post(
                f"/admin/providers/{cfg_id}/test",
                headers={"Origin": "http://testserver"},
            )
        # A bad token is a verdict, not a server error.
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["ok"])
        self.assertIn("403", data["error"])
        self.assertIn("Inference Providers", data["error"])


class ProviderModelsEndpointTests(unittest.TestCase):
    """POST /admin/providers/models lists a keyed provider's models for the
    admin form dropdown — using the typed key, or the stored key of the config
    being edited. A bad token / no-/models endpoint is a verdict, not a 500."""

    def _build(self):
        sys.modules.pop("reviewbot.webapp", None)
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
        }
        with patch.dict(os.environ, env, clear=True):
            webapp = importlib.import_module("reviewbot.webapp")
        return webapp, TestClient(webapp.app)

    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_typed_key_lists_models(self):
        webapp, client = self._build()
        seen = {}

        class _FakeClient:
            def __init__(self, api_base, api_key, *, bill_to=None, **kw):
                seen["api_base"] = api_base
                seen["api_key"] = api_key
                seen["bill_to"] = bill_to

            def list_models(self):
                return ["gpt-4o", "o3"]

        with patch.object(webapp, "ChatCompletionClient", _FakeClient):
            r = client.post(
                "/admin/providers/models",
                json={"provider": "openai", "api_key": "sk-typed"},
                headers={"Origin": "http://testserver"},
            )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["models"], ["gpt-4o", "o3"])
        self.assertEqual(seen["api_key"], "sk-typed")
        self.assertEqual(seen["api_base"], "https://api.openai.com/v1")
        # Non-HF providers never forward a billing org.
        self.assertIsNone(seen["bill_to"])

    def test_blank_key_falls_back_to_stored_config_key(self):
        webapp, client = self._build()
        webapp._store.insert_provider_config(
            id="cfg-openai",
            provider="openai",
            api_key="sk-stored",
            api_base=None,
            default_model=None,
            repo_pattern="acme/*",
            allowed_users=[],
            allowed_orgs=["acme"],
            created_by="dev",
        )
        seen = {}

        class _FakeClient:
            def __init__(self, api_base, api_key, **kw):
                seen["api_key"] = api_key

            def list_models(self):
                return ["gpt-4o"]

        with patch.object(webapp, "ChatCompletionClient", _FakeClient):
            r = client.post(
                "/admin/providers/models",
                json={"provider": "openai", "config_id": "cfg-openai"},
                headers={"Origin": "http://testserver"},
            )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertEqual(seen["api_key"], "sk-stored")

    def test_missing_key_is_400(self):
        webapp, client = self._build()
        r = client.post(
            "/admin/providers/models",
            json={"provider": "openai"},
            headers={"Origin": "http://testserver"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "api_key_required")

    def test_fetch_error_is_a_verdict_not_500(self):
        webapp, client = self._build()

        class _FakeClient:
            def __init__(self, *a, **kw):
                pass

            def list_models(self):
                raise RuntimeError("Failed to list models (status 401).")

        with patch.object(webapp, "ChatCompletionClient", _FakeClient):
            r = client.post(
                "/admin/providers/models",
                json={"provider": "anthropic", "api_key": "sk-bad"},
                headers={"Origin": "http://testserver"},
            )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["ok"])
        self.assertIn("401", data["error"])
        self.assertEqual(data["models"], [])


if __name__ == "__main__":
    unittest.main()
