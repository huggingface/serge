"""Tests for the dynamic serge-egress allowlist sync (reviewbot/k8s_sandbox.py
``sync_egress_allowlist`` + reviewbot/webapp.py ``_egress_llm_bases``).

Pod-based reviews dial their LLM provider through the serge-egress proxy, which
CONNECT-allows only the hosts in its tinyproxy filter. Provider configs are
dynamic, so serge keeps that filter in sync with the configured LLM hosts."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from reviewbot import k8s_sandbox
from reviewbot.store import JobStore


class HostRegexTests(unittest.TestCase):
    def test_url_to_exact_host_regex(self):
        self.assertEqual(
            k8s_sandbox._host_filter_regex("https://api.anthropic.com"),
            r"^api\.anthropic\.com$",
        )

    def test_strips_path_and_port_noise(self):
        self.assertEqual(
            k8s_sandbox._host_filter_regex("https://router.huggingface.co/v1"),
            r"^router\.huggingface\.co$",
        )

    def test_bare_host(self):
        self.assertEqual(
            k8s_sandbox._host_filter_regex("api.openai.com"), r"^api\.openai\.com$"
        )

    def test_empty_or_hostless_is_none(self):
        self.assertIsNone(k8s_sandbox._host_filter_regex(""))
        self.assertIsNone(k8s_sandbox._host_filter_regex("   "))


class _FakeApiException(Exception):
    def __init__(self, reason="boom"):
        super().__init__(reason)
        self.reason = reason


class _FakeCore:
    def __init__(self, filter_text):
        self.cm = types.SimpleNamespace(data={"filter": filter_text})
        self.patched_cm = None

    def read_namespaced_config_map(self, name, ns):
        self.read = (name, ns)
        return self.cm

    def patch_namespaced_config_map(self, name, ns, body):
        self.patched_cm = (name, ns, body)


class _FakeApps:
    def __init__(self):
        self.patched_deploy = None

    def patch_namespaced_deployment(self, name, ns, body):
        self.patched_deploy = (name, ns, body)


def _install_fake_kube(core, apps, *, raise_on_read=None):
    """Inject fake ``kubernetes`` modules so the lazy imports inside
    sync_egress_allowlist resolve. Returns a cleanup fn."""
    saved = {
        k: sys.modules.get(k)
        for k in ("kubernetes", "kubernetes.client", "kubernetes.client.rest")
    }
    if raise_on_read is not None:
        core.read_namespaced_config_map = mock.Mock(side_effect=raise_on_read)

    kmod = types.ModuleType("kubernetes")
    cmod = types.ModuleType("kubernetes.client")
    rmod = types.ModuleType("kubernetes.client.rest")
    rmod.ApiException = _FakeApiException
    cmod.CoreV1Api = lambda: core
    cmod.AppsV1Api = lambda: apps
    cmod.rest = rmod
    kmod.client = cmod
    kmod.config = types.SimpleNamespace(
        load_incluster_config=lambda: None, load_kube_config=lambda: None
    )
    sys.modules["kubernetes"] = kmod
    sys.modules["kubernetes.client"] = cmod
    sys.modules["kubernetes.client.rest"] = rmod

    def cleanup():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return cleanup


class SyncEgressAllowlistTests(unittest.TestCase):
    def _sync(self, filter_text, bases, **kw):
        core = _FakeCore(filter_text)
        apps = _FakeApps()
        self.addCleanup(_install_fake_kube(core, apps, **kw))
        changed = k8s_sandbox.sync_egress_allowlist(
            bases, egress_name="serge-egress", namespace="serge"
        )
        return changed, core, apps

    def test_no_egress_name_is_noop(self):
        # Must not even import the kube client.
        self.assertFalse(
            k8s_sandbox.sync_egress_allowlist(
                ["https://api.anthropic.com"], egress_name="", namespace="serge"
            )
        )

    def test_empty_bases_is_noop(self):
        self.assertFalse(
            k8s_sandbox.sync_egress_allowlist(
                [], egress_name="serge-egress", namespace="serge"
            )
        )

    def test_adds_missing_host_and_rolls_deployment(self):
        changed, core, apps = self._sync(
            r"^router\.huggingface\.co$" + "\n",
            ["https://api.anthropic.com", "https://router.huggingface.co/v1"],
        )
        self.assertTrue(changed)
        # ConfigMap patched with the union (existing preserved + new host).
        new_filter = core.patched_cm[2]["data"]["filter"]
        self.assertIn(r"^router\.huggingface\.co$", new_filter)
        self.assertIn(r"^api\.anthropic\.com$", new_filter)
        # Deployment rolled with the sync annotation.
        ann = apps.patched_deploy[2]["spec"]["template"]["metadata"]["annotations"]
        self.assertIn("serge.io/allowlist-synced-at", ann)

    def test_idempotent_when_all_present(self):
        changed, core, apps = self._sync(
            "^router\\.huggingface\\.co$\n^api\\.anthropic\\.com$\n",
            ["https://api.anthropic.com", "https://router.huggingface.co/v1"],
        )
        self.assertFalse(changed)
        self.assertIsNone(core.patched_cm)  # no write
        self.assertIsNone(apps.patched_deploy)  # no roll

    def test_api_error_is_fail_soft(self):
        changed, core, apps = self._sync(
            "",
            ["https://api.anthropic.com"],
            raise_on_read=_FakeApiException("forbidden"),
        )
        self.assertFalse(changed)
        self.assertIsNone(apps.patched_deploy)


class EgressBasesTests(unittest.TestCase):
    """webapp._egress_llm_bases unions the built-in provider bases with every
    provider_config's base (custom included)."""

    def _webapp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="serge-egress-")
        self.addCleanup(self._rm, self.tmpdir)
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy",
            "GITHUB_WEBHOOK_SECRET": "wh",
            "LLM_API_KEY": "llm",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
        }
        sys.modules.pop("reviewbot.webapp", None)
        self.addCleanup(lambda: sys.modules.pop("reviewbot.webapp", None))
        with mock.patch.dict(os.environ, env, clear=False):
            import reviewbot.webapp as webapp
        return webapp

    def _rm(self, tmp):
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    def test_bases_include_builtins_and_custom(self):
        webapp = self._webapp()
        store = JobStore(os.path.join(self.tmpdir, "providers.db"))
        store.insert_provider_config(
            id="c1",
            provider="custom",
            api_key="k",
            api_base="https://llm.internal.example.com/v1",
            default_model=None,
            repo_pattern="huggingface/x",
            allowed_users=[],
            allowed_orgs=[],
            created_by="admin",
        )
        with mock.patch.object(webapp, "_store", store):
            bases = webapp._egress_llm_bases()
        # Built-ins always present.
        self.assertIn("https://api.anthropic.com", bases)
        self.assertIn("https://api.openai.com/v1", bases)
        self.assertIn("https://router.huggingface.co/v1", bases)
        # The dynamic custom base is included.
        self.assertIn("https://llm.internal.example.com/v1", bases)


if __name__ == "__main__":
    unittest.main()
