import pytest

from reviewbot import toolchain_refresh as tr
from reviewbot.toolchain_refresh import (
    ImageRef,
    RefreshDispatcher,
    is_gate_broken,
    manifest_digest,
    resolve_runner_image,
)

IMAGE = "ghcr.io/huggingface/serge-task-runner:sha-6668325"
DIGEST = "sha256:" + "ab" * 32


@pytest.fixture(autouse=True)
def _fresh_cache():
    tr.clear_cache()
    yield
    tr.clear_cache()


def test_parse_image_refs():
    assert ImageRef.parse(IMAGE) == ImageRef(
        "ghcr.io", "huggingface/serge-task-runner", "sha-6668325"
    )
    assert ImageRef.parse("localhost:5000/x/y:t") == ImageRef(
        "localhost:5000", "x/y", "t"
    )
    assert ImageRef.parse("ubuntu:24.04") == ImageRef(
        "registry-1.docker.io", "library/ubuntu", "24.04"
    )
    assert ImageRef.parse(f"ghcr.io/a/b@{DIGEST}") is None
    assert ImageRef.parse("localhost:5000/x/y") is None


def test_pointer_present_launches_it_pinned_by_digest():
    calls = []

    def lookup(registry, repo, tag):
        calls.append((registry, repo, tag))
        return DIGEST

    r = resolve_runner_image(IMAGE, "-toolchain", lookup=lookup)
    assert r.refreshed
    assert (
        r.image
        == f"ghcr.io/huggingface/serge-task-runner:sha-6668325-toolchain@{DIGEST}"
    )
    assert calls == [
        ("ghcr.io", "huggingface/serge-task-runner", "sha-6668325-toolchain")
    ]
    assert "toolchain-refreshed" in r.note


def test_pointer_absent_keeps_configured_image():
    r = resolve_runner_image(IMAGE, "-toolchain", lookup=lambda *a: None)
    assert (r.image, r.refreshed, r.note) == (IMAGE, False, "")


def test_registry_failure_never_blocks_a_launch():
    def boom(*a):
        raise ConnectionError("ghcr down")

    r = resolve_runner_image(IMAGE, "-toolchain", lookup=boom)
    assert r.image == IMAGE and not r.refreshed
    assert "ghcr down" in r.note


def test_off_without_suffix_or_for_digest_refs():
    def never(*a):
        raise AssertionError("must not look up")

    assert resolve_runner_image(IMAGE, "", lookup=never).image == IMAGE
    pinned = f"ghcr.io/a/b@{DIGEST}"
    assert resolve_runner_image(pinned, "-toolchain", lookup=never).image == pinned


def test_lookup_is_cached_for_ttl():
    n = []
    clock = [0.0]

    def lookup(*a):
        n.append(1)
        return DIGEST

    for _ in range(5):
        resolve_runner_image(IMAGE, "-toolchain", lookup=lookup, now=lambda: clock[0])
    assert len(n) == 1
    clock[0] = 301.0
    resolve_runner_image(IMAGE, "-toolchain", lookup=lookup, now=lambda: clock[0])
    assert len(n) == 2


class _Resp:
    def __init__(self, status, headers=None, body=None, content=b""):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.content = content

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Registry:
    """GHCR's anonymous flow: 401 + Bearer challenge, token, then the manifest."""

    def __init__(self, tags):
        self.tags = tags
        self.calls = []

    def head(self, url, headers, timeout):
        self.calls.append(("HEAD", url, headers.get("Authorization")))
        if "Authorization" not in headers:
            return _Resp(
                401,
                {
                    "WWW-Authenticate": 'Bearer realm="https://ghcr.io/token",'
                    'service="ghcr.io",scope="repository:huggingface/serge-task-runner:pull"'
                },
            )
        tag = url.rsplit("/", 1)[1]
        if tag not in self.tags:
            return _Resp(404)
        return _Resp(200, {"Docker-Content-Digest": self.tags[tag]})

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(("GET", url, params))
        return _Resp(200, body={"token": "anon"})


def test_manifest_digest_follows_bearer_challenge():
    reg = _Registry({"sha-6668325-toolchain": DIGEST})
    d = manifest_digest(
        "ghcr.io", "huggingface/serge-task-runner", "sha-6668325-toolchain", session=reg
    )
    assert d == DIGEST
    assert reg.calls[1] == (
        "GET",
        "https://ghcr.io/token",
        {
            "service": "ghcr.io",
            "scope": "repository:huggingface/serge-task-runner:pull",
        },
    )
    assert reg.calls[2][2] == "Bearer anon"


def test_manifest_digest_absent_tag_is_none():
    reg = _Registry({})
    assert (
        manifest_digest("ghcr.io", "huggingface/serge-task-runner", "nope", session=reg)
        is None
    )


def test_gate_broken_detection_matches_tasks_messages():
    assert is_gate_broken(
        "normalize_error", "Preflight fails on the pristine checkout:\nx"
    )
    assert is_gate_broken(
        "normalize_error", "Normalizer fails on the pristine checkout:\nx"
    )
    # A bad PATCH is the model's problem, not the image's.
    assert not is_gate_broken(
        "normalize_error", "Normalizer failed (exit 2) for `make`"
    )
    assert not is_gate_broken("log", "Preflight fails on the pristine checkout")


def test_gate_broken_prefixes_are_what_tasks_emits():
    # If tasks.py rewords these, the dispatcher would silently never fire.
    import inspect

    from reviewbot import tasks

    src = inspect.getsource(tasks)
    for prefix in tr.GATE_BROKEN_PREFIXES:
        assert f'"normalize_error", f"{prefix}' in src


class _GitHub:
    def __init__(self, status=204):
        self.status = status
        self.posts = []

    def post(self, url, headers, json, timeout):
        self.posts.append((url, json))
        return _Resp(self.status)


def test_dispatcher_posts_once_per_cooldown():
    gh = _GitHub()
    clock = [0.0]
    d = RefreshDispatcher(
        token="t",
        repo="huggingface/serge",
        workflow="toolchain.yml",
        serge_ref="6668325",
        session=gh,
        now=lambda: clock[0],
    )
    assert d.request("task a") is True
    assert d.request("task b") is False  # the night's other tasks
    clock[0] = 1801.0
    assert d.request("task c") is True
    url, body = gh.posts[0]
    assert url.endswith(
        "/repos/huggingface/serge/actions/workflows/toolchain.yml/dispatches"
    )
    assert body == {"ref": "main", "inputs": {"serge_ref": "6668325"}}


def test_dispatcher_without_token_is_a_noop():
    gh = _GitHub()
    d = RefreshDispatcher(
        token=None, repo="r", workflow="w", serge_ref="abc", session=gh
    )
    assert d.request("x") is False and gh.posts == []


def test_failed_dispatch_does_not_start_cooldown():
    gh = _GitHub(status=500)
    d = RefreshDispatcher(
        token="t", repo="r", workflow="w", serge_ref="abc", session=gh
    )
    assert d.request("x") is False
    gh.status = 204
    assert d.request("y") is True
