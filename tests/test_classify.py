from reviewbot import classify


class _FakeLLM:
    """Minimal ChatCompletionClient stand-in: returns a canned content string,
    or raises when `boom` is set."""

    def __init__(self, content="", boom=False):
        self._content = content
        self._boom = boom
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self._boom:
            raise RuntimeError("router down")
        return type("R", (), {"content": self._content})()


TB = {
    "tests/models/glm46v/test_modeling_glm46v.py::T::test_x": (
        "E  RuntimeError: Expected tensor for argument #1 'indices' to have one "
        "of the following scalar types: Long, Int; but got CUDABFloat16Type"
    )
}


def test_parse_valid_labels():
    for label in ("product_issue", "test_issue", "unclear"):
        r = classify.parse_classify_response(f'{{"label": "{label}", "reason": "x"}}')
        assert r.label == label
        assert r.reason == "x"


def test_parse_unknown_label_is_unclear():
    r = classify.parse_classify_response('{"label": "banana", "reason": "y"}')
    assert r.label == classify.UNCLEAR


def test_parse_non_json_is_unclear():
    assert classify.parse_classify_response("not json at all").label == classify.UNCLEAR


def test_parse_json_wrapped_in_prose():
    # Some models wrap the object in fences/prose — the first {...} span wins.
    content = 'Sure!\n```json\n{"label": "test_issue", "reason": "stale"}\n```'
    r = classify.parse_classify_response(content)
    assert r.label == "test_issue"
    assert r.reason == "stale"


def test_build_messages_includes_traceback_and_nodeids():
    msgs = classify.build_classify_messages(
        list(TB.keys()), TB, context="12 tests failing (other)"
    )
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "glm46v" in user
    assert "CUDABFloat16Type" in user
    assert "12 tests failing" in user


def test_build_messages_truncates_long_traceback():
    long_tb = {"a.py::T::t": "X" * 10_000}
    user = classify.build_classify_messages(["a.py::T::t"], long_tb)[1]["content"]
    assert "truncated" in user
    assert len(user) < 6000


def test_classify_failure_product_issue():
    llm = _FakeLLM('{"label": "product_issue", "reason": "hard crash in embedding"}')
    r = classify.classify_failure(llm, node_ids=list(TB), tracebacks=TB)
    assert r.is_product_issue
    assert not r.is_test_issue
    # response_format must ask for a JSON object.
    assert llm.calls[0][1]["response_format"] == {"type": "json_object"}


def test_classify_failure_test_issue():
    llm = _FakeLLM('{"label": "test_issue", "reason": "stale expected values"}')
    r = classify.classify_failure(llm, node_ids=list(TB), tracebacks=TB)
    assert r.is_test_issue


def test_classify_failure_llm_error_degrades_to_unclear():
    r = classify.classify_failure(_FakeLLM(boom=True), node_ids=list(TB), tracebacks=TB)
    assert r.label == classify.UNCLEAR


def test_classify_failure_no_nodeids_is_unclear_without_calling_llm():
    llm = _FakeLLM('{"label": "product_issue"}')
    r = classify.classify_failure(llm, node_ids=[], tracebacks={})
    assert r.label == classify.UNCLEAR
    assert llm.calls == []
