import io
import json
import urllib.error

import pytest

from brandvis.collect import select_surfaces
from brandvis.config import Study, Surface
from brandvis.providers import ProviderError, QuotaExhausted
from brandvis.providers import anthropic as A


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok(body):
    return FakeResp(json.dumps(body).encode())


def http_error(code, body, headers=None):
    return urllib.error.HTTPError(A.URL, code, "err", headers or {}, io.BytesIO(json.dumps(body).encode()))


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(A.time, "sleep", lambda s: None)


def script(monkeypatch, outcomes):
    calls = []

    def fake(req, timeout=None):
        calls.append(json.loads(req.data))
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(A.urllib.request, "urlopen", fake)
    return calls


PLAIN = {"model": "claude-sonnet-5-5", "stop_reason": "end_turn", "usage": {"input_tokens": 20, "output_tokens": 300},
         "content": [{"type": "text", "text": "Try Aritzia and Zara."}]}

SEARCHED = {"model": "claude-sonnet-5-5", "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 400, "server_tool_use": {"web_search_requests": 2}},
            "content": [
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "best blazers women"}},
                {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [{"type": "web_search_result", "url": "https://x.com/a", "title": "A"}]},
                {"type": "text", "text": "Aritzia is popular", "citations": [{"type": "web_search_result_location", "url": "https://www.forbes.com/x", "title": "F"}]},
                {"type": "text", "text": " and Reformation too.", "citations": [{"type": "web_search_result_location", "url": "https://www.forbes.com/x"},
                                                                              {"type": "web_search_result_location", "url": "https://reddit.com/r/y"}]},
            ]}


def test_plain_answer_and_usage(monkeypatch):
    calls = script(monkeypatch, [ok(PLAIN)])
    r = A.AnthropicProvider("claude-sonnet-5-5").generate("q", temperature=1.0, grounded=False)
    assert r.text == "Try Aritzia and Zara." and r.model == "claude-sonnet-5-5" and r.finish_reason == "end_turn"
    assert r.usage["totalTokenCount"] == 320 and r.sources == [] and r.search_queries == []
    assert "tools" not in calls[0] and calls[0]["temperature"] == 1.0 and calls[0]["messages"][0]["content"] == "q"


def test_grounded_answer_collects_queries_and_unique_cited_domains(monkeypatch):
    calls = script(monkeypatch, [ok(SEARCHED)])
    r = A.AnthropicProvider("m").generate("q", grounded=True)
    assert calls[0]["tools"][0]["name"] == "web_search"
    assert r.text == "Aritzia is popular and Reformation too."
    assert r.search_queries == ["best blazers women"]
    assert r.sources == ["forbes.com", "reddit.com"]          # unique cited pages, www stripped, retrieved-but-uncited x.com excluded
    assert r.usage["web_search_requests"] == 2


def test_retries_overloaded_then_succeeds(monkeypatch):
    calls = script(monkeypatch, [http_error(529, {"error": {"type": "overloaded_error"}}), http_error(429, {}, {"retry-after": "2"}), ok(PLAIN)])
    assert A.AnthropicProvider("m").generate("q").text.startswith("Try")
    assert len(calls) == 3


def test_bad_key_is_a_clear_permanent_error(monkeypatch):
    script(monkeypatch, [http_error(401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}})])
    with pytest.raises(ProviderError, match="rejected the API key"):
        A.AnthropicProvider("m").generate("q")


def test_empty_balance_stops_the_run(monkeypatch):
    script(monkeypatch, [http_error(400, {"error": {"type": "invalid_request_error", "message": "Your credit balance is too low to access the Anthropic API."}})])
    with pytest.raises(QuotaExhausted):
        A.AnthropicProvider("m").generate("q")


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("BRANDVIS_ANTHROPIC_KEY", raising=False)
    with pytest.raises(ProviderError, match="no Anthropic API key"):
        A.AnthropicProvider("m").generate("q")


def test_pause_turn_is_resumed(monkeypatch):
    paused = {"model": "m", "stop_reason": "pause_turn", "usage": {"input_tokens": 10, "output_tokens": 5},
              "content": [{"type": "server_tool_use", "name": "web_search", "input": {"query": "q1"}}]}
    calls = script(monkeypatch, [ok(paused), ok(SEARCHED)])
    r = A.AnthropicProvider("m").generate("q", grounded=True)
    assert len(calls) == 2 and calls[1]["messages"][-1]["role"] == "assistant"
    assert r.search_queries == ["q1", "best blazers women"] and r.usage["input_tokens"] == 110


def test_empty_text_is_an_error(monkeypatch):
    script(monkeypatch, [ok({"model": "m", "stop_reason": "end_turn", "usage": {}, "content": []})])
    with pytest.raises(ProviderError, match="empty answer"):
        A.AnthropicProvider("m").generate("q")


def test_select_surfaces():
    st = Study("c", "f", 1, 1.0, (Surface("a", "p", "m"), Surface("b", "p", "m")))
    assert [s.id for s in select_surfaces(st, "b").surfaces] == ["b"]
    assert select_surfaces(st, None) is st
    with pytest.raises(SystemExit):
        select_surfaces(st, "zzz")


def test_key_is_cleaned_of_pasted_whitespace_and_quotes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", '  "sk-ant-abc123"\n ')
    assert A.read_key() == "sk-ant-abc123"
    calls = []
    monkeypatch.setattr(A.urllib.request, "urlopen", lambda req, timeout=None: (calls.append(req.headers), ok(PLAIN))[1])
    A.AnthropicProvider("m").generate("q")
    assert calls[0]["X-api-key"] == "sk-ant-abc123"


def test_check_keys_shape_never_reveals_the_key(monkeypatch):
    from brandvis.check_keys import shape

    monkeypatch.setenv("ANTHROPIC_API_KEY", " 'sk-ant-secretvalue...' ")
    out = shape(A.read_key())
    assert "secretvalue" not in out and "starts with sk-ant-" in out and "whitespace" in out and "'...'" in out
