import json

import httpx
import pytest
from langchain_core.tools import ToolException
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_bowmark import BowmarkGetLibrary, BowmarkRun, BowmarkToolkit
from langchain_bowmark import tools as mod


class TestBowmarkGetLibraryStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return BowmarkGetLibrary

    @property
    def tool_constructor_params(self):
        return {}

    @property
    def tool_invoke_params_example(self):
        return {"query": "flights"}


class TestBowmarkRunStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return BowmarkRun

    @property
    def tool_constructor_params(self):
        return {}

    @property
    def tool_invoke_params_example(self):
        return {"script": "return 1 + 1"}


def _mock(monkeypatch, handler):
    real_client, real_async = httpx.Client, httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(mod.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: real_async(transport=transport, **kw))


def test_toolkit_returns_both_tools_sharing_the_key(monkeypatch):
    monkeypatch.delenv("BOWMARK_API_KEY", raising=False)
    tools = BowmarkToolkit(api_key="bmk_test").get_tools()
    assert [t.name for t in tools] == ["bowmark_get_library", "bowmark_run"]
    assert tools[0].client is tools[1].client


def test_library_sends_query_and_bearer(monkeypatch):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, text="# flights\n...")

    _mock(monkeypatch, handler)
    out = BowmarkGetLibrary(client=mod._BowmarkClient(api_key="bmk_x")).invoke({"query": "flights"})
    assert out.startswith("# flights")
    assert seen["url"] == "https://api.bowmark.ai/v1/library?query=flights"
    assert seen["auth"] == "Bearer bmk_x"


async def test_run_posts_script_and_returns_envelope(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        assert json.loads(req.content) == {"script": "return 2"}
        return httpx.Response(200, json={"ok": True, "status": "ok", "result": 2, "logs": [], "error": None, "ms": 5, "trace": []})

    _mock(monkeypatch, handler)
    out = await BowmarkRun(client=mod._BowmarkClient(api_key="bmk_x")).ainvoke({"script": "return 2"})
    assert json.loads(out) == {"ok": True, "status": "ok", "result": 2, "logs": [], "error": None}


def test_failed_run_envelope_reaches_the_model(monkeypatch):
    _mock(monkeypatch, lambda req: httpx.Response(402, json={"ok": False, "status": "error", "result": None, "logs": [], "error": "add a card"}))
    out = BowmarkRun(client=mod._BowmarkClient(api_key="bmk_x")).invoke({"script": "return 1"})
    assert json.loads(out)["error"] == "add a card"


def test_missing_key_is_a_tool_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("BOWMARK_API_KEY", raising=False)
    out = BowmarkGetLibrary().invoke({"query": "flights"})
    assert "BOWMARK_API_KEY" in out
    with pytest.raises(ToolException):
        mod._BowmarkClient().get_library("flights")
