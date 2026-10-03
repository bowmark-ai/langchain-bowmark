import json
import os

import pytest

from langchain_bowmark import BowmarkToolkit

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("BOWMARK_API_KEY"), reason="needs BOWMARK_API_KEY"),
]


def test_library_then_run_against_the_real_api():
    lib, run = BowmarkToolkit().get_tools()
    text = lib.invoke({"query": "read a web page"})
    assert "bowmark" in text.lower()
    out = json.loads(run.invoke({"script": 'const p = await bowmark.read.page("https://example.com"); return p.title;'}))
    assert out["ok"] is True
    assert "Example Domain" in str(out["result"])
