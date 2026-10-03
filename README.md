# langchain-bowmark

LangChain tools for [Bowmark](https://bowmark.ai). Bowmark turns websites into typed
functions an AI agent can call: search, prices, availability, quotes, bookings, reading
a page. Your agent looks up the functions it needs, writes a short script against them,
and Bowmark runs it on the live sites and returns structured data. The agent never
drives a browser itself.

Full guide: [bowmark.ai/docs/langchain](https://bowmark.ai/docs/langchain)

## Install

```bash
pip install langchain-bowmark
```

Create an API key at [bowmark.ai/dashboard/keys](https://bowmark.ai/dashboard/keys) and
export it:

```bash
export BOWMARK_API_KEY="bmk_..."
```

## Tools

| Tool | What it does |
|---|---|
| `bowmark_get_library` | Returns the typed functions for a task or a site, with types and examples. Read-only, free, touches no site. |
| `bowmark_run` | Runs a short async JavaScript script against those functions on the live sites. Returns `{ok, status, result, logs, error}`. |

`BowmarkToolkit` returns both, sharing one key.

## Use with an agent

```python
from langchain.agents import create_agent
from langchain_bowmark import BowmarkToolkit

agent = create_agent(
    "openai:gpt-5.6-luna",
    tools=BowmarkToolkit().get_tools(),
    system_prompt=(
        "You can act on live websites through Bowmark. Call bowmark_get_library with what "
        "the user wants to do, write a short script against the functions it returns, and "
        "execute it with bowmark_run. Answer from the run's result."
    ),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the top story on Hacker News right now?"}]}
)
print(result["messages"][-1].content)
```

## Call the tools directly

```python
from langchain_bowmark import BowmarkGetLibrary, BowmarkRun

print(BowmarkGetLibrary().invoke({"query": "read a web page"}))
print(BowmarkRun().invoke({"script": 'return (await bowmark.read.page("https://example.com")).title;'}))
```

## Development

```bash
uv venv && uv pip install -e . --group test
pytest tests/unit_tests
BOWMARK_API_KEY=... pytest tests/integration_tests
```

## License

MIT
