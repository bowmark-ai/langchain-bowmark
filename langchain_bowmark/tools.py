"""LangChain tools for Bowmark: read the function library, then run a script against it."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

DEFAULT_BASE_URL = "https://api.bowmark.ai"
# A run executes on live websites and can take a minute or more; a library read is fast.
RUN_TIMEOUT_S = 300.0
LIBRARY_TIMEOUT_S = 60.0


class _BowmarkClient(BaseModel):
    """Shared auth and transport for the Bowmark REST API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[SecretStr] = None
    base_url: str = DEFAULT_BASE_URL

    @model_validator(mode="before")
    @classmethod
    def _key_from_env(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("api_key"):
            env = os.environ.get("BOWMARK_API_KEY")
            if env:
                values["api_key"] = env
        return values

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            raise ToolException(
                "No Bowmark API key. Set BOWMARK_API_KEY or pass api_key=. "
                "Create one at https://bowmark.ai/dashboard/keys"
            )
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "User-Agent": "langchain-bowmark",
        }

    def get_library(self, query: str) -> str:
        with httpx.Client(timeout=LIBRARY_TIMEOUT_S) as c:
            r = c.get(f"{self.base_url}/v1/library", params={"query": query}, headers=self._headers())
        return _library_text(r)

    async def aget_library(self, query: str) -> str:
        async with httpx.AsyncClient(timeout=LIBRARY_TIMEOUT_S) as c:
            r = await c.get(f"{self.base_url}/v1/library", params={"query": query}, headers=self._headers())
        return _library_text(r)

    def run(self, script: str) -> str:
        with httpx.Client(timeout=RUN_TIMEOUT_S) as c:
            r = c.post(f"{self.base_url}/v1/run", json={"script": script}, headers=self._headers())
        return _run_text(r)

    async def arun(self, script: str) -> str:
        async with httpx.AsyncClient(timeout=RUN_TIMEOUT_S) as c:
            r = await c.post(f"{self.base_url}/v1/run", json={"script": script}, headers=self._headers())
        return _run_text(r)


def _library_text(r: httpx.Response) -> str:
    if r.status_code != 200:
        raise ToolException(f"Bowmark library request failed ({r.status_code}): {r.text[:500]}")
    return r.text


def _run_text(r: httpx.Response) -> str:
    # The API answers every finished run with a JSON envelope, including a failed one,
    # a rate limit (429) and a payment requirement (402). Hand the envelope to the model
    # whenever there is one: its `error` says what to fix.
    try:
        envelope = r.json()
    except ValueError:
        raise ToolException(f"Bowmark run failed ({r.status_code}): {r.text[:500]}")
    if not isinstance(envelope, dict):
        raise ToolException(f"Bowmark run returned an unexpected body ({r.status_code})")
    keep = {k: envelope.get(k) for k in ("ok", "status", "result", "logs", "error", "runId") if k in envelope}
    return json.dumps(keep, ensure_ascii=False)


class GetLibraryInput(BaseModel):
    query: str = Field(
        description=(
            "What you want to do on the web ('flights', 'price a GPU', 'check stock'), "
            "or a company or site name ('Kayak', 'newegg.com')."
        )
    )


class RunInput(BaseModel):
    script: str = Field(
        description=(
            "An async JavaScript body written against the `bowmark` global, using only the "
            "functions bowmark_get_library returned. Always `await` calls and `return` the result."
        )
    )


class BowmarkGetLibrary(BaseTool):
    """Read Bowmark's typed function library for a task or a site.

    Read-only and free: it touches no website. Returns markdown with the matching
    functions, their TypeScript types and worked examples.
    """

    name: str = "bowmark_get_library"
    description: str = (
        "Look up which typed functions Bowmark has for a task on a live website (search, prices, "
        "availability, quotes, bookings, reading a page). Returns the functions with their types "
        "and examples. Call this before bowmark_run. Read-only; touches no site."
    )
    args_schema: Type[BaseModel] = GetLibraryInput
    client: _BowmarkClient = Field(default_factory=_BowmarkClient)
    handle_tool_error: bool = True

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.client.get_library(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return await self.client.aget_library(query)


class BowmarkRun(BaseTool):
    """Execute a short script against Bowmark's function library on the live websites."""

    name: str = "bowmark_run"
    description: str = (
        "Run a short async JavaScript script that calls the functions bowmark_get_library returned, "
        "on the live websites, and get back JSON: {ok, status, result, logs, error}. "
        "Use `await bowmark.<capability>.<fn>(...)` and `return` what you need."
    )
    args_schema: Type[BaseModel] = RunInput
    client: _BowmarkClient = Field(default_factory=_BowmarkClient)
    handle_tool_error: bool = True

    def _run(self, script: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.client.run(script)

    async def _arun(self, script: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return await self.client.arun(script)


class BowmarkToolkit(BaseToolkit):
    """Both Bowmark tools, sharing one API key.

    Example:
        .. code-block:: python

            from langchain_bowmark import BowmarkToolkit

            tools = BowmarkToolkit().get_tools()  # reads BOWMARK_API_KEY
    """

    api_key: Optional[SecretStr] = None
    base_url: str = DEFAULT_BASE_URL

    def get_tools(self) -> list[BaseTool]:
        client = _BowmarkClient(api_key=self.api_key, base_url=self.base_url)
        return [BowmarkGetLibrary(client=client), BowmarkRun(client=client)]
