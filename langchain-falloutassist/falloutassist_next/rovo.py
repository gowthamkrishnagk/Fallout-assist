"""Read-only semantic candidate search through Atlassian Rovo MCP."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import httpx

from .config import config_value, required_env

MCP_URL = "https://mcp.atlassian.com/v2/mcp"
ISSUE_KEY = re.compile(r"[A-Z][A-Z0-9_]{1,29}-[1-9][0-9]{0,9}")


def _auth_header() -> str:
    email = required_env("ROVO_EMAIL")
    token = required_env("ROVO_API_TOKEN")
    credentials = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    return f"Basic {credentials}"


def _issue_keys(result: Any) -> list[str]:
    structured = getattr(result, "structuredContent", None)
    blocks = getattr(result, "content", None) or []
    text = "\n".join(str(getattr(block, "text", "")) for block in blocks
                     if getattr(block, "text", ""))
    payload: Any = structured
    if payload is None and text:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = None

    found: list[str] = []
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            key = value.get("issueKey") or value.get("key")
            if isinstance(key, str) and ISSUE_KEY.fullmatch(key):
                found.append(key)
            url = value.get("url")
            if isinstance(url, str):
                match = re.search(r"/browse/([A-Z][A-Z0-9_]{1,29}-[1-9][0-9]{0,9})", url)
                if match:
                    found.append(match.group(1))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    if payload is not None:
        visit(payload)
    if not found and text:
        found.extend(re.findall(r"\b[A-Z][A-Z0-9_]{1,29}-[1-9][0-9]{0,9}\b", text))
    return list(dict.fromkeys(found))


async def _semantic_issue_keys(query: str, limit: int) -> list[str]:
    cloud_id = required_env("ROVO_CLOUD_ID")
    headers = {"Authorization": _auth_header()}
    ca_bundle = config_value("TLS_CA_BUNDLE")
    query = (query or "").strip()
    if len(query) > 900:
        # Atlassian's Rovo MCP search validates queries at 1,000 characters.
        # Preserve the opening context and the canonical failure fields appended
        # by the Jira plugin, which carry the most useful retrieval signals.
        query = query[:450] + "\n" + query[-449:]

    def httpx_client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth,
                                 verify=ca_bundle or True)

    async with streamablehttp_client(MCP_URL, headers=headers, timeout=45,
                                     httpx_client_factory=httpx_client_factory) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            search_tool = next((tool for tool in listed.tools
                                if "search" in tool.name.lower()
                                and ("semantic" in (tool.description or "").lower()
                                     or tool.name.lower().endswith("_search")
                                     or tool.name.lower() == "search")), None)
            if search_tool is None:
                raise RuntimeError("Rovo MCP did not expose its semantic search tool")
            result = await session.call_tool(search_tool.name, {
                "cloudId": cloud_id,
                "query": query,
                "targetApp": "JIRA",
                "mode": "STANDARD",
                # Atlassian's Rovo MCP search tool currently accepts at most
                # 30 results. Larger requests return an MCP input validation
                # error, which makes the combined retriever fall back to local.
                "limit": max(10, min(30, limit)),
            })
            if getattr(result, "isError", False):
                raise RuntimeError("Rovo semantic search returned an error")
            return _issue_keys(result)


def semantic_issue_keys(query: str, limit: int) -> list[str]:
    """Return visible Jira issue keys in Rovo's semantic result order."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_semantic_issue_keys(query, limit))
    raise RuntimeError("Rovo MCP search must run from the synchronous LangGraph workflow")
