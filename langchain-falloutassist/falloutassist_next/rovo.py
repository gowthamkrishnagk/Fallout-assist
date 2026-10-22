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
from .execution import remaining

MCP_URL = "https://mcp.atlassian.com/v2/mcp"
ISSUE_KEY = re.compile(r"[A-Z][A-Z0-9_]{1,29}-[1-9][0-9]{0,9}")


class RovoMCPError(RuntimeError):
    """A safe MCP diagnostic that can be shown without exposing credentials."""


class RovoMCPTimeout(RovoMCPError, TimeoutError):
    failure_status = "timeout"


class RovoMCPUnavailable(RovoMCPError):
    failure_status = "unavailable"


def _search_error(error: Exception) -> RovoMCPError:
    if isinstance(error, RovoMCPError):
        return error
    leaves = []

    def visit(exc):
        children = getattr(exc, "exceptions", ())
        if children:
            for child in children:
                visit(child)
        else:
            leaves.append(exc)

    visit(error)
    for exc in leaves:
        if isinstance(exc, RovoMCPError):
            return exc
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 401:
                return RovoMCPError(
                    "Atlassian MCP V2 rejected the saved API token/email (HTTP 401). "
                    "In Settings > Connections > Atlassian MCP V2 search, save a valid "
                    "MCP V2 API token and its owner's email. A2A OAuth does not authorize MCP search."
                )
            if status == 403:
                return RovoMCPError(
                    "Atlassian MCP V2 denied access (HTTP 403). Check the token's "
                    "search:rovo:agent-interface scope, site access, and organization API-token policy."
                )
            return RovoMCPError(f"Atlassian MCP V2 search failed (HTTP {status}). Try again shortly.")
    for exc in leaves:
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return RovoMCPTimeout("Atlassian MCP V2 search timed out. Try again shortly.")
        if isinstance(exc, httpx.RequestError):
            return RovoMCPUnavailable(
                "Could not connect to Atlassian MCP V2. Check network access and trusted TLS certificates."
            )
        if isinstance(exc, RuntimeError):
            labels = {"ROVO_CLOUD_ID": "Atlassian site (cloud ID or URL)",
                      "ROVO_EMAIL": "MCP token owner's email", "ROVO_API_TOKEN": "MCP V2 API token"}
            for name, label in labels.items():
                if str(exc) == f"{name} is required":
                    return RovoMCPUnavailable(f"Set {label} in Settings > Connections > Atlassian MCP V2 search.")
    return RovoMCPError(
        "Atlassian MCP V2 search failed. Check its site, API token, and search permissions."
    )


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

    async with streamablehttp_client(MCP_URL, headers=headers, timeout=remaining(45),
                                     httpx_client_factory=httpx_client_factory) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            remaining(45)
            listed = await session.list_tools()
            search_tool = next((tool for tool in listed.tools
                                if "search" in tool.name.lower()
                                and ("semantic" in (tool.description or "").lower()
                                     or tool.name.lower().endswith("_search")
                                     or tool.name.lower() == "search")), None)
            if search_tool is None:
                raise RovoMCPError(
                    "Atlassian MCP V2 did not expose Rovo semantic search. Check the "
                    "search:rovo:agent-interface scope and organization tool permissions."
                )
            remaining(45)
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
                raise RovoMCPError(
                    "Atlassian MCP V2's semantic search tool failed. Check its "
                    "search:rovo:agent-interface scope and access to the selected site."
                )
            return _issue_keys(result)


def semantic_issue_keys(query: str, limit: int) -> list[str]:
    """Return visible Jira issue keys in Rovo's semantic result order."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(_semantic_issue_keys(query, limit))
        except Exception as exc:
            raise _search_error(exc) from None
    raise RuntimeError("Rovo MCP search must run from the synchronous LangGraph workflow")
