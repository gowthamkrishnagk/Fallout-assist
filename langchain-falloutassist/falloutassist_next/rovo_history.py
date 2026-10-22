"""Scoped history for Rovo A2A previews only; no local retrieval or ingestion."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import re
import time

import httpx
import requests
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .config import config_value, required_env
from .history import extract_error_code, parse_failure
from .rovo import ISSUE_KEY, MCP_URL, RovoMCPError, _auth_header, _search_error

PROJECT = "SAC"
STATUSES = ("Resolved", "Closed")
SEARCH_LIMIT = 30
SOURCE_LIMIT = 5
CACHE_TTL_SECONDS = 300
CACHE_PREFIX = "rovo_a2a:history:"


def _cache_key(issue: dict) -> str:
    step, error = signature(issue)
    # Include all context that can change procedure applicability and site.
    from .config import config_value
    material = "\n".join((str(config_value("JIRA_BASE_URL") or "").casefold(),
                          str(config_value("JIRA_EMAIL") or "").casefold(),
                          str(config_value("ROVO_CLOUD_ID") or "").casefold(),
                          str(issue.get("order_type") or "").casefold(),
                          str(issue.get("order_reason") or "").casefold(),
                          str(issue.get("release") or "").casefold(),
                          _normalized_step(step), _normalized(stable_error(error))))
    return CACHE_PREFIX + hashlib.sha256(material.encode()).hexdigest()


def cached_sources(issue: dict) -> tuple[list[dict], dict] | None:
    """Return short-lived historical evidence for the same failure context."""
    from . import storage
    value = storage.integration_secret(_cache_key(issue))
    if not value:
        return None
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(value["cached_at"])).total_seconds()
        if age < 0 or age > CACHE_TTL_SECONDS:
            return None
        if not isinstance(value.get("sources"), list) or not value["sources"]:
            return None
        retrieval = dict(value.get("retrieval") or {})
        retrieval["cached_at"] = value["cached_at"]
        retrieval["cache_age_seconds"] = round(age)
        return value["sources"], retrieval
    except (KeyError, TypeError, ValueError):
        return None


def cache_sources(issue: dict, sources: list[dict], retrieval: dict) -> None:
    """Store a five-minute encrypted cache of validated historical evidence."""
    from . import storage
    storage.save_integration_secret(_cache_key(issue), {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "retrieval": retrieval,
    })


def signature(issue: dict) -> tuple[str, str]:
    parsed_step, parsed_error = parse_failure(issue.get("query_text", ""))
    return ((issue.get("step") or parsed_step).strip(),
            (issue.get("error") or parsed_error).strip())


def semantic_query(issue: dict) -> str:
    step, error = signature(issue)
    if not step and not error:
        return issue.get("query_text", "")[:900]
    # Ticket, account, and order IDs dominate the full description but do not
    # identify the recurring failure. Keep the operational signature instead.
    return f"Failed Step: {step[:400]}\nError Description: {stable_error(error)[:450]}"


def exact_jql(issue: dict) -> str:
    step, error = signature(issue)
    if not step or not error:
        return ""
    code = extract_error_code(error)
    searchable_error = stable_error(error)
    # JQL phrase escaping around punctuation-heavy Jira step names can return
    # zero results. Search the full step as ordinary text; strict step/error
    # validation happens after Jira hydrates each candidate. A short stable
    # error phrase improves recall when Jira interpolates per-ticket IDs.
    phrase = step[:150]
    error_term = code or " ".join(re.findall(r"[a-zA-Z0-9]+", searchable_error)[:6])
    return (f'project = {PROJECT} AND status in (Resolved, Closed) '
            f'AND key != {issue["key"]} AND summary ~ {json.dumps(phrase)} '
            f'AND text ~ {json.dumps(error_term)} ORDER BY resolutiondate DESC')


def issue_keys(result) -> list[str]:
    """Use issue records/links, never ticket-looking words in dashboard titles."""
    payload = getattr(result, "structuredContent", None)
    text = "\n".join(getattr(block, "text", "") or ""
                     for block in (getattr(result, "content", None) or []))
    if payload is None:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return list(dict.fromkeys(re.findall(
                r"/browse/([A-Z][A-Z0-9_]{1,29}-[1-9][0-9]{0,9})\b", text)))
    found = []

    def visit(value):
        if isinstance(value, dict):
            if str(value.get("type", "")).casefold() in {
                    "dashboard", "filter", "page", "space", "board", "project"}:
                return
            key = value.get("issueKey") or value.get("key")
            if isinstance(key, str) and ISSUE_KEY.fullmatch(key):
                found.append(key)
            url = value.get("url")
            if isinstance(url, str):
                found.extend(re.findall(r"/browse/([A-Z][A-Z0-9_]{1,29}-[1-9][0-9]{0,9})\b", url))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return list(dict.fromkeys(found))


async def _search(issue: dict) -> dict:
    cloud_id = required_env("ROVO_CLOUD_ID")
    query, jql = semantic_query(issue), exact_jql(issue)
    details = {"project": PROJECT, "statuses": list(STATUSES), "semantic_query": query,
               "exact_jql": jql, "search_limit": SEARCH_LIMIT, "warnings": []}
    semantic, exact = [], []

    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth,
                                 verify=config_value("TLS_CA_BUNDLE") or True)

    async with streamablehttp_client(MCP_URL, headers={"Authorization": _auth_header()},
                                     timeout=45, httpx_client_factory=factory) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            calls = []
            if jql:
                calls.append(("exact", "searchJiraIssuesUsingJql", {
                    "cloudId": cloud_id, "jql": jql, "maxResults": SEARCH_LIMIT,
                    "fields": ["summary", "status"], "view": "compact",
                }, "search:jira:agent-interface"))
            calls.append(("semantic", "search", {"cloudId": cloud_id, "query": query,
                          "targetApp": "JIRA", "mode": "STANDARD", "limit": SEARCH_LIMIT},
                          "search:rovo:agent-interface"))
            async def run(kind, tool, arguments, scope):
                try:
                    result = await session.call_tool(tool, arguments)
                    if getattr(result, "isError", False):
                        raise RovoMCPError(
                            f"Rovo MCP {kind} search tool failed. Check the {scope} scope and site access.")
                    return kind, issue_keys(result), None
                except Exception as exc:
                    return kind, [], str(_search_error(exc))

            results = await asyncio.gather(*(run(*call) for call in calls))
            for kind, keys, warning in results:
                if warning:
                    details["warnings"].append(warning)
                elif kind == "exact":
                    exact = keys
                else:
                    semantic = keys

    details.update(exact_candidates=len(exact), semantic_candidates=len(semantic))
    keys = list(dict.fromkeys(key for key in exact + semantic
                             if key.startswith(PROJECT + "-") and key != issue["key"]))
    if not keys and details["warnings"]:
        raise RovoMCPError(details["warnings"][0])
    return {"keys": keys, "retrieval": details}


def search(issue: dict, jira=None) -> dict:
    """Search with Rovo semantic matching plus Jira REST's scoped JQL index.

    Rovo MCP semantic search is valuable for recall, while Jira's own JQL index
    reliably finds exact step/error phrases. Merge both candidate lists before
    the same strict Jira hydration and applicability checks.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            result = asyncio.run(_search(issue))
        except Exception as exc:
            raise _search_error(exc) from None
        if jira is not None:
            jql = exact_jql(issue)
            try:
                rows = jira._search(jql, "status", SEARCH_LIMIT) if jql else []
                jira_keys = [row.get("key", "") for row in rows if isinstance(row, dict)]
                result["retrieval"]["jira_exact_candidates"] = len(jira_keys)
                result["keys"] = list(dict.fromkeys(jira_keys + result["keys"]))
            except requests.RequestException as exc:
                result["retrieval"].setdefault("warnings", []).append(
                    f"Jira exact history search unavailable: HTTP {getattr(getattr(exc, 'response', None), 'status_code', None)}")
        return result
    raise RovoMCPError("Rovo preview history must run in a synchronous worker.")


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def _normalized_step(value: str) -> str:
    # Jira sometimes appends this non-operational annotation to the step label.
    # Preserve operational qualifiers such as WITH/WITHOUT and Line Prepaid/PortIn.
    step = re.sub(r"\s*\(change plan\)\s*$", "", value or "", flags=re.IGNORECASE)
    return _normalized(step)


def stable_error(value: str) -> str:
    """Remove per-ticket identifiers while retaining the recurring failure text.

    Jira error descriptions often interpolate phone numbers and Salesforce
    record IDs. Those values make Rovo search and exact error comparison unique
    to one customer, even when the underlying failure is the same. Keep short
    numeric error codes and all surrounding wording intact.
    """
    text = value or ""
    text = re.sub(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", " ", text)
    text = re.sub(r"\b00[A-Za-z0-9]{13,16}\b", " ", text)  # Salesforce 15/18-char IDs
    text = re.sub(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)", " ", text)  # phone/line numbers
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return " ".join(text.split())


def select_sources(jira, issue: dict, keys: list[str], details: dict) -> list[dict]:
    """Validate scope and failure identity before selecting five usable comments."""
    candidates = list(dict.fromkeys(key for key in keys if ISSUE_KEY.fullmatch(key)
                                   and key.startswith(PROJECT + "-") and key != issue["key"]))
    rejected = {"scope": len(set(keys)) - len(candidates), "no_resolution": 0,
                "different_step": 0, "different_error": 0, "inaccessible": 0}
    details.update(project=PROJECT, statuses=list(STATUSES), rejected=rejected,
                   checked_candidates=0, selected_sources=0)
    if not candidates:
        return []
    scope_jql = (f'project = {PROJECT} AND status in (Resolved, Closed) AND key in ('
                 + ",".join(candidates) + ")")
    started = time.monotonic()
    try:
        rows = jira._search(scope_jql, "status", len(candidates))
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise RovoMCPError(f"Jira could not validate resolved/closed SAC sources (HTTP {status}).") from None
    eligible = {row["key"] for row in rows}
    details.setdefault("timings_ms", {})["scope_check"] = round((time.monotonic() - started) * 1000)
    rejected["scope"] += len(set(candidates) - eligible)
    step, error = signature(issue)
    code = extract_error_code(error)
    # Reuse the most recently validated historical set for five minutes.
    # Still check current Jira project/status scope before returning cached text.
    cached = cached_sources(issue) if details.get("mode") != "manual_keys" else None
    if cached:
        cached_items, cached_retrieval = cached
        valid_cached = [item for item in cached_items if item.get("key") in eligible]
        if valid_cached:
            for name, value in cached_retrieval.items():
                if name not in {"mode", "timings_ms", "cache", "cached_at"}:
                    details.setdefault(name, value)
            details["cache"] = "hit"
            details["checked_candidates"] = 0
            details["selected_sources"] = len(valid_cached)
            details["timings_ms"]["evidence_fetch"] = 0
            return valid_cached[:SOURCE_LIMIT]

    sources = []
    fetch_started = time.monotonic()
    # Fetch candidates in small concurrent batches, preserving search rank and
    # stopping once five usable resolution comments have been found.
    ordered = [key for key in candidates if key in eligible]
    for offset in range(0, len(ordered), 4):
        batch = ordered[offset:offset + 4]
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="rovo-history") as pool:
            futures = {pool.submit(jira.fetch_resolution_candidate, key): key for key in batch}
            fetched = {}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    fetched[key] = future.result()
                except requests.HTTPError as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status == 404:
                        fetched[key] = "__inaccessible__"
                        rejected["inaccessible"] += 1
                    else:
                        raise RovoMCPError(
                            f"Jira could not read historical source {key} (HTTP {status}).") from None
                except Exception:
                    raise RovoMCPError(
                        f"Jira could not read historical source {key}. Check the Jira connection and try again.") from None
        for key in batch:
            details["checked_candidates"] += 1
            source = fetched.get(key)
            if source == "__inaccessible__":
                continue
            if not source:
                rejected["no_resolution"] += 1
                continue
            if step and _normalized_step(source.get("step", "")) != _normalized_step(step):
                rejected["different_step"] += 1
                continue
            source_error = source.get("error", "")
            if ((code and extract_error_code(source_error) != code)
                    or (error and not code
                        and _normalized(stable_error(source_error)) != _normalized(stable_error(error)))):
                rejected["different_error"] += 1
                continue
            sources.append(source)
            if len(sources) == SOURCE_LIMIT:
                break
        if len(sources) == SOURCE_LIMIT:
            break
    details["timings_ms"]["evidence_fetch"] = round((time.monotonic() - fetch_started) * 1000)
    details["selected_sources"] = len(sources)
    details["cache"] = "miss"
    if sources and details.get("mode") != "manual_keys":
        cache_sources(issue, sources, details)
    return sources
