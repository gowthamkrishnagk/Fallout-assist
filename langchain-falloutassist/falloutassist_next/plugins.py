"""System plugin contract. Jira is the sole installed plugin in this phase."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from time import monotonic
from datetime import datetime, timedelta, timezone
from dataclasses import asdict, dataclass
from typing import Callable, Protocol
from urllib.parse import quote

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import requests

from . import storage
from .config import config_value, required_env
from .history import EMBEDDING_INDEX_VERSION, HistoryIndex, parse_failure, field
from . import retrieval_pool
from .execution import remaining
from .quality import (BOT_COMMENT_MARKERS, SOURCE_POLICY_VERSION, source_quality,
                      is_automatic_closure as _is_automatic_closure,
                      is_salesforce_update_comment as _is_salesforce_update_comment,
                      resolution_text as _resolution_text,
                      resolution_is_placeholder as _resolution_is_placeholder)
from .rulematch import match_approved

MARKER = BOT_COMMENT_MARKERS[0]
ISSUE_KEY = re.compile(r"[A-Z][A-Z0-9_]{1,29}-[1-9][0-9]{0,9}")


@dataclass(frozen=True)
class PluginManifest:
    name: str
    description: str
    read_only: bool
    capabilities: tuple[str, ...]


class SystemPlugin(Protocol):
    manifest: PluginManifest


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, SystemPlugin] = {}

    def register(self, plugin: SystemPlugin) -> None:
        name = plugin.manifest.name
        if name in self._plugins:
            raise ValueError(f"Duplicate plugin: {name}")
        self._plugins[name] = plugin

    def get(self, name: str) -> SystemPlugin:
        return self._plugins[name]

    def catalog(self) -> list[dict]:
        return [asdict(plugin.manifest) for plugin in self._plugins.values()]


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_text(part) for part in value.get("content", [])) or str(value.get("text", ""))
    if isinstance(value, list):
        return " ".join(_text(part) for part in value)
    return ""


def _timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (ValueError, TypeError):
        return None


def _select_resolution_comment(fields: dict, comments: list[dict],
                               step: str, error: str) -> dict:
    """One selection policy for owned ingestion and freshly hydrated Rovo cases."""
    assignee = (fields.get("assignee") or {}).get("accountId")
    status = str((fields.get("status") or {}).get("name") or "").strip().casefold()
    if "cancel" in status:
        return {"body": "", "workaround_text": "", "tier": "low",
                "reasons": ["Cancelled tickets are excluded from resolution evidence"],
                "created": "", "resolution_at": "", "comment_id": "",
                "selection_reason": "excluded_cancelled_ticket"}
    resolved = _timestamp(fields.get("resolutiondate") or "")
    choices = []
    for comment in comments:
        body = _text(comment.get("body")).strip()
        if (any(marker in body for marker in BOT_COMMENT_MARKERS)
                or _is_automatic_closure(body) or _is_salesforce_update_comment(body)):
            continue
        created_raw = str(comment.get("created") or "")
        # Apply the post-resolution age limit to edits as well as new comments.
        effective_raw = str(comment.get("updated") or created_raw)
        effective = _timestamp(effective_raw)
        if resolved and effective and effective > resolved + timedelta(days=14):
            continue
        # Quality and eventual quoting must refer to the same bounded evidence.
        if len(body) > 12000:
            continue
        tier, reasons = source_quality("ticket", {"step": step, "error": error}, body)
        selected = {"body": body, "workaround_text": _resolution_text(body),
                    "tier": tier, "reasons": reasons, "created": created_raw,
                    "resolution_at": effective_raw, "comment_id": str(comment.get("id") or "")}
        author = (comment.get("author") or {}).get("accountId")
        structured = _resolution_text(body) != body
        authored_by_assignee = bool(assignee and author == assignee)
        rank = ({"high": 2, "medium": 1, "low": 0}[tier],
                effective or datetime.min.replace(tzinfo=timezone.utc),
                int(authored_by_assignee), int(structured), selected["comment_id"])
        choices.append((rank, selected, structured, authored_by_assignee))
    if choices:
        # Prefer evidence quality first, then the newest comment at that tier.
        # This lets a newer strong procedure supersede an older strong one,
        # including when the assignee recorded it as free text. Authorship and
        # a labelled section break ties only. The fallback still passes the
        # same corrective-action quality checks.
        usable = [choice for choice in choices if choice[1]["tier"] != "low"]
        pool = usable or choices
        winner = max(pool, key=lambda choice: choice[0])
        selected = winner[1]
        reason = ("assignee_authored_fallback" if winner[3] and not winner[2]
                  else "structured_resolution" if winner[2]
                  else "comment_body_fallback" if usable else "low_quality_comment")
        selected["selection_reason"] = reason
        if reason == "assignee_authored_fallback":
            selected["reasons"] = [*selected["reasons"],
                                   "Assignee-authored free-form resolution used as fallback"]
        return selected
    return {"body": "", "workaround_text": "", "tier": "low",
            "reasons": ["No usable resolution comment"], "created": "",
            "resolution_at": "", "comment_id": "", "selection_reason": "none"}


def _resolution_context(query: str, body: str) -> dict:
    def value(*names: str) -> str:
        return field(body, *names) or field(query, *names)
    return {"order_reason": value("Order Reason", "OrderReason", "Reason"),
            "order_type": value("Order Type Failed", "Order Type", "OrderType"),
            "release": value("Release", "Release Version", "Software Release")}


def _query(issue: dict) -> tuple[str, str, str]:
    fields = issue.get("fields") or {}
    summary = _text(fields.get("summary"))
    description = _text(fields.get("description"))
    text = f"Summary: {summary}\n{description}"[:8000]
    step, error = parse_failure(text)
    # SAC BOT records the operational error in a comment field rather than in
    # the issue description. Search results already include inline comments;
    # issue lookups request them below as well.
    if not error:
        comments = (fields.get("comment") or {}).get("comments") or []
        for comment in reversed(comments):
            author = ((comment.get("author") or {}).get("displayName") or "").strip().lower()
            if author != "sac bot":
                continue
            body = _text(comment.get("body"))
            for line in body.splitlines():
                if "PR_Error_Description" not in line:
                    continue
                _, separator, value = line.partition(":")
                if separator and value.strip():
                    error = value.strip()[:500]
                    break
            if error:
                break
    # Add canonical labels so both local and Rovo retrieval receive the parsed
    # failure signature, and the downstream quality gate can compare it.
    canonical = [text]
    if step:
        canonical.append(f"Failed Step: {step}")
    if error:
        canonical.append(f"Error Description: {error}")
    return "\n".join(canonical)[:9000], step, error


class JiraPlugin:
    manifest = PluginManifest(
        "jira", "Jira ticket, owned history, approved rules, feedback and comments", False,
        ("fetch_issue", "search_history", "approved_rule", "ingest", "publish_comment"),
    )

    def __init__(self) -> None:
        self._local = threading.local()

    def _client(self) -> requests.Session:
        auth = (required_env("JIRA_EMAIL"), required_env("JIRA_API_TOKEN"))
        ca_bundle = config_value("TLS_CA_BUNDLE") or True
        if (not getattr(self._local, "session", None)
            or self._local.session.auth != auth
            or self._local.session.verify != ca_bundle):
            session = requests.Session()
            session.auth = auth
            session.verify = ca_bundle
            session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
            self._local.session = session
        return self._local.session

    @staticmethod
    def base_url() -> str:
        value = required_env("JIRA_BASE_URL").rstrip("/")
        if not value.startswith("https://"):
            raise RuntimeError("JIRA_BASE_URL must use HTTPS")
        return value

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self._client().request(
            method, self.base_url() + path, timeout=remaining(45), **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    def issue(self, key: str) -> tuple[dict, dict]:
        key = key.strip().upper()
        if not ISSUE_KEY.fullmatch(key):
            raise ValueError("Invalid Jira issue key")
        raw = self._request("GET", f"/rest/api/2/issue/{quote(key)}",
                            params={"fields": "summary,description,status,updated,assignee,comment"})
        fields = raw.setdefault("fields", {})
        comment_data = fields.get("comment") or {}
        inline_comments = comment_data.get("comments") or []
        if (not _query(raw)[2] and isinstance(comment_data.get("total"), int)
                and comment_data["total"] > len(inline_comments)):
            fields["comment"] = {"comments": self._comments(key)}
        query_text, step, error = _query(raw)
        return raw, {
            "key": raw["key"], "summary": _text(fields.get("summary")),
            "status": (fields.get("status") or {}).get("name", ""),
            "updated": fields.get("updated") or "", "query_text": query_text,
            "step": step, "error": error,
            "order_reason": field(query_text, "Order Reason", "OrderReason", "Reason"),
            "order_type": field(query_text, "Order Type", "OrderType"),
            "release": field(query_text, "Release", "Release Version", "Software Release"),
            "url": f"{self.base_url()}/browse/{raw['key']}",
        }

    def _iter_search(self, jql: str, fields: str, limit: int = 0):
        token = ""
        yielded = 0
        while True:
            params = {"jql": jql, "maxResults": min(100, limit - yielded) if limit else 100,
                      "fields": fields}
            if token:
                params["nextPageToken"] = token
            data = self._request("GET", "/rest/api/2/search/jql", params=params)
            batch = data.get("issues") or []
            for issue in batch:
                yield issue
                yielded += 1
                if limit and yielded >= limit:
                    return
            token = data.get("nextPageToken") or ""
            if not token or not batch:
                break

    def _search(self, jql: str, fields: str, limit: int = 0) -> list[dict]:
        return list(self._iter_search(jql, fields, limit))

    def test_jql(self, jql: str) -> dict:
        """Run a small, read-only Jira search to validate a query from Settings."""
        data = self._request("GET", "/rest/api/2/search/jql", params={
            "jql": jql, "maxResults": 5, "fields": "key"})
        issues = data.get("issues") or []
        total = data.get("total")
        return {"total": total if isinstance(total, int) else None,
                "sample_keys": [str(issue.get("key") or "") for issue in issues
                                if issue.get("key")],
                "has_more": bool(data.get("nextPageToken"))}

    def inflow_issues(self, limit: int = 0) -> list[dict]:
        jql = required_env("JIRA_INFLOW_JQL")
        field_ids = [config_value("JIRA_FEEDBACK_FIELD_ID"),
                     config_value("JIRA_SUGGESTED_WORKAROUND_FIELD_ID")]
        for field_id in field_ids:
            if field_id and not re.fullmatch(r"customfield_\d+", field_id):
                raise ValueError(f"Invalid Jira custom field ID: {field_id}")
        fields = ",".join(["summary", "updated", "status", *[f for f in field_ids if f]])
        issues = self._search(jql, fields, limit)
        def value(raw: object) -> str:
            if isinstance(raw, dict):
                return str(raw.get("value") or raw.get("name") or "")
            return _text(raw)
        return [{"key": issue["key"],
                 "updated": (issue.get("fields") or {}).get("updated") or "",
                 "feedback_value": value((issue.get("fields") or {}).get(field_ids[0]))
                    if field_ids[0] else "",
                 "suggested_workaround": value((issue.get("fields") or {}).get(field_ids[1]))
                    if field_ids[1] else ""}
                for issue in issues]

    def _comments(self, key: str) -> list[dict]:
        comments: list[dict] = []
        start = 0
        while True:
            result = self._request("GET", f"/rest/api/2/issue/{quote(key)}/comment",
                                   params={"startAt": start, "maxResults": 100})
            batch = result.get("comments") or []
            comments.extend(batch)
            start += len(batch)
            if not batch or start >= result.get("total", start):
                return comments

    def existing_comment(self, key: str) -> dict | None:
        for comment in self._comments(key):
            if MARKER in _text(comment.get("body")):
                return {"id": str(comment["id"]), "body": _text(comment.get("body"))}
        return None

    def fetch_resolution_candidate(self, key: str) -> dict | None:
        """Fetch the Rovo-selected Jira issue and its resolution evidence through Jira read APIs."""
        if not ISSUE_KEY.fullmatch(key):
            return None
        raw = self._request("GET", f"/rest/api/2/issue/{quote(key)}", params={
            "fields": "summary,description,status,resolution,updated,assignee,resolutiondate,comment"})
        fields = raw.get("fields") or {}
        inline = fields.get("comment") or {}
        comments = inline.get("comments") or []
        if not isinstance(inline.get("total"), int) or inline["total"] > len(comments):
            comments = self._comments(key)
        query_issue = {**raw, "fields": {**fields, "comment": {"comments": comments}}}
        query_text, step, error = _query(query_issue)
        selected = _select_resolution_comment(fields, comments, step, error)
        if selected["tier"] == "low":
            return None
        body = selected["body"]
        return {"type": "ticket", "key": key, "summary": _text(fields.get("summary")),
                "url": f"{self.base_url()}/browse/{key}", "comment": body,
                "workaround_text": selected["workaround_text"], "step": step, "error": error,
                **_resolution_context(query_text, body),
                "resolution_at": selected["resolution_at"], "source_tier": selected["tier"],
                "resolution_selection": selected.get("selection_reason", ""),
                "source_reasons": selected["reasons"], "retrieval_origins": ["rovo"],
                "score": 0.0, "retrieval_score": 0.0, "rank_score": 0.0,
                "workaround_version": storage.current_workaround_version(key),
                "feedback_net": storage.feedback_net(
                    key, step, error,
                    (storage.current_workaround_version(key) or {}).get("id", ""))}

    def publish_comment(self, issue_key: str, body: str, known_comment_id: str = "") -> str:
        existing = self.existing_comment(issue_key)
        if existing:
            self._request("PUT", f"/rest/api/2/issue/{quote(issue_key)}/comment/{quote(existing['id'])}",
                          json={"body": body})
            return existing["id"]
        data = self._request("POST", f"/rest/api/2/issue/{quote(issue_key)}/comment",
                             json={"body": body})
        return str(data["id"])

    def search_local_history(self, query_text: str, top_k: int, min_score: float,
                             exclude_keys: set[str]) -> dict:
        return self.search_history(query_text, top_k, min_score, exclude_keys, local_only=True)

    def search_rovo_candidates(self, query_text: str, top_k: int,
                               exclude_keys: set[str]) -> dict:
        """MCP-only adapter; graph edges own escalation decisions."""
        from .rovo import semantic_issue_keys
        keys = list(dict.fromkeys(key for key in semantic_issue_keys(query_text, min(30, top_k * 2))
                                  if ISSUE_KEY.fullmatch(key) and key not in exclude_keys))[:30]
        scoped = [key for key in keys if key.startswith("SAC-")]
        if scoped:
            rows = self._search('project = SAC AND status in (Resolved, Closed) AND key in ('
                                + ','.join(scoped) + ')', 'status', len(scoped))
            allowed = {row['key'] for row in rows}
            scoped = [key for key in scoped if key in allowed]
        rejected = set(keys) - set(scoped)
        candidates, failures = [], []
        queue, pending = deque(scoped), {}
        while queue or pending:
            while queue and len(pending) < 8:
                remaining(45)
                try:
                    future = retrieval_pool.submit(self.fetch_resolution_candidate, queue[0])
                except retrieval_pool.RetrievalBusy:
                    if not pending:
                        failures.append("unavailable")
                        queue.clear()
                    break
                pending[future] = queue.popleft()
            if not pending:
                break
            completed, _ = wait(pending, timeout=remaining(45), return_when=FIRST_COMPLETED)
            if not completed:
                for future in pending:
                    future.cancel()
                failures.append("timeout")
                break
            for future in completed:
                key = pending.pop(future)
                try:
                    candidate = future.result()
                    if candidate:
                        candidate["retrieval_origins"] = ["rovo"]
                        candidate["rovo_rank"] = keys.index(key) + 1
                        candidates.append(candidate)
                    else:
                        rejected.add(key)
                except Exception as exc:
                    from .retrieval_routing import failure_status
                    failures.append(failure_status(exc))
        candidates.sort(key=lambda item: item["rovo_rank"])
        return {"candidates": candidates[:top_k], "rejected_keys": sorted(rejected),
                "status": "partial" if failures and candidates else failures[0] if failures else "success"}

    def search_history(self, query_text: str, retrieval_top_k: int, min_score: float,
                       exclude_keys: set[str] | None = None,
                       local_only: bool = False) -> dict:
        excluded = exclude_keys or set()
        backend = config_value("JIRA_HISTORY_BACKEND").lower()
        if local_only or backend == "local":
            local = HistoryIndex().search(query_text, retrieval_top_k, min_score, excluded)
            for candidate in local["candidates"]:
                candidate["retrieval_origins"] = ["local"]
            local["strong"] = [c for c in local["candidates"] if c["score"] >= min_score]
            local["context"] = [c for c in local["candidates"] if c["score"] < min_score]
            local["best_score"] = local["candidates"][0]["score"] if local["candidates"] else 0.0
            local["backend"] = "local"
            return local
        if backend != "rovo":
            raise ValueError("JIRA_HISTORY_BACKEND must be 'rovo' or 'local'")

        from .rovo import semantic_issue_keys
        # Submit remote discovery first, then run local search concurrently. Do
        # not use an executor context manager here: its exit waits for timed-out
        # work and would silently defeat the local-fallback deadline.
        started = monotonic()
        budget = storage.settings()["rovo_retrieval_budget_seconds"]
        deadline = started + budget
        local = rovo_keys = rovo_future = None
        failures = {}
        try:
            rovo_future = retrieval_pool.submit(
                semantic_issue_keys, query_text, max(10, retrieval_top_k * 4))
        except Exception as exc:
            failures["rovo"] = type(exc).__name__
        try:
            local = HistoryIndex().search(query_text, 100, min_score, excluded)
        except Exception as exc:
            failures["local"] = type(exc).__name__
        if rovo_future is not None:
            try:
                rovo_keys = rovo_future.result(timeout=max(0, deadline - monotonic()))
            except Exception as exc:
                rovo_future.cancel()
                failures["rovo"] = type(exc).__name__

        if local is None and rovo_keys is None:
            failure_types = ", ".join(f"{name}={kind}" for name, kind in failures.items())
            raise RuntimeError(f"Both history retrieval sources failed ({failure_types})")
        local_available = local is not None
        rovo_available = rovo_keys is not None
        if local is None:
            step, error = parse_failure(query_text)
            local = {"candidates": [], "strong": [], "context": [], "best_score": 0.0,
                     "query_step": step, "query_error": error, "threshold": min_score}
        if rovo_keys is None:
            rovo_keys = []

        by_key = {item["key"]: item for item in local["candidates"]}
        for item in by_key.values():
            item["retrieval_origins"] = ["local"]
        local_rank = {item["key"]: rank for rank, item in enumerate(local["candidates"], 1)}
        rovo_rank = {key: rank for rank, key in enumerate(rovo_keys, 1)}
        candidates_by_key = dict(by_key)
        rovo_candidates: dict[str, dict] = {}
        hydrated_keys: set[str] = set()
        fetch_failures: dict[str, int] = {}
        fetch_keys = [key for key in rovo_keys[:min(30, max(10, retrieval_top_k * 2))]
                      if key not in excluded]
        fetch_futures, queue = {}, deque(fetch_keys)
        while (queue or fetch_futures) and monotonic() < deadline:
            while queue and len(fetch_futures) < 8 and monotonic() < deadline:
                try:
                    future = retrieval_pool.submit(self.fetch_resolution_candidate, queue[0])
                except retrieval_pool.RetrievalBusy:
                    if not fetch_futures:
                        fetch_failures["RetrievalBusy"] = len(queue)
                        queue.clear()
                    break
                except Exception as exc:
                    fetch_failures[type(exc).__name__] = fetch_failures.get(type(exc).__name__, 0) + 1
                    queue.popleft()
                    continue
                fetch_futures[future] = queue.popleft()
            if not fetch_futures:
                break
            completed, _ = wait(fetch_futures, timeout=max(0, deadline - monotonic()),
                                return_when=FIRST_COMPLETED)
            if not completed:
                break
            for future in completed:
                key = fetch_futures.pop(future)
                try:
                    candidate = future.result()
                    hydrated_keys.add(key)
                except Exception as exc:
                    fetch_failures[type(exc).__name__] = fetch_failures.get(type(exc).__name__, 0) + 1
                    candidate = None
                if candidate:
                    rovo_candidates[key] = candidate
        if fetch_futures or queue:
            for future in fetch_futures:
                future.cancel()
            fetch_failures["TimeoutError"] = len(fetch_futures) + len(queue)
        if fetch_failures:
            failures["rovo_fetch"] = ",".join(
                f"{name}:{count}" for name, count in sorted(fetch_failures.items()))
        for key in rovo_keys:
            if key in excluded:
                continue
            fetched = rovo_candidates.get(key)
            row = storage.source_document(key)
            if key in hydrated_keys and fetched is None:
                # A successful fetch rejected the evidence; an old local copy
                # must not reintroduce it. Network failures still allow fallback.
                candidates_by_key.pop(key, None)
                continue
            if key not in candidates_by_key and fetched:
                candidates_by_key[key] = fetched
            elif key in candidates_by_key:
                candidate = candidates_by_key[key]
                candidate["retrieval_origins"] = ["local", "rovo"]
                if fetched:
                    candidate.update({k: v for k, v in fetched.items()
                                      if k not in {"score", "retrieval_score", "rank_score", "feedback_net"}})
                candidate["retrieval_origins"] = ["local", "rovo"]
            elif not row or storage.source_tier(row["source_type"], key) == "low":
                continue
            elif key not in candidates_by_key:
                current_version = storage.current_workaround_version(key)
                candidates_by_key[key] = {
                    "type": row["source_type"], "key": key, "summary": row["summary"],
                    "url": row["url"], "comment": row["body"], "step": row["step"],
                    "error": row["error"], "score": 0.0, "retrieval_score": 0.0,
                    "retrieval_origins": ["rovo"],
                    "order_reason": (row.get("metadata") or {}).get("order_reason", ""),
                    "order_type": (row.get("metadata") or {}).get("order_type", ""),
                    "release": (row.get("metadata") or {}).get("release", ""),
                    "resolution_at": (row.get("metadata") or {}).get("resolution_at", ""),
                    "workaround_version": current_version,
                    "rank_score": 0.0, "feedback_net": storage.feedback_net(
                        key, local["query_step"], local["query_error"],
                        (current_version or {}).get("id", "")),
                }

        # Reciprocal rank fusion combines the independent rank lists without
        # assuming their score scales are comparable. Keep vector similarity
        # separately for the existing evidence quality gate.
        rrf_scores: dict[str, float] = {}
        for ranks in (local_rank, rovo_rank):
            for key, rank in ranks.items():
                if key in candidates_by_key:
                    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
        max_rrf = max(rrf_scores.values(), default=1.0)
        candidates = []
        for key, score in rrf_scores.items():
            candidate = candidates_by_key[key]
            candidate["retrieval_score"] = score / max_rrf
            candidate["rank_score"] = candidate["retrieval_score"]
            if key in local_rank:
                candidate["local_rank"] = local_rank[key]
            if key in rovo_rank:
                candidate["rovo_rank"] = rovo_rank[key]
            candidates.append(candidate)

        local_candidates = [candidates_by_key[key] for key in local_rank
                            if key in candidates_by_key]
        branch_rovo_candidates = [candidates_by_key[key] for key in rovo_rank
                                  if key in candidates_by_key]

        candidates.sort(key=lambda item: (item["retrieval_score"], item["score"]), reverse=True)
        candidates = candidates[:retrieval_top_k]
        return {
            **local,
            "candidates": candidates,
            "strong": [c for c in candidates if c["score"] >= min_score],
            "context": [c for c in candidates if c["score"] < min_score],
            "best_score": candidates[0]["score"] if candidates else 0.0,
            "backend": ("hybrid" if local_available and rovo_available
                        else "local" if local_available else "rovo"),
            "rovo_result_count": len(rovo_keys),
            "local_result_count": len(local_rank),
            "local_available": local_available,
            "rovo_available": rovo_available,
            "retrieval_failures": failures,
            "local_candidates": local_candidates[:retrieval_top_k],
            "rovo_candidates": branch_rovo_candidates[:retrieval_top_k],
            "rovo_fetched_count": len(rovo_candidates),
            "retrieval_elapsed_ms": round((monotonic() - started) * 1000),
            "rovo_budget_seconds": budget,
        }

    def approved_rule(self, query_text: str) -> dict | None:
        return match_approved(query_text)

    def ingest(self, heartbeat: Callable[[], bool] | None = None,
               actor: str = "worker", full: bool = False) -> dict:
        jql = required_env("JIRA_INGEST_JQL")
        # Jira includes the first page of comments with each search result. Use
        # those comments for the common case so a large history ingest does not
        # make one extra HTTP request per issue.
        previous_keys = set(storage.all_document_keys("ticket"))
        index = HistoryIndex()
        refresh_vectors = storage.get_state("embedding_index_version") != EMBEDDING_INDEX_VERSION
        if full:
            index.reset()
        if full or refresh_vectors:
            from .documents import restore_document_index
            restore_document_index(index)
        changed = skipped = 0
        new_documents = updated_documents = 0
        reindexed = 0
        seen: set[str] = set()
        pending_upserts: list[dict] = []
        pending_deletes: list[str] = []
        fetched = 0

        def flush_index() -> None:
            index.upsert_many(pending_upserts)
            index.delete_many(pending_deletes)
            pending_upserts.clear()
            pending_deletes.clear()

        for issue in self._iter_search(
                jql, "summary,description,status,resolution,updated,assignee,resolutiondate,comment"):
            if heartbeat and not heartbeat():
                raise RuntimeError("Jira ingestion lease was lost")
            fetched += 1
            key = issue["key"]
            seen.add(key)
            fields = issue.get("fields") or {}
            updated = fields.get("updated") or ""
            prior = storage.source_document(key)
            policy_current = bool(prior and (prior.get("metadata") or {}).get(
                "source_policy_version") == SOURCE_POLICY_VERSION)
            if prior and prior["issue_updated"] == updated and not full and policy_current and not refresh_vectors:
                skipped += 1
                continue
            inline = fields.get("comment") or {}
            inline_comments = inline.get("comments") or []
            total_comments = inline.get("total")
            if isinstance(total_comments, int) and total_comments <= len(inline_comments):
                comments = inline_comments
            else:
                comments = self._comments(key)
            query_issue = {**issue, "fields": {**fields, "comment": {"comments": comments}}}
            query, step, error = _query(query_issue)
            selected = _select_resolution_comment(fields, comments, step, error)
            body, workaround_text = selected["body"], selected["workaround_text"]
            tier, reasons = selected["tier"], selected["reasons"]
            resolved_raw = fields.get("resolutiondate") or ""
            selected_created = selected["resolution_at"]
            context = _resolution_context(query, body)
            order_reason, order_type, release = (context[name] for name in
                                                 ("order_reason", "order_type", "release"))
            item = {
                "source_key": key, "source_type": "ticket", "issue_updated": updated,
                "summary": _text(fields.get("summary")), "step": step, "error": error,
                "body": body[:12000], "url": f"{self.base_url()}/browse/{key}",
                "order_reason": order_reason,
                "workaround_text": workaround_text,
                "metadata": {"order_reason": order_reason, "order_type": order_type,
                             "release": release, "resolution_at": selected_created or resolved_raw,
                             "source_policy_version": SOURCE_POLICY_VERSION,
                             "resolution_comment_id": selected["comment_id"],
                             "resolution_selection": selected.get("selection_reason", "")},
                "content_hash": hashlib.sha256((query + body + order_reason + order_type + release +
                                                 (selected_created or resolved_raw) + workaround_text +
                                                 SOURCE_POLICY_VERSION).encode()).hexdigest(),
            }
            if (not full and not refresh_vectors and policy_current and prior and prior["content_hash"] == item["content_hash"]
                and storage.source_tier("ticket", key) == tier):
                storage.upsert_document(item, tier, reasons)
                skipped += 1
                continue
            # Only usable resolutions enter semantic retrieval. Low quality is kept
            # in SQLite for the dashboard and audit, never used to draft a fix.
            if tier == "low":
                if prior:
                    pending_deletes.append(key)
            else:
                pending_upserts.append(item)
                reindexed += int(full)
            if storage.upsert_document(item, tier, reasons):
                changed += 1
                if prior is None:
                    new_documents += 1
                else:
                    updated_documents += 1
            if tier in ("high", "medium"):
                version = storage.record_workaround(item, selected_created or resolved_raw)
                storage.remove_workaround_evidence(key, (version or {}).get("id", ""),
                                                   preserve_history=True)
            else:
                # A full refresh can downgrade previously accepted evidence. Remove
                # that source from old workaround versions so stale bot logs cannot
                # remain published as the current workaround.
                storage.remove_workaround_evidence(key)
            if len(pending_upserts) + len(pending_deletes) >= 128:
                flush_index()
        flush_index()
        # Full JQL scan is authoritative; tickets no longer matching are removed.
        missing_keys = previous_keys - seen
        if missing_keys and (not seen or len(missing_keys) > max(10, len(previous_keys) // 10)):
            if not storage.settings()["allow_mass_prune"]:
                raise RuntimeError("Ingestion JQL would remove many historical sources; review it before enabling mass prune in Settings")
        for key in missing_keys:
            index.delete(key)
            storage.remove_document(key)
            changed += 1
        storage.set_state("embedding_index_version", EMBEDDING_INDEX_VERSION)
        if changed or refresh_vectors or full:
            storage.set_state("corpus_version", storage.utc_now())
        result = {"ok": True, "full_rebuild": full, "fetched": fetched,
                  "indexed": reindexed if full else changed,
                  "changed_documents": changed, "skipped": skipped,
                  "new_documents": new_documents, "updated_documents": updated_documents,
                  "removed_documents": len(missing_keys),
                  "quality_counts": storage.source_counts()}
        storage.audit(actor, "ingest_completed", "jira", result)
        return result


registry = PluginRegistry()
registry.register(JiraPlugin())
