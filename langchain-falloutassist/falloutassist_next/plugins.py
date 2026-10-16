"""System plugin contract. Jira is the sole installed plugin in this phase."""

from __future__ import annotations

import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
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
from .history import HistoryIndex, parse_failure, field
from .quality import source_quality
from .rulematch import match_approved

MARKER = "FA-LANGCHAIN-8012"
BOT_COMMENT_MARKERS = (MARKER, "FA-LEGACY-8010", "FA-SUGGESTION")
ISSUE_KEY = re.compile(r"[A-Z][A-Z0-9_]{1,29}-[1-9][0-9]{0,9}")
_AUTO_CLOSURE = re.compile(
    r"(?is)(?:resolved for more than\s+\d+\s+days?\s+with no response|"
    r"will now be automatically closed|automatically closed due to inactivity)"
)


def _is_automatic_closure(body: str, author_name: str = "", account_type: str = "") -> bool:
    return bool(_AUTO_CLOSURE.search(body or ""))


def _is_salesforce_update_comment(body: str) -> bool:
    """Exclude Salesforce state snapshots and execution logs from Jira fixes."""
    value = (body or "").casefold()
    bot_attribution = "updated by the lcpr order fallout tickets bot" in value
    salesforce_snapshot = "update from salesforce" in value and any(
        marker in value for marker in (
            "orderstepid", "order step link", "pr_error_description__c",
            "lastmodifieddate", "execution log",
        )
    )
    return bot_attribution or salesforce_snapshot


def _resolution_text(body: str) -> str:
    value = field(body, "Solution Applied", "Resolution", "Workaround", "Fix")
    return value if value else body


def _resolution_is_placeholder(value: str) -> bool:
    return value.strip().casefold() in {
        "na", "n/a", "none", "completed the step", "completed step",
        "retry the step", "retried the step", "done", "fixed", "resolved",
    }


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
            method, self.base_url() + path, timeout=45, **kwargs)
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
        choices = []
        assignee_id = (fields.get("assignee") or {}).get("accountId")
        for comment in comments:
            body = _text(comment.get("body")).strip()
            author = comment.get("author") or {}
            if any(marker in body for marker in BOT_COMMENT_MARKERS) or _is_automatic_closure(
                    body, str(author.get("displayName") or "").casefold(),
                    str(author.get("accountType") or "")):
                continue
            solution = _resolution_text(body)
            if _resolution_is_placeholder(solution):
                continue
            tier, reasons = source_quality("ticket", {"step": step, "error": error}, solution)
            if tier == "low":
                continue
            choices.append(({"high": 2, "medium": 1}[tier],
                            int(bool(assignee_id and author.get("accountId") == assignee_id)),
                            str(comment.get("updated") or comment.get("created") or ""),
                            body, tier, reasons))
        if not choices:
            return None
        _, _, resolved_at, body, tier, reasons = max(choices)
        resolution = _resolution_text(body)
        return {"type": "ticket", "key": key, "summary": _text(fields.get("summary")),
                "url": f"{self.base_url()}/browse/{key}", "comment": body[:12000],
                "workaround_text": resolution[:5000], "step": step, "error": error,
                "order_reason": field(query_text, "Order Reason", "OrderReason", "Reason"),
                "order_type": field(query_text, "Order Type", "OrderType"),
                "release": field(query_text, "Release", "Release Version", "Software Release"),
                "resolution_at": resolved_at, "source_tier": tier,
                "source_reasons": reasons, "retrieval_origins": ["rovo"],
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

    def search_history(self, query_text: str, retrieval_top_k: int, min_score: float,
                       exclude_keys: set[str] | None = None) -> dict:
        excluded = exclude_keys or set()
        backend = config_value("JIRA_HISTORY_BACKEND").lower()
        if backend == "local":
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
        # Run the independent retrievers concurrently, then combine their ranks.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="falloutassist-retrieval") as pool:
            local_future = pool.submit(
                HistoryIndex().search, query_text, 100, min_score, excluded)
            rovo_future = pool.submit(
                semantic_issue_keys, query_text, max(10, retrieval_top_k * 4))
            local = None
            rovo_keys = None
            failures = {}
            try:
                local = local_future.result()
            except Exception as exc:
                failures["local"] = type(exc).__name__
            try:
                rovo_keys = rovo_future.result()
            except Exception as exc:
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
        fetch_failures: dict[str, int] = {}
        fetch_keys = [key for key in rovo_keys[:min(30, max(10, retrieval_top_k * 2))]
                      if key not in excluded]
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="falloutassist-rovo-fetch") as pool:
            fetch_futures = {pool.submit(self.fetch_resolution_candidate, key): key
                             for key in fetch_keys}
            for future in fetch_futures:
                key = fetch_futures[future]
                try:
                    candidate = future.result()
                except Exception as exc:
                    fetch_failures[type(exc).__name__] = fetch_failures.get(type(exc).__name__, 0) + 1
                    candidate = None
                if candidate:
                    rovo_candidates[key] = candidate
        if fetch_failures:
            failures["rovo_fetch"] = ",".join(
                f"{name}:{count}" for name, count in sorted(fetch_failures.items()))
        for key in rovo_keys:
            if key in excluded:
                continue
            fetched = rovo_candidates.get(key)
            row = storage.source_document(key)
            if key not in candidates_by_key and fetched:
                candidates_by_key[key] = fetched
            elif key in candidates_by_key:
                candidate = candidates_by_key[key]
                candidate["retrieval_origins"] = ["local", "rovo"]
                if fetched and fetched.get("resolution_at", "") > (row.get("metadata", {}).get("resolution_at", "") if row else ""):
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
        }

    def approved_rule(self, query_text: str) -> dict | None:
        return match_approved(query_text)

    def ingest(self, heartbeat: Callable[[], bool] | None = None,
               actor: str = "worker", full: bool = False) -> dict:
        jql = required_env("JIRA_INGEST_JQL")
        # Jira includes the first page of comments with each search result. Use
        # those comments for the common case so a large history ingest does not
        # make one extra HTTP request per issue.
        previous_keys = set(storage.all_document_keys())
        index = HistoryIndex()
        if full:
            index.reset()
        changed = skipped = 0
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
            if prior and prior["issue_updated"] == updated and not full:
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
            query_reason = field(query, "Order Reason", "OrderReason", "Reason")
            query_order_type = field(query, "Order Type", "OrderType")
            query_release = field(query, "Release", "Release Version", "Software Release")
            comments = [c for c in comments if not any(
                marker in _text(c.get("body")) for marker in BOT_COMMENT_MARKERS)]
            choices = []
            assignee_id = (fields.get("assignee") or {}).get("accountId")
            resolved_raw = fields.get("resolutiondate") or ""
            try:
                resolved_at = datetime.fromisoformat(resolved_raw.replace("Z", "+00:00"))
            except ValueError:
                resolved_at = None
            for comment in comments:
                body = _text(comment.get("body")).strip()
                author = comment.get("author") or {}
                author_name = str(author.get("displayName") or "").casefold()
                if _is_automatic_closure(body, author_name, str(author.get("accountType") or "")):
                    continue
                if _is_salesforce_update_comment(body):
                    continue
                created_raw = str(comment.get("created") or "")
                try:
                    created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                except ValueError:
                    created_at = None
                if resolved_at and created_at and created_at > resolved_at + timedelta(days=14):
                    continue
                solution = _resolution_text(body)
                if _resolution_is_placeholder(solution):
                    tier, reasons = "low", ["Resolution field contains only a placeholder"]
                else:
                    tier, reasons = source_quality("ticket", {"step": step, "error": error}, solution)
                author_id = (comment.get("author") or {}).get("accountId")
                from_assignee = int(bool(assignee_id and author_id == assignee_id))
                choices.append(({"high": 2, "medium": 1, "low": 0}[tier], from_assignee,
                                created_raw, body, solution, tier, reasons))
            if choices:
                _, _, selected_created, body, workaround_text, tier, reasons = max(choices)
            else:
                selected_created = ""
                workaround_text = ""
                body, tier, reasons = "", "low", ["No usable resolution comment"]
            resolution_context = f"{_text(fields.get('summary'))}\n{_text(fields.get('description'))}\n{body}"
            order_reason = field(resolution_context, "Order Reason", "OrderReason", "Reason") or query_reason
            order_type = field(resolution_context, "Order Type", "OrderType") or query_order_type
            release = field(resolution_context, "Release", "Release Version", "Software Release") or query_release
            item = {
                "source_key": key, "source_type": "ticket", "issue_updated": updated,
                "summary": _text(fields.get("summary")), "step": step, "error": error,
                "body": body[:12000], "url": f"{self.base_url()}/browse/{key}",
                "order_reason": order_reason,
                "workaround_text": workaround_text,
                "metadata": {"order_reason": order_reason, "order_type": order_type,
                             "release": release, "resolution_at": selected_created or resolved_raw},
                "content_hash": hashlib.sha256((query + body + order_reason + order_type + release +
                                                 (selected_created or resolved_raw) + workaround_text).encode()).hexdigest(),
            }
            if (not full and prior and prior["content_hash"] == item["content_hash"]
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
            if tier in ("high", "medium"):
                version = storage.record_workaround(item, selected_created or resolved_raw)
                storage.remove_workaround_evidence(key, (version or {}).get("id", ""))
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
        if changed:
            storage.set_state("corpus_version", storage.utc_now())
        result = {"ok": True, "full_rebuild": full, "fetched": fetched,
                  "indexed": reindexed if full else changed,
                  "changed_documents": changed, "skipped": skipped,
                  "quality_counts": storage.source_counts()}
        storage.audit(actor, "ingest_completed", "jira", result)
        return result


registry = PluginRegistry()
registry.register(JiraPlugin())
