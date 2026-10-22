"""Queued knowledge packages for downstream agents; Jira actions are never performed here."""
from __future__ import annotations

import json
import hashlib
import threading
import time
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import storage

_lock = threading.RLock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fa-knowledge-job")
_refresh_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fa-rovo-refresh")
_rovo_refreshes: set[str] = set()
_running_job = ""
_running_refresh = ""
QUEUE_KEY = "rovo_a2a:knowledge_queue"
JOB_PREFIX = "rovo_a2a:knowledge_job:"
REFRESH_QUEUE_KEY = "rovo_a2a:refresh_queue"
MAX_QUEUED_JOBS = 10
MAX_REFRESH_JOBS = 50
REFRESH_PREFIX = "rovo_a2a:refresh:"


class _KnowledgeState(TypedDict, total=False):
    job_id: str
    issue_key: str
    source_keys: list[str]
    actor: str
    started_at: str
    job_started: float
    falloutassist: dict
    rovo_a2a: dict
    rovo_cache_key: str
    package: dict


def _job(job_id: str) -> dict | None:
    return storage.integration_secret(JOB_PREFIX + job_id)


def get_job(job_id: str) -> dict | None:
    value = _job(job_id)
    if not value:
        return None
    response = {key: value.get(key) for key in
                ("job_id", "issue_key", "status", "created_at", "updated_at", "progress", "result", "error")}
    cache_key = value.get("rovo_cache_key")
    if cache_key and response.get("result"):
        from .rovo_a2a import cached_knowledge_by_key
        cached = cached_knowledge_by_key(cache_key)
        if cached:
            result = json.loads(json.dumps(response["result"]))
            rovo_package = cached.get("knowledge_package") or {}
            rovo_proposal = result.setdefault("proposals", {}).setdefault("rovo_a2a", {})
            rovo_proposal.update({
                "status": "completed",
                "suggestion": {key: cached.get(key) for key in
                               ("text", "task_id", "review_required", "engine", "reused") if key in cached},
                "sources": cached.get("sources", []),
                "knowledge_evidence": rovo_package.get("historical_evidence", []),
                "retrieval": cached.get("retrieval", {}),
            })
            result["status"] = "ready" if result.get("proposals", {}).get("falloutassist", {}).get("status") == "completed" else "partial"
            response["result"] = result
            response["status"] = result["status"]
            response.setdefault("progress", {})["rovo_a2a"] = "completed"
        else:
            proposal = response["result"].get("proposals", {}).get("rovo_a2a", {})
            refresh_key = _refresh_state_key(cache_key)
            refresh = storage.integration_secret(refresh_key) or {}
            if refresh.get("status") == "failed":
                result = json.loads(json.dumps(response["result"]))
                proposal = result.setdefault("proposals", {}).setdefault("rovo_a2a", {})
                proposal.update(status="unavailable", error="Rovo refresh failed; resubmit to retry.")
                result["status"] = "partial" if result.get("proposals", {}).get("falloutassist", {}).get("status") == "completed" else "failed"
                response["result"] = result
                response["status"] = result["status"]
                response.setdefault("progress", {})["rovo_a2a"] = "unavailable"
            elif proposal.get("status") == "pending":
                # FalloutAssist's partial result is usable now; keep the job
                # pollable until the independent Rovo refresh finishes.
                response["status"] = "running"
    return response


def submit(issue_key: str, source_keys: list[str], actor: str) -> dict:
    """Persist the request and return immediately; identical queued work is reused."""
    global _running_job
    issue_key = issue_key.strip().upper()
    with _lock:
        queue = storage.integration_secret(QUEUE_KEY) or {"job_ids": []}
        for job_id in queue.get("job_ids", []):
            item = _job(job_id)
            if (item and item.get("issue_key") == issue_key
                    and item.get("source_keys", []) == source_keys
                    and item.get("status") in {"queued", "running"}):
                return {"job_id": job_id, "issue_key": issue_key, "status": item["status"],
                        "deduplicated": True}
        if len(queue.get("job_ids", [])) >= MAX_QUEUED_JOBS:
            raise RuntimeError("The FalloutAssist knowledge queue is full. Retry shortly.")
        job_id = uuid.uuid4().hex
        now = storage.utc_now()
        storage.save_integration_secret(JOB_PREFIX + job_id, {
            "job_id": job_id, "issue_key": issue_key, "source_keys": source_keys,
            "actor": actor, "status": "queued", "created_at": now, "updated_at": now,
        })
        queue.setdefault("job_ids", []).append(job_id)
        storage.save_integration_secret(QUEUE_KEY, queue)
        _dispatch_locked()
        return {"job_id": job_id, "issue_key": issue_key, "status": "queued", "deduplicated": False}


def _dispatch_locked() -> None:
    global _running_job
    if _running_job:
        return
    queue = storage.integration_secret(QUEUE_KEY) or {"job_ids": []}
    job_ids = queue.get("job_ids", [])
    if not job_ids:
        return
    _running_job = job_ids[0]
    future = _executor.submit(_run, _running_job)
    future.add_done_callback(lambda completed, current=_running_job: _handle_unexpected(current, completed))


def _handle_unexpected(job_id: str, future) -> None:
    error = future.exception()
    if error is None:
        return
    item = _job(job_id)
    if item:
        item.update(status="failed", error=type(error).__name__, updated_at=storage.utc_now())
        storage.save_integration_secret(JOB_PREFIX + job_id, item)
    _finish_queue(job_id)


def _branch(work, *args):
    started = time.monotonic()
    try:
        value = work(*args)
        return {"status": "completed", "elapsed_ms": round((time.monotonic() - started) * 1000),
                "result": value}
    except Exception as exc:
        return {"status": "unavailable", "elapsed_ms": round((time.monotonic() - started) * 1000),
                "error": type(exc).__name__}


def _record_branch(job_id: str, branch: str, result: dict) -> None:
    with _lock:
        current = _job(job_id)
        if not current:
            return
        current.setdefault("progress", {})[branch] = result["status"]
        current["updated_at"] = storage.utc_now()
        storage.save_integration_secret(JOB_PREFIX + job_id, current)


def _falloutassist_node(state: _KnowledgeState) -> dict:
    from .workflow import investigate
    result = _branch(investigate, state["issue_key"], state["actor"], False)
    _record_branch(state["job_id"], "falloutassist", result)
    return {"falloutassist": result}


def _rovo_node(state: _KnowledgeState) -> dict:
    cache_key = ""
    if state.get("source_keys"):
        result = _branch(_rovo_resolution, state["issue_key"], state["source_keys"], state["actor"])
    else:
        started = time.monotonic()
        try:
            from .rovo_a2a import cached_knowledge_for_issue, issue_context
            issue = issue_context(state["issue_key"])
            cache_key, cached = cached_knowledge_for_issue(issue)
            if cached:
                result = {"status": "completed", "elapsed_ms": round((time.monotonic() - started) * 1000),
                          "result": cached}
            else:
                scheduled = _schedule_rovo_refresh(state["issue_key"], state["actor"], cache_key)
                result = {"status": "pending" if scheduled else "unavailable",
                          "elapsed_ms": round((time.monotonic() - started) * 1000),
                          "error": "Rovo knowledge refresh scheduled" if scheduled else
                                   "Rovo refresh queue is full; retry later"}
        except Exception:
            result = {"status": "unavailable", "elapsed_ms": round((time.monotonic() - started) * 1000),
                      "error": "Could not read the Jira failure context for Rovo."}
    _record_branch(state["job_id"], "rovo_a2a", result)
    return {"rovo_a2a": result, "rovo_cache_key": cache_key}


def _schedule_rovo_refresh(issue_key: str, actor: str, cache_key: str) -> bool:
    global _running_refresh
    identity = cache_key or issue_key.upper()
    with _lock:
        queue = storage.integration_secret(REFRESH_QUEUE_KEY) or {"jobs": []}
        if any(job.get("identity") == identity for job in queue.get("jobs", [])):
            return True
        if len(queue.get("jobs", [])) >= MAX_REFRESH_JOBS:
            return False
        _rovo_refreshes.add(identity)
        queue.setdefault("jobs", []).append({"identity": identity, "issue_key": issue_key,
                                             "actor": actor, "cache_key": cache_key})
        storage.save_integration_secret(REFRESH_QUEUE_KEY, queue)
        storage.save_integration_secret(_refresh_state_key(identity),
                                        {"status": "queued", "updated_at": storage.utc_now()})
        _dispatch_refresh_locked()
        return True


def _dispatch_refresh_locked() -> None:
    global _running_refresh
    if _running_refresh:
        return
    queue = storage.integration_secret(REFRESH_QUEUE_KEY) or {"jobs": []}
    jobs = queue.get("jobs", [])
    if not jobs:
        return
    job = jobs[0]
    _running_refresh = job["identity"]
    _refresh_executor.submit(_run_rovo_refresh, job)


def _run_rovo_refresh(job: dict) -> None:
    identity = job["identity"]
    state_key = _refresh_state_key(identity)
    storage.save_integration_secret(state_key, {"status": "running", "updated_at": storage.utc_now()})
    try:
        _rovo_resolution(job["issue_key"], [], job["actor"])
        storage.save_integration_secret(state_key, {"status": "completed", "updated_at": storage.utc_now()})
    except Exception:
        storage.save_integration_secret(state_key, {"status": "failed", "updated_at": storage.utc_now()})
    finally:
        _finish_rovo_refresh(identity)


def _finish_rovo_refresh(identity: str) -> None:
    global _running_refresh
    with _lock:
        queue = storage.integration_secret(REFRESH_QUEUE_KEY) or {"jobs": []}
        queue["jobs"] = [job for job in queue.get("jobs", []) if job.get("identity") != identity]
        storage.save_integration_secret(REFRESH_QUEUE_KEY, queue)
        _rovo_refreshes.discard(identity)
        if _running_refresh == identity:
            _running_refresh = ""
        _dispatch_refresh_locked()


def _refresh_state_key(identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return REFRESH_PREFIX + digest


def _assemble_node(state: _KnowledgeState) -> dict:
    local, rovo = state["falloutassist"], state["rovo_a2a"]
    local_result = local.get("result") or {}
    rovo_result = rovo.get("result") or {}
    rovo_package = rovo_result.get("knowledge_package") or {}
    issue = rovo_package.get("current_issue") or {"key": state["issue_key"]}
    package = {
        "schema_version": "1.0", "package_id": state["job_id"],
        "generated_at": storage.utc_now(),
        "total_elapsed_ms": round((time.monotonic() - state["job_started"]) * 1000),
        "queue_wait_ms": max(0, round((datetime.fromisoformat(state["started_at"])
                                         - datetime.fromisoformat(state["created_at"])).total_seconds() * 1000)),
        "issue": issue,
        "status": "ready" if local["status"] == rovo["status"] == "completed" else
                  "partial" if local["status"] == "completed" or rovo["status"] == "completed" else "failed",
        "proposals": {
            "falloutassist": {
                "status": local["status"], "confidence": local_result.get("tier", ""),
                "reasoning": local_result.get("reasons", []),
                "review_required": local_result.get("review_required", local_result.get("tier") != "high"),
                "retrieval_route": local_result.get("retrieval_route", []),
                "source_details": local_result.get("source_details", []),
                "final_suggestion": (local_result.get("supervisor_result") or {}).get("draft", {}),
                "local_retrieval_suggestion": (local_result.get("branch_drafts") or {}).get("local", {}),
                "rovo_search_suggestion": (local_result.get("branch_drafts") or {}).get("rovo", {}),
                "source_keys": local_result.get("source_keys", []),
                "branch_agreement": local_result.get("branch_agreement", ""),
                "investigation_id": local_result.get("investigation_id", ""),
                "elapsed_ms": local["elapsed_ms"],
                **({"error": local["error"]} if local.get("error") else {}),
            },
            "rovo_a2a": {
                "status": rovo["status"],
                "suggestion": {key: rovo_result.get(key) for key in
                               ("text", "task_id", "review_required", "engine") if key in rovo_result},
                "sources": rovo_result.get("sources", []),
                "knowledge_evidence": rovo_package.get("historical_evidence", []),
                "retrieval": rovo_result.get("retrieval", {}),
                "elapsed_ms": rovo["elapsed_ms"],
                **({"error": rovo["error"]} if rovo.get("error") else {}),
            },
        },
        "execution_guidance": {
            "may_execute": False,
            "required_checks": ["verify current order and identifiers", "check proposal prerequisites",
                                "confirm destructive actions before execution"],
            "note": "These are historical suggestions. The downstream agent must verify applicability and use its own authorised action tools.",
        },
    }
    return {"package": package}


def _build_graph():
    graph = StateGraph(_KnowledgeState)
    graph.add_node("falloutassist", _falloutassist_node)
    graph.add_node("rovo_a2a", _rovo_node)
    graph.add_node("assemble", _assemble_node)
    graph.add_edge(START, "falloutassist")
    graph.add_edge(START, "rovo_a2a")
    graph.add_edge(["falloutassist", "rovo_a2a"], "assemble")
    graph.add_edge("assemble", END)
    return graph.compile()


KNOWLEDGE_GRAPH = _build_graph()


def _rovo_resolution(issue_key: str, source_keys: list[str], actor: str) -> dict:
    from .rovo_a2a import A2ABusyError, resolution
    deadline = time.monotonic() + 300
    while True:
        try:
            return resolution(issue_key, source_keys, actor)
        except A2ABusyError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def _run(job_id: str) -> None:
    global _running_job
    job_started = time.monotonic()
    item = _job(job_id)
    if not item:
        _finish_queue(job_id)
        return
    started_at = storage.utc_now()
    item.update(status="running", started_at=started_at, updated_at=started_at,
                progress={"falloutassist": "running", "rovo_a2a": "running"})
    storage.save_integration_secret(JOB_PREFIX + job_id, item)
    state = KNOWLEDGE_GRAPH.invoke({
        "job_id": job_id,
        "issue_key": item["issue_key"],
        "source_keys": item.get("source_keys", []),
        "actor": item["actor"],
        "started_at": started_at,
        "created_at": item["created_at"],
        "job_started": job_started,
    })
    package = state["package"]
    local = state["falloutassist"]
    rovo = state["rovo_a2a"]
    item.update(status=package["status"], result=package,
                progress={"falloutassist": local["status"], "rovo_a2a": rovo["status"]},
                updated_at=storage.utc_now())
    item["rovo_cache_key"] = state.get("rovo_cache_key", "")
    storage.save_integration_secret(JOB_PREFIX + job_id, item)
    storage.audit(item["actor"], "knowledge_package_completed", item["issue_key"], {
        "job_id": job_id, "status": package["status"],
        "local_status": local["status"], "rovo_status": rovo["status"],
        "local_elapsed_ms": local["elapsed_ms"], "rovo_elapsed_ms": rovo["elapsed_ms"],
    })
    _finish_queue(job_id)


def _finish_queue(job_id: str) -> None:
    global _running_job
    with _lock:
        queue = storage.integration_secret(QUEUE_KEY) or {"job_ids": []}
        queue["job_ids"] = [value for value in queue.get("job_ids", []) if value != job_id]
        storage.save_integration_secret(QUEUE_KEY, queue)
        if _running_job == job_id:
            _running_job = ""
        _dispatch_locked()


def resume_pending() -> None:
    """Resume persisted queued work after app startup; mark a crashed running job queued."""
    global _running_job, _running_refresh
    with _lock:
        _running_job = ""
        queue = storage.integration_secret(QUEUE_KEY) or {"job_ids": []}
        valid = []
        for job_id in queue.get("job_ids", []):
            item = _job(job_id)
            if not item:
                continue
            if item.get("status") in {"ready", "partial", "failed"}:
                continue
            if item.get("status") == "running":
                item.update(status="queued", updated_at=storage.utc_now())
                storage.save_integration_secret(JOB_PREFIX + job_id, item)
            valid.append(job_id)
        queue["job_ids"] = valid
        storage.save_integration_secret(QUEUE_KEY, queue)
        _dispatch_locked()
        _running_refresh = ""
        refresh_queue = storage.integration_secret(REFRESH_QUEUE_KEY) or {"jobs": []}
        for refresh_job in refresh_queue.get("jobs", []):
            _rovo_refreshes.add(refresh_job.get("identity", ""))
            storage.save_integration_secret(_refresh_state_key(refresh_job["identity"]),
                                            {"status": "queued", "updated_at": storage.utc_now()})
        storage.save_integration_secret(REFRESH_QUEUE_KEY, refresh_queue)
        _dispatch_refresh_locked()
