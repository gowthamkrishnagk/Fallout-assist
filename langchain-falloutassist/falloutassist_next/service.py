"""Ingestion and Jira scanning with durable leases and idempotency."""

from __future__ import annotations

import hashlib
import json
import uuid

from . import storage
from .plugins import JiraPlugin, registry
from .workflow import investigate


def _jira() -> JiraPlugin:
    return registry.get("jira")  # type: ignore[return-value]


def _harvest_jira_feedback(item: dict) -> bool:
    raw_vote = item.get("feedback_value", "").strip().casefold()
    vote = ("up" if raw_vote in ("yes", "helpful", "thumbs up", "true") else
            "down" if raw_vote in ("no", "not helpful", "thumbs down", "false") else "")
    if not vote:
        return False
    publication = storage.publication(item["key"])
    investigation = (storage.investigation(publication["investigation_id"])
                     if publication else None)
    if not investigation:
        return False
    sources = json.loads(investigation["source_keys_json"])
    if not sources:
        return False
    candidate = sources[0]
    rerank = json.loads(investigation.get("rerank_json") or "[]")
    version_id = next((row.get("workaround_version_id", "") for row in rerank
                       if row.get("key") == candidate), "")
    note = item.get("suggested_workaround", "").strip()[:4000]
    digest = hashlib.sha256(f"{vote}|{candidate}|{version_id}|{note}".encode()).hexdigest()
    receipt_key = f"jira_feedback:{item['key']}"
    if storage.get_state(receipt_key) == digest:
        return False
    actor = f"jira_field_{hashlib.sha256(f'{vote}|{candidate}'.encode()).hexdigest()[:16]}"
    feedback_id, created = storage.add_feedback(
        item["key"], candidate, vote, note, actor, investigation["id"],
        investigation["query_step"], investigation["query_error"], version_id)
    if not created:
        storage.update_feedback_note(feedback_id, note, actor)
    storage.set_state(receipt_key, digest)
    return True


def ingest_once(actor: str = "worker", full: bool = False) -> dict:
    holder = uuid.uuid4().hex
    if not storage.claim_lease("jira_ingest", holder, 7200):
        return {"ok": False, "busy": True}
    try:
        storage.audit(actor, "ingest_started", "jira", {"full_rebuild": full})
        return _jira().ingest(lambda: storage.claim_lease("jira_ingest", holder, 7200), actor, full)
    except Exception as exc:
        storage.audit(actor, "ingest_failed", "jira", {"error_type": type(exc).__name__})
        raise
    finally:
        storage.release_lease("jira_ingest", holder)


def scan_once(actor: str = "worker", limit: int = 0) -> dict:
    holder = uuid.uuid4().hex
    if not storage.claim_lease("jira_scan", holder, 7200):
        return {"ok": False, "busy": True}
    counts = {"scanned": 0, "skipped": 0, "feedback_harvested": 0, "published": 0,
              "preview": 0, "unchanged": 0, "failed": 0}
    try:
        settings = storage.settings()
        corpus_version = storage.get_state("corpus_version")
        for item in _jira().inflow_issues(limit):
            if not storage.claim_lease("jira_scan", holder, 7200):
                raise RuntimeError("Jira scan lease was lost")
            key, updated = item["key"], item["updated"]
            try:
                counts["feedback_harvested"] += int(_harvest_jira_feedback(item))
            except Exception as exc:
                storage.audit(actor, "jira_feedback_harvest_failed", key,
                              {"error_type": type(exc).__name__})
            context = json.dumps({"settings": settings, "corpus_version": corpus_version,
                                  "connection_version": storage.get_state("connection_version"),
                                  "negative_feedback": storage.newest_negative_feedback(key)},
                                 sort_keys=True)
            context_hash = hashlib.sha256(context.encode()).hexdigest()
            previous = storage.scan_record(key)
            if (previous and previous["issue_updated"] == updated
                and previous["context_hash"] == context_hash):
                counts["skipped"] += 1
                continue
            counts["scanned"] += 1
            try:
                result = investigate(key, actor, publish=True)
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1
                storage.save_scan_record(key, updated, context_hash, status)
            except Exception as exc:
                counts["failed"] += 1
                storage.audit(actor, "scan_issue_failed", key,
                              {"error_type": type(exc).__name__})
        storage.audit(actor, "scan_completed", "jira", counts)
        return {"ok": counts["failed"] == 0, **counts}
    finally:
        storage.release_lease("jira_scan", holder)
