"""User-submitted workaround store — the queue of fixes proposed by users that are
held *pending* until an admin approves them, at which point they get embedded into
the searchable index (see ingest.ingest_suggestion).

Submissions arrive through Jira: a user sets the "Workaround helpful? = No" field and
fills the "Suggested workaround" field on the ticket, and jirabot.harvest_feedback
reads them on its next poll (no app-server access needed — see jirabot.py). The
in-app review panel approves/rejects them.

Mirrors feedback.py: a single append-style JSON file (default trackers/suggestions.json),
lock-guarded writes, reads cached on the file's mtime.
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

_DIR  = Path(__file__).parent
_lock = threading.Lock()

# Read cache: (mtime → list) so the review panel doesn't re-parse on every poll.
_cache: dict = {"mtime": None, "records": []}

STATUSES = ("pending", "approved", "rejected")


def _path(cfg: dict) -> Path:
    rel = cfg["workaround_finder"].get("suggestions_path", "trackers/suggestions.json")
    return _DIR / rel


def _load_raw(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _all(cfg: dict) -> list:
    """Every record, cached on the file's mtime."""
    path = _path(cfg)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    if _cache["mtime"] == mtime:
        return _cache["records"]
    records = _load_raw(path)
    _cache["mtime"], _cache["records"] = mtime, records
    return records


def add(step: str, error: str, source_key: str, disliked_key: str,
        suggestion: str, cfg: dict, origin: str = "jira_field") -> dict:
    """Append one pending suggestion. Returns the stored record."""
    rec = {
        "id":           uuid.uuid4().hex,
        "ts":           datetime.utcnow().isoformat(),
        "status":       "pending",
        "step":         step or "",
        "error":        error or "",
        "source_key":   (source_key or "").strip(),
        "disliked_key": (disliked_key or "").strip(),
        "suggestion":   (suggestion or "").strip(),
        "origin":       origin,
        "indexed_id":   "",
    }
    path = _path(cfg)
    with _lock:
        records = _load_raw(path)
        records.append(rec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        _cache["mtime"] = None   # force re-read next time
    return rec


def list_by_status(status: str, cfg: dict) -> list:
    """Records with the given status, newest first."""
    return sorted((r for r in _all(cfg) if r.get("status") == status),
                  key=lambda r: r.get("ts", ""), reverse=True)


def get(sid: str, cfg: dict) -> dict | None:
    return next((r for r in _all(cfg) if r.get("id") == sid), None)


def set_status(sid: str, status: str, cfg: dict, indexed_id: str = "") -> dict | None:
    """Flip a suggestion's status (and stamp the indexed chunk id on approval).
    Returns the updated record, or None if the id isn't found."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    path    = _path(cfg)
    updated = None
    with _lock:
        records = _load_raw(path)
        for r in records:
            if r.get("id") == sid:
                r["status"]     = status
                r["updated_ts"] = datetime.utcnow().isoformat()
                if indexed_id:
                    r["indexed_id"] = indexed_id
                updated = r
                break
        if updated is not None:
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            _cache["mtime"] = None
    return updated
