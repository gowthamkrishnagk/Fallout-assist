"""User feedback store + aggregator — the "training" signal for re-ranking.

Each 👍 / 👎 a user gives on a suggested workaround is recorded here, scoped to
*(workaround identity, failure)* so a ticket can be right for one error and wrong
for another:

  identity  = ticket key   (or document filename)
  failure   = the query's normalized error  (falls back to the step)

At search time `search.find_workarounds` calls `adjustments()` and nudges each
candidate's rank by the net votes it has accumulated for the current failure —
liked workarounds float up, disliked ones sink (softly). This never touches the
Chroma index, so no re-ingest is needed for feedback to take effect.

Storage is a single append-style JSON file (default trackers/feedback.json),
mirroring ingest's trackers/*.json state files. Writes are lock-guarded; reads
are cached and refreshed only when the file changes on disk.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

_DIR  = Path(__file__).parent
_lock = threading.Lock()

# Read cache: (mtime → aggregated dict) so search doesn't re-parse the file every
# query. Invalidated automatically when the file's mtime changes.
_cache: dict = {"mtime": None, "agg": {}}


def _path(cfg: dict) -> Path:
    rel = cfg["workaround_finder"].get("feedback_path", "trackers/feedback.json")
    return _DIR / rel


def signature(step: str, error: str) -> str:
    """The failure key a vote is scoped to. Error is the differentiator; fall back
    to the step when there's no error. Inputs are already normalized by the caller
    (search parses via clean_text), so identical failures collapse to one key."""
    return (error or step or "").strip()


def _key(kind: str, key: str, sig: str) -> tuple:
    return (kind, (key or "").strip(), sig)


def _load_raw(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def record(vote: str, kind: str, key: str, step: str, error: str,
           query_raw: str, cfg: dict) -> dict:
    """Append one vote. `vote` is "up" or "down". Returns the stored record."""
    v = 1 if vote == "up" else -1 if vote == "down" else 0
    if v == 0:
        raise ValueError(f"vote must be 'up' or 'down', got {vote!r}")

    rec = {
        "ts":        datetime.utcnow().isoformat(),
        "vote":      v,
        "kind":      kind or "ticket",
        "key":       (key or "").strip(),
        "step":      step or "",
        "error":     error or "",
        "query_raw": query_raw or "",
    }
    path = _path(cfg)
    with _lock:
        records = _load_raw(path)
        records.append(rec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        _cache["mtime"] = None   # force re-aggregate on next read
    return rec


def adjustments(cfg: dict) -> dict:
    """Aggregate net votes per (kind, key, signature). Cached on the file's mtime
    so repeated searches don't re-read/parse an unchanged file."""
    path = _path(cfg)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}   # no feedback yet

    if _cache["mtime"] == mtime:
        return _cache["agg"]

    agg: dict = {}
    for r in _load_raw(path):
        sig = signature(r.get("step", ""), r.get("error", ""))
        k   = _key(r.get("kind", "ticket"), r.get("key", ""), sig)
        agg[k] = agg.get(k, 0) + int(r.get("vote", 0))

    _cache["mtime"], _cache["agg"] = mtime, agg
    return agg


def net_votes(cfg: dict, kind: str, key: str, step: str, error: str) -> int:
    """Net votes for one candidate at the current failure (0 if none)."""
    agg = adjustments(cfg)
    return agg.get(_key(kind, key, signature(step, error)), 0)
