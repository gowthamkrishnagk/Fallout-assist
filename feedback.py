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

# Read cache: (mtime → aggregates) so search doesn't re-parse the file every query.
# Invalidated automatically when the file's mtime changes.
#
# TWO aggregates, built in ONE pass over the file and sharing the same mtime:
#   agg  net votes per candidate — what ranking uses (`adjustments`).
#   tal  up/down counts kept SEPARATE, plus the voter names — what display uses
#        (`tallies`). A net can't be split back apart: 3 up + 2 down and 1 up are
#        both net 1, and "3 👍 · 2 👎" says something very different from "1 👍".
_cache: dict = {"mtime": None, "agg": {}, "tal": {}}


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
           query_raw: str, cfg: dict, voter: str = "",
           voter_source: str = "") -> dict:
    """Append one vote. `vote` is "up" or "down". Returns the stored record.

    `voter` is WHO voted, when that is actually knowable — only the in-Jira field
    channel is (see jirabot.harvest_feedback). The two other channels genuinely cannot
    attribute a vote: the app has no auth, and one signed comment link is clicked by
    whoever is reading the ticket. Those stay '' — an unattributed vote is recorded
    honestly rather than guessed onto the assignee, because a wrong name in the
    training data is worse than a blank one.

    `voter_source` records HOW the name was obtained ("jira_field" = the changelog
    author who set the field, "assignee" = fell back to the ticket's assignee), so a
    later reader can tell a real attribution from an approximation."""
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
    # Only written when known, so the file doesn't fill with empty keys and old
    # records stay byte-identical in shape to new anonymous ones.
    if (voter or "").strip():
        rec["voter"]        = voter.strip()
        rec["voter_source"] = (voter_source or "").strip()
    path = _path(cfg)
    with _lock:
        records = _load_raw(path)
        records.append(rec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        _cache["mtime"] = None   # force re-aggregate on next read
    return rec


def _empty_tally() -> dict:
    return {"up": 0, "down": 0, "up_voters": [], "down_voters": [],
            "up_anon": 0, "down_anon": 0}


def _aggregate(path: Path) -> tuple[dict, dict]:
    """(net votes, up/down tallies) per (kind, key, signature), in one file pass."""
    agg: dict = {}
    tal: dict = {}
    for r in _load_raw(path):
        v = int(r.get("vote", 0))
        if v == 0:
            continue                  # not a real vote (corrupt / hand-edited record)
        sig = signature(r.get("step", ""), r.get("error", ""))
        k   = _key(r.get("kind", "ticket"), r.get("key", ""), sig)
        agg[k] = agg.get(k, 0) + v

        t    = tal.setdefault(k, _empty_tally())
        side = "up" if v > 0 else "down"
        t[side] += 1
        voter = (r.get("voter") or "").strip()
        if not voter:
            t[f"{side}_anon"] += 1                 # counted, but nobody to name
        elif voter not in t[f"{side}_voters"]:
            t[f"{side}_voters"].append(voter)      # one name per person, first vote wins
    return agg, tal


def _ensure(cfg: dict) -> tuple[dict, dict]:
    """Both aggregates, re-read only when the file changed on disk."""
    path = _path(cfg)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}, {}   # no feedback yet

    if _cache["mtime"] != mtime:
        agg, tal = _aggregate(path)
        _cache["mtime"], _cache["agg"], _cache["tal"] = mtime, agg, tal
    return _cache["agg"], _cache["tal"]


def adjustments(cfg: dict) -> dict:
    """Aggregate net votes per (kind, key, signature). Cached on the file's mtime
    so repeated searches don't re-read/parse an unchanged file."""
    return _ensure(cfg)[0]


def tallies(cfg: dict) -> dict:
    """Up/down counts + voter names per (kind, key, signature) — for DISPLAY, never
    for ranking (ranking uses the net from `adjustments`)."""
    return _ensure(cfg)[1]


def net_votes(cfg: dict, kind: str, key: str, step: str, error: str) -> int:
    """Net votes for one candidate at the current failure (0 if none)."""
    agg = adjustments(cfg)
    return agg.get(_key(kind, key, signature(step, error)), 0)


def history(cfg: dict, kind: str, key: str, step: str, error: str) -> list:
    """Every individual vote on one candidate at the current failure, newest first.

    Deliberately NOT cached and NOT aggregated: this backs the "who gave feedback" page,
    which wants one row per vote (who, which way, when, how it was attributed) rather than
    a total. It re-reads the file, which is fine — a human opened a page, versus
    `adjustments` running on every query."""
    sig  = signature(step, error)
    want = _key(kind, key, sig)
    out  = []
    for r in _load_raw(_path(cfg)):
        v = int(r.get("vote", 0))
        if v == 0:
            continue
        if _key(r.get("kind", "ticket"), r.get("key", ""),
                signature(r.get("step", ""), r.get("error", ""))) != want:
            continue
        out.append({
            "vote":         "up" if v > 0 else "down",
            "voter":        (r.get("voter") or "").strip(),
            "voter_source": (r.get("voter_source") or "").strip(),
            "ts":           r.get("ts", ""),
            # Which ticket the vote was cast from — the only provenance an anonymous
            # vote has, and often enough to work out who it was.
            "from_key":     (r.get("query_raw") or "").strip(),
        })
    out.sort(key=lambda r: r["ts"], reverse=True)
    return out


def vote_counts(cfg: dict, kind: str, key: str, step: str, error: str) -> dict:
    """{up, down, up_voters, down_voters, up_anon, down_anon} for one candidate at
    the current failure — all zeros when it has never been voted on.

    Scoped to (candidate, failure) like every other read here, so the number means
    "N people confirmed THIS fix for THIS error", not how popular the ticket is.
    Returns a fresh dict: callers render it, and must not be able to mutate the cache."""
    t = tallies(cfg).get(_key(kind, key, signature(step, error)))
    if not t:
        return _empty_tally()
    return {**t, "up_voters": list(t["up_voters"]),
                 "down_voters": list(t["down_voters"])}
