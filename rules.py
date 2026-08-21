"""Deterministic workaround rules — reviewed, exact-match fixes that short-circuit the
similarity search in search.py for a specific, known error signature.

Why this exists alongside suggestions.py: an approved SUGGESTION gets embedded into the
vector index and found by similarity later — same mechanism as every ticket, scored, never
guaranteed. A RULE is different on purpose: its Error Description pattern either matches
the incoming failure exactly, or it doesn't. When it matches, the same reviewed answer
comes back every time — no score, no "nearest neighbour", nothing for the model to blur.

Mirrors suggestions.py / auth.py: a single JSON file (default
trackers/workaround_rules.json), lock-guarded writes, reads cached on the file's mtime.

Match priority is intentionally NOT symmetric across fields:
  1. error_description (required on every rule) — the signature. Most unique, so it is
     the only field that ever GATES a match on its own.
  2. failed_step / order_reason (optional) — narrow an otherwise-wildcard rule to a
     specific slice, only when there's evidence the fix genuinely differs there. Absent
     (None/empty) means "any" — the common case.
  3. order_type (optional) — same "any means wildcard" treatment, lowest priority.

Because two APPROVED rules could in principle both match the same failure (one wildcard,
one narrowed), `check_overlap` is meant to be called before a rule is approved — reject or
narrow on a real conflict, so at runtime `match_query` never has to guess: it always
prefers the most specific match, but if authoring discipline is followed there is only
ever one candidate to begin with.

IMPORTANT — write patterns against the CLEANED error text, not the raw ticket string.
search.parse_input runs every parsed error through textclean.clean_error_text before
anything else sees it (same normalization ingest applies before embedding) — hyphens
become spaces, timestamps/record-IDs/MSISDNs are stripped, wiki '*' is removed, etc.
`_error_matches` below re-applies that same cleaning (idempotent, so it's safe even if
a caller already cleaned it) precisely so a pattern only ever has to agree with ONE
representation. Use the Rules tab's pattern tester — it cleans the sample the same way
before testing — rather than eyeballing the raw text, or 'MSISDN already defined -
[...]' silently never matches because the hyphen is gone by the time this runs."""

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

from textclean import clean_error_text

_DIR  = Path(__file__).parent
_lock = threading.Lock()

# Read cache: (mtime -> list), same pattern as suggestions.py / feedback.py.
_cache: dict = {"mtime": None, "records": []}

STATUSES     = ("draft", "approved", "deprecated")
MATCH_FIELDS = ("order_type", "order_reason", "failed_step")   # optional, in priority order


def _path(cfg: dict) -> Path:
    rel = cfg.get("workaround_finder", {}).get("rules_path", "trackers/workaround_rules.json")
    return _DIR / rel


def _load_raw(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _all(cfg: dict) -> list:
    """Every rule, cached on the file's mtime."""
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


def _save(records: list, cfg: dict):
    path = _path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    _cache["mtime"] = None   # force re-read next time


def list_rules(cfg: dict, status: str = "") -> list:
    """Every rule, optionally filtered by status, newest first."""
    records = _all(cfg)
    if status:
        records = [r for r in records if r.get("status") == status]
    return sorted(records, key=lambda r: r.get("created_ts", ""), reverse=True)


def get(rule_id: str, cfg: dict) -> dict | None:
    return next((r for r in _all(cfg) if r.get("id") == rule_id), None)


def _field_matches(cond: dict | None, value: str) -> bool:
    """One optional MATCH_FIELDS condition against the failure's actual value.
    `cond` is None/{} → wildcard, always matches. Otherwise {"op": "equals"/"in", ...}."""
    if not cond:
        return True
    op = cond.get("op")
    if op == "equals":
        return (value or "").strip().lower() == (cond.get("value") or "").strip().lower()
    if op == "in":
        allowed = {(v or "").strip().lower() for v in cond.get("values") or []}
        return (value or "").strip().lower() in allowed
    return True   # unknown op — fail open to wildcard rather than silently blocking a rule


def _error_matches(cond: dict, error_text: str) -> bool:
    """The one REQUIRED condition. `cond` must be {"op": "matches_pattern", "pattern": ...}
    (a regex, case-insensitive, DOTALL so a multi-line error still matches end-to-end).

    Cleans `error_text` the same way ingest/search do before matching — see the module
    docstring for why. clean_error_text is idempotent, so this is safe whether the
    caller already cleaned it (search.parse_input does) or passed something raw."""
    if not cond or cond.get("op") != "matches_pattern":
        return False
    pattern = cond.get("pattern", "")
    if not pattern:
        return False
    try:
        return re.search(pattern, clean_error_text(error_text or ""),
                         re.IGNORECASE | re.DOTALL) is not None
    except re.error:
        return False   # a malformed pattern never matches — never crashes the caller


def _specificity(rule: dict) -> int:
    """How many optional fields this rule narrows — used to break a tie if authoring
    discipline (check_overlap) was skipped and two approved rules both match."""
    m = rule.get("match", {})
    return sum(1 for f in MATCH_FIELDS if m.get(f))


def match_query(step: str, error: str, order_type: str = "", order_reason: str = "",
                cfg: dict = None, exclude_ids: frozenset = frozenset()) -> dict | None:
    """The single approved rule that fires for this failure, or None.

    Evaluates only status == "approved" rules. error_description is the gate — every
    other field is optional and defaults to "any". When (against authoring discipline)
    more than one approved rule matches, the most specific one wins (most narrowed
    fields); still ambiguous → the newest wins, so the result is always deterministic
    rather than order-dependent."""
    if cfg is None:
        raise ValueError("cfg is required")
    candidates = []
    for rule in _all(cfg):
        if rule.get("status") != "approved":
            continue
        if rule.get("id") in exclude_ids:
            continue
        m = rule.get("match", {})
        if not _error_matches(m.get("error_description"), error):
            continue
        if not _field_matches(m.get("order_type"), order_type):
            continue
        if not _field_matches(m.get("order_reason"), order_reason):
            continue
        if not _field_matches(m.get("failed_step"), step):
            continue
        candidates.append(rule)
    if not candidates:
        return None
    candidates.sort(key=lambda r: (_specificity(r), r.get("created_ts", "")), reverse=True)
    return candidates[0]


def _field_values(cond: dict | None) -> set | None:
    """A condition's value set, or None for wildcard (the universe)."""
    if not cond:
        return None
    if cond.get("op") == "equals":
        return {(cond.get("value") or "").strip().lower()}
    if cond.get("op") == "in":
        return {(v or "").strip().lower() for v in cond.get("values") or []}
    return None


def _field_relation(mine: dict | None, theirs: dict | None) -> str:
    """How one optional field's scope compares between two rules:
      'equal'          both wildcard, or both narrowed to the exact same set
      'disjoint'       both narrowed, no shared value — these rules can NEVER both
                       apply, regardless of every other field
      'mine_subset'    my allowed set is a strict subset of theirs (I'm the wildcard
                       side widening it, or I'm narrower and fully contained)
      'theirs_subset'  the mirror of the above
      'overlap'        both narrowed, share some values, but neither contains the other
    """
    mine_vals, theirs_vals = _field_values(mine), _field_values(theirs)
    if mine_vals is None and theirs_vals is None:
        return "equal"
    if mine_vals is None:                       # I'm wildcard, they're narrowed
        return "theirs_subset"
    if theirs_vals is None:                     # they're wildcard, I'm narrowed
        return "mine_subset"
    if not (mine_vals & theirs_vals):
        return "disjoint"
    if mine_vals == theirs_vals:
        return "equal"
    if mine_vals < theirs_vals:
        return "mine_subset"
    if theirs_vals < mine_vals:
        return "theirs_subset"
    return "overlap"


def check_overlap(match: dict, cfg: dict, exclude_id: str = "") -> list:
    """Approved rules that would be genuinely AMBIGUOUS against `match` — same error
    signature, and the optional fields don't resolve to a clean precedence. Call this
    before approving a rule; a non-empty result means a real conflict to resolve
    (narrow one of the two, or merge them) rather than something to guess about at
    runtime.

    What does NOT count as a conflict, by design:
      - A wildcard rule and a rule that narrows one more field than it — the narrower
        one is a strict subset of the wildcard one's scope, so it unambiguously takes
        precedence (match_query's specificity tie-break); the wildcard rule still
        covers everything outside that slice.
      - Two rules narrowed on the SAME field to disjoint values (e.g. order_reason=New
        vs. order_reason=Existing) — they can never both match the same failure.
    What DOES count as a conflict:
      - The exact same match spec twice (a literal duplicate).
      - Two rules each narrowing a DIFFERENT field (so neither is a subset of the
        other) — an event could satisfy both, and specificity alone can't order them.
      - The same field narrowed to two sets that share some values but neither
        contains the other (e.g. {New, Existing} vs. {Existing, Port In})."""
    conflicts = []
    pattern   = (match.get("error_description") or {}).get("pattern", "")
    if not pattern:
        return conflicts
    for rule in _all(cfg):
        if rule.get("status") != "approved" or rule.get("id") == exclude_id:
            continue
        other = rule.get("match", {})
        if pattern.strip() != (other.get("error_description") or {}).get("pattern", "").strip():
            continue   # different signature entirely — not this function's concern

        relations = [_field_relation(match.get(f), other.get(f)) for f in MATCH_FIELDS]
        if "disjoint" in relations:
            continue   # can never both apply, no matter what the other fields say
        if all(r == "equal" for r in relations):
            conflicts.append(rule)   # literal duplicate of an existing approved rule
            continue
        if all(r in ("equal", "mine_subset") for r in relations):
            continue   # `match` is a strict subset of `other` — clean precedence
        if all(r in ("equal", "theirs_subset") for r in relations):
            continue   # `other` is a strict subset of `match` — clean precedence
        conflicts.append(rule)       # mixed subset directions, or a partial overlap
    return conflicts


def create(match: dict, workaround_text: str, created_by: str, cfg: dict) -> dict:
    """A new DRAFT rule. Raises ValueError if error_description is missing/invalid —
    every rule must have the one required gating field."""
    if not (match.get("error_description") or {}).get("pattern"):
        raise ValueError("match.error_description.pattern is required")
    if not (workaround_text or "").strip():
        raise ValueError("workaround_text is required")
    rec = {
        "id":              uuid.uuid4().hex,
        "status":          "draft",
        "match":           match,
        "workaround_text": workaround_text.strip(),
        "created_by":      created_by,
        "created_ts":      datetime.utcnow().isoformat(),
        "approved_by":     "",
        "approved_ts":     "",
        "version":         1,
    }
    with _lock:
        records = _load_raw(_path(cfg))
        records.append(rec)
        _save(records, cfg)
    return rec


def approve(rule_id: str, approved_by: str, cfg: dict) -> dict:
    """Flip a draft to approved. Does NOT call check_overlap itself — the caller (the
    admin-only API route) does that first and surfaces conflicts before this commits,
    so a rule is never silently approved into an ambiguous state."""
    return _set_status(rule_id, "approved", cfg, approved_by=approved_by)


def deprecate(rule_id: str, cfg: dict) -> dict:
    """Retire a rule without deleting it — history (who approved what, when) is kept
    for audit rather than erased, same reasoning as suggestions.py never overwriting a
    record's own history."""
    return _set_status(rule_id, "deprecated", cfg)


def _set_status(rule_id: str, status: str, cfg: dict, approved_by: str = "") -> dict | None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    path    = _path(cfg)
    updated = None
    with _lock:
        records = _load_raw(path)
        for r in records:
            if r.get("id") == rule_id:
                r["status"] = status
                if status == "approved" and approved_by:
                    r["approved_by"] = approved_by
                    r["approved_ts"] = datetime.utcnow().isoformat()
                updated = r
                break
        if updated is not None:
            _save(records, cfg)
    return updated
