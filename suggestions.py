"""User-submitted workaround store — the queue of fixes proposed by users that are
held *pending* until an admin approves them, at which point they get embedded into
the searchable index (see ingest.ingest_suggestion).

Two ways in, both landing in the same pending queue:

  jira_field   a user sets "Workaround helpful? = No" and fills the "Suggested
               workaround" field on the ticket; jirabot.harvest_feedback reads them on
               its next poll (no app-server access needed — see jirabot.py).
  app_dislike  an agent clicks 👎 in the app and types what the fix actually is. The
               note is run through `synthesize()` into `=== FIX ===` steps first.

Either way a human approves in the in-app review panel before anything is indexed. That
gate matters more than it looks: `ingest.ingest_suggestion` embeds an approved fix on the
EXACT step+error it was submitted against, so it scores ~1.0 and outranks every real
ticket for that failure. An unreviewed note would become the top answer immediately.

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
        suggestion: str, cfg: dict, origin: str = "jira_field",
        raw: str = "", synth: dict | None = None, submitted_by: str = "") -> dict:
    """Append one pending suggestion. Returns the stored record.

    `suggestion` is the body that gets INDEXED on approval (ingest.ingest_suggestion
    reads this field). `raw` keeps the agent's own words when the text was rewritten, so
    a reviewer can always see what was actually submitted rather than only the model's
    reading of it. `synth` carries the synthesis outcome for the review panel.
    `submitted_by` is the signed-in user's email (the app_dislike path always has one now
    that submitting requires login; '' for the jira_field path, which has no app session)."""
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
        # '' when the text was not rewritten (the Jira-field path indexes verbatim).
        "raw":          (raw or "").strip(),
        "synth":        synth or {},
        "submitted_by": (submitted_by or "").strip(),
        "approved_by":  "",
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


# ── Turning an agent's note into a workaround ─────────────────────────────────────

# Returned by the model when the note doesn't describe a fix at all ("this is wrong",
# "ask L3"). Not a rejection — the note is still queued, just flagged, because the agent
# deliberately submitted it and a reviewer should be the one to bin it.
NOT_A_FIX = "NOT_A_WORKAROUND"


def build_prompt(note: str, step: str, error: str) -> str:
    """Prompt to rewrite an agent's free-text note as a `=== FIX ===` block.

    This is a REWRITE, not a synthesis from sources: the note is the only permitted input.
    The model may reorder into steps, drop chatter and fix grammar — it may not add a step
    the agent didn't describe, because the result is about to become indexed material that
    answers this failure for everyone else."""
    ctx = ""
    if step:
        ctx += f"Failed Step: {step}\n"
    if error:
        ctx += f"Error: {error}\n"
    return (
        "An engineer rejected the suggested workaround for the failure below and wrote "
        "what the correct fix actually is. Rewrite their note as a clean, ordered "
        "procedure.\n\n"
        f"{ctx}\n"
        f"THE ENGINEER'S NOTE:\n{note}\n\n"
        "Rules:\n"
        "- Use ONLY what the note says. Do NOT add steps, checks, causes or advice that "
        "are not in it. This will be indexed and shown to other engineers as the fix.\n"
        "- Turn it into imperative, numbered steps — one concrete action each.\n"
        "- Keep the engineer's field names, values and system names exactly as written.\n"
        "- Drop only greetings, hedging and chatter.\n"
        "- Include a 'Root Cause:' line ONLY if the note states a cause.\n"
        "- Never include an MSISDN, BAN-CAN, order number or record ID — describe the "
        "record generically ('the failing order item').\n"
        f"- If the note does not describe a fix at all (it is a complaint, a question, or "
        f"just says the suggestion was wrong), reply with exactly: {NOT_A_FIX}\n\n"
        "Output EXACTLY this block and nothing else:\n"
        "=== FIX ===\n"
        "Root Cause: <one line, only if the note states one>\n"
        "1. <first action>\n"
        "2. <next action>\n"
        "=== END ==="
    )


def synthesize(note: str, step: str, error: str, cfg: dict) -> dict:
    """An agent's note cleaned into `=== FIX ===` steps.

    Returns {body, raw, provider, model, flag, has_ids}. `body` is what gets indexed on
    approval.

    The agent's note is NEVER lost: if the LLM is switched off, every provider fails, or
    the model declines, `body` falls back to the raw note verbatim and `flag` says why.
    Losing a real fix because a rate limit was hit would be the worst possible outcome
    here — the whole point is to capture knowledge at the moment someone has it."""
    import watable as wt

    raw = (note or "").strip()
    out = {"body": raw, "raw": raw, "provider": "", "model": "", "flag": "",
           "has_ids": False}
    if not raw:
        return out

    wf = cfg.get("workaround_finder", {})
    if not (wf.get("llm_enabled", True) and wf.get("llm_synthesize", True)):
        out["flag"] = "llm_off"
        return out

    try:
        import generate as g
        gen = g.generate(build_prompt(raw, step, error), cfg["llm"], job="synthesis")
        ans = (gen.get("answer") or "").strip()
        if gen.get("provider") == "local":
            # Same distrust as the main synthesis path — a local model's rewrite of
            # someone's fix is not worth more than their own words.
            out["flag"] = "local_model_ignored"
        elif NOT_A_FIX in ans:
            # Queued anyway, flagged: the reviewer decides, not the model.
            out["flag"] = "llm_says_not_a_fix"
        elif len(ans) < 10:
            out["flag"] = "llm_empty"
        else:
            out.update(body=ans, provider=gen.get("provider", ""),
                       model=gen.get("model", ""))
    except Exception as e:
        out["flag"] = f"llm_failed: {str(e)[:120]}"

    # Deliberately FLAGGED, not scrubbed. Stripping digit runs here would also delete a
    # 6-digit error code — the sharpest discriminator in the corpus — from a fix the
    # engineer wrote by hand. A reviewer sees the warning and decides; that is exactly
    # what the approval gate is for.
    out["has_ids"] = wt.has_identifiers(out["body"])
    return out


def list_by_status(status: str, cfg: dict) -> list:
    """Records with the given status, newest first."""
    return sorted((r for r in _all(cfg) if r.get("status") == status),
                  key=lambda r: r.get("ts", ""), reverse=True)


def get(sid: str, cfg: dict) -> dict | None:
    return next((r for r in _all(cfg) if r.get("id") == sid), None)


def set_status(sid: str, status: str, cfg: dict, indexed_id: str = "",
               approved_by: str = "") -> dict | None:
    """Flip a suggestion's status (and stamp the indexed chunk id + approving admin's
    email on approval). Returns the updated record, or None if the id isn't found."""
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
                if approved_by:
                    r["approved_by"] = approved_by
                updated = r
                break
        if updated is not None:
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            _cache["mtime"] = None
    return updated
