"""Proactive Jira auto-suggest bot.

Polls inflow (open) Order Fallout tickets, and for each one that has a *confident*
match in the knowledge base, posts the suggested workaround as a comment — with two
signed feedback links baked in:

  👍  "good fix"      → confirm (boost this workaround for the failure), keep the comment.
  👎  "improve it"    → demote/train the shown fix, find the next-best PROPER match, and
                        UPDATE the same comment with it (repeat until accepted or exhausted).

Guarantees (per the product requirements):
  - Accurate / no spam: only comments when best_score >= score_threshold; one comment per
    ticket (idempotent via per-ticket state + a hidden marker); never re-posts on each poll.
  - No confident match → STAY SILENT (post nothing).
  - Dry-run first: when jira_suggest_dry_run is on, it only logs what it WOULD post.

Pure runtime layer — never touches the Chroma index, so no re-ingest is needed. Reuses
ingest._get_jira / _search_jql / _ticket_to_text, suggest.suggest_for_query, and
feedback.record (the same training store the in-app 👍/👎 feeds)."""

import hashlib
import hmac
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

_DIR        = Path(__file__).parent
_STATE_FILE = _DIR / "trackers" / "jira_suggestions.json"
_META_FILE  = _DIR / "trackers" / "jira_suggest_meta.json"
_lock       = threading.Lock()

MARKER = "FA-SUGGESTION"   # hidden token in the comment footer → idempotency backup


# ── State ──────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict):
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _save_meta(meta: dict):
    _META_FILE.parent.mkdir(parents=True, exist_ok=True)
    _META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def get_status(cfg: dict) -> dict:
    wf = cfg["workaround_finder"]
    try:
        meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    state = _load_state()
    return {
        "enabled":         bool(wf.get("jira_suggest_enabled", False)),
        "dry_run":         bool(wf.get("jira_suggest_dry_run", True)),
        "minutes":         int(wf.get("jira_suggest_minutes", 0) or 0),
        "jql":             wf.get("jira_suggest_jql", ""),
        "public_base_url": wf.get("public_base_url", ""),
        "tracked":         len(state),
        "last_run":        meta,
    }


# ── Link signing ───────────────────────────────────────────────────────────────

def _secret() -> bytes:
    """Stable HMAC key for feedback links. Prefer FEEDBACK_SECRET; otherwise derive a
    deterministic key from the Jira token so links survive restarts without extra config."""
    s = os.getenv("FEEDBACK_SECRET", "").strip()
    if s:
        return s.encode()
    tok = os.getenv("JIRA_API_TOKEN", "") or "fa-default-secret"
    return hashlib.sha256(("fa-fallback:" + tok).encode()).digest()


def sign(key: str, cand: str, action: str) -> str:
    msg = f"{key}|{cand}|{action}".encode()
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:16]


def verify(key: str, cand: str, action: str, sig: str) -> bool:
    return hmac.compare_digest(sign(key, cand, action), sig or "")


def _public_base(cfg: dict) -> str:
    return (cfg["workaround_finder"].get("public_base_url", "") or "").rstrip("/")


def _links(cfg: dict, key: str, cand: str) -> tuple[str, str]:
    base = _public_base(cfg)
    k, c = quote(key, safe=""), quote(cand, safe="")
    up   = f"{base}/api/jira-feedback?key={k}&cand={c}&action=up&sig={sign(key, cand, 'up')}"
    down = f"{base}/api/jira-feedback?key={k}&cand={c}&action=down&sig={sign(key, cand, 'down')}"
    return up, down


# ── Comment composition (Jira wiki markup; /2/ API is plain text) ──────────────

def _compose_comment(cfg: dict, key: str, cand: str, pct: int, answer: str) -> str:
    up, down = _links(cfg, key, cand)
    return (
        f"*💡 Suggested workaround* — matched {cand} · {pct}% confidence\n\n"
        f"{answer}\n\n"
        "----\n"
        f"*Was this helpful?*  [👍 Yes, good fix|{up}]    |    [👎 No, improve it|{down}]\n\n"
        f"_🤖 Auto-suggested by FalloutAssist · {MARKER}_"
    )


def _compose_manual_review(cfg: dict, key: str) -> str:
    return (
        "*💡 Suggested workaround*\n\n"
        "No further confident workaround was found for this failure after feedback — "
        "this one needs manual review.\n\n"
        f"_🤖 Auto-suggested by FalloutAssist · {MARKER}_"
    )


def _has_marker(issue) -> bool:
    try:
        comments = issue.fields.comment.comments if issue.fields.comment else []
    except Exception:
        return False
    return any(MARKER in (getattr(c, "body", "") or "") for c in comments)


def _cand_of(top: dict) -> tuple[str, str]:
    """(identity, kind) for the candidate that produced the answer."""
    if top.get("type") == "doc":
        return top.get("filename", ""), "doc"
    return top.get("key", ""), "ticket"


# ── Poll + post ────────────────────────────────────────────────────────────────

def run_once(cfg: dict) -> dict:
    """One pass over the inflow JQL. Returns a small summary dict."""
    import ingest as ing
    import suggest

    wf        = cfg["workaround_finder"]
    jql       = wf.get("jira_suggest_jql", "").strip()
    dry       = bool(wf.get("jira_suggest_dry_run", True))
    threshold = wf.get("score_threshold", 0.7)
    if not jql:
        return {"ok": False, "error": "jira_suggest_jql not configured"}

    with _lock:
        try:
            jira   = ing._get_jira(cfg)
            issues = ing._search_jql(jira, jql,
                                     "summary,description,comment,status,updated")
        except Exception as e:
            print(f"[JIRA-SUGGEST] search failed: {str(e)[:160]}")
            return {"ok": False, "error": str(e)[:200]}

        state = _load_state()
        posted = improved_silent = skipped = 0
        would  = 0

        for issue in issues:
            key = issue.key
            st  = state.get(key)

            # Idempotency: a live comment already exists for this ticket.
            if st and st.get("comment_id"):
                skipped += 1
                continue
            # Backup idempotency: our marker is already on the ticket (state lost / redeploy).
            if _has_marker(issue):
                skipped += 1
                state[key] = {**(st or {}), "key": key, "status": "posted",
                              "updated": _now()}
                continue
            # In dry-run, don't re-preview a ticket we already previewed this cycle set.
            if dry and st and st.get("status") == "dry_run":
                skipped += 1
                continue

            try:
                query = ing._ticket_to_text(issue)
                res   = suggest.suggest_for_query(query, cfg)
            except Exception as e:
                print(f"[JIRA-SUGGEST] suggest failed for {key}: {str(e)[:120]}")
                continue

            # Accurate / no-spam gate: only a confident strong match earns a comment.
            if res.get("mode") != "strong_match" or not res.get("top") \
                    or res.get("best_score", 0) < threshold:
                improved_silent += 1
                state[key] = {"key": key, "status": "no_match", "updated": _now()}
                continue

            top        = res["top"]
            cand, kind = _cand_of(top)
            pct        = round(res.get("best_score", 0) * 100)
            body       = _compose_comment(cfg, key, cand, pct, res["answer"])
            base_rec   = {"key": key, "suggested_key": cand, "cand_kind": kind,
                          "step": res.get("query_step", ""), "error": res.get("query_error", ""),
                          "rejected": [], "updated": _now()}

            if dry:
                print(f"[JIRA-SUGGEST] (dry-run) would comment on {key}: "
                      f"matched {cand} {pct}% — {res['answer'][:80]!r}")
                state[key] = {**base_rec, "status": "dry_run"}
                would += 1
                continue

            try:
                c = jira.add_comment(key, body)
                state[key] = {**base_rec, "comment_id": str(c.id), "status": "posted"}
                posted += 1
                print(f"[JIRA-SUGGEST] commented on {key} (matched {cand} {pct}%)")
            except Exception as e:
                print(f"[JIRA-SUGGEST] FAILED to comment on {key}: {str(e)[:140]}")

        _save_state(state)

    summary = {"ok": True, "scanned": len(issues), "posted": posted,
               "would_post": would, "silent": improved_silent, "skipped": skipped,
               "dry_run": dry, "last_run": _now()}
    _save_meta(summary)
    print(f"[JIRA-SUGGEST] run done — {summary}")
    return summary


# ── Feedback (called by the GET link handler) ──────────────────────────────────

def apply_vote(key: str, cand: str, action: str, cfg: dict) -> dict:
    """Handle a 👍/👎 click from a posted comment.
      up   → record + boost, mark accepted, leave the comment.
      down → record + demote, exclude this cand, re-match, and UPDATE the comment
             with the next-best proper workaround (or a manual-review note)."""
    import ingest as ing
    import suggest
    import feedback as fb

    wf  = cfg["workaround_finder"]
    dry = bool(wf.get("jira_suggest_dry_run", True))

    with _lock:
        state = _load_state()
        st    = state.get(key)
        if not st:
            # No tracked suggestion (state lost). Still record the training signal.
            st = {"key": key, "rejected": [], "step": "", "error": ""}

        kind = st.get("cand_kind", "ticket")
        step, error = st.get("step", ""), st.get("error", "")

        if action == "up":
            fb.record("up", kind, cand, step, error, key, cfg)
            st["status"], st["updated"] = "accepted", _now()
            state[key] = st
            _save_state(state)
            return {"ok": True, "action": "up", "key": key, "cand": cand}

        # ── down: train + improve ──────────────────────────────────────────────
        fb.record("down", kind, cand, step, error, key, cfg)
        rejected = list(dict.fromkeys((st.get("rejected") or []) + [cand]))

        # Re-match the live ticket, skipping everything rejected so far.
        try:
            query = ing.fetch_ticket_text(key, cfg)
            res   = suggest.suggest_for_query(query, cfg, exclude_keys=rejected)
        except Exception as e:
            res = {"mode": "error", "best_score": 0, "top": None,
                   "answer": f"(could not re-match: {str(e)[:80]})"}

        threshold = wf.get("score_threshold", 0.7)
        if res.get("mode") == "strong_match" and res.get("top") \
                and res.get("best_score", 0) >= threshold:
            new_cand, new_kind = _cand_of(res["top"])
            pct  = round(res.get("best_score", 0) * 100)
            body = _compose_comment(cfg, key, new_cand, pct, res["answer"])
            st.update(status="improved", suggested_key=new_cand, cand_kind=new_kind)
            improved, answer = True, res["answer"]
        else:
            new_cand = None
            body     = _compose_manual_review(cfg, key)
            st.update(status="exhausted", suggested_key=None)
            improved, answer = False, "No further confident workaround — needs manual review."

        st["rejected"], st["updated"] = rejected, _now()

        # Edit the existing comment in place (still one comment — never spam).
        if st.get("comment_id") and not dry:
            try:
                jira = ing._get_jira(cfg)
                jira.comment(key, st["comment_id"]).update(body=body)
            except Exception as e:
                print(f"[JIRA-SUGGEST] failed to update comment on {key}: {str(e)[:120]}")

        state[key] = st
        _save_state(state)
        return {"ok": True, "action": "down", "key": key,
                "improved": improved, "new_cand": new_cand, "answer": answer}
