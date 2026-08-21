"""
Workaround Finder — FastAPI app
Run: uvicorn app:app --reload --port 8010
"""

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

# Use the OS (Windows) certificate store for TLS so corporate proxy / inspection
# root CAs are trusted — fixes "CERTIFICATE_VERIFY_FAILED: unable to get local
# issuer certificate" on cloud API calls. Must run before any httpx/SSL use.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception as _e:   # truststore missing or unsupported → fall back to certifi
    print(f"[TLS] truststore not active ({_e}); using default certifi bundle")

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

_DIR        = Path(__file__).parent
CONFIG_PATH = _DIR / "config.json"

app = FastAPI(title="FalloutAssist")
app.mount("/static", StaticFiles(directory=str(_DIR / "static")), name="static")


def _session_secret() -> str:
    """Stable key for signing the login session cookie. Prefer SESSION_SECRET;
    otherwise derive a deterministic key from the Jira token so sessions survive
    restarts without extra config — same fallback pattern as jirabot._secret()."""
    s = os.getenv("SESSION_SECRET", "").strip()
    if s:
        return s
    tok = os.getenv("JIRA_API_TOKEN", "") or "fa-default-secret"
    return hashlib.sha256(("fa-session-fallback:" + tok).encode()).hexdigest()


app.add_middleware(SessionMiddleware, secret_key=_session_secret(),
                    session_cookie="fa_session", same_site="lax")


# ── Auth: who's signed in, and are they an admin ─────────────────────────────

def current_user(request: Request) -> dict | None:
    """The signed-in user (no password hash), or None if signed out / unknown session."""
    uid = request.session.get("uid")
    if not uid:
        return None
    import auth
    return auth.get_by_id(uid, load_config())


def require_login(request: Request) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(401, "sign in required")
    return user


def require_admin(request: Request) -> dict:
    user = require_login(request)
    if user.get("role") != "admin":
        raise HTTPException(403, "admin role required")
    return user


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Running flag for ingest (duplicate-run guard) ────────────────────────────

_running = False
_run_lock = threading.Lock()


def _try_acquire() -> bool:
    global _running
    with _run_lock:
        if _running:
            return False
        _running = True
        return True


def _release():
    global _running
    with _run_lock:
        _running = False


# ── Background auto-ingest scheduler ─────────────────────────────────────────
# Periodically pulls the LATEST tickets in the background. It re-reads config
# every minute, so changing the interval (or turning it off) via the schedule
# API takes effect within ~1 min without a restart. Each run is windowed +
# incremental: it fetches only tickets updated in the lookback window and
# re-embeds just the new/changed ones — never the whole project.

def _auto_ingest_loop():
    import ingest as ing
    elapsed = 0   # whole minutes since the last run
    while True:
        time.sleep(60)
        try:
            cfg      = load_config()
            interval = int(cfg["workaround_finder"].get("auto_ingest_minutes", 0) or 0)
        except Exception as e:
            print(f"[AUTO-INGEST] config read failed: {e}")
            continue
        if interval <= 0:        # disabled
            elapsed = 0
            continue
        elapsed += 1
        if elapsed < interval:
            continue
        elapsed = 0

        if not _try_acquire():   # a manual ingest/upload is in progress — skip this tick
            print("[AUTO-INGEST] skipped — another ingest is running")
            continue
        try:
            lookback = int(cfg["workaround_finder"].get("auto_ingest_lookback_minutes", 1440) or 1440)
            result   = ing.ingest_jira(cfg, since_minutes=lookback)
            if result.get("ok"):
                print(f"[AUTO-INGEST] {result.get('tickets_indexed', 0)} tickets indexed, "
                      f"{result.get('up_to_date', 0)} unchanged "
                      f"(last {lookback} min)")
            else:
                print(f"[AUTO-INGEST] FAILED: {result.get('error')}")
        except Exception as e:
            print(f"[AUTO-INGEST] EXCEPTION: {e}")
        finally:
            _release()


# ── Background Jira auto-suggest poller ──────────────────────────────────────
# Periodically scans inflow Order Fallout tickets and posts a suggested workaround
# (or, in dry-run, logs what it would post). Config is re-read every _SUGGEST_TICK
# seconds, so enabling/disabling or changing the cadence takes effect within that.
# Skips a tick while an ingest is running (the index is in flux).

_SUGGEST_TICK = 5          # seconds between config re-reads


def _suggest_interval_seconds(wf: dict) -> int:
    """The effective poll interval, delegated to jirabot so the scheduler, the API and
    the dashboard can never disagree about the cadence."""
    import jirabot
    return jirabot.interval_seconds(wf)


def _jira_suggest_loop():
    import jirabot
    waited = 0
    while True:
        time.sleep(_SUGGEST_TICK)
        try:
            cfg = load_config()
            wf  = cfg["workaround_finder"]
            enabled  = bool(wf.get("jira_suggest_enabled", False))
            interval = _suggest_interval_seconds(wf)
        except Exception as e:
            print(f"[JIRA-SUGGEST] config read failed: {e}")
            continue
        if not enabled or interval <= 0:
            waited = 0
            continue
        waited += _SUGGEST_TICK
        if waited < interval:
            continue
        waited = 0

        if _running:        # an ingest is in progress — index in flux, skip this tick
            print("[JIRA-SUGGEST] skipped — an ingest is running")
            continue
        try:
            jirabot.run_once(cfg)
        except Exception as e:
            print(f"[JIRA-SUGGEST] EXCEPTION: {e}")


@app.on_event("startup")
def _start_auto_ingest():
    threading.Thread(target=_auto_ingest_loop, daemon=True).start()
    print("[AUTO-INGEST] scheduler started (interval read live from config)")
    threading.Thread(target=_jira_suggest_loop, daemon=True).start()
    print("[JIRA-SUGGEST] scheduler started (interval read live from config)")


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (_DIR / "templates" / "index.html").read_text(encoding="utf-8")


# ── Auth: register / login / logout / who-am-i ───────────────────────────────
# The page itself still loads for everyone (the login form is part of it) — every
# other route below is what actually requires a session.

@app.post("/api/auth/register")
def auth_register(body: dict, request: Request):
    import auth
    cfg = load_config()
    try:
        user = auth.register(body.get("email", ""), body.get("password", ""), cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    request.session["uid"] = user["id"]
    return {"ok": True, "user": user}


@app.post("/api/auth/login")
def auth_login(body: dict, request: Request):
    import auth
    cfg  = load_config()
    user = auth.authenticate(body.get("email", ""), body.get("password", ""), cfg)
    if not user:
        raise HTTPException(401, "invalid email or password")
    request.session["uid"] = user["id"]
    return {"ok": True, "user": user}


@app.post("/api/auth/logout")
def auth_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(request: Request):
    """Always 200 — {"user": null} is the signed-out state, not an error, so the
    page's own JS can check this on load without a try/catch."""
    return {"ok": True, "user": current_user(request)}


# ── Jira ingest ───────────────────────────────────────────────────────────────

@app.post("/api/ingest/preview")
def ingest_preview(body: dict, user: dict = Depends(require_login)):
    import ingest as ing
    jql = body.get("jql", "").strip()
    if not jql:
        raise HTTPException(400, "jql required")
    cfg = load_config()
    return ing.preview_jql(jql, cfg)


@app.post("/api/ingest/start")
def ingest_start(body: dict, admin: dict = Depends(require_admin)):
    import ingest as ing
    if not _try_acquire():
        return JSONResponse({"ok": False, "status": "already_running"}, status_code=409)
    cfg  = load_config()
    jql  = body.get("jql", "").strip()
    # mode: "rebuild" → delete & full re-ingest | "refresh" → latest only
    # (windowed incremental) | "" → incremental full-scope sync.
    mode = body.get("mode", "")
    full = bool(body.get("full", False)) or mode == "rebuild"
    since_minutes = None
    if mode == "refresh" and not full:
        since_minutes = int(cfg["workaround_finder"].get("auto_ingest_lookback_minutes", 1440) or 1440)
    if jql:
        cfg["workaround_finder"]["ingest_jql"] = jql
        save_config(cfg)

    def _run():
        try:
            result = ing.ingest_jira(cfg, full=full, since_minutes=since_minutes)
            if result.get("ok"):
                print(f"[INGEST] Done — indexed {result['indexed']} chunks "
                      f"({result.get('tickets_indexed', 0)} tickets), "
                      f"skipped {result['skipped']}, "
                      f"up-to-date {result.get('up_to_date', 0)}, "
                      f"pruned {result.get('pruned', 0)}")
            else:
                print(f"[INGEST] FAILED: {result.get('error')}")
        except Exception as e:
            print(f"[INGEST] EXCEPTION: {e}")
        finally:
            _release()

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "status": "started"}


@app.get("/api/ingest/status")
def ingest_status(user: dict = Depends(require_login)):
    import ingest as ing
    cfg = load_config()
    s   = ing.get_status(cfg)
    s["running"] = _running
    return s


@app.get("/api/scorecard")
def scorecard_status(user: dict = Depends(require_login)):
    """Latest auto-graded accuracy (self-test, feedback-free) + recent history for a
    sparkline. Refreshes automatically after each ingest."""
    import json, scorecard
    cfg = load_config()
    try:
        hist = json.loads(scorecard._history_path(cfg).read_text(encoding="utf-8"))
    except Exception:
        hist = []
    return {"ok": True, "latest": (hist[-1] if hist else None), "history": hist[-30:]}


@app.get("/api/ingest/schedule")
def ingest_schedule_get(admin: dict = Depends(require_admin)):
    """Current background auto-ingest schedule."""
    wf = load_config()["workaround_finder"]
    return {
        "ok":               True,
        "minutes":          int(wf.get("auto_ingest_minutes", 0) or 0),   # 0 = off
        "lookback_minutes": int(wf.get("auto_ingest_lookback_minutes", 1440) or 1440),
        "running":          _running,
    }


@app.post("/api/ingest/schedule")
def ingest_schedule_set(body: dict, admin: dict = Depends(require_admin)):
    """Set the background auto-ingest cadence (persisted to config.json).
    minutes=0 disables it. Common: 10 (10 min), 60 (hourly), 1440 (daily).
    Takes effect within ~1 minute — no restart needed."""
    cfg = load_config()
    if "minutes" in body:
        cfg["workaround_finder"]["auto_ingest_minutes"] = max(0, int(body["minutes"]))
    if "lookback_minutes" in body:
        cfg["workaround_finder"]["auto_ingest_lookback_minutes"] = max(1, int(body["lookback_minutes"]))
    save_config(cfg)
    wf = cfg["workaround_finder"]
    return {"ok": True,
            "minutes":          int(wf.get("auto_ingest_minutes", 0) or 0),
            "lookback_minutes": int(wf.get("auto_ingest_lookback_minutes", 1440) or 1440)}


# ── Ask ───────────────────────────────────────────────────────────────────────

@app.post("/api/ask")
async def ask(body: dict, user: dict = Depends(require_login)):
    import ingest as ing
    import suggest

    ticket_id  = body.get("ticket_id", "").strip().upper()
    query_text = body.get("query_text", "").strip()
    cfg        = load_config()

    if not ticket_id and not query_text:
        raise HTTPException(400, "ticket_id or query_text required")

    # Fetch the ticket's text to use as the query
    if ticket_id:
        try:
            query_text = ing.fetch_ticket_text(ticket_id, cfg)
        except Exception as e:
            # Clean, actionable message — never dump the raw JiraError (headers,
            # request IDs, etc.) into the UI.
            code = getattr(e, "status_code", None)
            if code == 404:
                msg = (f"Ticket {ticket_id} not found, or your Jira account can't see it. "
                       f"Check the ID and that it's in a project you have access to.")
            elif code in (401, 403):
                msg = ("Jira authentication failed — verify JIRA_EMAIL and JIRA_API_TOKEN "
                       "in Settings (the token may have expired).")
            else:
                msg = f"Could not fetch ticket {ticket_id} from Jira (HTTP {code or 'error'})."
            raise HTTPException(400, msg)

    # Build the suggestion (retrieval + grounded synthesis). Shared verbatim with
    # the Jira auto-suggest bot so a posted comment matches the in-app answer.
    out = suggest.suggest_for_query(query_text, cfg)
    out.pop("top", None)            # internal handle — not part of the API response
    return out


# ── Feedback (👍/👎 → re-ranking signal) ───────────────────────────────────────

@app.post("/api/feedback")
def feedback_record(body: dict, user: dict = Depends(require_login)):
    """Record a 👍 (correct) / 👎 (wrong) vote on a suggested workaround. Scoped to
    (workaround identity, failure) so it trains ranking for that step+error only.
    The vote itself is a pure runtime signal — it does not touch the index.

    A 👎 may carry an optional `note`: what the fix actually is. That note is rewritten
    into `=== FIX ===` steps and queued as a PENDING suggestion (never indexed here — see
    suggestions.py for why the approval gate matters). The note is optional on purpose: a
    👎 has to stay one click, or people stop voting and the ranking signal dries up."""
    import feedback as fb
    vote = body.get("vote", "")
    key  = (body.get("key", "") or "").strip()
    note = (body.get("note", "") or "").strip()
    if vote not in ("up", "down"):
        raise HTTPException(400, "vote must be 'up' or 'down'")
    if not key:
        raise HTTPException(400, "key required (ticket key or document filename)")
    cfg   = load_config()
    step  = body.get("step", "")
    error = body.get("error", "")
    rec = fb.record(
        vote=vote,
        kind=body.get("kind", "ticket"),
        key=key,
        step=step,
        error=error,
        query_raw=body.get("query_raw", ""),
        cfg=cfg,
    )
    out = {"ok": True, "vote": rec["vote"]}

    if vote == "down" and note:
        # The vote is already saved above, so anything below costs the note, never the
        # training signal.
        out.update(_queue_fix_note(note, key, step, error,
                                   body.get("query_raw", ""), cfg, user["email"]))
    return out


def _queue_fix_note(note: str, disliked_key: str, step: str, error: str,
                    query_raw: str, cfg: dict, submitted_by: str = "") -> dict:
    """Rewrite an agent's 'here's the real fix' note and queue it for review.

    Shared by /api/feedback (one-shot: vote + note together) and /api/suggestions (the
    UI's path, where the 👎 vote has already been recorded and the note follows). Never
    raises — a failure here must not fail the caller's primary action."""
    try:
        if not cfg["workaround_finder"].get("suggestions_enabled", True):
            return {"note_status": "disabled"}
        if not (step or error):
            # ingest_suggestion embeds against the step and/or error; with neither there
            # is no failure to attach the fix to, so it could never be retrieved.
            return {"note_status": "no_failure_context"}
        import suggestions as sg
        syn  = sg.synthesize(note, step, error, cfg)
        srec = sg.add(step, error, source_key=(query_raw or "")[:80],
                      disliked_key=disliked_key, suggestion=syn["body"], cfg=cfg,
                      origin="app_dislike", raw=syn["raw"],
                      synth={k: syn[k] for k in ("provider", "model", "flag", "has_ids")},
                      submitted_by=submitted_by)
        print(f"[FEEDBACK] 👎 on {disliked_key} + fix note -> pending suggestion "
              f"{srec['id']} (flag={syn['flag'] or 'none'})")
        return {"note_status": "pending", "suggestion_id": srec["id"],
                "synthesized": syn["body"], "note_flag": syn["flag"],
                "note_has_ids": syn["has_ids"]}
    except Exception as e:
        print(f"[FEEDBACK] note capture failed for {disliked_key}: {str(e)[:160]}")
        return {"note_status": f"failed: {str(e)[:120]}"}


@app.post("/api/suggestions")
def suggestions_submit(body: dict, user: dict = Depends(require_login)):
    """An agent's corrected fix, submitted after a 👎 in the app.

    Separate from /api/feedback because the vote fires on the FIRST click (so it is never
    lost if the note is abandoned) and the note follows — routing both through the vote
    endpoint would record the 👎 twice."""
    note = (body.get("note", "") or "").strip()
    if not note:
        raise HTTPException(400, "note required")
    cfg = load_config()
    out = _queue_fix_note(note, (body.get("key", "") or "").strip(),
                          body.get("step", ""), body.get("error", ""),
                          body.get("query_raw", ""), cfg, user["email"])
    status = out.get("note_status", "")
    if status.startswith("failed"):
        raise HTTPException(500, status)
    if status == "disabled":
        raise HTTPException(409, "user-submitted fixes are disabled in config")
    if status == "no_failure_context":
        raise HTTPException(400, "this result has no step/error to attach a fix to")
    return {"ok": True, **out}


# ── User-submitted workarounds: pending review + approval ─────────────────────

@app.get("/api/suggestions/pending")
def suggestions_pending(admin: dict = Depends(require_admin)):
    """User-submitted fixes (from the in-Jira feedback fields) awaiting review.
    Admin-only — this is the approval queue, not a general activity feed."""
    import suggestions as sg
    cfg = load_config()
    if not cfg["workaround_finder"].get("suggestions_enabled", True):
        return {"ok": True, "pending": []}
    return {"ok": True, "pending": sg.list_by_status("pending", cfg)}


@app.post("/api/suggestions/{sid}/approve")
def suggestions_approve(sid: str, admin: dict = Depends(require_admin)):
    """Approve a pending suggestion → embed it into the searchable index as a
    verified user fix, then mark it approved. Admin-only: this is the gate that
    decides what becomes a trusted, indexed workaround for everyone else."""
    import suggestions as sg
    import ingest as ing
    cfg = load_config()
    rec = sg.get(sid, cfg)
    if not rec:
        raise HTTPException(404, "suggestion not found")
    if rec.get("status") != "pending":
        raise HTTPException(409, f"suggestion is already {rec.get('status')}")
    res = ing.ingest_suggestion(rec.get("step", ""), rec.get("error", ""),
                                rec.get("suggestion", ""), sid, cfg)
    if not res.get("ok"):
        raise HTTPException(400, res.get("error", "could not index suggestion"))
    sg.set_status(sid, "approved", cfg, indexed_id=res["indexed_id"],
                  approved_by=admin["email"])
    return {"ok": True, "indexed_id": res["indexed_id"]}


@app.post("/api/suggestions/{sid}/reject")
def suggestions_reject(sid: str, admin: dict = Depends(require_admin)):
    import suggestions as sg
    cfg = load_config()
    rec = sg.get(sid, cfg)
    if not rec:
        raise HTTPException(404, "suggestion not found")
    sg.set_status(sid, "rejected", cfg, approved_by=admin["email"])
    return {"ok": True}


# ── Deterministic workaround rules (exact-match, reviewed) ───────────────────
# Distinct from the suggestions queue above: a suggestion gets embedded and found by
# similarity later (a score, never guaranteed); a rule's error_description either
# matches the incoming failure exactly or it doesn't — see rules.py and
# suggest.suggest_for_query, which checks these before falling back to search.

@app.get("/api/rules")
def rules_list(status: str = "", user: dict = Depends(require_login)):
    import rules as rl
    cfg = load_config()
    return {"ok": True, "rules": rl.list_rules(cfg, status=status)}


@app.post("/api/rules")
def rules_create(body: dict, user: dict = Depends(require_login)):
    """Propose a new rule (draft) — any signed-in user; mirrors /api/suggestions."""
    import rules as rl
    cfg   = load_config()
    match = body.get("match") or {}
    try:
        rec = rl.create(match, body.get("workaround_text", ""), user["email"], cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "rule": rec}


@app.post("/api/rules/{rule_id}/approve")
def rules_approve(rule_id: str, admin: dict = Depends(require_admin)):
    """Admin-only. Refuses (409) when the rule would be genuinely ambiguous against
    an already-approved rule — narrow one of the two, or merge them, before retrying."""
    import rules as rl
    cfg  = load_config()
    rule = rl.get(rule_id, cfg)
    if not rule:
        raise HTTPException(404, "rule not found")
    if rule.get("status") == "approved":
        raise HTTPException(409, "rule is already approved")
    conflicts = rl.check_overlap(rule.get("match", {}), cfg, exclude_id=rule_id)
    if conflicts:
        raise HTTPException(409, {
            "message": "would be ambiguous against an already-approved rule",
            "conflicts": [{"id": c["id"], "workaround_text": c["workaround_text"][:200]}
                          for c in conflicts],
        })
    updated = rl.approve(rule_id, admin["email"], cfg)
    return {"ok": True, "rule": updated}


@app.post("/api/rules/clean-preview")
def rules_clean_preview(body: dict, user: dict = Depends(require_login)):
    """The exact string a rule's error_description pattern is tested against —
    error_matches cleans with the same function before matching (see rules.py's module
    docstring for why). Lets the Rules tab's pattern tester show the real thing instead
    of the raw text, which almost never matches (hyphens become spaces, etc.)."""
    import textclean as tc
    return {"ok": True, "cleaned": tc.clean_error_text(body.get("text", ""))}


@app.post("/api/rules/{rule_id}/deprecate")
def rules_deprecate(rule_id: str, admin: dict = Depends(require_admin)):
    import rules as rl
    cfg  = load_config()
    rule = rl.get(rule_id, cfg)
    if not rule:
        raise HTTPException(404, "rule not found")
    updated = rl.deprecate(rule_id, cfg)
    return {"ok": True, "rule": updated}


# ── Jira auto-suggest: feedback links + poller ────────────────────────────────

def _fb_page(title: str, body_html: str, tone: str = "ok") -> HTMLResponse:
    """Minimal self-contained page shown after a 👍/👎 link is clicked from Jira."""
    color = {"ok": "#22c55e", "warn": "#eab308", "err": "#ef4444"}.get(tone, "#22c55e")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FalloutAssist feedback</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:system-ui,Segoe UI,sans-serif;
          display:flex; min-height:100vh; align-items:center; justify-content:center; margin:0; }}
  .card {{ background:#1e293b; border:1px solid #334155; border-left:4px solid {color};
           border-radius:12px; padding:28px 32px; max-width:720px; }}
  h1 {{ font-size:18px; margin:0 0 12px; color:#f8fafc; }}
  p {{ line-height:1.6; color:#cbd5e1; }}
  pre {{ background:#0f172a; border:1px solid #334155; border-radius:8px; padding:12px;
         white-space:pre-wrap; color:#e2e8f0; font-size:13px; }}
  a.btn {{ display:inline-block; margin-top:14px; background:#ef4444; color:#fff;
           text-decoration:none; padding:10px 18px; border-radius:8px; font-weight:600; }}
  .muted {{ color:#64748b; font-size:12px; margin-top:16px; }}
  /* Feedback-details table. Scrolls inside its own box so a long voter name can never
     make the whole page scroll sideways on a phone. */
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:4px; }}
  th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid #334155;
            white-space:nowrap; }}
  th {{ color:#94a3b8; font-weight:600; font-size:11px; text-transform:uppercase;
        letter-spacing:.04em; }}
  td {{ color:#e2e8f0; }}
  td.dim {{ color:#94a3b8; }}
  /* Resolution-format page. The table is both what you read and what gets copied, so it
     carries real borders — they survive into the pasted Jira table. The textarea holds
     the wiki-markup flavour and is never shown; it is not `display:none`, because a
     hidden control's .value must stay readable and selectable. */
  table.wa {{ border:1px solid #334155; }}
  table.wa th, table.wa td {{ border:1px solid #334155; vertical-align:top;
                              white-space:normal; }}
  table.wa th {{ width:34%; color:#cbd5e1; text-transform:none; font-size:13px;
                 letter-spacing:0; background:#0f172a; }}
  /* The copy sources. Offscreen rather than display:none — hidden content cannot be
     selected, and the Ctrl+C fallback needs a real selection. #plain deliberately gets
     NO styling of its own, so nothing from this page can ride along into the paste. */
  .offscreen {{ position:absolute; left:-9999px; top:0; }}
  textarea {{ position:absolute; left:-9999px; width:1px; height:1px; opacity:0; }}
  .btn2 {{ margin-top:12px; background:#2563eb; color:#fff; border:0; cursor:pointer;
           padding:10px 18px; border-radius:8px; font-weight:600; font-size:14px; }}
  .btn2:hover {{ background:#1d4ed8; }}
  .ok-note {{ color:#22c55e; font-size:13px; margin-left:10px; font-weight:600; }}
</style></head><body><div class="card">
<h1>{title}</h1>{body_html}
<div class="muted">FalloutAssist · you can close this tab.</div>
</div></body></html>"""
    return HTMLResponse(html)


@app.get("/api/jira-feedback", response_class=HTMLResponse)
def jira_feedback(key: str = "", cand: str = "", action: str = "",
                  sig: str = "", confirm: int = 0):
    """Feedback links embedded in Jira comments land here.
      action=up                → confirm the workaround (boost), keep the comment.
      action=down (no confirm) → show a confirmation page (guards against link-prefetch).
      action=down&confirm=1    → demote + re-match + update the comment in place."""
    import jirabot
    key, cand, action = key.strip(), cand.strip(), action.strip()
    if action not in ("up", "down") or not key or not cand:
        return _fb_page("Invalid link", "<p>This feedback link is malformed.</p>", "err")
    if not jirabot.verify(key, cand, action, sig):
        return _fb_page("Invalid or expired link",
                        "<p>This feedback link could not be verified.</p>", "err")

    cfg = load_config()

    if action == "up":
        jirabot.apply_vote(key, cand, "up", cfg)
        return _fb_page("👍 Thanks — confirmed",
                        f"<p>Marked the workaround on <b>{key}</b> as a good fix. "
                        "It'll be favoured for similar failures from now on.</p>", "ok")

    # action == "down"
    if not confirm:
        # Confirmation step — a bare GET (email/link previewers) must not trigger an edit.
        url = (f"/api/jira-feedback?key={key}&cand={cand}&action=down"
               f"&sig={sig}&confirm=1")
        return _fb_page(
            "👎 Improve this suggestion?",
            f"<p>This will mark the current workaround on <b>{key}</b> as not right, "
            "and replace the comment with the next-best matched workaround.</p>"
            f'<a class="btn" href="{url}">Confirm &amp; improve</a>', "warn")

    out = jirabot.apply_vote(key, cand, "down", cfg)
    if out.get("improved"):
        return _fb_page("👎 Updated with a better match",
                        "<p>Thanks — the comment on <b>{0}</b> was replaced with the "
                        "next-best matched workaround:</p><pre>{1}</pre>".format(
                            key, _esc(out.get("answer", ""))), "ok")
    return _fb_page("👎 Noted — needs manual review",
                    f"<p>No further confident workaround was found for <b>{key}</b>. "
                    "It's been flagged for manual review.</p>", "warn")


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fb_when(ts: str) -> str:
    """'2026-07-29T18:30:00.123456' -> '2026-07-29 18:30 UTC'. Records are written with
    utcnow() (feedback.record), so the label is accurate, not assumed."""
    t = (ts or "").strip()
    if len(t) < 16:
        return t
    return f"{t[:10]} {t[11:16]} UTC"


@app.get("/api/resolution-format", response_class=HTMLResponse)
def resolution_format_page(key: str = "", cand: str = "", sig: str = ""):
    """The "Copy the resolution format" page behind the link in a Jira comment.

    Replaces the second comment the bot used to post: the 8-field table is shown here,
    pre-filled, with a one-click copy so the assignee pastes it as their resolution
    comment instead of retyping it. Read-only — it records nothing."""
    import jirabot
    key, cand = key.strip(), cand.strip()
    if not key:
        return _fb_page("Invalid link", "<p>This format link is malformed.</p>", "err")
    if not jirabot.verify(key, cand, jirabot.FORMAT_ACTION, sig):
        return _fb_page("Invalid or expired link",
                        "<p>This format link could not be verified.</p>", "err")

    d = jirabot.resolution_format(key, load_config())
    if not d["template"]:
        return _fb_page(f"Resolution format for {_esc(key)}",
                        "<p>Couldn't build the format for this ticket — Jira may be "
                        "unreachable. Open the app and search the failure to get it.</p>",
                        "err")

    if d["prefilled"]:
        intro = (f"<p>Pre-filled from the suggested workaround"
                 + (f" (matched <b>{_esc(d['matched'])}</b>)" if d["matched"] else "")
                 + ". <b>Check every row against what you actually did</b> and correct it "
                 "before posting — the bottom four rows are what get reused on other "
                 "tickets.</p>")
    else:
        intro = ("<p>No past fix matched this failure, so there's nothing to pre-fill — "
                 "<b>BAN CAN</b> and <b>MSISDN</b> are filled in from the ticket. Complete "
                 "the rest as you work it: yours would be the first indexed resolution for "
                 "this failure.</p>")

    # Rendered as a REAL html table, and copied as one. The stored template is Jira WIKI
    # markup (|label|value|), which only renders as a table in the old wiki editor — in
    # Jira Cloud's rich-text (ADF) editor, pasting it gives literal pipe characters, which
    # is exactly what it looked like before. The ADF editor DOES convert pasted HTML
    # tables into real tables, so the copy puts two flavours on the clipboard:
    #   text/html  -> this <table>, for the Cloud editor (and anything rich-text)
    #   text/plain -> the pipe markup, for a wiki-markup editor or a plain-text field
    # Whichever the target understands, the other is ignored. Both parse back through
    # watable.parse_table on ingest, which already handles the ADF-rendered form.
    import watable as wt
    trs   = list(wt.rows(d["template"]))
    # TWO renderings of the same rows, and the difference is the point:
    #   #tbl   — what you see. Styled for this dark page.
    #   #plain — what gets COPIED. No class, no styles, and <td> rather than <th>, so it
    #            pastes as an unstyled table that picks up Jira's own table look instead
    #            of dragging this page's colours into the ticket. <th> would also come
    #            back from the /2/ API as Jira's `||header||` form.
    # Kept as a real (offscreen) DOM node rather than a JS string, so the markup needs no
    # escaping and the Ctrl+C fallback has something plain to select.
    disp  = "".join(f"<tr><th>{_esc(l)}</th><td>{_esc(v)}</td></tr>" for l, v in trs)
    plain = "".join(f"<tr><td>{_esc(l)}</td><td>{_esc(v)}</td></tr>" for l, v in trs)

    body = f"""{intro}
<div class="scroll"><table id="tbl" class="wa">{disp}</table></div>
<button class="btn2" id="copy" onclick="doCopy()">📋 Copy to clipboard</button>
<span id="done" class="ok-note"></span>
<div class="offscreen" aria-hidden="true"><table id="plain">{plain}</table></div>
<textarea id="tpl" readonly aria-hidden="true" tabindex="-1"
          >{_esc(d['template'])}</textarea>
<p class="muted">Paste this as your resolution comment when you close
<b>{_esc(key)}</b> — it pastes as a table. Use <b>NA</b> for a row that genuinely
doesn't apply, and keep account numbers / MSISDNs out of the Cause, Solution applied and
Customer action rows — those rows get reused on other customers' orders.</p>
<script>
function doCopy() {{
  // Copy the PLAIN table, never the styled one on screen.
  var tbl   = document.getElementById('plain');
  var plain = document.getElementById('tpl').value;
  var html  = tbl.outerHTML;
  var note  = document.getElementById('done');
  function ok() {{
    note.textContent = 'Copied as a table';
    setTimeout(function () {{ note.textContent = ''; }}, 2500);
  }}
  // Secure context only (https/localhost). This app is served over http:// on a LAN
  // IP, so in practice the legacy path below is the one that runs.
  if (navigator.clipboard && window.isSecureContext && window.ClipboardItem) {{
    try {{
      navigator.clipboard.write([new ClipboardItem({{
        'text/html':  new Blob([html],  {{type: 'text/html'}}),
        'text/plain': new Blob([plain], {{type: 'text/plain'}})
      }})]).then(ok, legacy);
      return;
    }} catch (e) {{ /* fall through */ }}
  }}
  legacy();

  function legacy() {{
    // execCommand copies the SELECTION, so a selection has to exist — but the copy
    // event handler then overrides both flavours, so what is selected doesn't decide
    // what lands on the clipboard.
    function onCopy(e) {{
      var cd = e.clipboardData || window.clipboardData;
      if (!cd) {{ return; }}
      cd.setData('text/html',  html);
      cd.setData('text/plain', plain);
      e.preventDefault();
    }}
    document.addEventListener('copy', onCopy);
    var sel   = window.getSelection();
    var saved = sel.rangeCount ? sel.getRangeAt(0) : null;
    var rng   = document.createRange();
    rng.selectNodeContents(tbl);
    sel.removeAllRanges(); sel.addRange(rng);
    var done = false;
    try {{ done = document.execCommand('copy'); }} catch (e) {{}}
    document.removeEventListener('copy', onCopy);
    sel.removeAllRanges();
    if (saved) {{ sel.addRange(saved); }}
    if (done) {{ ok(); return; }}
    // Last resort: leave the table selected so Ctrl+C copies it as rich text anyway.
    sel.addRange(rng);
    note.textContent = 'Press Ctrl+C — the table is selected';
  }}
}}
</script>"""
    return _fb_page(f"Resolution format for {_esc(key)}", body, "ok")


@app.get("/api/feedback-details", response_class=HTMLResponse)
def feedback_details_page(key: str = "", cand: str = "", sig: str = ""):
    """The "who gave feedback" page behind the *Details* link in a Jira comment.

    Read-only: it records nothing and changes nothing, so unlike the 👎 link it needs no
    confirmation step. Still HMAC-verified — it discloses who voted, so it must not be a
    guessable public endpoint."""
    import jirabot
    key, cand = key.strip(), cand.strip()
    if not key or not cand:
        return _fb_page("Invalid link", "<p>This details link is malformed.</p>", "err")
    if not jirabot.verify(key, cand, jirabot.DETAILS_ACTION, sig):
        return _fb_page("Invalid or expired link",
                        "<p>This details link could not be verified.</p>", "err")

    d      = jirabot.feedback_details(key, cand, load_config())
    counts = d["counts"]
    votes  = d["votes"]

    head = (f"<p>Feedback on the workaround matched from <b>{_esc(cand)}</b> "
            f"for the failure on <b>{_esc(key)}</b>"
            + (f" — <b>{_esc(d['error'] or d['step'])}</b>" if (d["error"] or d["step"])
               else "") + ".</p>"
            f"<p style='font-size:20px;margin:4px 0 18px'>👍 {counts['up']} "
            f"&nbsp;·&nbsp; 👎 {counts['down']}</p>")

    if not votes:
        return _fb_page(f"Feedback on {_esc(key)}",
                        head + "<p class='muted'>No votes recorded for this workaround "
                        "on this failure yet.</p>", "warn")

    rows = []
    for v in votes:
        who = _esc(v["voter"]) if v["voter"] else "<i>anonymous</i>"
        # How the name was obtained, so an approximation is never mistaken for a
        # confirmed identity. Only the in-Jira field channel can attribute a vote;
        # in-app votes and comment-link clicks genuinely cannot.
        src = {"jira_field": "Jira field",
               "assignee":   "assignee (assumed)"}.get(v["voter_source"], "link / in-app")
        rows.append(
            "<tr>"
            f"<td>{'👍' if v['vote'] == 'up' else '👎'}</td>"
            f"<td>{who}</td>"
            f"<td class='dim'>{_esc(src)}</td>"
            f"<td class='dim'>{_esc(v['from_key'])}</td>"
            f"<td class='dim'>{_esc(_fb_when(v['ts']))}</td>"
            "</tr>")

    table = (
        "<div class='scroll'><table><thead><tr><th></th><th>Who</th><th>Via</th>"
        "<th>From ticket</th><th>When</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
        "<p class='muted'>Names come from the in-Jira <i>Workaround helpful?</i> field. "
        "A 👍/👎 clicked from a comment link, or given in the app, cannot be attributed "
        "to a person — one link is shared by everyone reading the ticket, and the app has "
        "no sign-in — so those are shown as anonymous rather than guessed.</p>")

    return _fb_page(f"Feedback on {_esc(key)}", head + table, "ok")


@app.get("/api/jira-suggest/status")
def jira_suggest_status(user: dict = Depends(require_login)):
    import jirabot
    return jirabot.get_status(load_config())


@app.get("/api/jira-suggest/previews")
def jira_suggest_previews(user: dict = Depends(require_login)):
    """The current per-ticket suggestions (would-post in dry-run / posted in live),
    so the UI can show them instead of grepping the log."""
    import jirabot
    return {"ok": True, "previews": jirabot.list_previews(load_config())}


@app.post("/api/jira-suggest/config")
def jira_suggest_config(body: dict, admin: dict = Depends(require_admin)):
    """Persist the auto-suggest settings (config.json). Takes effect within ~1 min
    (the poller re-reads config live). Mirrors the LLM-toggle endpoints."""
    import jirabot
    cfg = load_config()
    wf  = cfg["workaround_finder"]
    if "enabled" in body:
        wf["jira_suggest_enabled"] = bool(body["enabled"])
    if "dry_run" in body:
        wf["jira_suggest_dry_run"] = bool(body["dry_run"])
    # Interval: `seconds` is authoritative. A client that still posts `minutes` (the
    # older dashboard) is translated into seconds — otherwise its control would save
    # successfully and change nothing, because jira_suggest_seconds takes precedence.
    if "seconds" in body:
        secs = max(0, int(body["seconds"]))
        wf["jira_suggest_seconds"] = secs
        wf.pop("jira_suggest_minutes", None)
    elif "minutes" in body:
        mins = max(0, int(body["minutes"]))
        wf["jira_suggest_seconds"] = mins * 60
        wf.pop("jira_suggest_minutes", None)
    if "jql" in body and str(body["jql"]).strip():
        wf["jira_suggest_jql"] = str(body["jql"]).strip()
    if "public_base_url" in body:
        wf["public_base_url"] = str(body["public_base_url"]).strip()
    save_config(cfg)
    return jirabot.get_status(cfg)


@app.post("/api/jira-suggest/run")
def jira_suggest_run(admin: dict = Depends(require_admin)):
    """Trigger one auto-suggest pass on demand (background) instead of waiting for
    the timer. Honors the dry-run flag exactly like a scheduled run."""
    import jirabot
    cfg = load_config()
    if not cfg["workaround_finder"].get("jira_suggest_jql", "").strip():
        raise HTTPException(400, "Set the inflow JQL first.")

    def _run():
        try:
            jirabot.run_once(cfg)
        except Exception as e:
            print(f"[JIRA-SUGGEST] manual run failed: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "status": "started"}


# ── Documents ─────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".xls"}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), admin: dict = Depends(require_admin)):
    import ingest as ing
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    if not _try_acquire():
        return JSONResponse({"ok": False, "status": "already_running"}, status_code=409)
    try:
        content = await file.read()
        cfg     = load_config()
        result  = ing.ingest_document(content, file.filename, cfg)
        return result
    finally:
        _release()


@app.get("/api/documents")
def list_documents(user: dict = Depends(require_login)):
    import ingest as ing
    cfg = load_config()
    return {"ok": True, "documents": ing.list_documents(cfg)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, admin: dict = Depends(require_admin)):
    import ingest as ing
    cfg = load_config()
    return ing.delete_document(doc_id, cfg)


# ── LLM settings ─────────────────────────────────────────────────────────────

# Only Synapt (the org's governed Azure OpenAI deployment) is exposed in the UI:
# production org data must stay inside the org tenant, so the free-tier clouds
# (Groq / Cerebras / NVIDIA) and Claude/local are intentionally not selectable.
# generate.py still implements them for internal use, but they cannot be picked or
# set as a fallback — the save/fallback endpoints validate against this dict.
PROVIDERS = {
    "synapt":   {"label": "Synapt (Azure OpenAI)", "privacy": "governed", "needs_key": True},
}

PROVIDER_MODELS = {
    "synapt":   ["gpt-4o-mini"],
}


@app.get("/api/llm/providers")
def llm_providers(user: dict = Depends(require_login)):
    import generate as g
    cfg = load_config()
    return {
        "ok":        True,
        "providers": PROVIDERS,
        "models":    PROVIDER_MODELS,
        "current":   {"provider": cfg["llm"]["provider"], "model": cfg["llm"]["model"]},
        # Auto-failover order tried when the selected provider errors/rate-limits.
        "fallback":  cfg["llm"].get("fallback", []),
        # Which providers already have key(s) saved in .env → drives the masked
        # "Key saved" UI state per provider. key_counts powers the multi-key editor
        # (a provider may hold several rotating free-tier keys).
        "has_keys":   {p: bool(g._provider_keys(_key_env(p))) for p in PROVIDERS},
        "key_counts": {p: len(g._provider_keys(_key_env(p))) for p in PROVIDERS},
        "enabled":   cfg["workaround_finder"].get("llm_enabled", True),
        "rerank":    cfg["workaround_finder"].get("llm_rerank", False),
        "synthesize": cfg["workaround_finder"].get("llm_synthesize", True),
    }


@app.post("/api/llm/enabled")
def llm_set_enabled(body: dict, admin: dict = Depends(require_admin)):
    """Master switch for ALL LLM use (synthesis, re-rank, parse). When off, the
    tool does pure retrieval and shows matched comments verbatim — no LLM calls,
    cloud or local. Persisted to config.json."""
    cfg = load_config()
    cfg["workaround_finder"]["llm_enabled"] = bool(body.get("enabled", True))
    save_config(cfg)
    return {"ok": True, "enabled": cfg["workaround_finder"]["llm_enabled"]}


@app.post("/api/llm/rerank")
def llm_set_rerank(body: dict, admin: dict = Depends(require_admin)):
    """Toggle LLM relevance re-ranking on/off (persisted to config.json)."""
    cfg = load_config()
    cfg["workaround_finder"]["llm_rerank"] = bool(body.get("enabled", False))
    save_config(cfg)
    return {"ok": True, "enabled": cfg["workaround_finder"]["llm_rerank"]}


@app.post("/api/llm/synthesize")
def llm_set_synthesize(body: dict, admin: dict = Depends(require_admin)):
    """Toggle LLM workaround synthesis on/off (persisted to config.json). When
    off, a strong match is shown verbatim instead of being rewritten into the
    `=== FIX ===` format."""
    cfg = load_config()
    cfg["workaround_finder"]["llm_synthesize"] = bool(body.get("enabled", True))
    save_config(cfg)
    return {"ok": True, "enabled": cfg["workaround_finder"]["llm_synthesize"]}


@app.get("/api/llm/config")
def llm_config_get(admin: dict = Depends(require_admin)):
    cfg = load_config()
    return {
        "ok":       True,
        "provider": cfg["llm"]["provider"],
        "model":    cfg["llm"]["model"],
        "has_key":  bool(os.getenv(_key_env(cfg["llm"]["provider"]), "")),
    }


@app.post("/api/llm/config")
def llm_config_set(body: dict, admin: dict = Depends(require_admin)):
    provider = body.get("provider", "local")
    model    = body.get("model", "")
    api_key  = body.get("api_key", "").strip()

    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider: {provider}")

    cfg                = load_config()
    cfg["llm"]["provider"] = provider
    cfg["llm"]["model"]    = model
    # Optional fallback order: providers tried (in order) when the selected one
    # errors/rate-limits. Validated against known providers; unknown names dropped.
    if "fallback" in body and isinstance(body["fallback"], list):
        cfg["llm"]["fallback"] = [p for p in body["fallback"]
                                  if p in PROVIDERS and p != provider]
    save_config(cfg)

    # A key supplied here is APPENDED to the provider's rotation list (not an
    # overwrite) so saving provider/model never wipes other keys. Multi-key
    # management proper lives in /api/llm/keys.
    if api_key:
        _add_provider_key(provider, api_key)

    return {"ok": True, "provider": provider, "model": model}


@app.get("/api/llm/keys")
def llm_keys_get(provider: str, admin: dict = Depends(require_admin)):
    """Masked list of keys configured for a provider (fingerprint + last 4 chars).
    Never returns the secret itself."""
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider: {provider}")
    return {"ok": True, "provider": provider, "keys": _masked_keys(provider)}


@app.post("/api/llm/keys")
def llm_keys_mutate(body: dict, admin: dict = Depends(require_admin)):
    """Add or remove one key in a provider's rotation list (stored comma-separated
    in .env). Returns the updated masked list. body: {provider, action, api_key|fp}."""
    provider = body.get("provider", "")
    action   = body.get("action", "")
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider: {provider}")
    if action == "add":
        # Accept one OR many keys in a single request (comma- or newline-separated)
        # so the user can paste a whole batch of free-tier keys at once.
        raw   = body.get("api_key", "")
        added = sum(1 for k in re.split(r'[,\n]', raw) if _add_provider_key(provider, k))
        return {"ok": True, "added": added, "keys": _masked_keys(provider)}
    if action == "remove":
        import generate as g
        fp   = body.get("fp", "")
        keys = [k for k in _provider_keys_list(provider) if g._key_fp(k) != fp]
        _set_provider_keys(provider, keys)
        return {"ok": True, "removed": True, "keys": _masked_keys(provider)}
    raise HTTPException(400, f"Unknown action: {action}")


@app.post("/api/llm/test")
def llm_test(admin: dict = Depends(require_admin)):
    import generate as g
    cfg = load_config()
    return g.test_provider(cfg["llm"])


@app.get("/api/llm/status")
def llm_status(user: dict = Depends(require_login)):
    """Lightweight connection status for the active provider (UI indicator)."""
    import generate as g
    cfg = load_config()
    return g.provider_status(cfg["llm"])


@app.get("/api/llm/quota")
def llm_quota(user: dict = Depends(require_login)):
    """Latest Groq rate-limit snapshot (captured from real call headers — costs
    nothing). Empty until the first Groq call of the session."""
    import generate as g
    cfg = load_config()
    return {"ok": True, "provider": cfg["llm"]["provider"], "quota": g.get_groq_quota()}


def _key_env(provider: str) -> str:
    return {"groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY",
            "nvidia": "NVIDIA_API_KEY", "claude": "ANTHROPIC_API_KEY",
            "synapt": "AZURE_OPENAI_API_KEY"}.get(provider, "")


# ── Multi-key storage ────────────────────────────────────────────────────────
# Keys are stored comma-separated in one .env var per provider
# (GROQ_API_KEY=k1,k2,k3) — kept out of the git-committed config.json. generate.py
# rotates across them; these helpers add/remove/list them for the Settings UI.

def _provider_keys_list(provider: str) -> list[str]:
    import generate as g
    return g._provider_keys(_key_env(provider))


def _set_provider_keys(provider: str, keys: list[str]):
    env_var = _key_env(provider)
    if not env_var:
        return
    val = ",".join(keys)
    _write_env(env_var, val)
    os.environ[env_var] = val


def _add_provider_key(provider: str, api_key: str) -> bool:
    """Append a key to the provider's rotation list. No-op (returns False) if the
    key is blank or already present."""
    import generate as g
    api_key = (api_key or "").strip()
    if not api_key:
        return False
    keys = _provider_keys_list(provider)
    if any(g._key_fp(k) == g._key_fp(api_key) for k in keys):
        return False
    keys.append(api_key)
    _set_provider_keys(provider, keys)
    return True


def _masked_keys(provider: str) -> list[dict]:
    import generate as g
    return [{"fp": g._key_fp(k), "last4": k[-4:]} for k in _provider_keys_list(provider)]


def _write_env(key: str, value: str):
    env_path = _DIR / ".env"
    lines    = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated  = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            updated  = True
            break
    if not updated:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Config ────────────────────────────────────────────────────────────────────

@app.get("/api/config")
def config_get(admin: dict = Depends(require_admin)):
    cfg = load_config()
    wf  = cfg["workaround_finder"]
    return {
        "jql":               wf["ingest_jql"],
        "top_k":             wf["top_k"],
        "jira_email":        cfg["jira"].get("email", ""),
        "ollama_url":        cfg["llm"].get("ollama_url", "http://localhost:11434"),
        "has_token":         bool(os.getenv("JIRA_API_TOKEN", "").strip()),
        # Surfaced so the tab reflects what the app is actually doing. score_threshold
        # is the knob behind the `no_match` rate — the single biggest reason a ticket
        # gets no suggestion — so it belongs in the UI rather than config.json only.
        # No synthesis_format: every suggestion now produces BOTH the === FIX === steps
        # (the workaround) and the pre-filled 8-field table (the resolution comment), so
        # there is no longer a format to pick.
        "synthesis_sources": int(wf.get("synthesis_sources", 4) or 4),
        "score_threshold":   float(wf.get("score_threshold", 0.7)),
        "embed_model":       cfg.get("embed", {}).get("model", ""),
    }


@app.post("/api/config")
def config_set(body: dict, admin: dict = Depends(require_admin)):
    cfg = load_config()
    if "jql" in body:
        cfg["workaround_finder"]["ingest_jql"] = body["jql"]
    if "top_k" in body:
        cfg["workaround_finder"]["top_k"] = int(body["top_k"])
    if "synthesis_sources" in body:
        cfg["workaround_finder"]["synthesis_sources"] = max(1, min(10, int(body["synthesis_sources"])))
    if "score_threshold" in body:
        # Clamped: below ~0.5 the lexical error gate is doing all the work and near-miss
        # failures start surfacing as confident suggestions; above 0.95 almost nothing
        # qualifies and the no_match rate goes to 100%.
        cfg["workaround_finder"]["score_threshold"] = max(0.5, min(0.95, float(body["score_threshold"])))
    if "jira_email" in body:
        cfg["jira"]["email"] = body["jira_email"]
    if body.get("ollama_url", "").strip():
        cfg["llm"]["ollama_url"] = body["ollama_url"].strip()
    save_config(cfg)

    if "jira_api_token" in body and body["jira_api_token"].strip():
        _write_env("JIRA_API_TOKEN", body["jira_api_token"])
        os.environ["JIRA_API_TOKEN"] = body["jira_api_token"]
    if "jira_email" in body:
        _write_env("JIRA_EMAIL", body["jira_email"])

    return {"ok": True}
