"""
Workaround Finder — FastAPI app
Run: uvicorn app:app --reload --port 8010
"""

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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

_DIR        = Path(__file__).parent
CONFIG_PATH = _DIR / "config.json"

app = FastAPI(title="FalloutAssist")
app.mount("/static", StaticFiles(directory=str(_DIR / "static")), name="static")


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
# (or, in dry-run, logs what it would post). Config is re-read every minute, so
# enabling/disabling or changing the cadence takes effect within ~1 min. Skips a
# tick while an ingest is running (the index is in flux).

def _jira_suggest_loop():
    import jirabot
    elapsed = 0
    while True:
        time.sleep(60)
        try:
            cfg = load_config()
            wf  = cfg["workaround_finder"]
            enabled  = bool(wf.get("jira_suggest_enabled", False))
            interval = int(wf.get("jira_suggest_minutes", 1) or 0)
        except Exception as e:
            print(f"[JIRA-SUGGEST] config read failed: {e}")
            continue
        if not enabled or interval <= 0:
            elapsed = 0
            continue
        elapsed += 1
        if elapsed < interval:
            continue
        elapsed = 0

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


# ── Jira ingest ───────────────────────────────────────────────────────────────

@app.post("/api/ingest/preview")
def ingest_preview(body: dict):
    import ingest as ing
    jql = body.get("jql", "").strip()
    if not jql:
        raise HTTPException(400, "jql required")
    cfg = load_config()
    return ing.preview_jql(jql, cfg)


@app.post("/api/ingest/start")
def ingest_start(body: dict):
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
def ingest_status():
    import ingest as ing
    cfg = load_config()
    s   = ing.get_status(cfg)
    s["running"] = _running
    return s


@app.get("/api/scorecard")
def scorecard_status():
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
def ingest_schedule_get():
    """Current background auto-ingest schedule."""
    wf = load_config()["workaround_finder"]
    return {
        "ok":               True,
        "minutes":          int(wf.get("auto_ingest_minutes", 0) or 0),   # 0 = off
        "lookback_minutes": int(wf.get("auto_ingest_lookback_minutes", 1440) or 1440),
        "running":          _running,
    }


@app.post("/api/ingest/schedule")
def ingest_schedule_set(body: dict):
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
async def ask(body: dict):
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
def feedback_record(body: dict):
    """Record a 👍 (correct) / 👎 (wrong) vote on a suggested workaround. Scoped to
    (workaround identity, failure) so it trains ranking for that step+error only.
    Pure runtime signal — does not touch the index, so no re-ingest is needed."""
    import feedback as fb
    vote = body.get("vote", "")
    key  = (body.get("key", "") or "").strip()
    if vote not in ("up", "down"):
        raise HTTPException(400, "vote must be 'up' or 'down'")
    if not key:
        raise HTTPException(400, "key required (ticket key or document filename)")
    cfg = load_config()
    rec = fb.record(
        vote=vote,
        kind=body.get("kind", "ticket"),
        key=key,
        step=body.get("step", ""),
        error=body.get("error", ""),
        query_raw=body.get("query_raw", ""),
        cfg=cfg,
    )
    return {"ok": True, "vote": rec["vote"]}


# ── User-submitted workarounds: pending review + approval ─────────────────────

@app.get("/api/suggestions/pending")
def suggestions_pending():
    """User-submitted fixes (from the in-Jira feedback fields) awaiting review."""
    import suggestions as sg
    cfg = load_config()
    if not cfg["workaround_finder"].get("suggestions_enabled", True):
        return {"ok": True, "pending": []}
    return {"ok": True, "pending": sg.list_by_status("pending", cfg)}


@app.post("/api/suggestions/{sid}/approve")
def suggestions_approve(sid: str):
    """Approve a pending suggestion → embed it into the searchable index as a
    verified user fix, then mark it approved."""
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
    sg.set_status(sid, "approved", cfg, indexed_id=res["indexed_id"])
    return {"ok": True, "indexed_id": res["indexed_id"]}


@app.post("/api/suggestions/{sid}/reject")
def suggestions_reject(sid: str):
    import suggestions as sg
    cfg = load_config()
    rec = sg.get(sid, cfg)
    if not rec:
        raise HTTPException(404, "suggestion not found")
    sg.set_status(sid, "rejected", cfg)
    return {"ok": True}


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
           border-radius:12px; padding:28px 32px; max-width:560px; }}
  h1 {{ font-size:18px; margin:0 0 12px; color:#f8fafc; }}
  p {{ line-height:1.6; color:#cbd5e1; }}
  pre {{ background:#0f172a; border:1px solid #334155; border-radius:8px; padding:12px;
         white-space:pre-wrap; color:#e2e8f0; font-size:13px; }}
  a.btn {{ display:inline-block; margin-top:14px; background:#ef4444; color:#fff;
           text-decoration:none; padding:10px 18px; border-radius:8px; font-weight:600; }}
  .muted {{ color:#64748b; font-size:12px; margin-top:16px; }}
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


@app.get("/api/jira-suggest/status")
def jira_suggest_status():
    import jirabot
    return jirabot.get_status(load_config())


@app.get("/api/jira-suggest/previews")
def jira_suggest_previews():
    """The current per-ticket suggestions (would-post in dry-run / posted in live),
    so the UI can show them instead of grepping the log."""
    import jirabot
    return {"ok": True, "previews": jirabot.list_previews(load_config())}


@app.post("/api/jira-suggest/config")
def jira_suggest_config(body: dict):
    """Persist the auto-suggest settings (config.json). Takes effect within ~1 min
    (the poller re-reads config live). Mirrors the LLM-toggle endpoints."""
    import jirabot
    cfg = load_config()
    wf  = cfg["workaround_finder"]
    if "enabled" in body:
        wf["jira_suggest_enabled"] = bool(body["enabled"])
    if "dry_run" in body:
        wf["jira_suggest_dry_run"] = bool(body["dry_run"])
    if "minutes" in body:
        wf["jira_suggest_minutes"] = max(0, int(body["minutes"]))
    if "jql" in body and str(body["jql"]).strip():
        wf["jira_suggest_jql"] = str(body["jql"]).strip()
    if "public_base_url" in body:
        wf["public_base_url"] = str(body["public_base_url"]).strip()
    save_config(cfg)
    return jirabot.get_status(cfg)


@app.post("/api/jira-suggest/run")
def jira_suggest_run():
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
async def upload_document(file: UploadFile = File(...)):
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
def list_documents():
    import ingest as ing
    cfg = load_config()
    return {"ok": True, "documents": ing.list_documents(cfg)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
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
def llm_providers():
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
def llm_set_enabled(body: dict):
    """Master switch for ALL LLM use (synthesis, re-rank, parse). When off, the
    tool does pure retrieval and shows matched comments verbatim — no LLM calls,
    cloud or local. Persisted to config.json."""
    cfg = load_config()
    cfg["workaround_finder"]["llm_enabled"] = bool(body.get("enabled", True))
    save_config(cfg)
    return {"ok": True, "enabled": cfg["workaround_finder"]["llm_enabled"]}


@app.post("/api/llm/rerank")
def llm_set_rerank(body: dict):
    """Toggle LLM relevance re-ranking on/off (persisted to config.json)."""
    cfg = load_config()
    cfg["workaround_finder"]["llm_rerank"] = bool(body.get("enabled", False))
    save_config(cfg)
    return {"ok": True, "enabled": cfg["workaround_finder"]["llm_rerank"]}


@app.post("/api/llm/synthesize")
def llm_set_synthesize(body: dict):
    """Toggle LLM workaround synthesis on/off (persisted to config.json). When
    off, a strong match is shown verbatim instead of being rewritten into the
    `=== FIX ===` format."""
    cfg = load_config()
    cfg["workaround_finder"]["llm_synthesize"] = bool(body.get("enabled", True))
    save_config(cfg)
    return {"ok": True, "enabled": cfg["workaround_finder"]["llm_synthesize"]}


@app.get("/api/llm/config")
def llm_config_get():
    cfg = load_config()
    return {
        "ok":       True,
        "provider": cfg["llm"]["provider"],
        "model":    cfg["llm"]["model"],
        "has_key":  bool(os.getenv(_key_env(cfg["llm"]["provider"]), "")),
    }


@app.post("/api/llm/config")
def llm_config_set(body: dict):
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
def llm_keys_get(provider: str):
    """Masked list of keys configured for a provider (fingerprint + last 4 chars).
    Never returns the secret itself."""
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider: {provider}")
    return {"ok": True, "provider": provider, "keys": _masked_keys(provider)}


@app.post("/api/llm/keys")
def llm_keys_mutate(body: dict):
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
def llm_test():
    import generate as g
    cfg = load_config()
    return g.test_provider(cfg["llm"])


@app.get("/api/llm/status")
def llm_status():
    """Lightweight connection status for the active provider (UI indicator)."""
    import generate as g
    cfg = load_config()
    return g.provider_status(cfg["llm"])


@app.get("/api/llm/quota")
def llm_quota():
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
def config_get():
    cfg = load_config()
    return {
        "jql":        cfg["workaround_finder"]["ingest_jql"],
        "top_k":      cfg["workaround_finder"]["top_k"],
        "jira_email": cfg["jira"].get("email", ""),
        "ollama_url": cfg["llm"].get("ollama_url", "http://localhost:11434"),
        "has_token":  bool(os.getenv("JIRA_API_TOKEN", "").strip()),
    }


@app.post("/api/config")
def config_set(body: dict):
    cfg = load_config()
    if "jql" in body:
        cfg["workaround_finder"]["ingest_jql"] = body["jql"]
    if "top_k" in body:
        cfg["workaround_finder"]["top_k"] = int(body["top_k"])
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
