"""
Workaround Finder — FastAPI app
Run: uvicorn app:app --reload --port 8010
"""

import json
import os
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


@app.on_event("startup")
def _start_auto_ingest():
    threading.Thread(target=_auto_ingest_loop, daemon=True).start()
    print("[AUTO-INGEST] scheduler started (interval read live from config)")


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
    import search as s
    import generate as g
    import ingest as ing

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

    result = s.find_workarounds(query_text, cfg)
    strong  = result["strong"]
    context = result["context"]
    threshold  = result["threshold"]
    best_score = result["best_score"]

    # Nothing matched. Either the KB is empty, or no ticket matches this error
    # (a step-only match with a different error is dropped, not shown as a lead).
    if not strong and not context:
        empty  = ing.get_status(cfg).get("total_chunks", 0) == 0
        answer = ("No tickets are indexed yet — ingest first."
                  if empty else
                  "No past resolution matches this failure. No ticket has this "
                  "error, so there's no workaround to suggest (a different error on "
                  "the same step is treated as a different problem).")
        return {"ok": True, "mode": "no_data", "answer": answer,
                "strong": [], "context": [], "best_score": 0}

    wf         = cfg["workaround_finder"]
    llm_on     = wf.get("llm_enabled", True)            # master LLM switch
    synthesize = llm_on and wf.get("llm_synthesize", True)
    llm_note   = ""

    # Strong match(es) found — produce a grounded `=== FIX ===` recommendation.
    if strong:
        top_body = strong[0]["comment"]
        # Hybrid: if the LLM is disabled, or the best source is ALREADY a clean
        # === FIX === block, show it verbatim — no generation.
        if not synthesize or "=== fix ===" in top_body.lower():
            answer, provider, model = top_body, "direct_match", ""
        else:
            # Legacy / multiple comments → LLM synthesizes into the FIX format,
            # grounded in sources only. On decline (NO_RELIABLE_WORKAROUND), empty
            # output, a local-model answer, or any error → fall back to the raw
            # best comment, never invent.
            try:
                prompt = s.build_prompt(query_text, result)
                gen    = g.generate(prompt, cfg["llm"], job="synthesis")
                ans    = (gen.get("answer") or "").strip()
                if gen.get("provider") == "local":
                    print("[LLM] synthesis answered by local model — distrust, verbatim")
                    answer, provider, model = top_body, "direct_match", ""
                elif s.NO_FIX_SENTINEL in ans or len(ans) < 10:
                    print("[LLM] declined / empty — showing raw best comment")
                    answer, provider, model = top_body, "direct_match", ""
                else:
                    answer, provider, model = ans, gen["provider"], gen["model"]
            except Exception as llm_err:
                # All cloud providers failed — show the raw comment and tell the
                # user why, instead of silently dropping to a local model.
                reason   = str(llm_err)[:160]
                llm_note = f"LLM unavailable, showing the raw matched comment: {reason}"
                print(f"[LLM] synthesis failed, verbatim fallback — {reason}")
                answer, provider, model = top_body, "direct_match", ""
        mode = "strong_match"

    else:
        # No strong match — abstain from synthesis (the sources are below the
        # relevance bar; generating from them is where hallucination happens).
        # Show the nearest weak match's actual comment as a lead instead.
        top      = context[0] if context else None
        answer   = top["comment"] if top else "No similar tickets found."
        provider = "direct_match"
        model    = ""
        mode     = "low_confidence"

    return {
        "ok":        True,
        "mode":      mode,          # strong_match | low_confidence | no_data
        "answer":    answer,
        "provider":  provider,
        "model":     model,
        "llm_note":  llm_note,      # set when cloud LLM failed (verbatim fallback)
        "threshold": threshold,
        "best_score": best_score,
        "strong":    strong,        # comments scoring >= threshold
        "context":   context,       # weaker matches shown as reference
    }


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

PROVIDERS = {
    "local":    {"label": "Local (Ollama)",     "privacy": "private",    "needs_key": False},
    "groq":     {"label": "Groq",               "privacy": "sends_data", "needs_key": True},
    "cerebras": {"label": "Cerebras",           "privacy": "sends_data", "needs_key": True},
    "nvidia":   {"label": "NVIDIA NIM",         "privacy": "sends_data", "needs_key": True},
    "claude":   {"label": "Claude (Anthropic)", "privacy": "governed",   "needs_key": True},
}

PROVIDER_MODELS = {
    "groq":     ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    "cerebras": ["gpt-oss-120b", "zai-glm-4.7"],
    "nvidia":   ["meta/llama-3.1-8b-instruct", "meta/llama-3.3-70b-instruct"],
    "claude":   ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"],
}


@app.get("/api/llm/providers")
def llm_providers():
    import generate as g
    cfg          = load_config()
    local_models = g.available_local_models(cfg["llm"].get("ollama_url", "http://localhost:11434"))
    return {
        "ok":        True,
        "providers": PROVIDERS,
        "models":    {**PROVIDER_MODELS, "local": [m["name"] for m in local_models]},
        "current":   {"provider": cfg["llm"]["provider"], "model": cfg["llm"]["model"]},
        # Which providers already have a key saved in .env → drives the masked
        # "Key saved" UI state per provider.
        "has_keys":  {p: bool(os.getenv(_key_env(p), "").strip()) for p in PROVIDERS},
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
    save_config(cfg)

    if api_key:
        env_var = _key_env(provider)
        _write_env(env_var, api_key)
        os.environ[env_var] = api_key

    return {"ok": True, "provider": provider, "model": model}


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
            "nvidia": "NVIDIA_API_KEY", "claude": "ANTHROPIC_API_KEY"}.get(provider, "")


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
