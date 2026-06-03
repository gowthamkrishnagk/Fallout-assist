"""
Workaround Finder — FastAPI app
Run: uvicorn app:app --reload --port 8010
"""

import json
import os
import threading
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
    cfg = load_config()
    jql = body.get("jql", "").strip()
    if jql:
        cfg["workaround_finder"]["ingest_jql"] = jql
        save_config(cfg)

    def _run():
        try:
            result = ing.ingest_jira(cfg)
            if result.get("ok"):
                print(f"[INGEST] Done — indexed {result['indexed']} tickets, skipped {result['skipped']}")
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
            raise HTTPException(400, f"Could not fetch ticket {ticket_id}: {e}")

    result = s.find_workarounds(query_text, cfg)
    strong  = result["strong"]
    context = result["context"]
    threshold  = result["threshold"]
    best_score = result["best_score"]

    # No data in KB at all
    if not strong and not context:
        return {"ok": True,
                "mode": "no_data",
                "answer": "No similar tickets found. Please ingest tickets first.",
                "strong": [], "context": [], "best_score": 0}

    # Strong match(es) found — return actual comment(s) as workaround
    if strong:
        if len(strong) == 1:
            answer   = strong[0]["comment"]
            provider = "direct_match"
            model    = ""
        else:
            # Multiple strong matches — try LLM synthesis, fall back to top comment
            try:
                prompt   = s.build_prompt(query_text, result)
                gen      = g.generate(prompt, cfg["llm"])
                answer   = gen["answer"]
                provider = gen["provider"]
                model    = gen["model"]
            except Exception as llm_err:
                print(f"[LLM] fallback to direct match — {llm_err}")
                answer   = strong[0]["comment"]
                provider = "direct_match"
                model    = ""
        mode = "strong_match"

    else:
        # No strong match — show top result's actual comment directly.
        # Never use LLM for low-confidence: small local models invert or misread
        # short resolution comments, producing answers worse than the raw text.
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
    "local":  {"label": "Local (Ollama)",    "privacy": "private",   "needs_key": False},
    "groq":   {"label": "Groq",              "privacy": "sends_data", "needs_key": True},
    "gemini": {"label": "Google Gemini",     "privacy": "sends_data", "needs_key": True},
    "claude": {"label": "Claude (Anthropic)","privacy": "governed",   "needs_key": True},
}

PROVIDER_MODELS = {
    "groq":   ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    "gemini": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"],
    "claude": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"],
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
        "rerank":    cfg["workaround_finder"].get("llm_rerank", False),
    }


@app.post("/api/llm/rerank")
def llm_set_rerank(body: dict):
    """Toggle LLM relevance re-ranking on/off (persisted to config.json)."""
    cfg = load_config()
    cfg["workaround_finder"]["llm_rerank"] = bool(body.get("enabled", False))
    save_config(cfg)
    return {"ok": True, "enabled": cfg["workaround_finder"]["llm_rerank"]}


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
    return {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
            "claude": "ANTHROPIC_API_KEY"}.get(provider, "")


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
