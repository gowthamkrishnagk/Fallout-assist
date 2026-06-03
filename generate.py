"""
Swappable LLM generation backend.
Provider is read from config — switching provider = one config change, no code change.

Supported providers:
  local   → Ollama at localhost:11434 (default, fully private)
  groq    → Groq cloud API  (GROQ_API_KEY in .env)
  gemini  → Google Gemini   (GEMINI_API_KEY in .env)
  claude  → Anthropic       (ANTHROPIC_API_KEY in .env)
"""

import os
import time
import httpx

# Trust the OS certificate store (covers corporate TLS-inspection root CAs).
# Idempotent with app.py's injection; guarded so a missing dep never breaks import.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass


# Latest Groq rate-limit snapshot, captured from real API response headers so the
# UI can show remaining quota without spending an extra request. Updated in-place
# by _groq(); read via get_groq_quota().
_groq_quota: dict = {}


def _capture_groq_quota(resp: httpx.Response):
    h = resp.headers
    if "x-ratelimit-remaining-requests" not in h and "x-ratelimit-remaining-tokens" not in h:
        return
    _groq_quota.update({
        "remaining_requests": h.get("x-ratelimit-remaining-requests"),
        "limit_requests":     h.get("x-ratelimit-limit-requests"),      # RPD
        "remaining_tokens":   h.get("x-ratelimit-remaining-tokens"),
        "limit_tokens":       h.get("x-ratelimit-limit-tokens"),        # TPM
        "reset_requests":     h.get("x-ratelimit-reset-requests"),
        "reset_tokens":       h.get("x-ratelimit-reset-tokens"),
        "updated":            time.time(),
    })


def get_groq_quota() -> dict:
    return dict(_groq_quota)


# Default model per provider — used for fallback providers (the primary uses the
# model from config). All are free-tier friendly.
DEFAULT_MODELS = {
    "groq":   "llama-3.1-8b-instant",
    "gemini": "gemini-2.0-flash",
    "claude": "claude-haiku-4-5-20251001",
    "local":  "llama3.2:1b",
}


def _generate_one(prompt: str, provider: str, model: str, cfg: dict) -> dict:
    model = model or DEFAULT_MODELS.get(provider, "")
    if provider == "local":
        return _local(prompt, model, cfg.get("ollama_url", "http://localhost:11434"))
    if provider == "groq":
        return _groq(prompt, model)
    if provider == "gemini":
        return _gemini(prompt, model)
    if provider == "claude":
        return _claude(prompt, model)
    raise ValueError(f"Unknown provider: {provider}")


def generate(prompt: str, cfg: dict) -> dict:
    """Generate with automatic provider failover.

    Tries the configured provider first, then each provider in cfg["fallback"]
    (e.g. ["gemini", "local"]) until one succeeds — so a Groq rate-limit (429)
    transparently rolls to Gemini, then to local Ollama (unlimited). The primary
    uses cfg["model"]; fallbacks use cfg["fallback_models"][p] or DEFAULT_MODELS.

    Returns {"answer", "provider", "model"} — provider/model reflect whoever
    actually answered, so the UI shows which one was used. Raises only if ALL fail.
    cfg = config["llm"]"""
    primary  = cfg.get("provider", "local")
    fallback = cfg.get("fallback", []) or []
    order    = [primary] + [p for p in fallback if p != primary]

    errors = []
    for i, prov in enumerate(order):
        model = cfg.get("model") if prov == primary else \
                (cfg.get("fallback_models", {}) or {}).get(prov) or DEFAULT_MODELS.get(prov)
        try:
            result = _generate_one(prompt, prov, model, cfg)
            if i > 0:
                print(f"[LLM] failover -> answered by '{prov}' "
                      f"(after {', '.join(order[:i])} failed)")
            return result
        except Exception as e:
            errors.append(f"{prov}: {str(e)[:120]}")
            nxt = f" -- trying {order[i+1]}" if i < len(order) - 1 else ""
            print(f"[LLM] provider '{prov}' failed ({str(e)[:120]}){nxt}")

    raise ValueError("All LLM providers failed: " + " | ".join(errors))


def _local(prompt: str, model: str, ollama_url: str) -> dict:
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        return {"answer": resp.json().get("response", "").strip(),
                "provider": "local", "model": model}
    except httpx.ConnectError:
        raise ValueError(f"Ollama is not running at {ollama_url}. Start it with: ollama serve")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 500:
            raise ValueError(
                f"Ollama model '{model}' is not installed. "
                f"Pull it first: ollama pull {model}"
            )
        raise ValueError(f"Ollama error {e.response.status_code}: {e.response.text[:200]}")


def _groq(prompt: str, model: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model or "llama-3.3-70b-versatile",
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2},
        timeout=60,
    )
    _capture_groq_quota(resp)   # record remaining quota even on a 429
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "provider": "groq", "model": model}


def _gemini(prompt: str, model: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")
    m    = model or "gemini-1.5-flash"
    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}",
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    resp.raise_for_status()
    answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return {"answer": answer, "provider": "gemini", "model": m}


def _claude(prompt: str, model: str) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        json={"model": model or "claude-haiku-4-5-20251001",
              "max_tokens": 1024,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    resp.raise_for_status()
    answer = resp.json()["content"][0]["text"].strip()
    return {"answer": answer, "provider": "claude", "model": model}


def available_local_models(ollama_url: str = "http://localhost:11434") -> list[dict]:
    """Return installed Ollama models (non-cloud only)."""
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        return [
            {"name": m["name"], "size_gb": round(m["size"] / 1e9, 1)}
            for m in models
            if ":cloud" not in m["name"]
        ]
    except Exception:
        return []


def provider_status(cfg: dict) -> dict:
    """Lightweight reachability check for the active provider — powers the UI
    status indicator. Avoids a full generation: pings a cheap endpoint (model
    list / tags) instead. Returns {connected, provider, model, detail}."""
    provider = cfg.get("provider", "local")
    model    = cfg.get("model", "")

    def _ok(detail):   return {"connected": True,  "provider": provider, "model": model, "detail": detail}
    def _bad(detail):  return {"connected": False, "provider": provider, "model": model, "detail": detail}

    try:
        if provider == "local":
            url  = cfg.get("ollama_url", "http://localhost:11434")
            resp = httpx.get(f"{url}/api/tags", timeout=4)
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            base  = model.split(":")[0]
            if model and not any(n == model or n.startswith(base) for n in names):
                return _bad(f"Ollama running, but model '{model}' not installed — run: ollama pull {model}")
            return _ok("Ollama reachable")

        env_var = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
                   "claude": "ANTHROPIC_API_KEY"}.get(provider, "")
        key = os.getenv(env_var, "")
        if not key:
            return _bad(f"No API key set ({env_var})")

        if provider == "groq":
            resp = httpx.get("https://api.groq.com/openai/v1/models",
                             headers={"Authorization": f"Bearer {key}"}, timeout=6)
        elif provider == "gemini":
            resp = httpx.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
                timeout=6)
        else:  # claude
            resp = httpx.get("https://api.anthropic.com/v1/models",
                             headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                             timeout=6)
        resp.raise_for_status()
        return _ok("API reachable")

    except httpx.ConnectError:
        return _bad("Connection failed" if provider != "local"
                    else f"Ollama not running at {cfg.get('ollama_url', 'localhost:11434')}")
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        return _bad("Invalid or unauthorized API key" if code in (401, 403) else f"HTTP {code}")
    except Exception as e:
        return _bad(str(e)[:100])


def test_provider(cfg: dict) -> dict:
    """Quick connectivity test — returns {ok, latency_ms, provider, model, response}."""
    import time
    start = time.time()
    try:
        result  = generate("Reply in one word: ready", cfg)
        elapsed = round((time.time() - start) * 1000)
        return {"ok": True, "latency_ms": elapsed,
                "provider": result["provider"], "model": result["model"],
                "response": result["answer"]}
    except Exception as e:
        return {"ok": False, "error": str(e),
                "provider": cfg.get("provider"), "model": cfg.get("model")}
