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
import re
import time
import hashlib
import httpx


def _redact(msg) -> str:
    """Strip any `key=...` query value (Gemini puts the API key in the URL, so it
    leaks into httpx error messages) before logging or showing it in the UI."""
    return re.sub(r'key=[\w.\-]+', 'key=***', str(msg))

# Trust the OS certificate store (covers corporate TLS-inspection root CAs).
# Idempotent with app.py's injection; guarded so a missing dep never breaks import.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass


# Latest Groq rate-limit snapshot, captured from real API response headers so the
# UI can show remaining quota without spending an extra request. `_groq_quota` is
# the latest snapshot (whichever key answered last); `_groq_quota_by_key` keeps a
# per-key snapshot so the UI can show which of several rotating keys is live.
_groq_quota: dict = {}
_groq_quota_by_key: dict = {}   # key_fp -> snapshot


def _capture_groq_quota(resp: httpx.Response, key_fp: str = "", key_last4: str = ""):
    h = resp.headers
    if "x-ratelimit-remaining-requests" not in h and "x-ratelimit-remaining-tokens" not in h:
        return
    snap = {
        "remaining_requests": h.get("x-ratelimit-remaining-requests"),
        "limit_requests":     h.get("x-ratelimit-limit-requests"),      # RPD
        "remaining_tokens":   h.get("x-ratelimit-remaining-tokens"),
        "limit_tokens":       h.get("x-ratelimit-limit-tokens"),        # TPM
        "reset_requests":     h.get("x-ratelimit-reset-requests"),
        "reset_tokens":       h.get("x-ratelimit-reset-tokens"),
        "updated":            time.time(),
        "key":                key_last4,
    }
    _groq_quota.update(snap)
    if key_fp:
        _groq_quota_by_key[key_fp] = snap


def get_groq_quota() -> dict:
    q = dict(_groq_quota)
    if _groq_quota_by_key:
        q["by_key"] = {fp: dict(s) for fp, s in _groq_quota_by_key.items()}
    return q


# ── Multi-key rotation ───────────────────────────────────────────────────────
# A provider's env var may hold SEVERAL free-tier keys (comma- or newline-
# separated): GROQ_API_KEY=k1,k2,k3. We round-robin across them and, when one is
# rate-limited (429), put just that key on cooldown until its reset and move to
# the next key. Only when ALL of a provider's keys are exhausted does the call
# raise — which then triggers the existing provider failover in generate().
_key_cursor:   dict = {}   # provider -> monotonic counter (round-robin start point)
_key_cooldown: dict = {}   # (provider, key_fp) -> unix ts until which the key is skipped


def _provider_keys(env_var: str) -> list[str]:
    """All keys configured for a provider, in order. Accepts comma- or newline-
    separated values so one env var can hold several free-tier keys."""
    return [k.strip() for k in re.split(r'[,\n]', os.getenv(env_var, "")) if k.strip()]


def _key_fp(key: str) -> str:
    """Stable short fingerprint of a key — used as a cooldown/quota map id without
    storing the secret itself."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _ordered_keys(provider: str, env_var: str) -> list[str]:
    """Keys to try this call, in rotation order. Round-robins the start point so
    load spreads across keys; keys still in cooldown are moved to the back so they
    are only retried if every other key is also cooled."""
    keys = _provider_keys(env_var)
    if not keys:
        return []
    n = _key_cursor.get(provider, 0)
    _key_cursor[provider] = n + 1
    start   = n % len(keys)
    rotated = keys[start:] + keys[:start]
    now     = time.time()
    live    = [k for k in rotated if _key_cooldown.get((provider, _key_fp(k)), 0) <= now]
    cooled  = [k for k in rotated if _key_cooldown.get((provider, _key_fp(k)), 0) >  now]
    return live + cooled


def _parse_reset(val: str) -> float:
    """Seconds from a rate-limit reset header: plain seconds ('2.5'), a duration
    ('1m30s', '500ms'), or a Retry-After integer. Falls back to 60s."""
    val = str(val or "").strip().lower()
    if re.fullmatch(r'\d+(?:\.\d+)?', val):
        return float(val)
    total = 0.0
    for num, unit in re.findall(r'(\d+(?:\.\d+)?)\s*(ms|s|m|h)', val):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total or 60.0


def _cool_key(provider: str, key: str, resp: httpx.Response | None):
    """Mark a key rate-limited until its reset (header-driven, default 60s)."""
    secs = 60.0
    if resp is not None:
        h = resp.headers
        hint = (h.get("x-ratelimit-reset-requests") or h.get("x-ratelimit-reset-tokens")
                or h.get("retry-after"))
        if hint:
            secs = _parse_reset(hint)
    _key_cooldown[(provider, _key_fp(key))] = time.time() + min(secs, 3600)


# OpenAI-compatible cloud providers — all speak the same /chat/completions schema
# with Bearer auth, so one helper (_openai_chat) serves all of them. Free tier,
# no card required: Groq, Cerebras, NVIDIA NIM.
OPENAI_COMPAT = {
    "groq":     {"url": "https://api.groq.com/openai/v1/chat/completions",     "key": "GROQ_API_KEY"},
    "cerebras": {"url": "https://api.cerebras.ai/v1/chat/completions",         "key": "CEREBRAS_API_KEY"},
    "nvidia":   {"url": "https://integrate.api.nvidia.com/v1/chat/completions", "key": "NVIDIA_API_KEY"},
}

# Default model per provider — used for fallback providers (the primary uses the
# model from config). All are free-tier friendly.
DEFAULT_MODELS = {
    "groq":     "llama-3.1-8b-instant",
    "cerebras": "gpt-oss-120b",
    "nvidia":   "meta/llama-3.1-8b-instruct",
    "claude":   "claude-haiku-4-5-20251001",
    "local":    "llama3.2:1b",
}


class LLMRetryable(Exception):
    """A transient LLM failure — a service-capacity 429 ('high traffic'), a 5xx, or
    a network timeout — that is worth retrying after a short backoff, as opposed to
    a hard error (bad key/model/quota) that won't change on retry."""


# 429 bodies that mean "the SERVICE is momentarily busy", not "THIS key is out of
# quota". A short retry clears these; cooling the key would needlessly bench it.
_TRANSIENT_429 = ("high traffic", "try again", "overloaded", "temporarily",
                  "capacity", "please retry", "service unavailable", "is busy")


def _is_transient_429(reason: str) -> bool:
    r = (reason or "").lower()
    return any(t in r for t in _TRANSIENT_429)


def _err_reason(resp: httpx.Response) -> str:
    """Extract the human error message from a provider's error response. Providers
    shape errors differently: OpenAI/Groq use {"error":{"message":...}}, Cerebras/
    NVIDIA use {"detail":...} or {"message":...}. Try all so it's never lost."""
    try:
        j   = resp.json()
        err = j.get("error")
        return ((err.get("message") if isinstance(err, dict) else err)
                or j.get("detail") or j.get("message") or "")
    except Exception:
        return resp.text[:200]


def _is_retryable(e: Exception) -> bool:
    """True for transient failures worth an in-request retry: our LLMRetryable plus
    raw httpx transport errors (timeout / connection reset) from a cold or flaky
    network path."""
    return isinstance(e, (LLMRetryable, httpx.TimeoutException, httpx.ConnectError,
                          httpx.ReadError, httpx.RemoteProtocolError))


def _generate_one(prompt: str, provider: str, model: str, cfg: dict) -> dict:
    model = model or DEFAULT_MODELS.get(provider, "")
    if provider == "local":
        return _local(prompt, model, cfg.get("ollama_url", "http://localhost:11434"))
    if provider in OPENAI_COMPAT:
        return _openai_chat(prompt, model, provider)
    if provider == "claude":
        return _claude(prompt, model)
    raise ValueError(f"Unknown provider: {provider}")


def _resolve_job(cfg: dict, job: str):
    """Per-job provider/model override so different jobs can spread across free
    providers (Groq / Cerebras / NVIDIA) instead of draining one quota.

    Only an EXPLICIT cfg["jobs"][job] override applies — there is no built-in
    preference, so every job uses the user's selected provider by default (the
    user's choice is always respected). Returns (provider_or_None, model_or_None)."""
    jobs = cfg.get("jobs") or {}
    if job in jobs:
        return jobs[job].get("provider"), jobs[job].get("model")
    return None, None


def generate(prompt: str, cfg: dict, job: str | None = None) -> dict:
    """Generate with automatic provider failover.

    Tries the configured provider first, then each provider in cfg["fallback"]
    (e.g. ["cerebras", "local"]) until one succeeds — so a Groq rate-limit (429)
    transparently rolls to Cerebras, then to local Ollama (unlimited). The primary
    uses cfg["model"]; fallbacks use cfg["fallback_models"][p] or DEFAULT_MODELS.

    job: optional job name ("synthesis"/"rerank"/"parse") — routes that job to its
    own provider/model (see _resolve_job) so quota spreads across providers. The
    fallback chain still applies, so a job's provider 429 rolls onward as usual.

    Transient failures (a 'high traffic' capacity 429, a 5xx, or a network timeout)
    are retried IN-REQUEST after a short backoff — up to cfg["max_retries"] extra
    passes over the whole provider order (default 2) — so a brief provider blip is
    absorbed here instead of surfacing as "LLM unavailable" and forcing the user to
    click search again. A hard error (bad key/model, real quota) is not retried.

    Returns {"answer", "provider", "model"} — provider/model reflect whoever
    actually answered, so the UI shows which one was used. Raises only if ALL fail.
    cfg = config["llm"]"""
    job_prov, job_model = _resolve_job(cfg, job) if job else (None, None)
    primary  = job_prov or cfg.get("provider", "local")
    fallback = cfg.get("fallback", []) or []
    # Never auto-fall to the local model: a weak local model ignores grounding and
    # hallucinates. `local` is used ONLY when it is the explicitly selected primary.
    order    = [primary] + [p for p in fallback if p != primary and p != "local"]
    max_retries = int(cfg.get("max_retries", 2))

    def _model_for(prov: str) -> str:
        if prov == primary:
            return job_model or (cfg.get("model") if not job_prov else None) \
                   or DEFAULT_MODELS.get(prov)
        return (cfg.get("fallback_models", {}) or {}).get(prov) or DEFAULT_MODELS.get(prov)

    last_errors: list = []
    for attempt in range(max_retries + 1):
        errors, retryable = [], False
        for i, prov in enumerate(order):
            try:
                result = _generate_one(prompt, prov, _model_for(prov), cfg)
                if i > 0 or attempt > 0:
                    where = (f" (after {', '.join(order[:i])} failed)" if i else "")
                    when  = (f" on retry {attempt}" if attempt else "")
                    print(f"[LLM] answered by '{prov}'{where}{when}")
                return result
            except Exception as e:
                msg = _redact(e)[:120]
                errors.append(f"{prov}: {msg}")
                retryable = retryable or _is_retryable(e)
                nxt = f" -- trying {order[i+1]}" if i < len(order) - 1 else ""
                print(f"[LLM] provider '{prov}' failed ({msg}){nxt}")
        last_errors = errors
        # Only retry the whole order when at least one failure was transient — a
        # hard error (bad key/model, real quota) won't change on a retry.
        if retryable and attempt < max_retries:
            backoff = 1.5 * (attempt + 1)
            print(f"[LLM] transient failure — retrying in {backoff:.1f}s "
                  f"(attempt {attempt + 2}/{max_retries + 1})")
            time.sleep(backoff)
            continue
        break

    raise ValueError("All LLM providers failed: " + " | ".join(last_errors))


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


def _openai_chat(prompt: str, model: str, provider: str) -> dict:
    """One client for every OpenAI-compatible provider (Groq / Cerebras / NVIDIA).
    Surfaces the API's real error message (key-redacted) instead of a bare status.

    Rotates across every key configured for the provider: a rate-limited (429) key
    is put on cooldown and the next key is tried. Only a 429 burns through to the
    next key — any other 4xx/5xx is a real failure (bad model, bad key) and raises
    immediately so it isn't masked by trying more keys. Raises when all keys are
    rate-limited, which lets generate() fall through to the next provider."""
    spec = OPENAI_COMPAT[provider]
    keys = _ordered_keys(provider, spec["key"])
    if not keys:
        raise ValueError(f"{spec['key']} not set in .env")

    rate_limited, any_transient = [], False
    for api_key in keys:
        resp = httpx.post(
            spec["url"],
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2},
            timeout=60,
        )
        if provider == "groq":
            _capture_groq_quota(resp, _key_fp(api_key), api_key[-4:])   # quota even on 429
        if resp.status_code == 429:
            reason    = _err_reason(resp)
            transient = _is_transient_429(reason)   # service busy, not a key quota
            if not transient:
                _cool_key(provider, api_key, resp)  # real per-key RPM/TPM limit
            any_transient = any_transient or transient
            rate_limited.append(f"…{api_key[-4:]}{' (busy)' if transient else ''}")
            if len(keys) > 1:
                kind = "busy" if transient else "rate-limited"
                print(f"[LLM] {provider} key …{api_key[-4:]} {kind} — rotating to next key")
            continue
        if resp.status_code >= 400:
            # Non-429 4xx/5xx is a hard error (bad model/key) — raise immediately
            # so trying more keys doesn't mask the real reason. (5xx is wrapped as
            # retryable so generate() can give it one more pass.)
            reason = _err_reason(resp)
            err = ValueError(f"{provider} {resp.status_code}: {_redact(reason)}")
            raise LLMRetryable(str(err)) if resp.status_code >= 500 else err
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        return {"answer": answer, "provider": provider, "model": model}

    msg = (f"{provider} 429: all {len(keys)} key(s) rate-limited "
           f"({', '.join(rate_limited)})")
    # If any failure was a transient capacity blip, let generate() retry the order.
    raise LLMRetryable(msg) if any_transient else ValueError(msg)


def _claude(prompt: str, model: str) -> dict:
    """Anthropic Claude with the same multi-key rotation as the OpenAI-compatible
    providers: a 429'd key is cooled and the next key tried; any other error
    raises so it isn't masked. Raises when every key is rate-limited."""
    keys = _ordered_keys("claude", "ANTHROPIC_API_KEY")
    if not keys:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

    rate_limited, any_transient = [], False
    for api_key in keys:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": model or "claude-haiku-4-5-20251001",
                  "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        if resp.status_code == 429:
            reason    = _err_reason(resp)
            transient = _is_transient_429(reason)
            if not transient:
                _cool_key("claude", api_key, resp)
            any_transient = any_transient or transient
            rate_limited.append(f"…{api_key[-4:]}{' (busy)' if transient else ''}")
            if len(keys) > 1:
                kind = "busy" if transient else "rate-limited"
                print(f"[LLM] claude key …{api_key[-4:]} {kind} — rotating to next key")
            continue
        if resp.status_code >= 500:
            raise LLMRetryable(f"claude {resp.status_code}: {_redact(_err_reason(resp))}")
        resp.raise_for_status()
        answer = resp.json()["content"][0]["text"].strip()
        return {"answer": answer, "provider": "claude", "model": model}

    msg = (f"claude 429: all {len(keys)} key(s) rate-limited "
           f"({', '.join(rate_limited)})")
    raise LLMRetryable(msg) if any_transient else ValueError(msg)


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

        env_var = (OPENAI_COMPAT.get(provider, {}).get("key")
                   or {"claude": "ANTHROPIC_API_KEY"}.get(provider, ""))
        keys = _provider_keys(env_var)
        if not keys:
            return _bad(f"No API key set ({env_var})")
        key = keys[0]   # any configured key proves reachability + auth

        if provider in OPENAI_COMPAT:
            # OpenAI-compatible /models endpoint = cheap reachability + auth check
            models_url = OPENAI_COMPAT[provider]["url"].replace("/chat/completions", "/models")
            resp = httpx.get(models_url, headers={"Authorization": f"Bearer {key}"}, timeout=6)
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
        return _bad(_redact(e)[:100])


def test_provider(cfg: dict) -> dict:
    """Quick connectivity test for the SELECTED provider only — deliberately
    bypasses the failover chain so a real error (bad model, missing key) surfaces
    instead of silently succeeding via a fallback. Returns
    {ok, latency_ms, provider, model, response}."""
    import time
    provider = cfg.get("provider", "local")
    model    = cfg.get("model") or DEFAULT_MODELS.get(provider)
    start    = time.time()
    try:
        result  = _generate_one("Reply in one word: ready", provider, model, cfg)
        elapsed = round((time.time() - start) * 1000)
        return {"ok": True, "latency_ms": elapsed,
                "provider": result["provider"], "model": result["model"],
                "response": result["answer"]}
    except Exception as e:
        return {"ok": False, "error": _redact(e), "provider": provider, "model": model}
