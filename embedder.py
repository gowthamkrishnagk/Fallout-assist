"""Embeddings — Azure OpenAI (text-embedding-ada-002 by default) via the same REST
API shape as generate.py's `_synapt` chat provider: `api-key` header auth, the
deployment + api-version embedded in the URL. Reuses generate.py's key rotation,
cooldown and error-parsing helpers rather than duplicating them.

Replaces the old local `sentence-transformers` model — that dependency (and the
torch/tokenizers/chroma-hnswlib chain it drags in) had no working wheels on this
machine's Python version, and this way embeddings share the same Azure deployment
already configured for chat.

IMPORTANT: switching embedding models/providers changes the vector dimension —
`text-embedding-ada-002` is 1536-dim, the old `all-MiniLM-L6-v2` was 384-dim.
Anything already in trackers/workaround_index was embedded at the OLD dimension, so
this change needs a fresh ingest (re-run from Jira), not just a config edit — adding a
1536-dim vector to a collection created at 384-dim will error, not silently work."""

import os
import time

import httpx

import generate as g   # _ordered_keys / _cool_key / _err_reason / _redact / _is_transient_429 / _parse_reset

DEFAULT_DEPLOYMENT = "text-embedding-ada-002"

# A full ingest can hand embed() many thousands of texts at once (one ticket per
# chunk, thousands of tickets) — sending that as ONE request risks tripping Azure's
# per-request/per-minute limits outright. Batching keeps each call small regardless
# of how large the caller's list is.
_MAX_BATCH = 100
# Retries on the SAME key before rotating/giving up on it — a capacity 429 often
# clears in seconds, so this is worth doing before treating the key as exhausted.
_MAX_RETRIES_PER_KEY = 3
_MAX_BACKOFF = 30.0   # seconds — cap so one stuck batch can't stall an ingest for long

# Proactive pacing BETWEEN batches — stay under the deployment's TPM/RPM ceiling in
# the first place, rather than only reacting after a 429. Retry-with-backoff (above)
# handles a burst; this is for a SUSTAINED run (a full ingest can be hundreds of
# batches back to back) that would otherwise hit the ceiling repeatedly. Azure rate
# limits are per DEPLOYMENT, not per key, so — unlike Groq/Cerebras in generate.py —
# rotating keys here doesn't help; pacing and retry are the only real levers.
_BATCH_PACING_SECONDS = 0.5


def _deployment(model_name: str) -> str:
    return model_name or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", DEFAULT_DEPLOYMENT)


def _post_with_retry(url: str, api_key: str, texts: list[str]) -> httpx.Response:
    """One embeddings call for a single (already-chunked) batch, retrying a 429 on
    THIS key with header-driven backoff before the caller gives up on it."""
    resp = None
    for attempt in range(_MAX_RETRIES_PER_KEY):
        resp = httpx.post(url, headers={"api-key": api_key},
                          json={"input": texts}, timeout=60)
        if resp.status_code != 429:
            return resp
        hint = (resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-requests")
               or resp.headers.get("x-ratelimit-reset-tokens") or "")
        wait = min(_MAX_BACKOFF, g._parse_reset(hint)) if hint else min(_MAX_BACKOFF, 5.0 * (attempt + 1))
        print(f"[EMBED] 429 ({g._err_reason(resp)[:80]}) — retry {attempt + 1}/{_MAX_RETRIES_PER_KEY} "
              f"in {wait:.1f}s")
        time.sleep(wait)
    return resp   # still 429 after every retry — caller rotates key or gives up


def _embed_batch(texts: list[str], deployment: str) -> list[list[float]]:
    """Embed ONE batch (already <= _MAX_BATCH — see embed()) with retry-then-rotate."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT not set in .env")
    api_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    keys    = g._ordered_keys("synapt", "AZURE_OPENAI_API_KEY")
    if not keys:
        raise ValueError("AZURE_OPENAI_API_KEY not set in .env")
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_ver}"

    for api_key in keys:
        resp = _post_with_retry(url, api_key, texts)
        if resp.status_code == 429:
            reason    = g._err_reason(resp)
            transient = g._is_transient_429(reason)
            if not transient:
                g._cool_key("synapt", api_key, resp)
            if len(keys) > 1:
                kind = "busy" if transient else "rate-limited"
                print(f"[EMBED] synapt key …{api_key[-4:]} {kind} after retries — rotating to next key")
            continue
        if resp.status_code >= 400:
            reason = g._err_reason(resp)
            raise ValueError(f"embeddings {resp.status_code}: {g._redact(reason)}")
        data = resp.json()["data"]
        # Azure returns items in submission order, but `index` is authoritative —
        # sort on it rather than trust that, since a caller might rely on ordering
        # to zip embeddings back up with their source texts.
        data.sort(key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    raise ValueError(f"embeddings: all {len(keys)} key(s) rate-limited (after retrying each)")


def embed(texts: list[str], model_name: str = "") -> list[list[float] | None]:
    """Embed any number of texts — chunked into <= _MAX_BATCH-sized requests so a
    caller (e.g. a full re-ingest handing this thousands of texts at once) never
    sends one oversized request. Each chunk gets its own retry/rotation, and
    consecutive chunks are paced (_BATCH_PACING_SECONDS) so a long run stays under
    the deployment's rate limit instead of only reacting after crossing it.

    A blank/whitespace-only entry gets None back WITHOUT calling the API for it —
    Azure/OpenAI reject an empty string in `input` outright (400 on '$.input'),
    which would otherwise fail the ENTIRE batch over one placeholder. Callers that
    pad a list with '' purely to keep positions aligned (e.g. ingest.py's
    error_texts_clean, for tickets with no error context) already discard whatever
    embedding comes back for that position anyway, so None is exactly as usable."""
    if not texts:
        return []
    deployment = _deployment(model_name)
    real = [(i, t) for i, t in enumerate(texts) if (t or "").strip()]
    out: list[list[float] | None] = [None] * len(texts)
    if not real:
        return out
    idxs, real_texts = zip(*real)
    embedded: list[list[float]] = []
    chunks = range(0, len(real_texts), _MAX_BATCH)
    for n, i in enumerate(chunks):
        if n > 0:
            time.sleep(_BATCH_PACING_SECONDS)
        embedded.extend(_embed_batch(list(real_texts[i:i + _MAX_BATCH]), deployment))
    for idx, emb in zip(idxs, embedded):
        out[idx] = emb
    return out


def embed_one(text: str, model_name: str = "") -> list[float]:
    return _embed_batch([text], _deployment(model_name))[0]
