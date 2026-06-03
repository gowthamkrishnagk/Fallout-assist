"""
Search the vector DB for matching comments.

Strategy:
- One chunk per assignee comment in the index.
- Search returns the most similar individual comments across all tickets.
- Comments with score >= threshold (default 0.80) are treated as strong matches
  and returned as the primary workaround (real text, no LLM invention).
- Weaker matches are shown as supporting context only.
- Results are de-duplicated per ticket (best comment per ticket wins).
"""

import json
import re
from pathlib import Path
import embedder
import vectordb

DEFAULT_THRESHOLD = 0.70   # cosine-similarity scale; live value from config.json (score_threshold)


def _clean_query(text: str) -> str:
    """Strip MSISDNs, order numbers, Salesforce IDs from raw query input."""
    text = re.sub(r'\b[0-9][A-Za-z0-9]{14,17}\b', '', text)   # SF record IDs
    text = re.sub(r'\b\d{6,}\b', '', text)                      # MSISDNs / order numbers
    text = re.sub(r'(?:Order\s+Id|Order)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\s*-\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text




def find_workarounds(query: str, cfg: dict) -> dict:
    """
    Returns:
      {
        strong:  [ {key, summary, url, assignee, comment, score} ],  # score >= threshold
        context: [ {key, summary, url, assignee, comment, score} ],  # score < threshold
        threshold: float,
        best_score: float,
      }
    """
    wf          = cfg["workaround_finder"]
    index_path  = str(Path(__file__).parent / wf["index_path"])
    embed_model = cfg["embed"]["model"]
    top_k       = wf.get("top_k", 10)
    threshold   = wf.get("score_threshold", DEFAULT_THRESHOLD)

    # Hybrid parse: regex for labeled input, LLM fallback for free-form prose
    step, error = parse_input(query, cfg)

    # Build embeddings only for the fields actually present → dynamic routing in search_dual
    step_emb  = embedder.embed_one(f"Failed Step: {step}", embed_model) if step else None
    error_emb = embedder.embed_one(f"Error: {error}", embed_model)       if error else None

    # Both-collections fallback when neither step nor error could be extracted:
    # search on the raw cleaned query so something still surfaces.
    if step_emb is None and error_emb is None:
        raw_emb = embedder.embed_one(_clean_query(query), embed_model)
        step_emb = error_emb = raw_emb   # search both collections, best side wins

    # Tickets and docs are both matched on the same (step, error) dual basis, so a
    # doc only ranks high when its failed step / error match — not on shared prose.
    ticket_hits = vectordb.search_dual(step_emb, error_emb, top_k, index_path)
    doc_hits    = vectordb.search_docs_dual(step_emb, error_emb, max(2, top_k // 2), index_path)

    # Build a single cosine-sorted candidate list (dedup tickets, drop weak docs).
    candidates   = []
    seen_tickets = set()
    for h in sorted(ticket_hits + doc_hits, key=lambda x: x["score"], reverse=True):
        meta   = h["meta"]
        source = meta.get("source", "unknown")
        score  = h["score"]

        if source == "ticket":
            key = meta.get("key")
            if key in seen_tickets:
                continue
            seen_tickets.add(key)
            candidates.append({
                "type":        "ticket",
                "key":         key,
                "summary":     meta.get("summary", ""),
                "url":         meta.get("url", ""),
                "assignee":    meta.get("assignee", ""),
                "author":      meta.get("comment_author", ""),
                "comment":     meta.get("comment_body", h["doc"]),
                "description": meta.get("description", ""),
                "error":       meta.get("error", ""),
                "step":        meta.get("step", ""),
                "score":       score,
            })
        else:
            # Drop unrelated docs entirely: a document below the match threshold
            # is never shown. We don't dump the whole document body as a weak
            # "reference" — that just produces unrelated answers.
            if score < threshold:
                continue
            candidates.append({
                "type":     "doc",
                "filename": meta.get("filename", ""),
                "chunk":    meta.get("chunk", 0),
                "comment":  h["doc"],
                "score":    score,
            })

    # LLM relevance re-rank: drop semantic near-misses the cosine score lets
    # through, keeping only candidates about the SAME failure. No-op if disabled
    # or the LLM is unavailable.
    candidates = _llm_rerank(step, error, candidates, cfg)

    strong  = [c for c in candidates if c["score"] >= threshold]
    context = [c for c in candidates if c["score"] <  threshold]

    all_hits   = ticket_hits + doc_hits
    best_score = all_hits[0]["score"] if all_hits else 0.0
    return {
        "strong":     strong,
        "context":    context,
        "threshold":  threshold,
        "best_score": round(best_score, 3),
    }


def _llm_rerank(step: str, error: str, candidates: list, cfg: dict) -> list:
    """Ask the LLM which candidates address the SAME failure (step + error kind),
    returning the kept subset in the model's preferred order.

    Best-effort and safe:
      - disabled via config (llm_rerank=false) → unchanged
      - LLM down / unparseable / empty output → unchanged (cosine order kept).
        Re-rank only ever REMOVES candidates the model explicitly names; it never
        collapses the result set, so a weak model can't make results worse than
        plain similarity search.
    Only step/error/summary are sent — never full comment bodies — so the call
    stays cheap."""
    # Nothing to disambiguate with 0 or 1 candidate — skip the LLM call entirely
    # (saves a request against the daily quota; the result can't change).
    if len(candidates) <= 1 or not cfg["workaround_finder"].get("llm_rerank", False):
        return candidates

    import generate as g
    lines = []
    for i, c in enumerate(candidates, 1):
        if c["type"] == "ticket":
            lines.append(f"[{i}] ticket — step: {c.get('step','')} | "
                         f"error: {c.get('error','')} | summary: {c.get('summary','')}")
        else:
            lines.append(f"[{i}] doc '{c.get('filename','')}' "
                         f"(matched on its own step + error)")

    prompt = (
        "You filter search results for a Salesforce order-fallout support query.\n"
        f"Query failed step: {step or '(unknown)'}\n"
        f"Query error: {error or '(unknown)'}\n\n"
        "Candidates:\n" + "\n".join(lines) + "\n\n"
        "Return ONLY a JSON array of the candidate numbers that address the SAME "
        "failure (same step AND same kind of error), most-relevant first. "
        "If none are relevant, return []. Output nothing but the JSON array."
    )
    try:
        out = g.generate(prompt, cfg["llm"]).get("answer", "")
        m   = re.search(r'\[[\d,\s]*\]', out)
        if not m:
            print("[RERANK] no JSON array returned — keeping cosine order")
            return candidates
        idxs = json.loads(m.group(0))
        kept = [candidates[i - 1] for i in idxs
                if isinstance(i, int) and 1 <= i <= len(candidates)]
        if not kept:
            # Empty/unusable selection (common with tiny local models) — never
            # collapse the result set; fall back to the full cosine ranking.
            print("[RERANK] no usable selection — keeping cosine order")
            return candidates
        print(f"[RERANK] kept {len(kept)}/{len(candidates)} candidates")
        return kept
    except Exception as e:
        print(f"[RERANK] skipped ({e}) — keeping cosine order")
        return candidates


def _extract_fields(text: str):
    """Return (step, error, cleaned_text) from raw query."""
    lines = [re.sub(r'\s+', ' ', l).strip() for l in text.splitlines()]
    text  = '\n'.join(l for l in lines if l)

    step_match = re.search(
        r'(?:Failed\s+)?Step:\s*([^\n\|]{5,120}?)(?:\s*[\n\|]|$)',
        text, re.IGNORECASE
    )
    step = _clean_query(step_match.group(1).strip()) if step_match else ''

    # Error code pattern: "11 | …", "5 | …", "14081 | …" — allow 1-6 leading digits
    error_match = re.search(r'(\d{1,6}\s*\|[^\n]{3,150})', text)
    if not error_match:
        error_match = re.search(
            r'Error(?:\s+Description)?\s*:\s*([^\n]{3,200})',
            text, re.IGNORECASE
        )
    error = _clean_query(error_match.group(1).strip()) if error_match else ''

    return step, error, _clean_query(text)


def parse_input(raw: str, cfg: dict) -> tuple[str, str]:
    """Hybrid parser → (step, error).
    1. Regex extraction first — handles labeled input (Step:/Error:), the 'N | text'
       error pattern, and the labeled blob produced by _ticket_to_text (ticket-ID path).
    2. If regex finds nothing (free-form prose), fall back to the LLM parser using the
       configured provider. On any LLM failure → ('', '') so the caller searches raw text."""
    step, error, _ = _extract_fields(raw)
    if step or error:
        return step, error
    return _llm_parse(raw, cfg)


def _llm_parse(raw: str, cfg: dict) -> tuple[str, str]:
    """Use the configured LLM to extract {step, error} from unstructured input.
    Tolerant of failure — returns ('', '') if the model is down or output isn't parseable."""
    import generate as g
    prompt = (
        "You extract two fields from a Salesforce order-fallout support query.\n"
        "Return ONLY a JSON object: {\"step\": \"...\", \"error\": \"...\"}\n"
        "- step  = the failed orchestration step name, or \"\" if not mentioned\n"
        "- error = the error message or code, or \"\" if not mentioned\n"
        "Do not add any text outside the JSON.\n\n"
        f"Query:\n{raw[:800]}\n\nJSON:"
    )
    try:
        out = g.generate(prompt, cfg["llm"]).get("answer", "")
        m   = re.search(r'\{.*\}', out, re.DOTALL)
        if not m:
            print("[PARSE] LLM returned no JSON — falling back to raw search")
            return "", ""
        data  = json.loads(m.group(0))
        step  = _clean_query(str(data.get("step", "")).strip())
        error = _clean_query(str(data.get("error", "")).strip())
        print(f"[PARSE] LLM extracted step='{step}' error='{error}'")
        return step, error
    except Exception as e:
        print(f"[PARSE] LLM parse failed ({e}) — falling back to raw search")
        return "", ""


def build_prompt(query: str, result: dict) -> str:
    """Build LLM prompt only when needed (multiple strong matches → synthesize).

    Only strong (>= threshold) matches are given to the model. Weak context is
    deliberately excluded: it is below the relevance bar and small local models
    can't tell it apart from the real sources, so it contaminates the answer
    (e.g. a 49% doc about a different error leaking into a Suspend workaround)."""
    strong = result["strong"][:6]

    sources = ""
    for i, h in enumerate(strong, 1):
        if h["type"] == "ticket":
            label = f"{h['key']} (score {h['score']:.2f}) — comment by {h['author']}"
            body  = h["comment"]
        else:
            label = f"Doc '{h['filename']}' (score {h['score']:.2f})"
            body  = h["comment"]
        sources += f"\n--- Source {i}: {label} ---\n{body[:600]}\n"

    return (
        "You are a Jira order-fallout support assistant. "
        "A new issue needs a workaround:\n\n"
        f"{query}\n\n"
        "The following past resolved ticket comments are the most similar to this issue. "
        "Synthesize a clear, actionable workaround based ONLY on these comments. "
        "Do not invent steps not present in the sources.\n"
        f"{sources}\n"
        "Workaround:"
    )
