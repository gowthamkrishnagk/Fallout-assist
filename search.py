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
from math import tanh
from pathlib import Path
import embedder
import feedback
import vectordb
from textclean import clean_text, is_pointer_comment, referenced_ticket

DEFAULT_THRESHOLD = 0.70   # cosine-similarity scale; live value from config.json (score_threshold)


def _clean_query(text: str) -> str:
    """Normalize raw query input the SAME way ingest normalizes stored step/error
    (shared canonical cleaner) so a query and the matching ticket embed alike."""
    return clean_text(text)




def _resolution_quality(c: dict) -> float:
    """Cheap, LLM-FREE score of how usable a candidate's resolution is, so
    synthesis draws from the best-resolved tickets first instead of an arbitrary
    cosine order. Costs nothing (no network / no Groq quota).

    Rewards: the structured '=== FIX ===' block, numbered/bulleted steps, and
    substance. Penalizes one-word closers ('done', 'fixed') and very short text.
    """
    if c.get("type") == "doc":
        return 3.0   # uploaded docs are curated workarounds — treat as solid
    body = (c.get("comment") or "").strip()
    if not body:
        return 0.0
    score = 0.0
    if "=== fix ===" in body.lower():
        score += 5.0                                   # follows the resolution format
    steps = len(re.findall(r'(?m)^\s*(?:\d+[.)]|[-*•])\s+\S', body))
    score += min(steps, 5) * 0.6                        # actionable, ordered steps
    score += min(len(body) / 200.0, 3.0)                # substance (capped)
    if re.fullmatch(r'(?i)\s*(done|fixed|closing|closed|resolved|n/?a|ok)[.!]?\s*', body):
        score -= 5.0                                    # non-resolution closer
    if len(body) < 60:
        score -= 1.0
    return score


def find_workarounds(query: str, cfg: dict, exclude_keys=frozenset()) -> dict:
    """
    Returns:
      {
        strong:  [ {key, summary, url, assignee, comment, score} ],  # score >= threshold
        context: [ {key, summary, url, assignee, comment, score} ],  # score < threshold
        threshold: float,
        best_score: float,
      }

    `exclude_keys` is a set of ticket keys to drop from the results — used by the
    Jira 👎 "improve" loop to skip workarounds already rejected on a ticket so the
    re-match advances to a genuinely different, properly-matched fix.
    """
    exclude_keys = {k.upper() for k in exclude_keys}
    wf          = cfg["workaround_finder"]
    index_path  = str(Path(__file__).parent / wf["index_path"])
    embed_model = cfg["embed"]["model"]
    top_k       = wf.get("top_k", 10)
    threshold   = wf.get("score_threshold", DEFAULT_THRESHOLD)
    # Error is the differentiator (the step name is shared by many tickets), so it
    # carries more weight when both step and error are present — and a candidate
    # whose error is below err_floor is dropped: a wrong error is a different
    # failure, not a weak match (step-only matches must not surface).
    err_weight  = wf.get("error_weight", 0.65)
    err_floor   = wf.get("error_floor", 0.55)

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
    ticket_hits = vectordb.search_dual(step_emb, error_emb, top_k, index_path, err_weight, err_floor)
    doc_hits    = vectordb.search_docs_dual(step_emb, error_emb, max(2, top_k // 2), index_path, err_weight, err_floor)
    all_hits    = ticket_hits + doc_hits

    # Hybrid retrieval: fuse a BM25 keyword ranking with the vector ranking (RRF).
    # Lifts exact-token matches (error codes, step names) that embeddings blur and
    # pulls in keyword-only hits the vector pool missed — each scored with an honest
    # cosine so the threshold/display stay truthful. LLM-free, so it costs no quota.
    rrf = {}
    if wf.get("hybrid_enabled", True):
        rrf, all_hits = _hybrid_augment(step, error, step_emb, error_emb,
                                        all_hits, index_path, wf, err_weight, err_floor)

    # Graph expansion: pull in sibling tickets that share this failure's signature
    # (same step+error) or are pointer-linked, even when their comment wording
    # matched neither vector nor keyword search. Surfaces a fix that lives on a
    # related ticket. LLM-free — costs no quota.
    if wf.get("graph_enabled", True):
        all_hits = _graph_augment(step_emb, error_emb, all_hits, index_path,
                                  wf, err_weight, err_floor)

    # Build a single cosine-sorted candidate list (dedup tickets, drop weak docs).
    candidates   = []
    seen_tickets = set()
    for h in sorted(all_hits, key=lambda x: x["score"], reverse=True):
        h_id   = h.get("id")
        meta   = h["meta"]
        source = meta.get("source", "unknown")
        score  = h["score"]

        if source == "ticket":
            key = meta.get("key")
            if key in seen_tickets:
                continue
            if key and key.upper() in exclude_keys:
                continue                      # rejected on this ticket — skip to next match
            comment = meta.get("comment_body", h["doc"])
            url     = meta.get("url", "")
            author  = meta.get("comment_author", "")

            # A pointer comment ("duplicate, refer to SAC-231619") is not a fix.
            # Resolve it to the referenced ticket's real resolution — from the index
            # first, else a live Jira fetch. Drop the candidate if neither yields a
            # usable fix (referenced ticket missing / still open / also a pointer).
            if is_pointer_comment(comment):
                ref = referenced_ticket(comment)
                if not ref or ref in seen_tickets:
                    continue
                ref_meta = vectordb.get_ticket_meta(ref, index_path)
                ref_body = ref_meta.get("comment_body", "")
                if ref_body and not is_pointer_comment(ref_body):
                    # Referenced ticket is indexed — use its stored resolution.
                    key, comment = ref_meta.get("key", ref), ref_body
                    url, author  = ref_meta.get("url", ""), ref_meta.get("comment_author", "")
                    meta         = ref_meta
                else:
                    # Not indexed → follow to Jira live for the real resolution.
                    import ingest as ing
                    fetched = ing.fetch_ticket_resolution(ref, cfg)
                    if not fetched:
                        continue
                    key, comment = fetched["key"], fetched["comment"]
                    url, author  = fetched["url"], fetched.get("author", "")
                    # keep meta (the duplicate's step/error/summary — same failure)
                if key in seen_tickets or (key and key.upper() in exclude_keys):
                    continue

            seen_tickets.add(key)
            candidates.append({
                "type":        "ticket",
                "_id":         h_id,
                "key":         key,
                "summary":     meta.get("summary", ""),
                "url":         url,
                "assignee":    meta.get("assignee", ""),
                "author":      author,
                "comment":     comment,
                "description": meta.get("description", ""),
                "error":       meta.get("error", ""),
                "step":        meta.get("step", ""),
                "score":       score,
                "updated_ts":  meta.get("updated_ts", 0),
            })
        else:
            # Drop unrelated docs entirely: a document below the match threshold
            # is never shown. We don't dump the whole document body as a weak
            # "reference" — that just produces unrelated answers.
            if score < threshold:
                continue
            candidates.append({
                "type":     "doc",
                "_id":      h_id,
                "filename": meta.get("filename", ""),
                "chunk":    meta.get("chunk", 0),
                "comment":  h["doc"],
                "score":    score,
            })

    # LLM relevance re-rank: drop semantic near-misses the cosine score lets
    # through, keeping only candidates about the SAME failure. No-op if disabled
    # or the LLM is unavailable.
    candidates = _llm_rerank(step, error, candidates, cfg)

    # Feedback "training": nudge each candidate by the net 👍/👎 it has earned for
    # THIS failure. rank_score = raw cosine + a bounded delta — used only for
    # ordering and the strong/context split; the raw `score` is left untouched so
    # the UI's "% match" badge stays honest. Likes float a workaround up, dislikes
    # sink it (softly — it can drop below threshold but is never erased).
    _apply_feedback(step, error, candidates, cfg, threshold)

    # Hybrid boost: lift candidates that also rank high on lexical (BM25) match, so
    # an exact error-code / step-name hit the embedding buried floats up. Bounded
    # (like feedback) so it nudges ordering and can cross the strong threshold for a
    # near-miss, but can't manufacture a strong match from an unrelated ticket.
    if rrf:
        _apply_hybrid(candidates, rrf, wf)

    # Order matches of similar strength (same 0.01 score bucket — e.g. the many
    # tickets that tie at 1.0 for an identical error) by, in priority:
    #   1. feedback-adjusted match strength (likes/dislikes reorder ties)
    #   2. resolution quality      (best-resolved ticket feeds the answer/synthesis)
    #   3. recency                 (newest among equally-good resolutions)
    # All LLM-free, so it costs no Groq/OpenAI quota.
    candidates.sort(
        key=lambda c: (round(c["rank_score"], 2),
                       round(_resolution_quality(c)),
                       c.get("updated_ts", 0)),
        reverse=True)

    strong  = [c for c in candidates if c["rank_score"] >= threshold]
    context = [c for c in candidates if c["rank_score"] <  threshold]

    # Best score reflects the ACTUAL candidates shown — not the raw hits — so a
    # dropped pointer/duplicate (e.g. an 82% "refer to SAC-x" whose target isn't
    # indexed) doesn't leave a misleading "best: 82%" badge on the result.
    best_score = max((c["score"] for c in candidates), default=0.0)
    return {
        "strong":      strong,
        "context":     context,
        "threshold":   threshold,
        "best_score":  round(best_score, 3),
        "query_step":  step,
        "query_error": error,
    }


def _hybrid_augment(step: str, error: str, step_emb, error_emb, vector_hits: list,
                    index_path: str, wf: dict, err_weight: float, err_floor: float):
    """Run BM25 keyword search, fuse it with the vector hits (RRF), and pull in
    keyword-only hits the vector pool missed — scoring them with the same dual
    cosine so they're directly comparable to the vector candidates.

    Returns (rrf_scores_by_id, augmented_hits). Best-effort: if the keyword index
    is unavailable (rank-bm25 missing / empty KB) it returns ({}, vector_hits)."""
    import retrieval
    top_k   = wf.get("top_k", 10)
    kw_hits = retrieval.keyword_search(step, error, max(top_k * 3, 30), index_path)
    if not kw_hits:
        return {}, vector_hits

    vector_sorted = sorted(vector_hits, key=lambda x: x["score"], reverse=True)
    rrf = retrieval.rrf_fuse(vector_sorted, kw_hits, wf.get("rrf_k", 60))

    have  = {h.get("id") for h in vector_hits}
    extra = []
    for src in ("ticket", "doc"):
        only = [h for h in kw_hits if h["id"] not in have and h.get("source") == src][:top_k]
        if not only:
            continue
        ids = [h["id"] for h in only]
        sc  = vectordb.scores_for_ids(ids, step_emb, error_emb, index_path,
                                      source=src, error_weight=err_weight, error_floor=err_floor)
        by_id = {h["id"]: h for h in only}
        for id_, score in sc.items():
            h = by_id[id_]
            extra.append({"id": id_, "doc": h["doc"], "meta": h["meta"], "score": score})
    return rrf, vector_hits + extra


def _graph_augment(step_emb, error_emb, all_hits: list, index_path: str, wf: dict,
                   err_weight: float, err_floor: float):
    """Expand the top ticket hits to their graph siblings (same failure signature /
    pointer-linked), score the new chunks with the same dual cosine, and add the best
    chunk per sibling ticket. Best-effort: returns all_hits unchanged on any failure.

    Returns the augmented hit list."""
    import graph
    seed_cap = wf.get("graph_seed", 6)
    cap      = wf.get("graph_neighbor_cap", 8)

    seeds, have_keys = [], set()
    for h in sorted(all_hits, key=lambda x: x["score"], reverse=True):
        m = h.get("meta", {})
        if m.get("source") != "ticket":
            continue
        k = m.get("key")
        if k:
            have_keys.add(k)
            if k not in seeds and len(seeds) < seed_cap:
                seeds.append(k)
    if not seeds:
        return all_hits

    neighbors = graph.expand(seeds, index_path, cap=cap)
    new_ids   = [cid for nb in neighbors for cid in nb["chunk_ids"]]
    if not new_ids:
        return all_hits

    scores = vectordb.scores_for_ids(new_ids, step_emb, error_emb, index_path,
                                     source="ticket", error_weight=err_weight,
                                     error_floor=err_floor)
    if not scores:
        return all_hits
    detail = vectordb.chunks_by_ids(list(scores), index_path, source="ticket")

    # Best-scoring chunk per NEW sibling ticket (skip tickets already in the pool).
    best: dict = {}
    for id_, sc in scores.items():
        d   = detail.get(id_)
        key = d["meta"].get("key") if d else None
        if not key or key in have_keys:
            continue
        if key not in best or sc > best[key][0]:
            best[key] = (sc, id_)

    extra = [{"id": id_, "doc": detail[id_]["doc"], "meta": detail[id_]["meta"],
              "score": sc, "via_graph": True}
             for sc, id_ in best.values()]
    return all_hits + extra


def _apply_hybrid(candidates: list, rrf: dict, wf: dict):
    """Add a bounded RRF-derived delta to each candidate's rank_score (annotated as
    hybrid_adj). Normalized to the strongest fused candidate so the boost is capped
    at hybrid_weight (default 0.1)."""
    weight  = wf.get("hybrid_weight", 0.1)
    max_rrf = max(rrf.values(), default=0.0) or 1.0
    for c in candidates:
        r = rrf.get(c.get("_id"), 0.0)
        adj = round((r / max_rrf) * weight, 4) if r else 0.0
        c["hybrid_adj"] = adj
        c["rank_score"] = c.get("rank_score", c["score"]) + adj


def _apply_feedback(step: str, error: str, candidates: list, cfg: dict, threshold: float):
    """Annotate each candidate in place with feedback-derived fields:
      feedback_net — net 👍/👎 this workaround has earned for THIS failure
      feedback_adj — the signed ranking delta applied (0.0 when no votes)
      rank_score   — cosine score plus the delta (used for ordering + strong split)
      curated      — True once net likes cross feedback_curate_min (a repeatedly
                     confirmed "verified" fix)
      feedback     — "curated" | "boosted" | "demoted" | "" (UI tag + synthesis signal)
    The delta is tanh(net_votes / 2) * weight, so it saturates: one bad vote nudges,
    a pile of them caps out — a workaround can never be hard-buried by a single noisy
    click. A *curated* fix is pinned at/above threshold so a repeatedly-confirmed
    workaround stays a strong match (and keeps feeding synthesis) for its failure."""
    wf         = cfg["workaround_finder"]
    weight     = wf.get("feedback_weight", 0.15)
    curate_min = wf.get("feedback_curate_min", 3)
    for c in candidates:
        key = c.get("key") if c.get("type") == "ticket" else c.get("filename")
        net = feedback.net_votes(cfg, c.get("type", "ticket"), key, step, error)
        adj = tanh(net / 2) * weight if net else 0.0
        curated = net >= curate_min
        rank = c["score"] + adj
        if curated:
            rank = max(rank, threshold + 0.001)   # pin verified fixes as strong
        c["feedback_net"] = net
        c["feedback_adj"] = round(adj, 4)
        c["rank_score"]   = rank
        c["curated"]      = curated
        c["feedback"]     = ("curated" if curated else
                             "boosted" if net > 0 else
                             "demoted" if net < 0 else "")


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
    # (saves a request against the daily quota; the result can't change). Also
    # skipped when the master LLM switch is off or re-rank is disabled.
    wf = cfg["workaround_finder"]
    if (len(candidates) <= 1
            or not wf.get("llm_enabled", True)
            or not wf.get("llm_rerank", False)):
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
        out = g.generate(prompt, cfg["llm"], job="rerank").get("answer", "")
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
    # Free-form prose with nothing extractable: only call the LLM parser if the
    # master LLM switch is on; otherwise search the raw text.
    if not cfg["workaround_finder"].get("llm_enabled", True):
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
        out = g.generate(prompt, cfg["llm"], job="parse").get("answer", "")
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


NO_FIX_SENTINEL = "NO_RELIABLE_WORKAROUND"


def build_prompt(query: str, result: dict) -> str:
    """Prompt the LLM to produce a grounded workaround in the `=== FIX ===`
    format — a clean, paste-ready resolution comment — WITHOUT hallucinating.

    Anti-hallucination guardrails:
      - Only strong (>= threshold) matches are given as sources. Weak context is
        excluded so a near-miss can't leak into the answer.
      - The model is told to use ONLY steps present in the sources and to invent
        nothing (no field names / values not in the sources).
      - It has an explicit escape hatch: if the sources don't contain a clear,
        applicable fix it must reply exactly NO_RELIABLE_WORKAROUND, so the caller
        shows the raw best comment instead of a made-up one.

    Feedback-aware: each source is tagged with the human 👍/👎 it earned for THIS
    failure (from feedback.py), and the model is told to prefer VERIFIED/confirmed
    sources and steer away from ones marked wrong — so accumulated feedback raises
    the quality of the generated workaround, not just the ordering.
    """
    strong = result["strong"][:6]

    sources    = ""
    has_verified = False
    for i, h in enumerate(strong, 1):
        if h["type"] == "ticket":
            label = f"{h['key']} (score {h['score']:.2f}) — comment by {h['author']}"
            body  = h["comment"]
        else:
            label = f"Doc '{h['filename']}' (score {h['score']:.2f})"
            body  = h["comment"]
        # Human feedback signal for this source on this failure.
        net = h.get("feedback_net", 0)
        if h.get("curated"):
            label += f"  [✅ VERIFIED — confirmed working {net}× for this failure; PREFER THIS]"
            has_verified = True
        elif net > 0:
            label += f"  [👍 confirmed helpful {net}×]"
        elif net < 0:
            label += f"  [👎 marked wrong {-net}× — use only if clearly applicable]"
        sources += f"\n--- Source {i}: {label} ---\n{body[:600]}\n"

    feedback_rule = (
        "- Sources are tagged with human feedback. PREFER steps from VERIFIED / "
        "👍-confirmed sources, and avoid relying on a 👎 source unless it is the only "
        "one that clearly fits.\n"
        if any(h.get("feedback_net") for h in strong) or has_verified else ""
    )

    return (
        "You are a Salesforce order-fallout support assistant. A new issue needs a "
        "workaround:\n\n"
        f"{query}\n\n"
        "Below are the most similar PAST RESOLVED tickets. Write the workaround "
        "based STRICTLY on these sources.\n"
        "Rules:\n"
        "- Use ONLY actions, field names, and values that appear in the sources.\n"
        "- Do NOT invent steps. Do NOT add generic advice.\n"
        "- Write clear, imperative, step-by-step actions — each numbered step is ONE "
        "concrete action an engineer can follow.\n"
        "- STRIP all noise from the source text: greetings/sign-offs, person names and "
        "@mentions like [~accountid:...], ticket/PR links and URLs, MSISDNs / order IDs / "
        "record IDs, and any 'please check / can you look into this' chatter. Keep only "
        "the actual resolution actions.\n"
        f"{feedback_rule}"
        f"- If the sources do not contain a clear, applicable fix for THIS step and error "
        f"(e.g. they are just questions, status chatter, or a bare 'done/fixed' with no "
        f"actions), reply with exactly: {NO_FIX_SENTINEL}\n"
        f"{sources}\n"
        "Output EXACTLY this block and nothing else (omit the Root Cause line if the "
        "sources don't state a cause):\n"
        "=== FIX ===\n"
        "Root Cause: <one line, only if supported by the sources>\n"
        "1. <first action, imperative>\n"
        "2. <next action, imperative>\n"
        "3. <continue as needed>\n"
        "=== END ==="
    )
