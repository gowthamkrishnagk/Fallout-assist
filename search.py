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
import watable
from textclean import clean_text, clean_error_text, is_pointer_comment, referenced_ticket

DEFAULT_THRESHOLD = 0.70   # cosine-similarity scale; live value from config.json (score_threshold)


def _clean_query(text: str) -> str:
    """Normalize raw query input the SAME way ingest normalizes stored step/error
    (shared canonical cleaner) so a query and the matching ticket embed alike."""
    return clean_text(text)




def _resolution_quality(c: dict) -> float:
    """Cheap, LLM-FREE score of how usable a candidate's resolution is, so
    synthesis draws from the best-resolved tickets first instead of an arbitrary
    cosine order. Costs nothing (no network / no Groq quota).

    Rewards: the team's 8-field workaround table, the structured '=== FIX ===' block,
    numbered/bulleted steps, and substance. Penalizes one-word closers ('done',
    'fixed') and very short text.
    """
    if c.get("type") == "doc":
        return 3.0   # uploaded docs are curated workarounds — treat as solid
    body = (c.get("comment") or "").strip()
    if not body:
        return 0.0
    score = 0.0
    # The 8-field table is the CURRENT resolution format and the richest thing in the
    # corpus — a named cause, the action, the system touched and the follow-up. Without
    # this it scored ~2.0 (no === FIX ===, no numbered steps) and ranked below far
    # thinner prose resolutions, which is backwards.
    #
    # Rows come from what ingest already parsed (table_of), so this costs no parsing. Only
    # the CONTENT rows are counted — Cause / Solution applied / System modified / Customer
    # action / Category. The ticket-owned rows are filled from the description on every
    # table ever written, so counting them measured nothing and flattered a table whose
    # actual content rows were blank. A fully-filled table still tops out at +2.0.
    table = watable.table_of(c)
    if table:
        score += 5.0
        score += min(sum(1 for f, v in table.items()
                         if f not in watable.TICKET_FIELDS and watable.clean_value(v)),
                     5) * 0.4                               # content rows filled in
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


def find_workarounds(query: str, cfg: dict, exclude_keys=frozenset(),
                     offline: bool = False) -> dict:
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

    `offline=True` skips the live-Jira lookup used to resolve a pointer comment whose
    target isn't indexed (such a candidate is just dropped). Used by the self-test /
    any batch run so it stays fast and needs no network / API token.
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
    # error_match: "lexical" (default) decides the error leg vectorlessly — a different
    # error CODE is a different failure (see errormatch.py). In that mode the vector
    # error no longer GATES recall (err_floor → 0): vectors cast a wide net, the
    # lexical layer does the gating. "vector" restores the old cosine-floor behavior.
    err_mode    = wf.get("error_match", "lexical")
    err_floor   = 0.0 if err_mode == "lexical" else wf.get("error_floor", 0.55)

    # Hybrid parse: regex for labeled input, LLM fallback for free-form prose
    step, error = parse_input(query, cfg)

    # Categorical disambiguators (Order Type / Order Reason). Pulled from the query
    # text when present (the Jira bot's query carries them; free-form input may not).
    # Used as a keyword signal (BM25) + a tie-breaker boost — never an embedding leg.
    order_type   = _grab_field(query, "Order Type")
    order_reason = _grab_field(query, "Order Reason")

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
        kw_extra = " ".join(p for p in (order_type, order_reason) if p)
        rrf, all_hits = _hybrid_augment(step, error, step_emb, error_emb,
                                        all_hits, index_path, wf, err_weight, err_floor,
                                        extra_query=kw_extra)

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
            # True while `meta` still describes `comment`. The live-Jira pointer branch
            # below breaks that: it keeps the pointer's metadata but swaps in the
            # referenced ticket's body, so the stored table rows would be the wrong ones.
            meta_fits = True

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
                    if offline:
                        continue          # offline/self-test: no live Jira lookups
                    # Not indexed → follow to Jira live for the real resolution.
                    import ingest as ing
                    fetched = ing.fetch_ticket_resolution(ref, cfg)
                    if not fetched:
                        continue
                    key, comment = fetched["key"], fetched["comment"]
                    url, author  = fetched["url"], fetched.get("author", "")
                    # keep meta (the duplicate's step/error/summary — same failure)
                    meta_fits = False
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
                # Whether this comment is the ASSIGNEE's own (83% of the corpus). The
                # remaining 17% come from the resolver/any-human fallback — a shift
                # handover or escalation. Carried through so synthesis can weight the
                # engineer who actually worked the ticket above a bystander's note.
                "is_assignee": meta.get("is_assignee", ""),
                "comment":     comment,
                # The 8-field table rows, parsed ONCE at ingest. Empty for a prose
                # resolution and for a live-fetched pointer target; `watable.table_of`
                # falls back to parsing the body in both cases.
                "table":       watable.from_meta(meta) if meta_fits else {},
                "description": meta.get("description", ""),
                "order_type":  meta.get("order_type", ""),
                "order_reason": meta.get("order_reason", ""),
                "error":       meta.get("error", ""),
                "step":        meta.get("step", ""),
                "score":       score,
                "step_score":  h.get("step_score"),
                "updated_ts":  meta.get("updated_ts", 0),
            })
        else:
            # Docs are scored on the same (step, error) basis as tickets, so the
            # vectorless error leg applies here too — carry step/error + step_score.
            # The weak-doc drop (a doc below threshold is never shown as a vague
            # "reference") happens AFTER the lexical rescore, since the score moves.
            candidates.append({
                "type":     "doc",
                "_id":      h_id,
                "filename": meta.get("filename", ""),
                "chunk":    meta.get("chunk", 0),
                # "user_fix" marks an approved user-submitted workaround so the UI can
                # badge it. It's embedded on the exact failure's step+error, so it
                # already scores ~1.0 and ranks at top via the normal doc path — no
                # special boost needed here.
                "kind":     meta.get("kind", ""),
                "comment":  h["doc"],
                "error":    meta.get("error", ""),
                "step":     meta.get("step", ""),
                "score":    score,
                "step_score": h.get("step_score"),
            })

    # Vectorless decision: among the candidates the vector net gathered, re-score the
    # step + error legs lexically (input is copied from the system, never paraphrased,
    # so exact comparison is sharper than a blurred embedding) and drop any whose error
    # CODE contradicts the query's. No-op in "vector" mode or when the query has neither
    # field. Done before everything below so the lexical score flows through re-rank /
    # feedback / threshold split untouched.
    candidates = _lexical_rescore(step, error, candidates, wf, err_weight, err_mode)

    # Weak-doc drop: a doc below the match threshold is never shown as a vague
    # "reference". Deferred to here because the lexical rescore moves the score.
    candidates = [c for c in candidates
                  if c.get("type") != "doc" or c["score"] >= threshold]

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

    # Categorical match/mismatch: among candidates that tie on step+error, lift the one
    # whose Order Type / Order Reason matches the query and push down one that
    # contradicts it — the right fix for THIS kind of order (a Disconnect fix is wrong
    # for a New order).
    _apply_order_match(candidates, order_type, order_reason, wf)

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
                    index_path: str, wf: dict, err_weight: float, err_floor: float,
                    extra_query: str = ""):
    """Run BM25 keyword search, fuse it with the vector hits (RRF), and pull in
    keyword-only hits the vector pool missed — scoring them with the same dual
    cosine so they're directly comparable to the vector candidates.

    Returns (rrf_scores_by_id, augmented_hits). Best-effort: if the keyword index
    is unavailable (rank-bm25 missing / empty KB) it returns ({}, vector_hits)."""
    import retrieval
    top_k   = wf.get("top_k", 10)
    kw_hits = retrieval.keyword_search(step, error, max(top_k * 3, 30), index_path,
                                       extra=extra_query)
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
        for id_, d in sc.items():
            h = by_id[id_]
            extra.append({"id": id_, "doc": h["doc"], "meta": h["meta"],
                          "score": d["score"], "step_score": d["step_score"],
                          "error_score": d["error_score"]})
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
    best: dict = {}   # key -> (combined_score, id)
    for id_, sd in scores.items():
        d   = detail.get(id_)
        key = d["meta"].get("key") if d else None
        if not key or key in have_keys:
            continue
        sc = sd["score"]
        if key not in best or sc > best[key][0]:
            best[key] = (sc, id_)

    extra = [{"id": id_, "doc": detail[id_]["doc"], "meta": detail[id_]["meta"],
              "score": scores[id_]["score"], "step_score": scores[id_]["step_score"],
              "error_score": scores[id_]["error_score"], "via_graph": True}
             for _sc, id_ in best.values()]
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


def _lexical_rescore(query_step: str, query_error: str, candidates: list, wf: dict,
                     err_weight: float, err_mode: str) -> list:
    """Score the matched candidates VECTORLESSLY (vectors only gathered them):
      step  → errormatch.text_similarity  (exact/tolerant; input is copy-pasted)
      error → errormatch.error_similarity (hard CODE gate + cleaned-message overlap)

    Dynamic weighting — whichever fields the query carries decide the score, and a
    single field counts 100% (so a perfect error-only OR step-only search isn't capped
    below the strong threshold):
      both  present → step_lex*(1 - err_weight) + error_lex*err_weight
      error only    → error_lex
      step  only    → step_lex
    A candidate whose error CODE contradicts the query's is dropped (different failure).

    No-op (returns candidates unchanged) when err_mode != "lexical" (the config
    off-switch restores cosine behavior), or the query has neither a step nor an error
    (nothing to score lexically — the vector recall ordering stands)."""
    if err_mode != "lexical":
        return candidates
    import errormatch
    has_step, has_error = bool(query_step), bool(query_error)
    if not has_step and not has_error:
        return candidates
    kept = []
    for c in candidates:
        e_lex = s_lex = None
        if has_error:
            e_lex, gated = errormatch.error_similarity(query_error, c.get("error", ""))
            if gated:
                continue                   # different error code → different failure
        if has_step:
            s_lex = errormatch.text_similarity(query_step, c.get("step", ""))
        if has_step and has_error:
            c["score"] = round(s_lex * (1 - err_weight) + e_lex * err_weight, 3)
        else:
            c["score"] = e_lex if has_error else s_lex
        c["error_lex"], c["step_lex"] = e_lex, s_lex
        kept.append(c)
    return kept


def _apply_order_match(candidates: list, q_type: str, q_reason: str, wf: dict):
    """Categorical match/mismatch on Order Type / Order Reason. These are low-cardinality
    CATEGORICAL fields — useless as a weighted embedding (a 384-dim vector of 'Modify' ≈
    'New'), but decisive among the many candidates that tie on the SAME step+error: a
    Disconnect fix is wrong for a New order.

      match    (both sides present, equal)   → +order_match_weight   (lift the right kind)
      mismatch (both sides present, unequal) → -order_mismatch_penalty (sink the wrong kind)
      either side absent                     → no-op (absence is not a contradiction, so a
                                               free-form query with no Order Type is untouched)

    Bounded and additive like the feedback/hybrid nudges, applied to rank_score only — the
    raw score and display threshold stay truthful. A mismatch demotes (strong→context, or
    lower in context) but never erases. The penalty is larger than the boost so a
    contradicting category outweighs an incidental match elsewhere.

    Candidate categories come from the dedicated order_type/order_reason metadata, falling
    back to parsing the description blob so this still works on an index that predates the
    dedicated fields (i.e. not yet re-ingested)."""
    if not q_type and not q_reason:
        return
    w_type    = wf.get("order_match_weight", 0.05)
    p_type    = wf.get("order_mismatch_penalty", 0.12)
    w_reason  = w_type * 0.6   # Order Type is the sharper discriminator of the two
    p_reason  = p_type * 0.6

    def _cand(c, field, label):
        return c.get(field) or _grab_field(c.get("description", ""), label)

    for c in candidates:
        if c.get("type") != "ticket":
            continue
        adj = 0.0
        c_type = _cand(c, "order_type", "Order Type")
        if q_type and c_type:
            adj += w_type if _order_eq(q_type, c_type) else -p_type
        c_reason = _cand(c, "order_reason", "Order Reason")
        if q_reason and c_reason:
            adj += w_reason if _order_eq(q_reason, c_reason) else -p_reason
        if adj:
            c["order_adj"]  = round(adj, 4)
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


def _grab_field(text: str, field: str) -> str:
    """Pull a single labeled categorical field (e.g. 'Order Type', 'Order Reason')
    out of query text or a candidate's stored description. Mirrors ingest's
    _extract_description_fields matcher so the query and the stored chunk read a
    field the same way. '' when absent or a bare number (an id, not a category)."""
    m = re.search(
        rf'(?:^|\n|\|)\s*\*?{re.escape(field)}\*?\s*[:\|]\s*\*?([^*\n\|]{{1,80}}?)\*?\s*(?:\||$|\n)',
        text or "", re.IGNORECASE | re.MULTILINE
    )
    if not m:
        return ""
    val = m.group(1).strip().strip('*').strip()
    if not val or re.match(r'^\d+$', val):
        return ""
    return _clean_query(val)


def _order_eq(a: str, b: str) -> bool:
    """Categorical equality for Order Type / Order Reason — exact after normalization,
    or one's WORD set is a subset of the other's (tolerates 'Modify' vs 'Modify
    Service' without the substring trap where 'New' would match 'Renewal')."""
    ta = set(re.findall(r'[a-z0-9]+', a.lower()))
    tb = set(re.findall(r'[a-z0-9]+', b.lower()))
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


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
            r'Error(?:\s+(?:Description|Code))?\s*:\s*([^\n]{1,200})',
            text, re.IGNORECASE
        )
    # clean_error_text, not _clean_query: the leading code must survive on the QUERY
    # side too, or a 6-digit code is stripped here while ingest keeps it and the two
    # sides stop matching. Also matches an 'Error Code:' label, which the old pattern
    # missed entirely.
    error = clean_error_text(error_match.group(1).strip()) if error_match else ''

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


# Acknowledgement / status notes that are NOT workarounds: "already activated",
# "fixed by L3", "done", "no issue". Matched only when the WHOLE (short) comment is
# such a note — a comment that also describes an action ("re-triggered the step") is kept.
_NON_RESOLUTION_RE = re.compile(
    r'^\W*(?:'
    r'already\s+\w+'                                             # already activated / done
    r'|(?:fixed|done|handled|resolved|completed)\s+by\s+l?\d'    # fixed by L3, done by 2
    r'|no\s+(?:issue|action|fix|workaround)s?(?:\s+\w+){0,3}'   # no issue found / no action needed
    r'|done|fixed|closing|closed|resolved|n/?a|ok'
    r')\W*$',
    re.IGNORECASE)


def _is_non_resolution(body: str) -> bool:
    """True when a SHORT comment is pure acknowledgement / status ('already activated',
    'fixed by L3', 'done', 'no issue') rather than a workaround. A longer comment, or
    one whose first line describes an action, is kept."""
    b = (body or "").strip()
    if not b:
        return True
    head = b.splitlines()[0].strip().strip('*').strip()
    return bool(_NON_RESOLUTION_RE.match(head)) and len(b) < 80


def select_resolution(strong: list) -> dict | None:
    """The most relevant comment that is an ACTUAL workaround: the first ranked strong
    match whose comment isn't pure status chatter. Falls back to the top match only if
    every candidate is chatter (the LLM then declines on it). Avoids the trade-off of
    blindly feeding rank-1 — a 'fixed by L3' note no longer hides a real fix below it."""
    for c in strong:
        if not _is_non_resolution(c.get("comment", "")):
            return c
    return strong[0] if strong else None


# Jira media embeds — "!image-20260320-124732.png|width=687,alt="..."!" — and attachment
# links "[^file.png]". A comment that is ONLY a screenshot carries no actions (119 such
# chunks in the corpus) and would silently consume a synthesis slot.
_JIRA_MEDIA_RE  = re.compile(
    r'!\s*[^!\n]*?\.(?:png|jpe?g|gif|bmp|svg|pdf|docx?|xlsx?)[^!\n]*!?', re.IGNORECASE)
_JIRA_ATTACH_RE = re.compile(r'\[\^[^\]\n]+\]')

# Workflow / status notes that _NON_RESOLUTION_RE deliberately does NOT catch, because it
# is anchored to the WHOLE comment (^...$). These have trailing text, so they slip past:
#   "SLA Breached. Reason: Request from L3 to Hold"
#   "Bulk resolved - Nokia Delete Line Order Fallout"
# Both were observed occupying a synthesis slot.
_STATUS_NOTE_RE = re.compile(
    r'(?i)^\W*(?:'
    r'sla\s+breach\w*'
    r'|bulk\s+(?:resolved|closed|updated|completed)'
    r'|(?:on|placed\s+on|kept\s+on|keeping\s+on)\s+hold'
    r'|request\s+from\s+l\d'
    r'|waiting\s+(?:for|on)\b'
    r'|pending\s+(?:from|with|on)\b'
    r'|reopened\b'
    r'|will\s+(?:check|update|verify|revert)\b'
    r')')

# Stopwords matter for near-duplicate detection because resolutions are terse: 'retried
# the step' vs 'retried the order' share two of three tokens on stopwords alone, which
# would make two different fixes look like restatements of each other.
_STOPWORDS = frozenset("""the a an and or of to in on for is was be been being it its this
that these those with as at by from we i he she they you please kindly so then now have
has had do does did can could should would will shall not no yes ok okay after before
again per via out up""".split())


def _strip_media(body: str) -> str:
    """Drop Jira image/attachment markup, keeping the prose around it.

    Applied both when judging a source and when rendering it into the prompt: a
    screenshot is invisible to the model, so the markup is pure noise it would otherwise
    have to be instructed to ignore, and it burns tokens for nothing. Blank runs left
    behind are collapsed so the body reads as written."""
    out = _JIRA_ATTACH_RE.sub(' ', _JIRA_MEDIA_RE.sub(' ', body or ''))
    return re.sub(r'\n\s*\n+', '\n\n', out).strip()


def _content_tokens(body: str) -> frozenset:
    """Content words of a resolution — canonically cleaned, lowercased, stopwords and
    single characters dropped. The unit of comparison for near-duplicate detection."""
    toks = re.findall(r'[a-z0-9]+', clean_text(body or "").lower())
    return frozenset(t for t in toks if len(t) > 1 and t not in _STOPWORDS)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _is_unusable_source(body: str) -> bool:
    """True when a comment cannot contribute ACTIONS to a synthesized workaround.

    Deliberately stricter than _is_non_resolution. This only decides which sources fill
    the synthesis slots, so over-filtering merely means fewer sources — whereas
    _is_non_resolution also gates the raw comment shown to a user when the LLM declines,
    where over-filtering would change user-visible output.

    Rejects: empty, whatever _is_non_resolution already rejects, workflow/status notes,
    pointer comments ('duplicate, refer to SAC-x' — the follow-reference logic upstream
    has already resolved those to a real fix or dropped them), and bodies that are only
    an image/attachment embed."""
    b = (body or "").strip()
    if not b or _is_non_resolution(b):
        return True
    if _STATUS_NOTE_RE.match(b.splitlines()[0].strip().strip('*').strip()):
        return True
    if is_pointer_comment(b):
        return True
    # Only treat thin text as unusable when media markup was actually present — a terse
    # but real fix ("reprocessed") has few content tokens too and MUST be kept, since
    # over half the corpus is under 60 characters.
    stripped = _strip_media(b)
    if stripped != b and len(_content_tokens(stripped)) < 3:
        return True
    return False


# Two sources saying the same thing is CORROBORATION (the whole point of multi-source);
# five is waste. Admit near-duplicates up to _NEAR_DUP_ALLOW, then require later slots to
# bring something new. Threshold is deliberately loose — wrongly collapsing two distinct
# fixes loses information, while keeping one paraphrase only costs a few tokens.
_NEAR_DUP_SIM   = 0.65
_NEAR_DUP_ALLOW = 2


def select_resolutions(strong: list, limit: int) -> list:
    """The top `limit` strong matches that are ACTUAL workarounds, best rank first.

    Multi-source on purpose. The median stored resolution is ~56 characters
    ("reprocessed", "retried the step") and over half are under 60, so ONE comment is
    usually too thin to synthesize a real procedure from. Several resolutions of the
    SAME failure corroborate each other — one names the step, another the field to
    change, a third the order to do them in — and the retriever is accurate enough
    (93.6% Hit@1) that the strong set is genuinely the same failure, not a grab bag.

    Slots are scarce, so what fills them matters more than how many there are. Two
    filters keep the extra sources signal rather than noise:
      - _is_unusable_source drops chatter, workflow/status notes, pointer comments and
        screenshot-only bodies outright — they contribute no actions, and with 4 slots a
        wasted one is 25% of the evidence;
      - restatements are capped, not banned: identical content is dropped, and
        near-duplicates are admitted only up to _NEAR_DUP_ALLOW, so slot 3+ has to bring
        something new instead of a fourth rewording of "retried the step".

    Falls back to [strong[0]] when EVERY candidate is unusable, preserving the previous
    behaviour: the model sees it, recognises it isn't a fix, and returns the sentinel."""
    picked, seen, dups = [], [], 0
    for c in strong:
        body = (c.get("comment") or "").strip()
        if _is_unusable_source(body):
            continue
        toks = _content_tokens(body)
        if not toks or any(toks == p for p in seen):
            continue                       # nothing to add / exact restatement
        if any(_jaccard(toks, p) >= _NEAR_DUP_SIM for p in seen):
            if dups >= _NEAR_DUP_ALLOW:
                continue                   # enough corroboration already
            dups += 1
        seen.append(toks)
        picked.append(c)
        if len(picked) >= max(1, limit):
            break
    return picked or (strong[:1] if strong else [])


def build_prompt(query: str, result: dict, cfg: dict | None = None) -> str:
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

    MULTI-SOURCE: several strong matches are passed, not one. Individual resolutions
    are terse (median ~56 chars), so corroborating same-failure comments is what makes
    a complete procedure possible — see select_resolutions. Each source carries its OWN
    Failed Step / Error so the model can verify it really describes the query's failure
    instead of trusting the ranker blindly, and is told how to merge them.

    Feedback-aware: each source is tagged with the human 👍/👎 it earned for THIS
    failure (from feedback.py), plus whether the author was the ticket's ASSIGNEE (the
    engineer who actually worked it, 83% of the corpus). The model is told to prefer
    VERIFIED/confirmed sources and steer away from ones marked wrong — so accumulated
    feedback raises the quality of the generated workaround, not just the ordering.
    """
    strong, sources, has_verified = render_sources(result, cfg)
    return _assemble_prompt(query, strong, sources, has_verified)


def render_sources(result: dict, cfg: dict | None = None) -> tuple[list, str, bool]:
    """(picked_sources, rendered_source_block, any_verified) for a search result.

    Shared by both synthesis formats — the `=== FIX ===` prompt and the 8-field table
    prompt in watable.py — so a workaround is grounded in exactly the same evidence
    whichever output format is configured."""
    limit  = int((cfg or {}).get("workaround_finder", {}).get("synthesis_sources", 4) or 4)
    strong = select_resolutions(result["strong"], limit)

    sources    = ""
    has_verified = False
    for i, h in enumerate(strong, 1):
        if h["type"] == "ticket":
            label = f"{h['key']} (score {h['score']:.2f}) — comment by {h['author']}"
            if str(h.get("is_assignee")) == "True":
                label += " [ASSIGNEE — the engineer this ticket was assigned to]"
        else:
            label = f"Doc '{h['filename']}' (score {h['score']:.2f})"
        # Human feedback signal for this source on this failure.
        net = h.get("feedback_net", 0)
        if h.get("curated"):
            label += f"  [✅ VERIFIED — confirmed working {net}× for this failure; PREFER THIS]"
            has_verified = True
        elif net > 0:
            label += f"  [👍 confirmed helpful {net}×]"
        elif net < 0:
            label += f"  [👎 marked wrong {-net}× — use only if clearly applicable]"
        # The source's OWN failure, so a mis-ranked source can be recognised and
        # ignored rather than silently merged into the answer.
        ctx = ""
        if h.get("step"):
            ctx += f"Its Failed Step: {h['step']}\n"
        if h.get("error"):
            ctx += f"Its Error: {h['error']}\n"
        # 1000 matches the stored comment_body cap, so nothing retrievable is cut here.
        sources += (f"\n--- Source {i}: {label} ---\n{ctx}"
                    f"Resolution:\n{_strip_media(h['comment'])[:1000]}\n")

    has_feedback = bool(any(h.get("feedback_net") for h in strong) or has_verified)
    return strong, sources, has_feedback


def _assemble_prompt(query: str, strong: list, sources: str, has_verified: bool) -> str:
    """The `=== FIX ===` prompt proper, given an already-rendered source block."""
    feedback_rule = (
        "- Sources are tagged with human feedback. PREFER steps from VERIFIED / "
        "👍-confirmed sources, and avoid relying on a 👎 source unless it is the only "
        "one that clearly fits.\n"
        if has_verified else ""
    )

    plural = len(strong) > 1
    merge_rule = (
        "- The sources are DIFFERENT past tickets that hit the SAME failure. Merge them "
        "into ONE procedure: keep the actions they agree on, and where they conflict "
        "prefer the higher-scored source, then a VERIFIED/👍 one, then an ASSIGNEE one.\n"
        "- Each source shows its own Failed Step / Error. IGNORE any source whose own "
        "failure clearly differs from the issue above, even though it was retrieved.\n"
        "- Output ONE merged procedure. Do NOT write a section per source and do NOT "
        "repeat the same action twice.\n"
        if plural else ""
    )
    no_fix_rule = (
        f"- If NONE of the sources contain a real workaround — they are only "
        f"acknowledgements or status notes like 'already activated', 'fixed by L3', "
        f"'handled by L2', 'done', questions, or bare 'fixed/closed' with no concrete "
        f"actions — reply with exactly: {NO_FIX_SENTINEL}\n"
        if plural else
        f"- If this comment is NOT a real workaround — e.g. an acknowledgement or status "
        f"note like 'already activated', 'fixed by L3', 'handled by L2', 'done', a "
        f"question, or a bare 'fixed/closed' with no concrete actions — reply with "
        f"exactly: {NO_FIX_SENTINEL}\n"
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
        f"{merge_rule}"
        f"{feedback_rule}"
        f"{no_fix_rule}"
        f"{sources}\n"
        "Output EXACTLY this, and nothing else (omit the Root Cause line if the "
        "sources don't state a cause):\n"
        "=== FIX ===\n"
        "Root Cause: <one line, only if supported by the sources>\n"
        "1. <first action, imperative>\n"
        "2. <next action, imperative>\n"
        "3. <continue as needed>\n"
        "=== END ===\n"
        f"System modified: <which system the steps CHANGE: {watable._SYSTEM_CHOICES}, "
        "or NA>\n"
        "Customer action: <what the agent/customer does next to verify, or NA>"
    )


# The two labelled lines requested after `=== END ===`. They exist so ONE generation can
# fill the prefilled resolution table as well as the steps block — 'System modified' and
# 'Customer action' are table rows with no place inside a numbered procedure, and a second
# LLM call to get them would double the cost and let the two artifacts disagree.
_TRAILING_FIELDS = re.compile(
    r'(?:^|\n)[ \t*_]*(system(?:s)?[ \t]+modified|customer[ \t]+action)[ \t]*:',
    re.IGNORECASE)


def split_fix_answer(answer: str) -> tuple[str, dict]:
    """(fix_block, {table field: value}) for a synthesis reply.

    The block is what gets shown/posted as the workaround; the trailing metadata lines
    only feed the resolution-comment draft, so they are cut out of the visible answer.
    A model that ignored the trailing lines simply yields an empty dict."""
    text = (answer or "").strip()
    if not text or NO_FIX_SENTINEL in text:
        return text, {}
    m = _TRAILING_FIELDS.search(text)
    if not m:
        return text, {}
    block, trailing = text[:m.start()].strip(), text[m.start():]
    return (block or text), watable.parse_llm_fields(trailing)
