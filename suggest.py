"""Build a suggested workaround for a query — the single source of truth shared by
the web UI (`/api/ask`) and the Jira auto-suggest bot (`jirabot`).

Wraps `search.find_workarounds` and the grounded `=== FIX ===` synthesis (with the
same anti-hallucination fallbacks the UI has always used) so a comment posted on a
Jira ticket is identical to what a user would see in the app for that ticket.
"""


def suggest_for_query(query_text: str, cfg: dict, exclude_keys=frozenset()) -> dict:
    """Returns:
      {
        ok, mode,                 # strong_match | low_confidence | no_data
        answer,                   # the workaround text (synthesized or verbatim)
        provider, model, llm_note,
        threshold, best_score,
        strong, context,          # the ranked candidates
        query_step, query_error,  # normalized failure (for feedback scoping)
        top,                      # the candidate that produced the answer, or None
      }

    `exclude_keys` is threaded into the search so the Jira 👎 "improve" loop skips
    workarounds already rejected on a ticket and advances to the next-best match.
    """
    import search as s
    import generate as g
    import ingest as ing

    result     = s.find_workarounds(query_text, cfg, exclude_keys=exclude_keys)
    strong     = result["strong"]
    context    = result["context"]
    threshold  = result["threshold"]
    best_score = result["best_score"]

    base = {
        "ok":          True,
        "threshold":   threshold,
        "best_score":  best_score,
        "strong":      strong,
        "context":     context,
        "query_step":  result.get("query_step", ""),
        "query_error": result.get("query_error", ""),
    }

    # Nothing matched — KB empty, or no ticket shares this error (a step-only match
    # with a different error is dropped, not surfaced as a weak lead).
    if not strong and not context:
        empty  = ing.get_status(cfg).get("total_chunks", 0) == 0
        answer = ("No tickets are indexed yet — ingest first."
                  if empty else
                  "No past resolution matches this failure. No ticket has this "
                  "error, so there's no workaround to suggest (a different error on "
                  "the same step is treated as a different problem).")
        return {**base, "mode": "no_data", "answer": answer,
                "provider": "", "model": "", "llm_note": "",
                "best_score": 0, "declined": False, "top": None}

    wf         = cfg["workaround_finder"]
    llm_on     = wf.get("llm_enabled", True)            # master LLM switch
    synthesize = llm_on and wf.get("llm_synthesize", True)
    llm_note   = ""
    declined   = False   # True only when the LLM judged the sources have no real fix

    # Strong match(es) found — produce a grounded `=== FIX ===` recommendation.
    if strong:
        # Same selection the LLM prompt uses: the most relevant REAL workaround, not
        # blindly rank-1 (so a 'fixed by L3' note doesn't become the verbatim answer).
        top      = s.select_resolution(strong) or strong[0]
        top_body = top["comment"]
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
                    # The model judged the matched sources contain no real workaround.
                    # The UI still shows the raw best comment as a lead; the Jira bot
                    # uses `declined` to STAY SILENT instead of posting a non-fix.
                    print("[LLM] declined / empty — no reliable workaround")
                    answer, provider, model = top_body, "direct_match", ""
                    declined = True
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

    return {**base, "mode": mode, "answer": answer,
            "provider": provider, "model": model, "llm_note": llm_note,
            "declined": declined, "top": top}
