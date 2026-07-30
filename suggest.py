"""Build a suggested workaround for a query — the single source of truth shared by
the web UI (`/api/ask`) and the Jira auto-suggest bot (`jirabot`).

Wraps `search.find_workarounds` and the grounded `=== FIX ===` synthesis (with the
same anti-hallucination fallbacks the UI has always used) so a comment posted on a
Jira ticket is identical to what a user would see in the app for that ticket.

TWO ARTIFACTS come back from every call, and they are for different readers:

  answer              the workaround itself, as `=== FIX ===` numbered steps. This is
                      the main output — what an engineer reads and acts on, shown in the
                      app and posted as the bot's suggestion comment.
  resolution_template the 8-field workaround table (watable.py), pre-filled from this
                      ticket plus the same synthesis. Not a fix to act on: it is the
                      RESOLUTION COMMENT the assignee is asked to post when they close
                      the ticket, handed over as a ready-to-edit draft.

They are derived from one generation, never two, so the suggested steps and the drafted
resolution can't describe different fixes.
"""


def _fix_steps(wt, body: str, table: dict) -> str:
    """A matched resolution shown as `=== FIX ===` steps, with no LLM.

    Used whenever synthesis doesn't run — the LLM is off, every provider failed, or the
    match is already a clean block. Falls back to the comment's own text when there are
    no actions to turn into steps (a terse resolution with nothing procedural in it),
    since showing it verbatim beats showing nothing."""
    return wt.to_fix_block(body, table) or (body or "").strip()


def suggest_for_query(query_text: str, cfg: dict, exclude_keys=frozenset()) -> dict:
    """Returns:
      {
        ok, mode,                 # strong_match | low_confidence | no_data
        answer,                   # the workaround, as === FIX === steps
        resolution_template,      # prefilled 8-field table for the closing comment
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
    import watable as wt

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
        # Still hand over the blank template: the ticket has to be closed in the team's
        # format whether or not we could suggest anything, and the identifier rows are
        # ours to fill either way.
        return {**base, "mode": "no_data", "answer": answer,
                "resolution_template": wt.blank_template(query_text),
                "provider": "", "model": "", "llm_note": "",
                "best_score": 0, "declined": False, "top": None}

    wf         = cfg["workaround_finder"]
    llm_on     = wf.get("llm_enabled", True)            # master LLM switch
    synthesize = llm_on and wf.get("llm_synthesize", True)
    llm_note   = ""
    declined   = False   # True only when the LLM judged the sources have no real fix
    template   = ""      # filled below; blank template when there's no fix to draft

    # Strong match(es) found — produce a grounded `=== FIX ===` recommendation.
    if strong:
        # Same selection the LLM prompt uses: the most relevant REAL workaround, not
        # blindly rank-1 (so a 'fixed by L3' note doesn't become the verbatim answer).
        top      = s.select_resolution(strong) or strong[0]
        top_body = top["comment"]
        # The matched resolution's table rows, parsed once at ingest (watable.table_of
        # falls back to the body for anything the stored fields don't cover). {} means
        # this resolution is prose.
        top_table = wt.table_of(top)
        # Hybrid: if the LLM is disabled, or the best source is ALREADY a clean
        # === FIX === block, show it verbatim — no generation. A source already in the
        # team's 8-field table format is likewise the finished article, so its rows are
        # turned into steps directly rather than paraphrased by a model.
        if not synthesize or "=== fix ===" in top_body.lower() or top_table:
            answer, provider, model = _fix_steps(wt, top_body, top_table), "direct_match", ""
            # from_raw (not from_fix_answer): it reuses the matched table's own four
            # generated rows verbatim where they exist, which is strictly better than
            # re-deriving them from steps we just rendered.
            template = wt.from_raw(query_text, top_body, top_table or None)
        else:
            # Legacy / multiple comments → LLM synthesizes the block, grounded in
            # sources only. On decline (NO_RELIABLE_WORKAROUND), empty output, a
            # local-model answer, or any error → fall back to the raw best comment,
            # never invent.
            try:
                prompt = s.build_prompt(query_text, result, cfg)
                gen    = g.generate(prompt, cfg["llm"], job="synthesis")
                # The reply is the block plus two trailing table-only lines; the lines
                # are split off here so they never show up in the workaround itself.
                ans, extra = s.split_fix_answer(gen.get("answer") or "")
                if gen.get("provider") == "local":
                    print("[LLM] synthesis answered by local model — distrust, verbatim")
                    answer, provider, model = _fix_steps(wt, top_body, top_table), "direct_match", ""
                    template = wt.from_raw(query_text, top_body, top_table or None)
                elif s.NO_FIX_SENTINEL in ans or len(ans) < 10:
                    # The model judged the matched sources contain no real workaround.
                    # The UI still shows the raw best comment as a lead; the Jira bot
                    # uses `declined` to post the no-match pair instead of a non-fix.
                    # Left as raw text on purpose — this is NOT a workaround, so
                    # dressing it up in the fix format would misrepresent it.
                    print("[LLM] declined / empty — no reliable workaround")
                    answer, provider, model = top_body, "direct_match", ""
                    template = wt.blank_template(query_text)
                    declined = True
                else:
                    answer, provider, model = ans, gen["provider"], gen["model"]
                    template = wt.from_fix_answer(query_text, ans, extra)
            except Exception as llm_err:
                # All cloud providers failed — show the matched comment and tell the
                # user why, instead of silently dropping to a local model. Still in the
                # steps format, so the output shape never depends on LLM uptime.
                reason   = str(llm_err)[:160]
                llm_note = f"LLM unavailable, showing the raw matched comment: {reason}"
                print(f"[LLM] synthesis failed, verbatim fallback — {reason}")
                answer, provider, model = _fix_steps(wt, top_body, top_table), "direct_match", ""
                template = wt.from_raw(query_text, top_body, top_table or None)
        mode = "strong_match"

    else:
        # No strong match — abstain from synthesis (the sources are below the
        # relevance bar; generating from them is where hallucination happens).
        # Show the nearest weak match's actual comment as a lead instead. The template
        # stays blank: drafting a resolution off a below-threshold guess would put a
        # wrong fix into the corpus the moment someone posted it unread.
        top      = context[0] if context else None
        answer   = top["comment"] if top else "No similar tickets found."
        provider = "direct_match"
        model    = ""
        mode     = "low_confidence"
        template = wt.blank_template(query_text)

    return {**base, "mode": mode, "answer": answer,
            "resolution_template": template,
            "provider": provider, "model": model, "llm_note": llm_note,
            "declined": declined, "top": top}
