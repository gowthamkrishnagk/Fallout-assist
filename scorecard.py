"""Accuracy scorecard — measures retrieval accuracy against accumulated up/down feedback.

The up/down votes users give (feedback.py / trackers/feedback.json) are already labeled
ground truth: an up-vote on ticket K for failure F means "K's fix is correct for F"; a
down-vote means "K is wrong for F". This script turns that into a numeric report so every
change (vectorless vs vector, weights, threshold) is a MEASURED before/after, not a guess.

It evaluates RAW retrieval with the feedback re-ranking DISABLED — otherwise the
feedback would boost the very tickets it's being scored against (circular). So the
numbers reflect how well step/error/order matching ALONE puts the right fix on top.

Metrics (per failure that has at least one up-vote, then averaged):
  Hit@1   — was an up-voted ticket ranked #1?
  Hit@3   — was an up-voted ticket in the top 3?
  MRR     — 1 / rank of the first up-voted ticket (0 if absent)
  Neg-leak— fraction of failures where a down-voted ticket outranks every up-voted one

Usage:  py scorecard.py            # full report
        py scorecard.py --verbose  # also list each failure's outcome
"""

import copy
import sys

import feedback


def _load_labels(cfg: dict) -> dict:
    """Group feedback into per-failure labels:
       sig -> {"step", "error", "query", "pos": set(keys), "neg": set(keys)}
    `sig` is feedback's failure signature (error, else step). Only (kind, key) pairs
    with a non-zero NET vote become a label, so a ticket up- then down-voted nets out."""
    path = feedback._path(cfg)
    records = feedback._load_raw(path)
    if not records:
        return {}

    # Net vote per (sig, kind, key), plus a representative step/error/query per sig.
    nets: dict = {}
    meta: dict = {}
    for r in records:
        step, error = r.get("step", ""), r.get("error", "")
        sig = feedback.signature(step, error)
        if not sig:
            continue
        ident = (r.get("kind", "ticket"), (r.get("key", "") or "").strip())
        nets[(sig, ident)] = nets.get((sig, ident), 0) + int(r.get("vote", 0))
        # Prefer the real query the user typed; fall back to a labeled step/error blob.
        if sig not in meta:
            q = (r.get("query_raw") or "").strip()
            if not q:
                q = "\n".join(p for p in (f"Failed Step: {step}" if step else "",
                                          f"Error: {error}" if error else "") if p)
            meta[sig] = {"step": step, "error": error, "query": q}

    labels: dict = {}
    for (sig, ident), net in nets.items():
        if net == 0:
            continue
        lab = labels.setdefault(sig, {**meta[sig], "pos": set(), "neg": set()})
        (lab["pos"] if net > 0 else lab["neg"]).add(ident)
    # Keep only failures that have at least one positive (something to find).
    return {s: l for s, l in labels.items() if l["pos"]}


def _ranked_idents(result: dict) -> list:
    """Candidate identities (kind, key) in ranked order: strong first, then context."""
    out = []
    for c in result["strong"] + result["context"]:
        if c.get("type") == "doc":
            out.append(("doc", (c.get("filename", "") or "").strip()))
        else:
            out.append(("ticket", (c.get("key", "") or "").strip()))
    return out


def _eval_one(label: dict, cfg: dict) -> dict:
    import search
    result = search.find_workarounds(label["query"], cfg)
    ranked = _ranked_idents(result)
    pos, neg = label["pos"], label["neg"]

    pos_rank = next((i + 1 for i, ident in enumerate(ranked) if ident in pos), None)
    neg_rank = next((i + 1 for i, ident in enumerate(ranked) if ident in neg), None)

    return {
        "hit1":  pos_rank == 1,
        "hit3":  pos_rank is not None and pos_rank <= 3,
        "mrr":   (1.0 / pos_rank) if pos_rank else 0.0,
        "found": pos_rank is not None,
        "leak":  neg_rank is not None and (pos_rank is None or neg_rank < pos_rank),
        "pos_rank": pos_rank,
        "neg_rank": neg_rank,
    }


def run(verbose: bool = False, cfg: dict = None) -> dict:
    import json
    if cfg is None:
        cfg = json.load(open("config.json"))

    labels = _load_labels(cfg)
    if not labels:
        print("No usable feedback yet - give up/down votes in the app, then re-run.\n"
              "(The scorecard needs at least one up-vote to know the correct fix for a failure.)")
        return {}

    # Honest measurement: turn OFF the feedback re-rank and the LLM (we're scoring
    # raw step/error/order retrieval, not the feedback boost or LLM synthesis).
    cfg = copy.deepcopy(cfg)
    wf = cfg["workaround_finder"]
    wf["feedback_weight"]     = 0.0
    wf["feedback_curate_min"] = 10**9
    wf["llm_enabled"]         = False

    rows = []
    for sig, label in sorted(labels.items()):
        m = _eval_one(label, cfg)
        rows.append((sig, label, m))
        if verbose:
            tag = "OK " if m["hit1"] else ("hit" if m["found"] else "MISS")
            leak = "  NEG-LEAK" if m["leak"] else ""
            print(f"  [{tag}] pos_rank={m['pos_rank']} mrr={m['mrr']:.2f}{leak}  "
                  f"err={ (label['error'] or label['step'])[:55]!r}")

    n = len(rows)
    summary = {
        "failures":  n,
        "hit@1":     sum(r[2]["hit1"]  for r in rows) / n,
        "hit@3":     sum(r[2]["hit3"]  for r in rows) / n,
        "mrr":       sum(r[2]["mrr"]   for r in rows) / n,
        "found":     sum(r[2]["found"] for r in rows) / n,
        "neg_leak":  sum(r[2]["leak"]  for r in rows) / n,
        "mode":      wf.get("error_match", "lexical"),
    }

    print("\n================ ACCURACY SCORECARD ================")
    print(f" error_match mode : {summary['mode']}")
    print(f" labeled failures : {summary['failures']}  (from up/down-vote feedback)")
    print(f" Hit@1            : {summary['hit@1']:.1%}   (right fix ranked #1)")
    print(f" Hit@3            : {summary['hit@3']:.1%}   (right fix in top 3)")
    print(f" MRR              : {summary['mrr']:.3f}   (1/rank of right fix)")
    print(f" Found at all     : {summary['found']:.1%}")
    print(f" Neg-leak         : {summary['neg_leak']:.1%}   (down-voted fix outranks up-voted - lower=better)")
    print("====================================================")
    print(" Tip: flip workaround_finder.error_match between 'lexical' and 'vector'")
    print("      and re-run to compare the two head-to-head on YOUR feedback.")
    return summary


# ── Self-test (feedback-free) ────────────────────────────────────────────────────
#
# Feedback is sparse — most users never vote. But every indexed ticket is already a
# labeled example: it knows its OWN error and its OWN fix, and many tickets share a
# failure. So we grade the matcher against itself (leave-one-out): hide a ticket,
# search with its error, and check whether a SAME-FAILURE sibling comes back on top.
# Thousands of test cases, zero feedback required.

def _failure_sig(step: str, error: str) -> str:
    """The 'same failure' key two tickets are grouped under. Error code first (the
    sharp discriminator), else the cleaned error text, else the cleaned step."""
    import errormatch
    from textclean import clean_text
    code, _ = errormatch.error_signature(error)
    if code:
        return "code:" + code
    ce = clean_text(error or "").lower()
    if ce:
        return "err:" + ce
    return "step:" + clean_text(step or "").lower()


def _ticket_corpus(cfg: dict):
    """All indexed tickets once as {key, step, error, comment}, plus a map
    failure_sig -> set(keys) so each ticket's same-failure siblings are known."""
    import vectordb
    from pathlib import Path
    from textclean import referenced_ticket, is_pointer_comment
    index_path = str(Path(__file__).parent / cfg["workaround_finder"]["index_path"])

    tickets, by_sig, pointer = {}, {}, {}
    for c in vectordb.all_chunks(index_path):
        if c.get("source") != "ticket":
            continue
        m   = c.get("meta", {})
        key = (m.get("key") or "").strip()
        if not key or key in tickets:
            continue
        step, error = m.get("step", ""), m.get("error", "")
        body = m.get("comment_body", "") or c.get("doc", "")
        tickets[key] = {"key": key, "step": step, "error": error, "comment": body}
        by_sig.setdefault(_failure_sig(step, error), set()).add(key)
        # A 'duplicate, refer to SAC-x' comment is an explicit same-failure label.
        if is_pointer_comment(body):
            ref = referenced_ticket(body)
            if ref:
                pointer[key] = ref
    return tickets, by_sig, pointer


def run_selftest(sample: int = 250, verbose: bool = False, cfg: dict = None,
                 offline: bool = True, quiet: bool = False) -> dict:
    import json, random
    import search
    if cfg is None:
        cfg = json.load(open("config.json"))
    cfg = copy.deepcopy(cfg)
    wf = cfg["workaround_finder"]
    wf["feedback_weight"]     = 0.0       # honest: no feedback boost
    wf["feedback_curate_min"] = 10**9
    wf["llm_enabled"]         = False     # measure retrieval, not synthesis

    tickets, by_sig, pointer = _ticket_corpus(cfg)

    # Evaluable = tickets that HAVE at least one same-failure sibling (something the
    # leave-one-out search could correctly find). Singletons are unscorable, skipped.
    evaluable = []
    for key, t in tickets.items():
        sig      = _failure_sig(t["step"], t["error"])
        siblings = by_sig.get(sig, set()) - {key}
        if pointer.get(key):
            siblings = siblings | {pointer[key]}
        if not (t["step"] or t["error"]) or not siblings:
            continue
        evaluable.append((key, t, siblings))

    if not evaluable:
        print("No evaluable tickets (need indexed tickets that share a failure). "
              "Ingest first.")
        return {}

    total = len(evaluable)
    if sample and sample < total:
        random.seed(42)                   # reproducible across lexical/vector runs
        chosen = random.sample(evaluable, sample)
    else:
        chosen = evaluable

    hit1 = hit3 = found = wrong1 = 0
    mrr_sum = 0.0
    for key, t, siblings in chosen:
        query  = "\n".join(p for p in (f"Failed Step: {t['step']}" if t["step"] else "",
                                       f"Error: {t['error']}" if t["error"] else "") if p)
        result = search.find_workarounds(query, cfg, exclude_keys={key},   # hide self
                                         offline=offline)
        ranked = [c["key"] for c in result["strong"] + result["context"]
                  if c.get("type") != "doc" and c.get("key")]
        rank = next((i + 1 for i, k in enumerate(ranked) if k in siblings), None)
        if rank:
            found += 1
            mrr_sum += 1.0 / rank
            hit1 += rank == 1
            hit3 += rank <= 3
        if ranked and ranked[0] not in siblings:
            wrong1 += 1                    # #1 is a DIFFERENT failure
        if verbose:
            tag = "OK " if rank == 1 else ("hit" if rank else "MISS")
            print(f"  [{tag}] {key} rank={rank} sig_siblings={len(siblings)} "
                  f"err={(t['error'] or t['step'])[:50]!r}")

    n = len(chosen)
    summary = {
        "mode": wf.get("error_match", "lexical"),
        "evaluable": total, "tested": n,
        "hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": mrr_sum / n,
        "found": found / n, "wrong@1": wrong1 / n,
    }
    if not quiet:
        print("\n============ SELF-TEST SCORECARD (no feedback needed) ============")
        print(f" error_match mode : {summary['mode']}")
        print(f" tickets tested   : {n} of {total} evaluable  (rest are unique failures)")
        print(f" Hit@1            : {summary['hit@1']:.1%}   (a same-failure ticket ranked #1)")
        print(f" Hit@3            : {summary['hit@3']:.1%}   (same-failure ticket in top 3)")
        print(f" MRR              : {summary['mrr']:.3f}")
        print(f" Found at all     : {summary['found']:.1%}")
        print(f" Wrong@1          : {summary['wrong@1']:.1%}   (#1 was a DIFFERENT failure - lower=better)")
        print("==================================================================")
        print(" Tip: flip workaround_finder.error_match ('lexical' vs 'vector'),")
        print("      re-run, and compare Hit@1 - same sample each time (seeded).")
    return summary


# ── Automatic logging (called after each ingest) ─────────────────────────────────

def _history_path(cfg: dict):
    from pathlib import Path
    rel = cfg["workaround_finder"].get("selftest_history_path",
                                       "trackers/scorecard_history.json")
    return Path(__file__).parent / rel


def latest(cfg: dict) -> dict:
    """The most recent logged self-test score, or {} if none yet."""
    import json
    try:
        hist = json.loads(_history_path(cfg).read_text(encoding="utf-8"))
        return hist[-1] if hist else {}
    except Exception:
        return {}


def log_selftest(cfg: dict, sample: int = None) -> dict:
    """Run the self-test quietly and APPEND the score (+ timestamp) to the history
    file, so accuracy is tracked automatically over time. Called after each ingest.
    Best-effort: returns {} and logs nothing on any failure. Keeps the last 200 runs."""
    import json
    from datetime import datetime
    wf = cfg["workaround_finder"]
    if sample is None:
        sample = int(wf.get("selftest_sample", 200) or 200)
    summary = run_selftest(sample=sample, cfg=cfg, offline=True, quiet=True)
    if not summary:
        return {}
    entry = {"ts": datetime.utcnow().isoformat(),
             "mode": summary["mode"], "tested": summary["tested"],
             "hit1": round(summary["hit@1"], 4), "hit3": round(summary["hit@3"], 4),
             "mrr": round(summary["mrr"], 4), "found": round(summary["found"], 4),
             "wrong1": round(summary["wrong@1"], 4)}
    path = _history_path(cfg)
    try:
        hist = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(hist, list):
            hist = []
    except Exception:
        hist = []
    hist.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hist[-200:], indent=2), encoding="utf-8")
    print(f"[SELFTEST] {entry['mode']} Hit@1={entry['hit1']:.1%} "
          f"Hit@3={entry['hit3']:.1%} (tested {entry['tested']})")
    return entry


def _usage():
    print("Usage:")
    print("  py scorecard.py                  feedback-based scorecard")
    print("  py scorecard.py --selftest       leave-one-out self-test (no feedback needed)")
    print("  py scorecard.py --selftest --full   test ALL evaluable tickets (slow)")
    print("  py scorecard.py --selftest --sample 500")
    print("  add --verbose / -v to either for a per-case breakdown")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if "--help" in sys.argv or "-h" in sys.argv:
        _usage()
    elif "--selftest" in sys.argv:
        n = 0 if "--full" in sys.argv else 250
        if "--sample" in sys.argv:
            try:
                n = int(sys.argv[sys.argv.index("--sample") + 1])
            except (IndexError, ValueError):
                pass
        run_selftest(sample=n, verbose=verbose)
    else:
        run(verbose=verbose)
