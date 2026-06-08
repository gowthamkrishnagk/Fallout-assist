"""Ticket graph — the "graph RAG" layer.

The fix for a failure is often scattered: several tickets hit the SAME failure but
word their resolution differently (so neither vector nor keyword search on the
comment connects them), or a ticket's own comment is just a pointer ("duplicate,
refer to SAC-x"). This builds a lightweight graph that links those siblings so a
fix living on a related ticket can be surfaced.

Edges (no new data — built from metadata already in Chroma):
  - Failure signature: tickets sharing the same (normalized step, error) — the
    structured "what failed / what was the error", independent of comment wording.
  - Pointer: a "refer to SAC-x" comment links the ticket to its referent.

Built lazily and cached; rebuilds when the indexed chunk count changes. Edges are
NOT materialized pairwise (a common signature can be shared by hundreds of tickets,
which would be O(n^2)); instead we keep signature->keys buckets and expand on
demand, skipping buckets too large to be a meaningful "same failure" cluster.
"""

import re
import vectordb

_cache: dict = {}   # index_path -> {count, key_sigs, sig_keys, ptr_adj, key_ids}

# A signature bucket bigger than this is too generic to mean "the same specific
# failure" (e.g. every "Internal Server Error" ticket), so it's skipped on expand.
_MAX_BUCKET = 200


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _sig(meta: dict) -> str:
    step = _norm(meta.get("step", ""))
    err  = _norm(meta.get("error", ""))
    if not step and not err:
        return ""
    return f"{step}||{err}"


def _build(index_path: str, count: int) -> dict:
    from textclean import is_pointer_comment, referenced_ticket

    key_sigs: dict = {}   # ticket key -> set(signature)
    sig_keys: dict = {}   # signature -> set(ticket key)
    ptr_adj:  dict = {}   # ticket key -> set(referenced key)  (and reverse)
    key_ids:  dict = {}   # ticket key -> [chunk ids] (for scoring an expanded sibling)

    for c in vectordb.all_chunks(index_path):
        if c["source"] != "ticket":
            continue
        meta = c["meta"]
        key  = meta.get("key")
        if not key:
            continue
        key_ids.setdefault(key, []).append(c["id"])

        sig = _sig(meta)
        if sig and sig != "||":
            key_sigs.setdefault(key, set()).add(sig)
            sig_keys.setdefault(sig, set()).add(key)

        body = meta.get("comment_body") or c["doc"] or ""
        if is_pointer_comment(body):
            ref = referenced_ticket(body)
            if ref and ref != key:
                ptr_adj.setdefault(key, set()).add(ref)
                ptr_adj.setdefault(ref, set()).add(key)

    print(f"[GRAPH] built over {len(key_ids)} tickets, "
          f"{len(sig_keys)} failure signatures")
    return {"count": count, "key_sigs": key_sigs, "sig_keys": sig_keys,
            "ptr_adj": ptr_adj, "key_ids": key_ids}


def _get(index_path: str) -> dict | None:
    count  = vectordb.count(index_path)
    cached = _cache.get(index_path)
    if cached and cached["count"] == count:
        return cached
    try:
        data = _build(index_path, count)
    except Exception as e:
        print(f"[GRAPH] build failed ({e}) — graph expansion disabled")
        return None
    _cache[index_path] = data
    return data


def build_graph(index_path: str) -> int:
    """Force a rebuild (called at the end of an ingest pass). Returns ticket count."""
    _cache.pop(index_path, None)
    data = _get(index_path)
    return len(data["key_ids"]) if data else 0


def expand(seed_keys, index_path: str, cap: int = 8) -> list[dict]:
    """1-hop neighbours of the seed tickets — siblings sharing a failure signature,
    plus pointer-linked tickets. Returns [{key, chunk_ids, via}] for keys NOT in the
    seed set, capped. `via` is "signature" or "pointer" for transparency."""
    data = _get(index_path)
    if not data:
        return []
    seeds = {k for k in seed_keys if k}
    found: dict = {}   # key -> via
    for k in seeds:
        for sig in data["key_sigs"].get(k, ()):
            bucket = data["sig_keys"].get(sig, ())
            if len(bucket) > _MAX_BUCKET:
                continue                      # too generic to be the same failure
            for nb in bucket:
                if nb not in seeds:
                    found.setdefault(nb, "signature")
        for nb in data["ptr_adj"].get(k, ()):
            if nb not in seeds:
                found[nb] = "pointer"         # pointer is a stronger, explicit link
    out = []
    for key, via in found.items():
        ids = data["key_ids"].get(key)
        if ids:                               # only siblings we actually have indexed
            out.append({"key": key, "chunk_ids": ids, "via": via})
        if len(out) >= cap:
            break
    return out
