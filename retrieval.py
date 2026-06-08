"""Lexical (BM25) retrieval + Reciprocal Rank Fusion — the keyword half of hybrid
search.

Vector search alone blurs exact tokens: an error code like "14081" or a step name
like "Cancel Addon" embeds close to its neighbours, so the exact-match ticket can
rank below a semantic near-miss. BM25 scores on literal token overlap, and RRF
fuses the two rankings so a candidate strong on EITHER signal surfaces.

The keyword index is built lazily from whatever is already in Chroma (no re-ingest
needed) and cached in-process; it auto-rebuilds when the indexed chunk count
changes. rank-bm25 is a pure-Python dependency — if it's missing, keyword_search
returns [] and the caller degrades cleanly to pure vector search.
"""

import re
import vectordb

_TOKEN = re.compile(r"[a-z0-9]+")
_cache: dict = {}   # index_path -> {count, bm25, ids, doc_by_id, meta_by_id, source_by_id}


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _entry_text(doc: str, meta: dict) -> str:
    """The text a chunk is indexed under for keyword search: its failed step, error,
    summary, and resolution comment — the same fields vector search embeds."""
    parts = [meta.get("step", ""), meta.get("error", ""),
             meta.get("summary", ""), meta.get("comment_body") or doc or ""]
    return " ".join(p for p in parts if p)


def _build(index_path: str, count: int) -> dict | None:
    try:
        from rank_bm25 import BM25Okapi
    except Exception as e:
        print(f"[BM25] rank-bm25 not installed ({e}) — keyword search disabled")
        return None

    ids, corpus, doc_by_id, meta_by_id, source_by_id = [], [], {}, {}, {}
    for c in vectordb.all_chunks(index_path):
        toks = _tokenize(_entry_text(c["doc"], c["meta"]))
        if not toks:
            continue
        ids.append(c["id"])
        corpus.append(toks)
        doc_by_id[c["id"]]    = c["doc"]
        meta_by_id[c["id"]]   = c["meta"]
        source_by_id[c["id"]] = c["source"]
    if not ids:
        return None
    data = {"count": count, "bm25": BM25Okapi(corpus), "ids": ids,
            "doc_by_id": doc_by_id, "meta_by_id": meta_by_id, "source_by_id": source_by_id}
    print(f"[BM25] built keyword index over {len(ids)} chunks")
    return data


def _get(index_path: str) -> dict | None:
    """Cached BM25 index, rebuilt when the indexed chunk count changes (so an ingest
    that adds/removes tickets is reflected without a manual refresh)."""
    count = vectordb.count(index_path)
    cached = _cache.get(index_path)
    if cached and cached["count"] == count:
        return cached
    data = _build(index_path, count)
    if data is not None:
        _cache[index_path] = data
    else:
        _cache.pop(index_path, None)
    return data


def build_keyword_index(index_path: str) -> int:
    """Force a rebuild (called at the end of an ingest pass). Returns chunk count."""
    _cache.pop(index_path, None)
    data = _get(index_path)
    return len(data["ids"]) if data else 0


def keyword_search(step: str, error: str, top_k: int, index_path: str) -> list[dict]:
    """BM25 hits for the query's step+error, best first, as
    {id, doc, meta, source, kw_score, kw_rank}. [] if the index is empty/unavailable
    or the query has no usable tokens."""
    try:
        data = _get(index_path)
    except Exception as e:
        print(f"[BM25] keyword search unavailable ({e})")
        return []
    if not data:
        return []
    query = _tokenize(f"{step} {error}")
    if not query:
        return []
    scores = data["bm25"].get_scores(query)
    order  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = []
    for rank, i in enumerate(order[:top_k], 1):
        if scores[i] <= 0:
            break                       # no token overlap beyond here
        id_ = data["ids"][i]
        hits.append({"id": id_, "doc": data["doc_by_id"][id_],
                     "meta": data["meta_by_id"][id_], "source": data["source_by_id"][id_],
                     "kw_score": float(scores[i]), "kw_rank": rank})
    return hits


def rrf_fuse(vector_hits: list[dict], keyword_hits: list[dict], k: int = 60) -> dict:
    """Reciprocal Rank Fusion: id -> sum of 1/(k+rank) across both ranked lists.
    Rank-based, so it needs no score calibration between cosine and BM25. Inputs
    must be ordered best-first and carry an "id"."""
    rrf: dict = {}
    for rank, h in enumerate(vector_hits, 1):
        if h.get("id") is not None:
            rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (k + rank)
    for rank, h in enumerate(keyword_hits, 1):
        if h.get("id") is not None:
            rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (k + rank)
    return rrf
