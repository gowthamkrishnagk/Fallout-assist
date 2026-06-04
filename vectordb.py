"""Chroma vector DB wrapper.

Tickets and documents are both indexed the same way — by step and error
separately — so matching is driven by *what failed* and *what the error was*,
not by incidental prose overlap:

  workarounds_step       / workarounds_error       — Jira ticket chunks
  workarounds_docs_step  / workarounds_docs_error  — uploaded document chunks

_search_dual() searches a step + error collection pair separately and averages
the scores so neither field can dominate the other.

Scoring note: collections use Chroma's default L2 (squared-euclidean) space and
MiniLM vectors are unit-length, so dist = 2 - 2*cos. We recover the true cosine
similarity as cos = 1 - dist/2.
"""

from pathlib import Path
import chromadb

_clients: dict = {}   # index_path → PersistentClient
_cols:    dict = {}   # (index_path, col_name) → Collection

TICKET_COLS = ("workarounds_step", "workarounds_error")
DOC_COLS    = ("workarounds_docs_step", "workarounds_docs_error")


# hnswlib raises "Cannot return the results ... ef or M is too small" when a
# query asks for more results (n_results) than the index's search_ef. Chroma's
# default search_ef is 10, which is below our over-fetch (top_k * 3), so set a
# larger ef on new collections. (Ignored for collections that already exist —
# those are handled by the back-off retry in _search_dual.)
_HNSW_META = {"hnsw:search_ef": 256}

# Per-collection candidate pool for dual (step+error) search. Must stay <= the
# collection's search_ef (256). Large enough to cover a same-step cluster so the
# step-list and error-list overlap and true both-sides matches score ~1.0.
_DUAL_FETCH = 250


def _get_col(index_path: str, col_name: str):
    key = (index_path, col_name)
    if key not in _cols:
        if index_path not in _clients:
            Path(index_path).mkdir(parents=True, exist_ok=True)
            _clients[index_path] = chromadb.PersistentClient(path=index_path)
        _cols[key] = _clients[index_path].get_or_create_collection(
            col_name, metadata=_HNSW_META)
    return _cols[key]


# ── Indexing ───────────────────────────────────────────────────────────────────

def _add_dual(step_col, error_col, ids, step_embs, error_embs, documents, metadatas, index_path):
    """Add chunks to a (step, error) collection pair. Either embedding may be
    None per-chunk → that chunk simply isn't placed in that side's collection."""
    def _add(col_name, embs):
        keep = [i for i, e in enumerate(embs) if e is not None]
        if not keep:
            return
        _get_col(index_path, col_name).add(
            ids=[ids[i] for i in keep],
            embeddings=[embs[i] for i in keep],
            documents=[documents[i] for i in keep],
            metadatas=[metadatas[i] for i in keep],
        )
    _add(step_col,  step_embs)
    _add(error_col, error_embs)


def add_tickets(ids, step_embs, error_embs, documents, metadatas, index_path):
    _add_dual(*TICKET_COLS, ids, step_embs, error_embs, documents, metadatas, index_path)


def add_docs_dual(ids, step_embs, error_embs, documents, metadatas, index_path):
    _add_dual(*DOC_COLS, ids, step_embs, error_embs, documents, metadatas, index_path)


def _delete_by_source(col_names, source_id, index_path):
    total = 0
    for col_name in col_names:
        col = _get_col(index_path, col_name)
        ids = col.get(where={"source_id": source_id}).get("ids", [])
        if ids:
            col.delete(ids=ids)
            total += len(ids)
    return total


def delete_tickets(index_path: str) -> int:
    """Delete all ticket chunks from both ticket collections."""
    total = 0
    for col_name in TICKET_COLS:
        col = _get_col(index_path, col_name)
        ids = col.get(where={"source": "ticket"}).get("ids", [])
        if ids:
            col.delete(ids=ids)
            total += len(ids)
    return total


def delete_ticket_by_source(source_id: str, index_path: str) -> int:
    """Delete one ticket's chunks (by its Jira key) from both ticket collections —
    used for incremental upsert: clear a changed ticket before re-adding it."""
    return _delete_by_source(TICKET_COLS, source_id, index_path)


def reset_ticket_collections(index_path: str):
    """Drop and recreate the ticket collections so a full rebuild gets correct
    HNSW params (search_ef) even if the old collections were built with the small
    default ef. Document collections are left untouched."""
    if index_path not in _clients:
        Path(index_path).mkdir(parents=True, exist_ok=True)
        _clients[index_path] = chromadb.PersistentClient(path=index_path)
    client = _clients[index_path]
    for col_name in TICKET_COLS:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass
        _cols.pop((index_path, col_name), None)
        _get_col(index_path, col_name)   # recreate eagerly with _HNSW_META


def delete_docs_by_source(source_id: str, index_path: str) -> int:
    return _delete_by_source(DOC_COLS, source_id, index_path)


# ── Search ─────────────────────────────────────────────────────────────────────

def _search_dual(step_col_name, error_col_name, step_emb, error_emb, top_k, index_path):
    """Search a step + error collection pair with dynamic weighting:
      - both embeddings present → average 50 / 50
      - only one side present   → 100% that side
    Either embedding may be None. Returns [] if both are None or both empty."""
    if step_emb is None and error_emb is None:
        return []

    step_col  = _get_col(index_path, step_col_name)
    error_col = _get_col(index_path, error_col_name)
    step_count, error_count = step_col.count(), error_col.count()

    doc_map, meta_map = {}, {}

    def _query(col, emb, count):
        scores = {}
        if emb is None or count == 0:
            return scores
        # Fetch a LARGE candidate pool, not just top_k*3: step and error are
        # scored from separate collections and averaged, so a ticket only gets
        # credit on a side if it appears in THAT side's results. When many tickets
        # share a step (e.g. dozens of "Cancel Addon" tickets with different
        # errors), a small pool lets the step-list and error-list miss each other,
        # scoring a genuine both-sides match at only ~0.5. A big pool (capped at
        # the collection's search_ef) makes the lists overlap so true matches
        # score ~1.0. Back off on the hnswlib "ef too small" error rather than crash.
        n   = min(_DUAL_FETCH, count)
        res = None
        while n >= 1:
            try:
                res = col.query(query_embeddings=[emb], n_results=n)
                break
            except RuntimeError as e:
                if "ef or M" not in str(e) and "Cannot return" not in str(e):
                    raise
                n //= 2
        if not res:
            return scores
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0]
        ):
            # cos = 1 - dist/2 (L2 space + unit-length MiniLM vectors).
            scores[id_] = 1 - dist / 2
            if id_ not in doc_map:
                doc_map[id_]  = doc
                meta_map[id_] = meta
        return scores

    step_scores  = _query(step_col,  step_emb,  step_count)
    error_scores = _query(error_col, error_emb, error_count)

    use_step  = step_emb  is not None and step_count  > 0
    use_error = error_emb is not None and error_count > 0

    combined = []
    for id_ in set(step_scores) | set(error_scores):
        s = step_scores.get(id_, 0.0)
        e = error_scores.get(id_, 0.0)
        if use_step and use_error:
            score = (s + e) / 2          # equal weight
        elif use_step:
            score = s                     # step-only → 100% step
        else:
            score = e                     # error-only → 100% error
        combined.append((id_, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [
        {"doc": doc_map[id_], "meta": meta_map[id_], "score": round(score, 3)}
        for id_, score in combined[:top_k]
        if id_ in doc_map
    ]


def search_dual(step_emb, error_emb, top_k, index_path):
    return _search_dual(*TICKET_COLS, step_emb, error_emb, top_k, index_path)


def search_docs_dual(step_emb, error_emb, top_k, index_path):
    """Match documents on step + error — the same basis as tickets — so a doc
    only ranks high when its failed step and error are semantically close to the
    query, not when it merely shares boilerplate prose."""
    return _search_dual(*DOC_COLS, step_emb, error_emb, top_k, index_path)


# ── Misc ───────────────────────────────────────────────────────────────────────

def get_ticket_meta(key: str, index_path: str) -> dict:
    """Stored metadata (step/error/summary…) for a ticket key, or {} if the
    ticket isn't indexed. Lets the ticket-ID search fall back to the index when
    a live Jira fetch fails."""
    try:
        got = _get_col(index_path, "workarounds_step").get(where={"key": key})
        metas = got.get("metadatas", [])
        return metas[0] if metas else {}
    except Exception:
        return {}


def count(index_path: str) -> int:
    try:
        return (
            _get_col(index_path, "workarounds_step").count() +
            _get_col(index_path, "workarounds_docs_step").count()
        )
    except Exception:
        return 0


def reset(index_path: str):
    global _cols
    for col_name in (*TICKET_COLS, *DOC_COLS, "workarounds_docs", "workarounds"):
        try:
            col = _get_col(index_path, col_name)
            all_ids = col.get()["ids"]
            if all_ids:
                col.delete(ids=all_ids)
        except Exception:
            pass
    _cols = {}
