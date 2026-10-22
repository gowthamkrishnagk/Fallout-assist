"""Owned historical index using LangChain documents and Chroma retrieval."""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from . import storage
from .config import VECTOR_DIR, config_value
from .embedding_text import EmbeddingWindow

EMBEDDING_INDEX_VERSION = "token-passages-v2"


def field(text: str, *names: str) -> str:
    for name in names:
        label = rf"\*{{0,2}}\s*{re.escape(name)}\s*\*{{0,2}}\s*:\s*\*{{0,2}}"
        patterns = (
            rf"(?im)^\s*{label}\s*([^\n]+)$",
            rf"(?im)^\s*\|{{1,2}}\s*{re.escape(name)}\s*\|{{1,2}}\s*([^\n|]+)",
        )
        for pattern in patterns:
            found = re.search(pattern, text)
            if found:
                return found.group(1).strip(" |*\t")[:500]
    return ""


def parse_failure(text: str) -> tuple[str, str]:
    step = field(text, "Failed Step", "Step")
    error = field(text, "Error Description", "Error Code", "Error")
    # Fallout Jira summaries put the step inline after other text, for example:
    # "ORDER_FALLOUT: ... - Failed Step: Activate Account - Order: ...".
    # The structured field parser above intentionally handles standalone labels;
    # this fallback extracts the inline summary form used by the live project.
    if not step:
        match = re.search(
            r"(?i)\bFailed\s+Step\s*:\s*(.+?)(?:\s+-\s*Order\s*:|\n|$)", text)
        if match:
            step = match.group(1).strip(" -*|\t")[:500]
    return step, error


def extract_error_code(text: str) -> str:
    """Extract explicit short error codes without mistaking a leading year for one."""
    value = (text or "").strip()
    patterns = (
        r"^\s*\[?(-?\d{1,6})\]?\s*$",
        r"(?i)\b(?:error\s*code|status\s*code|http(?:\s+status)?)\s*[:=# ]+([A-Z][A-Z0-9_-]*|-?\d{1,6})\b",
        r"(?i)\berror\s*[:=#]\s*([A-Z][A-Z0-9_-]*|-?\d{1,6})\b",
        r"^\s*\[?([A-Z][A-Z0-9_-]{1,30}|-?\d{1,6})\]?\s*(?:(?:\||:|-|–)|(?=\s+[A-Z]))",
        r"\b(UNABLE_TO_LOCK_ROW|DUPLICATE_VALUE|REQUEST_LIMIT_EXCEEDED)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            code = match.group(1).upper()
            if re.match(r"^\d{4}$", code) and re.search(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}", value):
                continue
            return code
    return ""


def _code(text: str) -> str:
    return extract_error_code(text)


@lru_cache(maxsize=2)
def _embeddings_for_model(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=2)
def _store_for_model(model_name: str) -> Chroma:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = _embeddings_for_model(model_name)
    return Chroma(collection_name="jira_history_v1", embedding_function=embeddings,
                  persist_directory=str(VECTOR_DIR),
                  collection_metadata={"hnsw:space": "cosine"})


def _store() -> Chroma:
    return _store_for_model(config_value("EMBEDDING_MODEL"))


def _embedding_window() -> EmbeddingWindow:
    client = _embeddings_for_model(config_value("EMBEDDING_MODEL"))._client
    return EmbeddingWindow(client.tokenizer, client.max_seq_length)


class HistoryIndex:
    def reset(self) -> None:
        """Drop the current vector collection before an explicit full rebuild."""
        store = _store()
        store.delete_collection()
        _store_for_model.cache_clear()

    def upsert(self, item: dict) -> None:
        self.upsert_many([item])

    def upsert_many(self, items: list[dict]) -> None:
        if not items:
            return
        store, window = _store(), _embedding_window()
        keys = list(dict.fromkeys(item["source_key"] for item in items))
        # Source identity is stable even when its vector passage count changes.
        # Delete by metadata so retries/re-indexes cannot leave old tail vectors.
        store.delete(where={"source_key": {"$in": keys}})
        documents, identifiers = [], []
        for item in items:
            key = item["source_key"]
            context = {**item, "error_code": extract_error_code(item.get("error", ""))}
            metadata = {"source_key": key, "embedding_version": EMBEDDING_INDEX_VERSION}
            documents.append(Document(page_content=window.failure_text(context), metadata=metadata))
            identifiers.append(key)
            body = item.get("workaround_text") or item.get("body") or ""
            prefix = window.failure_text(context, max(24, window.limit // 3))
            for number, passage in enumerate(window.passages(body, prefix), 1):
                documents.append(Document(page_content=passage, metadata=metadata))
                identifiers.append(f"{key}::passage:{number}")
            # Bound embedding batches rather than holding a corpus of vectors.
            while len(documents) >= 64:
                store.add_documents(documents[:64], ids=identifiers[:64])
                del documents[:64]
                del identifiers[:64]
        if documents:
            store.add_documents(documents, ids=identifiers)

    def delete(self, key: str) -> None:
        self.delete_many([key])

    def delete_many(self, keys: list[str]) -> None:
        if keys:
            _store().delete(where={"source_key": {"$in": keys}})

    def search(self, query_text: str, top_k: int, min_score: float,
               exclude_keys: set[str]) -> dict:
        step, error = parse_failure(query_text)
        if not storage.usable_document_count():
            return {"candidates": [], "strong": [], "context": [], "best_score": 0.0,
                    "query_step": step, "query_error": error, "threshold": min_score}
        summary_match = re.search(r"(?im)^\s*Summary\s*:\s*(.+)$", query_text)
        summary = summary_match.group(1).strip()[:300] if summary_match else ""
        order_reason = field(query_text, "Order Reason", "OrderReason", "Reason")
        vector_scores: dict[str, float] = {}
        dense_keys: list[str] = []
        try:
            query = _embedding_window().failure_text({
                "error_code": extract_error_code(error), "error": error, "step": step,
                "order_reason": order_reason, "summary": summary})
            for doc, distance in _store().similarity_search_with_score(
                    query, k=min(400, max(16, top_k * 8))):
                key = doc.metadata.get("source_key", "")
                if key:
                    if key not in vector_scores:
                        dense_keys.append(key)
                    vector_scores[key] = max(vector_scores.get(key, 0.0),
                                             max(0.0, min(1.0, 1.0 - float(distance))))
        except Exception as exc:
            # An empty index is valid before the first ingest; other errors must surface.
            if storage.all_document_keys():
                raise RuntimeError("Historical vector search failed") from exc
        lexical_keys = storage.lexical_keys(f"{step} {error}" if step or error else summary,
                                            min(100, top_k * 4), error=error)
        # Reciprocal rank fusion combines ranks without assuming vector and text
        # search scores share a scale: score(d) = sum(1 / (60 + rank(d))).
        rrf_scores: dict[str, float] = {}
        for result_list in (dense_keys, lexical_keys):
            for rank, key in enumerate(result_list, 1):
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)
        max_rrf = max(rrf_scores.values(), default=1.0)
        candidates = []
        for key, rrf_score in rrf_scores.items():
            if key in exclude_keys:
                continue
            row = storage.source_document(key)
            if not row or storage.source_tier(row["source_type"], key) == "low":
                continue
            current_workaround = storage.current_workaround_version(key)
            query_code, candidate_code = _code(error), _code(row["error"])
            if query_code and candidate_code and query_code != candidate_code:
                continue
            net = storage.feedback_net(key, step, error,
                                       (current_workaround or {}).get("id", ""))
            retrieval_score = min(1.0, rrf_score / max_rrf)
            candidates.append({
                "type": row["source_type"], "key": key, "summary": row["summary"],
                "url": (storage.settings()["public_base_url"].rstrip("/") + row["url"]
                        if row["source_type"] == "doc" and row["url"].startswith("/") else row["url"]),
                "comment": row["body"], "step": row["step"],
                "filename": (row.get("metadata") or {}).get("filename", ""),
                "kind": (row.get("metadata") or {}).get("kind", ""),
                "chunk_count": (row.get("metadata") or {}).get("chunk_count", 1),
                "error": row["error"], "order_reason": (row.get("metadata") or {}).get("order_reason", ""),
                "order_type": (row.get("metadata") or {}).get("order_type", ""),
                "release": (row.get("metadata") or {}).get("release", ""),
                "resolution_at": (row.get("metadata") or {}).get("resolution_at", ""),
                "workaround_version": current_workaround,
                "score": vector_scores.get(key, 0.0),
                "retrieval_score": retrieval_score, "rank_score": retrieval_score,
                "feedback_net": net,
            })
        candidates.sort(key=lambda c: (c["rank_score"], c["score"]), reverse=True)
        candidates = candidates[:top_k]
        return {"candidates": candidates,
                "strong": [c for c in candidates if c["score"] >= min_score],
                "context": [c for c in candidates if c["score"] < min_score],
                "best_score": candidates[0]["score"] if candidates else 0.0,
                "query_step": step, "query_error": error,
                "threshold": min_score}
