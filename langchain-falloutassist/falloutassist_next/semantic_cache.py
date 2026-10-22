"""Bounded, persistent reuse of ranking scores; never caches an investigation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass

from . import storage
from .config import config_value

logger = logging.getLogger(__name__)
CACHE_VERSION = "rerank-semantic-v1"


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _guards(query: str) -> tuple[dict, bool]:
    from .history import field, parse_failure
    step, error = parse_failure(query)
    context = {
        "step": step, "error": error,
        "order_type": field(query, "Order Type", "OrderType"),
        "order_reason": field(query, "Order Reason", "OrderReason", "Reason"),
        "release": field(query, "Release", "Release Version", "Software Release"),
    }
    # Keep exact failure wording, codes, identifiers, counts and operation/negation
    # tokens even if their embeddings are almost identical. Unknown order context
    # permits exact reuse only. No heuristic ID stripping is used here.
    context["numbers"] = sorted(set(re.findall(r"\b[\w./:-]*\d[\w./:-]*\b", query)))
    context["operators"] = sorted(set(re.findall(
        r"\b(?:no|not|never|without|cannot|can't|isn't|isnt|disabled|enabled|enable|disable|"
        r"create|delete|activate|deactivate|suspend|resume|cancel|add|remove|prepaid|postpaid|"
        r"disconnect|connect|update|increase|decrease)\b", query.casefold())))
    eligible = all(context[key] for key in ("step", "error", "order_type", "order_reason"))
    return context, eligible


def embed_query(text: str, model_name: str) -> list[float]:
    from .history import _embeddings_for_model
    vector = [float(value) for value in _embeddings_for_model(model_name).embed_query(text)]
    if not vector or not all(math.isfinite(value) for value in vector) or not any(vector):
        raise ValueError("Invalid cache embedding")
    return vector


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right) or not all(math.isfinite(x) for x in right):
        return -1.0
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(a*b for a, b in zip(left, right)) / denominator if denominator else -1.0


def _scores(raw: str, expected: set[str]) -> dict | None:
    try:
        scores = json.loads(raw)
        if not isinstance(scores, dict) or set(scores) != expected:
            return None
        for value in scores.values():
            score = value["llm_score"]
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1
                    or not isinstance(value["llm_reason"], str)):
                return None
        return scores
    except (ValueError, KeyError, TypeError):
        return None


@dataclass
class CacheRequest:
    scope: str
    query_hash: str
    vector: list[float]
    settings: dict
    backend: object | None = None

    def save(self, scores: dict, trace_id: str | None, info: dict) -> None:
        try:
            item = {"id": _hash([self.scope, self.query_hash]),
                "scope": self.scope, "query_hash": self.query_hash,
                "vector": self.vector, "result": scores, "trace_id": trace_id}
            writer = self.backend.save if self.backend is not None else storage.put_semantic_cache
            writer(item, self.settings["semantic_cache_ttl_minutes"],
                self.settings["semantic_cache_max_entries"])
        except Exception as exc:
            info["write_error"] = type(exc).__name__
            logger.warning("Ranking cache write failed (%s)", type(exc).__name__)


def lookup(query: str, embedding_text: str, candidates: list[dict], temperature: float,
           prompt_version: str, settings: dict, info: dict
           ) -> tuple[CacheRequest | None, dict | None, str | None]:
    """Return a validated hit or a save handle. Cache failures leave model calls available."""
    info.update(status="disabled", backend=settings["semantic_cache_backend"])
    if not settings["semantic_cache_enabled"]:
        return None, None, None
    try:
        guards, eligible = _guards(query)
        model_name = config_value("EMBEDDING_MODEL")
        connection = {key: config_value(key) for key in (
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_API_VERSION")}
        # Hash complete evidence, not truncated excerpts, so changed or withdrawn
        # procedures cannot retain old scores. Retrieval ranks and feedback are
        # deliberately excluded: both are recalculated by the caller on every run.
        evidence = [{key: item.get(key) for key in (
            "key", "filename", "summary", "step", "error", "order_type", "order_reason",
            "release", "comment", "resolution_at", "workaround_version", "source_tier")}
            for item in candidates]
        evidence.sort(key=lambda item: item.get("key") or item.get("filename") or "")
        scope = _hash([CACHE_VERSION, prompt_version, connection, model_name,
                       storage.get_state("connection_version"),
                       storage.get_state("semantic_cache_generation"), temperature, evidence, guards])
        request = CacheRequest(scope, _hash(query), [], settings)
        expected = {item.get("key") or item.get("filename") or "" for item in candidates}
        if settings["semantic_cache_backend"] == "redis":
            from .redis_cache import lookup as redis_lookup
            return redis_lookup(request, embedding_text, model_name, expected, eligible, info)
        rows = storage.semantic_cache_entries(scope, settings["semantic_cache_ttl_minutes"],
                                               settings["semantic_cache_max_entries"])
        valid = [(row, _scores(row["result_json"], expected)) for row in rows]
        for row, scores in valid:
            if scores is not None and row["query_hash"] == request.query_hash:
                info.update(status="exact_hit", similarity=1.0, original_trace_id=row["trace_id"])
                return request, scores, None
        info.update(status="miss", semantic_eligible=eligible)
        if eligible:
            try:
                request.vector = embed_query(embedding_text, model_name)
                best = None
                for row, scores in valid:
                    if scores is None:
                        continue
                    try:
                        similarity = _cosine(request.vector, json.loads(row["vector_json"]))
                    except (ValueError, TypeError):
                        continue
                    if similarity >= settings["semantic_cache_threshold"] and (
                            best is None or similarity > best[0]):
                        best = (similarity, row, scores)
                if best:
                    info.update(status="semantic_hit", similarity=round(best[0], 6),
                                original_trace_id=best[1]["trace_id"])
                    return request, best[2], None
            except Exception as exc:
                info["embedding_error"] = type(exc).__name__
                logger.warning("Ranking cache embedding failed (%s)", type(exc).__name__)
        return request, None, None
    except Exception as exc:
        info.update(status="unavailable", error_type=type(exc).__name__)
        logger.warning("Ranking cache lookup failed (%s)", type(exc).__name__)
        return None, None, None
