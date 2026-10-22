"""RedisVL-backed ranking cache with exact evidence partitions and bounded retention."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from functools import lru_cache
from urllib.parse import urlsplit

from .config import config_value

# All application keys share the namespace's Redis Cluster hash tag. Only cache
# keys registered by this application can be evicted; no database-wide flush.
TRIM_SCRIPT = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', tonumber(ARGV[1])-86400)
redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2])
local overflow = redis.call('ZCARD', KEYS[1])-tonumber(ARGV[3])
if overflow > 0 then
    local victims = redis.call('ZRANGE', KEYS[1], 0, overflow-1)
    for _, key in ipairs(victims) do
        redis.call('UNLINK', key)
        redis.call('ZREM', KEYS[1], key)
    end
end
redis.call('EXPIRE', KEYS[1], 86400)
return overflow
"""


def _connection() -> tuple[str, str]:
    url = config_value("SEMANTIC_CACHE_REDIS_URL")
    namespace = config_value("SEMANTIC_CACHE_NAMESPACE")
    parsed = urlsplit(url)
    if (parsed.scheme not in {"redis", "rediss"} or not parsed.hostname
            or parsed.query or parsed.fragment):
        raise ValueError("Redis cache connection is not configured correctly")
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,59}", namespace):
        raise ValueError("Invalid Redis cache namespace")
    return url, namespace


@lru_cache(maxsize=4)
def _client(url: str):
    from redis import Redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry
    return Redis.from_url(url, socket_connect_timeout=2, socket_timeout=3,
                          retry=Retry(NoBackoff(), 0))


@lru_cache(maxsize=4)
def _cache(url: str, namespace: str, model_name: str):
    from redisvl.extensions.cache.llm import SemanticCache
    from redisvl.utils.vectorize import CustomVectorizer
    from .semantic_cache import embed_query
    client = _client(url)
    client.ping()  # Fail before loading an embedding model if Redis is down.
    vectorizer = CustomVectorizer(embed=lambda text: embed_query(text, model_name))
    model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    cache = SemanticCache(
        name=f"fa_rerank_{{{namespace}}}_{model_hash}_v1", redis_client=client,
        vectorizer=vectorizer, ttl=None, overwrite=False,
        filterable_fields=[{"name": "scope", "type": "tag"},
                           {"name": "query_hash", "type": "tag"},
                           {"name": "mode", "type": "tag"}])
    return RedisRankingCache(cache, client, namespace, vectorizer.dims)


class RedisRankingCache:
    def __init__(self, cache, client, namespace: str, dims: int):
        self.cache, self.client, self.namespace, self.dims = cache, client, namespace, dims

    def partition(self, scope: str, ttl_minutes: int):
        from redisvl.query.filter import Num, Tag
        return ((Tag("scope") == scope) &
                (Num("inserted_at") >= time.time() - ttl_minutes * 60))

    def exact(self, scope: str, query_hash: str, ttl_minutes: int) -> list[dict]:
        from redisvl.query import FilterQuery
        from redisvl.query.filter import Tag
        return self.cache.index.query(FilterQuery(
            filter_expression=self.partition(scope, ttl_minutes) & (Tag("query_hash") == query_hash),
            return_fields=["response", "metadata"], num_results=2))

    def similar(self, scope: str, vector: list[float], settings: dict) -> list[dict]:
        from redisvl.query.filter import Tag
        return self.cache.check(
            vector=vector, num_results=3,
            filter_expression=self.partition(scope, settings["semantic_cache_ttl_minutes"]) &
                              (Tag("mode") == "semantic"),
            # RedisVL uses cosine distance, whereas the UI uses similarity.
            distance_threshold=1.0 - settings["semantic_cache_threshold"])

    def save(self, item: dict, ttl_minutes: int, max_entries: int) -> None:
        vector = item["vector"]
        key = self.cache.store(
            prompt=item["query_hash"],  # Never store the original query text.
            response=json.dumps(item["result"], allow_nan=False),
            vector=vector or [1.0] + [0.0] * (self.dims - 1),
            filters={"scope": item["scope"], "query_hash": item["query_hash"],
                     "mode": "semantic" if vector else "exact"},
            metadata={"scope": item["scope"], "created_at": time.time(),
                      "trace_id": item.get("trace_id")}, ttl=ttl_minutes * 60)
        try:
            self.client.eval(TRIM_SCRIPT, 1, f"fa_rerank_{{{self.namespace}}}_entries",
                             time.time(), key, max_entries)
        except Exception:
            # A failed capacity update must not leave an untracked new entry.
            self.client.unlink(key)
            raise


def _valid_hit(row: dict, request, expected: set[str]) -> tuple[dict, dict] | None:
    from .semantic_cache import _scores
    try:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        created = float(metadata["created_at"])
        now = time.time()
        if (metadata.get("scope") != request.scope or not math.isfinite(created)
                or not now - request.settings["semantic_cache_ttl_minutes"] * 60 < created <= now):
            return None
        scores = _scores(row["response"], expected)
        if scores is not None:
            return scores, metadata
    except (KeyError, ValueError, TypeError):
        pass
    return None


def lookup(request, embedding_text: str, model_name: str, expected: set[str],
           eligible: bool, info: dict):
    from .semantic_cache import embed_query
    url, namespace = _connection()
    request.backend = backend = _cache(url, namespace, model_name)
    for row in backend.exact(request.scope, request.query_hash,
                             request.settings["semantic_cache_ttl_minutes"]):
        hit = _valid_hit(row, request, expected)
        if hit:
            info.update(status="exact_hit", similarity=1.0, original_trace_id=hit[1].get("trace_id"))
            return request, hit[0], None
    info.update(status="miss", semantic_eligible=eligible)
    if eligible:
        try:
            request.vector = embed_query(embedding_text, model_name)
        except Exception as exc:
            info["embedding_error"] = type(exc).__name__
            return request, None, None
        for row in backend.similar(request.scope, request.vector, request.settings):
            hit = _valid_hit(row, request, expected)
            distance = float(row.get("vector_distance", float("nan")))
            if hit and math.isfinite(distance) and distance <= 1.0 - request.settings["semantic_cache_threshold"]:
                info.update(status="semantic_hit", similarity=round(1.0 - distance, 6),
                            original_trace_id=hit[1].get("trace_id"))
                return request, hit[0], None
    return request, None, None


def clear_remote() -> bool:
    """Remove only this application's namespace; leave indexes available for other workers."""
    url, namespace = _connection()
    client = _client(url)
    keys = []
    for key in client.scan_iter(match=f"fa_rerank_{{{namespace}}}_*:*", count=200):
        keys.append(key)
        if len(keys) == 200:
            client.unlink(*keys)
            keys.clear()
    if keys:
        client.unlink(*keys)
    client.unlink(f"fa_rerank_{{{namespace}}}_entries")
    return True
