"""Pure route metadata and rank fusion; searches live in the plugin adapters."""

from copy import deepcopy
from time import monotonic

import httpx
import requests

from . import execution, retrieval_pool


def failure_status(error: Exception) -> str:
    if getattr(error, "failure_status", None) in {"timeout", "unavailable"}:
        return error.failure_status
    if isinstance(error, (TimeoutError, requests.Timeout, httpx.TimeoutException)):
        return "timeout"
    if isinstance(error, (execution.StageUnavailable, retrieval_pool.RetrievalBusy,
                          requests.ConnectionError, httpx.ConnectError)):
        return "unavailable"
    return "error"


def stage(state: dict, limit: float, function, *args, reserve: float = 0):
    seconds = min(limit, state["deadline"] - monotonic() - reserve)
    return execution.call(function, seconds, *args)


def event(state: dict, name: str, status: str, reason: str, started: float,
          *, candidate_count: int | None = None, requested_top_k: int | None = None) -> list[dict]:
    counts = {key: value for key, value in {"candidate_count": candidate_count,
              "requested_top_k": requested_top_k}.items() if value is not None}
    return [*state.get("retrieval_route", []), {"stage": name, "status": status,
            "reason": reason, "elapsed_ms": round((monotonic() - started) * 1000), **counts}]


def merge(local: list[dict], remote: dict, top_k: int) -> list[dict]:
    rejected = set(remote.get("rejected_keys") or [])
    by_key = {row["key"]: deepcopy(row) for row in local if row["key"] not in rejected}
    ranks = [{row["key"]: rank for rank, row in enumerate(local, 1)}, {}]
    for rank, item in enumerate(remote.get("candidates") or [], 1):
        key = item["key"]
        if key in rejected:
            continue
        ranks[1][key] = item.get("rovo_rank", rank)
        old = by_key.get(key)
        updated = {**(old or {}), **deepcopy(item)}
        updated["retrieval_origins"] = list(dict.fromkeys(
            [*((old or {}).get("retrieval_origins") or []), *(item.get("retrieval_origins") or ["rovo"])]))
        if old:
            updated["score"] = old.get("score", 0.0)
        updated["relevance_verified"] = False  # New combined evidence must be assessed.
        by_key[key] = updated
    scores = {key: sum(1 / (60 + ranking[key]) for ranking in ranks if key in ranking)
              for key in by_key}
    maximum = max(scores.values(), default=1)
    for key, row in by_key.items():
        row["retrieval_score"] = scores[key] / maximum
        row["rank_score"] = row["retrieval_score"]
        if key in ranks[0]:
            row["local_rank"] = ranks[0][key]
    return sorted(by_key.values(), key=lambda row: row["retrieval_score"], reverse=True)[:top_k]
