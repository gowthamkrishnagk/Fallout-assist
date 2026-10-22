"""Read-only retrieval evaluation against independently verified source labels."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter


def load_cases(path: Path) -> list[dict]:
    cases, identifiers = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        identifier = case.get("id")
        expected = case.get("expected_source_keys")
        if (not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers
                or case.get("verified") is not True
                or not isinstance(expected, list) or not expected
                or any(not isinstance(key, str) or not key.strip() for key in expected)
                or not (case.get("query_text") or case.get("issue_key"))):
            raise ValueError(f"Invalid or unverified evaluation case at line {line_number}")
        excluded = set(case.get("exclude_keys") or [])
        if case.get("issue_key"):
            excluded.add(case["issue_key"])
        if excluded.intersection(expected):
            raise ValueError(f"Evaluation case {identifier} uses its own excluded ticket as a label")
        identifiers.add(identifier)
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation cases are empty")
    return cases


def evaluate(cases: list[dict], search, top_k: int = 5) -> dict:
    if not cases or top_k < 1:
        raise ValueError("Provide at least one verified case and a positive top-k")
    results, latencies = [], []
    for case in cases:
        if case.get("verified") is not True or not case.get("query_text"):
            raise ValueError(f"Case {case.get('id')} needs verified labels and query text")
        excluded = set(case.get("exclude_keys") or [])
        if case.get("issue_key"):
            excluded.add(case["issue_key"])
        started = perf_counter()
        found = search(case["query_text"], top_k, 0.0, excluded)
        elapsed = (perf_counter() - started) * 1000
        # Ignore duplicate source passages and disallow a current-ticket self-hit.
        keys = list(dict.fromkeys(item["key"] for item in found["candidates"]
                                  if item["key"] not in excluded))[:top_k]
        expected = set(case["expected_source_keys"])
        hits = expected.intersection(keys)
        rank = next((rank for rank, key in enumerate(keys, 1) if key in expected), None)
        latencies.append(elapsed)
        results.append({"id": case["id"], "returned_keys": keys,
                        "expected_source_keys": sorted(expected), "hit": bool(hits),
                        "recall": len(hits) / len(expected),
                        "reciprocal_rank": 1 / rank if rank else 0,
                        "elapsed_ms": round(elapsed, 2)})
    ordered = sorted(latencies)
    percentile = lambda p: round(ordered[max(0, math.ceil(len(ordered) * p) - 1)], 2)
    return {"case_count": len(results), "top_k": top_k,
            "hit_rate_at_k": mean(int(row["hit"]) for row in results),
            "recall_at_k": mean(row["recall"] for row in results),
            "mrr_at_k": mean(row["reciprocal_rank"] for row in results),
            "latency_ms": {"p50": percentile(.5), "p95": percentile(.95)},
            "scope": "local candidate retrieval; not answer accuracy or fix success",
            "cases": results}
