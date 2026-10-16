"""Explainable source-quality and publishing decisions."""

from __future__ import annotations

import re
from .history import extract_error_code

from .redact import scrub


_ACTION = re.compile(
    r"\b(retry|retried|retrigger|re-trigger|create|created|update|updated|change|changed|"
    r"remove|removed|restore|restored|cancel|cancelled|provision|provisioned|"
    r"activate|activated|sync|synced|correct|corrected|fix|fixed|set|reset|"
    r"verify|verified|confirm|confirmed|check|checked|ensure|mark|select|delete|"
    r"configure|configured|enter|submit|complete|completed|restart|restarted)\b",
    re.IGNORECASE,
)
_BARE = re.compile(r"(?is)^\s*(done|fixed|resolved|closed|completed|ok|n/?a)[.!]?\s*$")


def source_quality(source_type: str, meta: dict, body: str) -> tuple[str, list[str]]:
    """A source-quality tier, never a probability of a future fix being correct."""
    step = (meta.get("step") or "").strip()
    error = (meta.get("error") or "").strip()
    body = (body or "").strip()
    reasons: list[str] = []
    if not step:
        reasons.append("Missing failed step")
    if not error:
        reasons.append("Missing error details")
    if not body or _BARE.fullmatch(body) or len(body) < 20:
        reasons.append("Resolution is empty or too vague")
        return "low", reasons
    fix_text = body.lower().split("=== fix ===", 1)[-1].strip()
    numbered_actions = any(
        re.match(r"\s*\d+[.)]\s+", line) and _ACTION.search(line)
        for line in body.splitlines())
    structured = (("=== fix ===" in body.lower() and len(fix_text) >= 20
                   and bool(_ACTION.search(fix_text))) or numbered_actions)
    actionable = structured or bool(_ACTION.search(body))
    if not actionable:
        reasons.append("No clear action in resolution")
        return "low", reasons
    if source_type == "doc" and meta.get("kind") == "user_fix":
        reasons.append("Approved user-submitted fix")
        return "high", reasons
    if step and error and structured:
        reasons.append("Step, error, and structured action present")
        return "high", reasons
    reasons.append("Action exists, but context or structure is incomplete")
    return "medium", reasons


def clean_excerpt(value: str) -> str:
    return " ".join((value or "").split()).casefold()


def evidence_quote_is_present(quote: str, body: str) -> bool:
    quote_clean = clean_excerpt(quote)
    return len(quote_clean) >= 12 and quote_clean in clean_excerpt(body)


def step_is_supported(text: str, quote: str) -> bool:
    if not text or len(text) > 800 or len(quote) > 500:
        return False
    stop = {"the", "and", "for", "with", "then", "this", "that", "from", "into",
            "after", "before", "should", "must", "your", "their", "have", "was"}
    words = {word for word in re.findall(r"[a-z0-9]+", text.casefold())
             if len(word) >= 3 and word not in stop}
    source_words = set(re.findall(r"[a-z0-9]+", quote.casefold()))
    return bool(words) and len(words & source_words) / len(words) >= 0.55


def same_failure(query_step: str, query_error: str, source: dict) -> bool:
    def normalize(value: str) -> str:
        value = re.sub(r"\b[A-Fa-f0-9]{8}-[A-Fa-f0-9-]{27}\b", " [id] ", value or "")
        value = re.sub(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b", " [time] ", value)
        value = re.sub(r"\b\d{7,}\b", " [id] ", value)
        return " ".join(re.findall(r"[a-z0-9_-]+", value.casefold()))
    if not query_step or not query_error or not source.get("step") or not source.get("error"):
        return False
    if normalize(query_step) != normalize(source["step"]):
        return False
    query_code = extract_error_code(query_error)
    source_code = extract_error_code(source["error"])
    if query_code or source_code:
        return bool(query_code and source_code and query_code == source_code)
    query_words = set(normalize(query_error).split())
    source_words = set(normalize(source["error"]).split())
    return bool(query_words) and len(query_words & source_words) / len(query_words) >= 0.7


def proposal_quality(rule: dict | None, candidates: list[dict], draft: dict | None,
                     min_score: float, source_tiers: dict[str, str],
                     query_step: str = "", query_error: str = "") -> tuple[str, list[str]]:
    """High permits a fix; medium/low produce a lead or no-match comment."""
    if rule:
        return "high", ["Approved rule matched the failure"]
    if not candidates:
        return "low", ["No historical candidates"]
    top = candidates[0]
    if not same_failure(query_step, query_error, top):
        return "medium", ["Historical source does not match the exact failed step and error"]
    raw_score = float(top.get("score") or 0)
    if raw_score < min_score:
        return "low", ["Best raw historical match is below the posting threshold"]
    key = (top.get("key") or top.get("filename") or "").strip()
    top_tier = source_tiers.get(key, "low")
    if top_tier == "low":
        return "low", ["Best historical resolution has low source quality"]
    if int(top.get("feedback_net") or 0) < 0:
        return "medium", ["Best source has negative feedback for this failure"]
    if not draft or not draft.get("steps"):
        return "medium", ["No evidence-grounded fix steps were drafted"]
    if len(draft["steps"]) > 5:
        return "medium", ["Draft contains too many steps for automatic posting"]
    allowed = {(c.get("key") or c.get("filename") or ""): c for c in candidates}
    for step in draft["steps"]:
        source_key = step.get("source_key", "")
        source = allowed.get(source_key)
        if (not source or not evidence_quote_is_present(step.get("source_excerpt", ""),
                                                        scrub(source.get("comment", "")))
            or not step_is_supported(step.get("text", ""), step.get("source_excerpt", ""))):
            return "medium", ["A drafted step lacks an exact supporting source excerpt"]
        if source_tiers.get(source_key, "low") != "high":
            return "medium", ["A drafted step relies on a source without a reviewed high-quality resolution"]
        for support in step.get("supporting_sources", []):
            support_source = allowed.get(support.get("source_key", ""))
            if (not support_source or not evidence_quote_is_present(
                    support.get("source_excerpt", ""), scrub(support_source.get("comment", "")))):
                return "medium", ["A cross-retriever source lacks an exact supporting excerpt"]
            if source_tiers.get(support.get("source_key", ""), "low") != "high":
                return "medium", ["An agreeing retrieval path has no reviewed high-quality resolution"]
    if top_tier != "high":
        return "medium", ["Best source needs a clearer reviewed resolution"]
    return "high", ["Strong same-failure match with structured source and cited steps"]

