"""Conservative, explainable source eligibility and evidence checks."""

from __future__ import annotations

import math
import re

from .history import extract_error_code
from .redact import scrub


# Canonical forms permit tense changes, not changes of operation or polarity.
_VERBS = {
    "retry": "retry retried retrying", "retrigger": "retrigger retriggered re-trigger re-triggered",
    "create": "create created creating", "update": "update updated updating",
    "change": "change changed changing", "remove": "remove removed removing",
    "restore": "restore restored restoring", "cancel": "cancel cancelled canceled cancelling canceling",
    "provision": "provision provisioned provisioning", "activate": "activate activated activating",
    "deactivate": "deactivate deactivated deactivating", "suspend": "suspend suspended suspending",
    "resume": "resume resumed resuming", "sync": "sync synced syncing",
    "correct": "correct corrected correcting", "fix": "fix fixed fixing",
    "set": "set setting", "reset": "reset resetting", "verify": "verify verified verifying",
    "confirm": "confirm confirmed confirming", "check": "check checked checking",
    "ensure": "ensure ensured ensuring", "mark": "mark marked marking",
    "select": "select selected selecting", "delete": "delete deleted deleting",
    "configure": "configure configured configuring", "enter": "enter entered entering",
    "submit": "submit submitted submitting", "restart": "restart restarted restarting",
}
_CANONICAL = {form: verb for verb, forms in _VERBS.items() for form in forms.split()}
_ACTION = re.compile(r"\b(?:" + "|".join(map(re.escape, _CANONICAL)) + r")\b", re.I)
_PLACEHOLDER = re.compile(
    r"(?is)^(?:n/?a|none|done|fixed|resolved|closed|ok|completed?|"
    r"(?:completed?|retr(?:y|ied))\s+(?:the\s+)?(?:failed\s+)?step)[.! ]*$")
_ADMIN = re.compile(r"(?i)\b(?:new|another|support)\s+(?:ticket|case)\b|"
                    r"\b(?:ticket closure|automatically closed|no response)\b")
_NOISE = {"a", "an", "the", "please", "we", "have", "has", "had", "was", "were", "been"}
_LEXEMES = re.compile(r"!=|<>|<=|>=|==|[=<>!+*/&|]|(?<!\w)[+-]?\d+(?:\.\d+)*(?!\w)|"
                      r"[a-z0-9_]+(?:[./:-][a-z0-9_]+)*", re.I)
BOT_COMMENT_MARKERS = ("FA-LANGCHAIN-8012", "FA-LEGACY-8010", "FA-SUGGESTION")
SOURCE_POLICY_VERSION = "resolution-evidence-v3"


def is_automatic_closure(body: str, author_name: str = "", account_type: str = "") -> bool:
    return bool(re.search(r"(?is)(?:resolved for more than\s+\d+\s+days?\s+with no response|"
                          r"will now be automatically closed|automatically closed due to inactivity)", body or ""))


def is_salesforce_update_comment(body: str) -> bool:
    value = (body or "").casefold()
    return ("updated by the lcpr order fallout tickets bot" in value or
            ("update from salesforce" in value and any(marker in value for marker in (
                "orderstepid", "order step link", "pr_error_description__c",
                "lastmodifieddate", "execution log"))))


def _resolution_field(body: str, *names: str) -> tuple[bool, str]:
    lines = (body or "").splitlines()
    for name in names:
        pattern = re.compile(r"^\s*(?:\|{1,2}\s*|\*{0,2}\s*)" + re.escape(name)
                             + r"\s*\*{0,2}\s*(?:\|{1,2}|:)\s*(.*)$", re.I)
        for index, line in enumerate(lines):
            match = pattern.match(line)
            if not match:
                continue
            parts = [match.group(1).rstrip(" |")]
            for following in lines[index + 1:]:
                if re.match(r"^\s*(?:\||\*{0,2}[\w /]{2,40}\*{0,2}\s*:)", following):
                    break
                parts.append(following.rstrip(" |"))
            return True, "\n".join(parts).strip()
    return False, ""


def resolution_text(body: str) -> str:
    """Assess the solution field, not action-like words elsewhere in a table."""
    found, value = _resolution_field(body, "Solution Applied", "Resolution", "Workaround", "Fix")
    if found:
        return value
    match = re.search(r"(?is)===\s*fix\s*===\s*(.*?)(?:===\s*end\s*===|$)", body or "")
    if match:
        return match.group(1).strip()
    return (body or "").strip()


def resolution_is_placeholder(value: str) -> bool:
    lines = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
             for line in (value or "").splitlines() if line.strip()]
    return not lines or all(_PLACEHOLDER.fullmatch(line) for line in lines)


def resolution_evidence(body: str) -> str:
    solution = resolution_text(body)
    _, customer_action = _resolution_field(body, "Customer Action")
    return "\n".join(part for part in (solution, customer_action) if part)


def _tokens(value: str) -> list[str]:
    value = re.sub(r"(?i)\b(can't|cannot)\b", "can not", value or "")
    value = re.sub(r"(?i)n't\b", " not", value)
    return [_CANONICAL.get(word.casefold(), word) for word in _LEXEMES.findall(value)
            if word.casefold() not in _NOISE]


def _clauses(value: str) -> list[str]:
    value = re.sub(r"(?m)^\s*\d+[.)]\s*", "", value or "")
    return [part.strip() for part in re.split(
        r"[;\n]+|(?<=[.!?])\s+|\s+(?:and\s+)?then\s+|\s+and\s+(?=" + _ACTION.pattern + ")",
        value, flags=re.I) if part.strip()]


def source_quality(source_type: str, meta: dict, body: str) -> tuple[str, list[str]]:
    """An automatic evidence assessment; never human approval or fix probability."""
    solution = resolution_text(body)
    reasons = []
    if (is_automatic_closure(body) or is_salesforce_update_comment(body)
            or any(marker in (body or "") for marker in BOT_COMMENT_MARKERS)):
        return "low", ["Administrative or generated activity is not resolution evidence"]
    if not (meta.get("step") or "").strip():
        reasons.append("Missing failed step")
    if not (meta.get("error") or "").strip():
        reasons.append("Missing error details")
    if resolution_is_placeholder(solution) or len(solution) < 20:
        return "low", [*reasons, "Resolution is empty or too vague"]
    if _ADMIN.search(solution) or not _ACTION.search(solution):
        return "low", [*reasons, "No clear corrective action in resolution"]
    clauses = _clauses(solution)
    # Require a named action with a substantive object; numbering adds no quality.
    diagnostic = {"verify", "confirm", "check", "ensure", "mark", "fix"}
    corrective = any(len(_tokens(clause)) >= 4 and any(
        _CANONICAL[match.group().casefold()] not in diagnostic for match in _ACTION.finditer(clause))
        for clause in clauses)
    clear = corrective and all(_ACTION.search(clause) and not clause.rstrip().endswith("?")
                               for clause in clauses)
    if source_type == "doc" and meta.get("chunk_count", 1) > 1:
        return "medium", [*reasons, "Passage belongs to a longer document; verify the complete procedure"]
    if not reasons and clear:
        return "high", ["Automatically assessed: failure context and specific corrective actions present"]
    return "medium", [*reasons, "Resolution requires clarification or additional failure context"]


def clean_excerpt(value: str) -> str:
    return " ".join((value or "").split())


def evidence_quote_is_present(quote: str, body: str) -> bool:
    quote_clean = clean_excerpt(quote)
    return len(quote_clean) >= 12 and quote_clean in clean_excerpt(body)


def step_is_supported(text: str, quote: str) -> bool:
    """Allow faithful action clauses and tense changes; ambiguous rewrites need review.

    Token overlap cannot prove support. Retain objects, values, negation and
    conditions, and do not extract an operation name from a retry instruction.
    """
    if not text or len(text) > 800 or not quote or len(quote) > 12000:
        return False
    proposed = _clauses(text)
    quoted = _clauses(quote)
    if not proposed or not _ACTION.search(text):
        return False
    # Splitting must not discard a condition/negation shared by several clauses.
    qualifiers = r"\b(?:not|no|never|without|before|after|if|unless|until|only|except)\b"
    if set(re.findall(qualifiers, " ".join(_tokens(quote)).casefold())) != set(
            re.findall(qualifiers, " ".join(_tokens(text)).casefold())):
        return False
    positions = []
    for action in proposed:
        matches = [index for index, evidence in enumerate(quoted) if _tokens(action) == _tokens(evidence)]
        if not matches:
            return False
        positions.append(matches[0])
    return positions == sorted(positions)


def _failure_text(value: str) -> str:
    # Only recognized instance identifiers and timestamps are interchangeable.
    value = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
                   r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b", " timestamp ", value or "")
    value = re.sub(r"\b[A-Fa-f0-9]{8}-(?:[A-Fa-f0-9]{4}-){3}[A-Fa-f0-9]{12}\b", " instanceid ", value)
    value = re.sub(r"(?i)\b(orderid|requestid|correlationid|executionid|externalid|"
                   r"accountid|enrollmentid|msisdn|imei|iccid)\s*[:=]\s*[\w-]+",
                   r"\1=instanceid", value)
    # Preserve word order, negation, technical values and unlabelled numbers.
    return " ".join(_LEXEMES.findall(value.casefold()))


def same_failure(query_step: str, query_error: str, source: dict,
                 query_context: dict | None = None) -> bool:
    if not all((query_step, query_error, source.get("step"), source.get("error"))):
        return False
    if _failure_text(query_step) != _failure_text(source["step"]):
        return False
    query_code, source_code = extract_error_code(query_error), extract_error_code(source["error"])
    if (query_code or source_code) and query_code != source_code:
        return False
    # A shared HTTP/error code alone never proves the message has the same meaning.
    if _failure_text(query_error) != _failure_text(source["error"]):
        return False
    context = query_context or {}
    for name in ("order_type", "order_reason", "release"):
        actual, historical = context.get(name) or "", source.get(name) or ""
        if name == "order_type":
            # Explicit aliases only; preserve operation/subtype distinctions.
            aliases = {f"{kind} order": kind for kind in ("change", "sales", "add", "disconnect")}
            actual = aliases.get(actual.strip().casefold(), actual)
            historical = aliases.get(historical.strip().casefold(), historical)
        if historical and _failure_text(actual) != _failure_text(historical):
            return False
    return True


def routing_source_quality(candidate: dict, step: str, error: str, context: dict | None,
                           min_relevance: float) -> tuple[str, list[str]]:
    """Source-level routing proof, independent of normalized retrieval scores."""
    from urllib.parse import urlsplit
    key = candidate.get("key") or ""
    url = urlsplit(candidate.get("url") or "")
    if not key or url.scheme not in {"http", "https"} or not url.netloc:
        return "low", ["Source identity or URL is missing"]
    if not same_failure(step, error, candidate, context):
        return "medium", ["Failure or applicability context does not match"]
    score = candidate.get("llm_score")
    if (candidate.get("relevance_verified") is not True or isinstance(score, bool)
            or not isinstance(score, (float, int)) or not math.isfinite(score)
            or not min_relevance <= score <= 1):
        return "low", ["A successful real reranker assessment above the routing threshold is missing"]
    tier, reasons = source_quality(candidate.get("type", "ticket"), candidate,
                                   candidate.get("comment", ""))
    if tier != "high":
        return tier, reasons
    if (candidate.get("feedback_net", 0) < 0
            or candidate.get("automated_failures", 0) > candidate.get("automated_successes", 0)):
        return "medium", ["Feedback or verified outcomes do not support this procedure"]
    return "high", ["Applicable actionable source with verified relevance and traceable evidence"]


def procedure_identity(candidate: dict) -> tuple:
    """Preserve action order, technical values and qualifiers when checking agreement."""
    return tuple(tuple(_tokens(clause)) for clause in _clauses(
        resolution_evidence(candidate.get("comment", ""))))


def assess_routing_evidence(candidates: list[dict], step: str, error: str,
                            context: dict, settings: dict) -> dict:
    eligible = {candidate["key"]: candidate for candidate in candidates
                if routing_source_quality(candidate, step, error, context,
                                          settings["routing_min_llm_relevance"])[0] == "high"}
    procedures = {procedure_identity(candidate) for candidate in eligible.values()}
    conflicting = len(procedures) > 1
    sufficient = len(eligible) >= settings["routing_min_sources"] and not conflicting
    return {"sufficient": sufficient, "eligible_source_keys": sorted(eligible),
            "conflicting": conflicting, "status": "sufficient" if sufficient else
            "conflicting" if conflicting else "empty" if not candidates else "weak",
            "reason": "Applicable source evidence passed the routing rule" if sufficient else
            "Applicable historical procedures disagree" if conflicting else
            "No usable matches" if not candidates else "Evidence does not meet the routing rule"}


def candidate_quality(candidate: dict, min_score: float, source_tiers: dict[str, str],
                      query_step: str, query_error: str,
                      query_context: dict | None = None,
                      routing_policy: dict | None = None) -> tuple[str, list[str]]:
    if routing_policy is not None:
        return routing_source_quality(candidate, query_step, query_error, query_context,
                                      routing_policy["min_relevance"])
    if not same_failure(query_step, query_error, candidate, query_context):
        return "medium", ["Historical source does not match the failed step, error and applicability context"]
    score = candidate.get("score")
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not min_score <= score <= 1):
        return "low", ["Best raw historical match is below the posting threshold"]
    key = candidate.get("key") or candidate.get("filename") or ""
    # Reassess actual evidence so stale source tiers cannot authorize a fix.
    assessed, _ = source_quality(candidate.get("type", "ticket"), candidate,
                                 candidate.get("comment", ""))
    tier = source_tiers.get(key, "low")
    if "low" in (tier, assessed):
        return "low", ["Historical resolution has low source quality"]
    if tier != "high" or assessed != "high":
        return "medium", ["Historical resolution needs clearer corrective evidence"]
    if int(candidate.get("feedback_net") or 0) < 0:
        return "medium", ["Source has negative feedback for this failure"]
    if int(candidate.get("automated_failures", 0)) > int(candidate.get("automated_successes", 0)):
        return "medium", ["Verified automated failures exceed successes for this workaround version"]
    return "high", ["Applicable source with automatically assessed corrective evidence"]


def prioritize_candidates(candidates: list[dict], min_score: float, source_tiers: dict[str, str],
                          query_step: str, query_error: str,
                          query_context: dict | None = None,
                          routing_policy: dict | None = None) -> list[dict]:
    """Keep rank order within eligibility groups, before the drafting shortlist is cut."""
    def priority(candidate: dict) -> tuple[bool, bool, bool]:
        tier, reasons = candidate_quality(candidate, min_score, source_tiers,
                                         query_step, query_error, query_context, routing_policy)
        candidate["eligibility_tier"] = tier
        candidate["eligibility_reasons"] = reasons
        return (tier == "high", same_failure(query_step, query_error, candidate, query_context),
                tier != "low")
    return sorted(candidates, key=priority, reverse=True)


def preserve_supporting_sources(original: dict, revised: dict) -> dict:
    """A supervisor rewrite cannot silently drop evidence for a retained action."""
    for step in revised.get("steps") or []:
        supports = list(step.get("supporting_sources") or [])
        for previous in original.get("steps") or []:
            if (step.get("source_key") == previous.get("source_key")
                    and _tokens(step.get("text", "")) == _tokens(previous.get("text", ""))):
                for citation in previous.get("supporting_sources") or []:
                    if citation not in supports:
                        supports.append(citation)
        step["supporting_sources"] = supports
    return revised


def draft_support_reasons(candidates: list[dict], draft: dict) -> list[str]:
    """Validate action fidelity independently of a source's confidence tier."""
    allowed = {(c.get("key") or c.get("filename") or ""): c for c in candidates}
    last_position: dict[str, int] = {}
    used_positions: dict[str, set[int]] = {}
    for step in draft.get("steps", []):
        step_positions: dict[str, set[int]] = {}
        citations = [step, *step.get("supporting_sources", [])]
        for citation in citations:
            key = citation.get("source_key", "")
            source = allowed.get(key)
            if source is None:
                return ["A drafted step cites a source outside the selected candidates"]
            evidence = scrub(resolution_evidence(source.get("comment", "")))
            quote, action = citation.get("source_excerpt", ""), step.get("text", "")
            if (len(quote) > 500 or not evidence_quote_is_present(quote, evidence)
                    or not step_is_supported(action, quote)
                    or not step_is_supported(action, evidence)):
                return ["A drafted action changes or omits source details, negation or conditions",
                        f"Cited source: {key}"]
            positions = [index for index, clause in enumerate(_clauses(evidence))
                         if any(_tokens(clause) == _tokens(part) for part in _clauses(action))]
            step_positions.setdefault(key, set()).update(positions)
        for key, positions in step_positions.items():
            if positions and min(positions) < last_position.get(key, -1):
                return ["Draft reorders the documented procedure", f"Cited source: {key}"]
            if positions:
                last_position[key] = max(positions)
                used_positions.setdefault(key, set()).update(positions)
    for key, positions in used_positions.items():
        if positions != set(range(max(positions) + 1)):
            return ["Draft omits a prerequisite action from the cited procedure", f"Cited source: {key}"]
    return []


def proposal_quality(rule: dict | None, candidates: list[dict], draft: dict | None,
                     min_score: float, source_tiers: dict[str, str],
                     query_step: str = "", query_error: str = "",
                     query_context: dict | None = None,
                     routing_policy: dict | None = None) -> tuple[str, list[str]]:
    """Check every cited source. No unrelated top candidate can authorize a step."""
    if rule:
        answer = (rule.get("workaround_text") or "").strip()
        steps = (draft or {}).get("steps") or []
        if (answer and not (draft or {}).get("decline") and len(steps) == 1
                and steps[0].get("source_key") == f"RULE:{rule.get('id', '')}"
                and steps[0].get("text", "").strip() == answer):
            return "high", ["Approved rule matched the failure; action preserved exactly"]
        return "medium", ["Draft does not preserve the approved rule action"]
    if not candidates:
        return "low", ["No historical candidates"]
    ranked = prioritize_candidates(candidates, min_score, source_tiers, query_step, query_error,
                                   query_context, routing_policy)
    if not draft or not draft.get("steps") or draft.get("decline"):
        tier, reasons = candidate_quality(ranked[0], min_score, source_tiers,
                                         query_step, query_error, query_context, routing_policy)
        return ("medium" if tier == "high" else tier), [*reasons, "No approved evidence-grounded fix steps"]
    if len(draft["steps"]) > 5:
        return "medium", ["Draft contains too many steps for automatic posting"]
    support_reasons = draft_support_reasons(candidates, draft)
    if support_reasons:
        return "medium", support_reasons
    allowed = {(c.get("key") or c.get("filename") or ""): c for c in candidates}
    for step in draft["steps"]:
        citations = [{"source_key": step.get("source_key", ""),
                      "source_excerpt": step.get("source_excerpt", "")},
                     *step.get("supporting_sources", [])]
        for citation in citations:
            key = citation.get("source_key", "")
            source = allowed.get(key)
            if source is None:
                return "medium", ["A drafted step cites a source outside the selected candidates"]
            tier, reasons = candidate_quality(source, min_score, source_tiers,
                                             query_step, query_error, query_context, routing_policy)
            if tier != "high":
                return tier, [*reasons, f"Cited source: {key}"]
    return "high", ["Every action has applicable historical evidence and a faithful supporting excerpt"]
