"""Jira-first LangGraph investigation, quality gate, and audited publisher."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from copy import deepcopy
from time import monotonic
from typing import Annotated, TypedDict

from itsdangerous import URLSafeSerializer
from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph

from . import storage
from .agent import DraftResponse, PROMPT_VERSION, _scrub, chat_model
from .config import feedback_secret, config_value
from .plugins import JiraPlugin, registry
from .quality import (preserve_supporting_sources, prioritize_candidates,
                      proposal_quality, source_quality, assess_routing_evidence,
                      draft_support_reasons, resolution_evidence,
                      routing_source_quality, same_failure)
from .history import field, parse_failure
from . import retrieval_routing as routing
from .reranker import rerank_candidates
from .specialist_registry import specialist_registry


def _merge_specialist_findings(left: dict, right: dict) -> dict:
    return {**(left or {}), **(right or {})}


class InvestigationState(TypedDict, total=False):
    id: str
    issue_key: str
    actor: str
    publish_requested: bool
    local_only: bool
    settings: dict
    issue: dict
    history: dict
    rule: dict | None
    candidates: list[dict]
    rerank: list[dict]
    source_tiers: dict[str, str]
    draft: dict
    trace_id: str | None
    rerank_trace_id: str | None
    tier: str
    reasons: list[str]
    comment_body: str
    status: str
    comment_id: str
    selected_systems: list[str]
    specialist_findings: Annotated[dict[str, dict], _merge_specialist_findings]
    supervisor_summary: str
    branch_drafts: dict
    branch_agreement: str
    supervisor_result: dict
    deadline: float
    routing_applied: bool
    retrieval_route: list[dict]
    retrieval_phase: str
    routing_assessment: dict
    search_status: dict
    rerank_status: str
    pool_candidates: list[dict]
    excluded_sources: list[str]
    remote_history: dict
    review_required: bool
    issue_load_status: str
    allow_rovo: bool
    agent_deadline: float
    agent_proposal: dict
    agent_draft: dict
    agent_status: str
    agent_new_sources: bool


def _jira() -> JiraPlugin:
    return registry.get("jira")  # type: ignore[return-value]


def _query_context(state: InvestigationState) -> dict:
    text = state["issue"]["query_text"]
    return {"order_type": field(text, "Order Type", "OrderType", "Order Type Failed"),
            "order_reason": field(text, "Order Reason", "OrderReason", "Reason"),
            "release": field(text, "Release", "Release Version", "Software Release")}


def _route(state: InvestigationState) -> dict:
    if "jira" not in {item["name"] for item in registry.catalog()}:
        raise RuntimeError("Jira plugin is not registered")
    systems = ["jira"]
    systems.extend(item["name"] for item in specialist_registry.catalog()
                   if item["name"] != "jira" and item["configured"])
    return {"selected_systems": systems, "routing_applied": True,
            "deadline": state.get("deadline", monotonic() + state["settings"]["investigation_timeout_seconds"]),
            "retrieval_route": [], "review_required": False,
            "allow_rovo": not state.get("local_only", False) and config_value("JIRA_HISTORY_BACKEND") == "rovo"}


def _load(state: InvestigationState) -> dict:
    started = monotonic()
    try:
        _, issue = routing.stage(state, 10, _jira().issue, state["issue_key"], reserve=2)
        status, reason = "success", "Current Jira ticket loaded"
    except Exception as exc:
        status, reason = routing.failure_status(exc), "Current Jira ticket could not be loaded; review required"
        issue = {"key": state["issue_key"], "updated": "", "query_text": "", "summary": ""}
    return {"issue": issue, "issue_key": issue["key"], "issue_load_status": status,
            "retrieval_route": routing.event(state, "ticket", status, reason, started)}


def _run_specialist(state: InvestigationState, name: str) -> dict:
    specialist = specialist_registry.get(name)
    if name not in state.get("selected_systems", []):
        result = {"specialist": name, "status": "skipped", "findings": [],
                  "summary": "Not selected by the router."}
    else:
        try:
            result = routing.stage(state, 10, specialist.invoke, {"query_text": state["issue"]["query_text"],
                                        "issue": state["issue"],
                                        "temperature": state["settings"]["temperature"],
                                        "langfuse_enabled": state["settings"]["langfuse_enabled"]}, reserve=2)
        except Exception as exc:
            result = {"specialist": name, "status": "unavailable", "findings": [],
                      "summary": f"Specialist unavailable ({type(exc).__name__})."}
            storage.audit("system", "specialist_failed", state["issue_key"],
                          {"investigation_id": state["id"], "specialist": name,
                           "error_type": type(exc).__name__})
    return {"specialist_findings": {name: result}}


def _candidate_update(candidates: list[dict]) -> dict:
    candidates = deepcopy(candidates)
    source_tiers = {}
    for candidate in candidates:
        key = candidate.get("key") or candidate.get("filename") or ""
        tier, _ = source_quality(candidate.get("type", "ticket"), candidate,
                                 candidate.get("comment", ""))
        source_tiers[key] = tier
        candidate["source_tier"] = tier
    return {"candidates": candidates, "pool_candidates": candidates,
            "source_tiers": source_tiers, "rerank": []}


def _local_retrieve(state: InvestigationState) -> dict:
    started = monotonic()
    issue, settings = state["issue"], state["settings"]
    step, error = parse_failure(issue["query_text"])
    history = {"candidates": [], "query_step": step, "query_error": error, "backend": "local"}
    excluded = {issue["key"], *storage.rejected_candidates(issue["key"])}
    rule = None
    status = "unavailable"
    if state["issue_load_status"] == "success":
        rule = _jira().approved_rule(issue["query_text"])
        if rule and f"RULE:{rule['id']}" in excluded:
            rule = None
        try:
            history = routing.stage(state, settings["local_retrieval_timeout_seconds"],
                                    _jira().search_local_history, issue["query_text"],
                                    settings["retrieval_top_k"], settings["min_match_score"],
                                    excluded, reserve=2)
            status = "success"
        except Exception as exc:
            status = routing.failure_status(exc)
    history.update(local_available=status == "success", rovo_available=False)
    candidates = history["candidates"][:settings["rerank_pool_k"]]
    return {**_candidate_update(candidates), "history": history, "rule": rule,
            "excluded_sources": sorted(excluded), "retrieval_phase": "local",
            "search_status": {"local": status},
            "retrieval_route": routing.event(state, "local_search", status,
                "Local semantic and lexical search completed" if status == "success" else
                "Local retrieval had a technical failure", started,
                candidate_count=len(history["candidates"]), requested_top_k=settings["retrieval_top_k"])}


def _routing_policy(state: InvestigationState) -> dict | None:
    return {"min_relevance": state["settings"]["routing_min_llm_relevance"]} if state.get("routing_applied") else None


def _after_local_search(state: InvestigationState) -> str:
    # Default: retrieve top K from both sources, then pay for one combined
    # relevance assessment. Local-first remains an explicit configurable mode.
    if (state["allow_rovo"] and state["issue_load_status"] == "success"
            and state["settings"].get("rovo_mcp_always_search", True) and not state.get("rule")):
        return "mcp"
    return "rerank"


def _assess(state: InvestigationState) -> dict:
    started = monotonic()
    result = assess_routing_evidence(state.get("pool_candidates", state.get("candidates", [])),
        state["history"].get("query_step", ""), state["history"].get("query_error", ""),
        _query_context(state), state["settings"])
    if state.get("rule"):
        result.update(sufficient=True, status="sufficient", conflicting=False,
                      reason="An approved matching rule supplies the procedure")
    phase = state["retrieval_phase"]
    review = not result["sufficient"] and (not state["allow_rovo"] or state["issue_load_status"] != "success")
    return {"routing_assessment": result, "review_required": review,
            "retrieval_route": routing.event(state, phase + "_evidence", result["status"],
                                              result["reason"], started)}


def _after_local(state: InvestigationState) -> str:
    if (state["routing_assessment"]["sufficient"] or not state["allow_rovo"]
            or state["issue_load_status"] != "success"):
        return "ready"
    return "mcp"


def _mcp_retrieve(state: InvestigationState) -> dict:
    started = monotonic()
    try:
        remote = routing.stage(state, state["settings"]["rovo_retrieval_budget_seconds"],
            _jira().search_rovo_candidates, state["issue"]["query_text"],
            state["settings"]["retrieval_top_k"], set(state["excluded_sources"]), reserve=2)
    except Exception as exc:
        remote = {"candidates": [], "status": routing.failure_status(exc)}
    return {"remote_history": remote,
            "search_status": {**state["search_status"], "mcp": remote["status"]},
            "retrieval_route": routing.event(state, "rovo_mcp", remote["status"],
                "MCP search and evidence fetch completed" if remote["status"] == "success" else
                "MCP retrieval was incomplete; useful local evidence retained", started,
                candidate_count=len(remote.get("candidates", [])), requested_top_k=state["settings"]["retrieval_top_k"])}


def _merge_history(state: InvestigationState) -> dict:
    candidates = routing.merge(state.get("pool_candidates", []), state["remote_history"],
                                state["settings"]["rerank_pool_k"])
    history = {**state["history"], "backend": "local_then_rovo",
               "rovo_available": state["remote_history"]["status"] in {"success", "partial"}}
    return {**_candidate_update(candidates), "history": history, "retrieval_phase": "combined"}


def _after_combined(state: InvestigationState) -> str:
    return "ready" if state["routing_assessment"]["sufficient"] else "agent"


def _agent_investigate(state: InvestigationState) -> dict:
    from . import rovo_a2a
    started = monotonic()
    budget = min(state["settings"]["rovo_agent_budget_seconds"], state["deadline"] - started - 2)
    deadline = started + max(0, budget)
    reserve = min(10, max(0, budget) / 3)
    proposal = {}
    try:
        proposal = routing.stage(state, budget - reserve, rovo_a2a.investigate_evidence,
            deepcopy(state["issue"]), deepcopy(state.get("pool_candidates") or []),
            deadline - reserve, reserve=2)
        status = "success"
    except Exception as exc:
        status = routing.failure_status(exc)
    return {"agent_proposal": proposal, "agent_status": status, "agent_deadline": deadline,
            "retrieval_phase": "agent", "agent_new_sources": False,
            "retrieval_route": routing.event(state, "rovo_agent", status,
                "Rovo agent returned a proposal for independent validation" if status == "success" else
                "Rovo agent did not complete; manual review required", started)}


def _agent_citations(draft: dict) -> set[str]:
    return {citation["source_key"] for step in draft.get("steps", [])
            for citation in [step, *step.get("supporting_sources", [])]}


def _hydrate_agent(state: InvestigationState) -> dict:
    from .plugins import ISSUE_KEY
    draft = DraftResponse.model_validate(state["agent_proposal"].get("draft")).model_dump()
    keys = _agent_citations(draft)
    known = {source["key"] for source in state.get("pool_candidates", [])}
    added = keys - known
    if (len(added) > 5 or keys.intersection(state["excluded_sources"])
            or any(not ISSUE_KEY.fullmatch(key) or not key.startswith("SAC-") for key in added)):
        raise ValueError("Agent citations are outside the permitted source scope")
    candidates, rejected = [], []
    if added:
        jira = _jira()
        allowed = {row["key"] for row in jira._search(
            'project = SAC AND status in (Resolved, Closed) AND key in (' + ','.join(sorted(added)) + ')',
            "status", len(added))}
        for key in sorted(added):
            candidate = jira.fetch_resolution_candidate(key) if key in allowed else None
            if candidate:
                candidate["retrieval_origins"] = ["rovo_agent"]
                candidates.append(candidate)
            else:
                rejected.append(key)
    if rejected:
        raise ValueError("Agent cited unavailable or unusable resolution evidence")
    return {"draft": draft, "candidates": candidates, "added": bool(added)}


def _validate_agent_sources(state: InvestigationState) -> dict:
    if state["agent_status"] != "success":
        return {"review_required": True}
    started = monotonic()
    try:
        result = routing.stage(state, state["agent_deadline"] - started, _hydrate_agent,
                                deepcopy(state), reserve=2)
        merged = routing.merge(state.get("pool_candidates", []), result,
                                state["settings"]["rerank_pool_k"])
        update = _candidate_update(merged)
        update.update(agent_draft=result["draft"], agent_new_sources=result["added"])
        status, reason = "success", "Agent source IDs were resolved to application-verified evidence"
    except Exception as exc:
        update = {"review_required": True, "agent_status": routing.failure_status(exc), "agent_draft": {}}
        status, reason = routing.failure_status(exc), "Agent source validation failed; manual review required"
    update["retrieval_route"] = routing.event(state, "agent_sources", status, reason, started)
    return update


def _agent_rerank(state: InvestigationState) -> dict:
    if state.get("review_required") or not state.get("agent_draft"):
        return {}
    if not state.get("agent_new_sources"):
        return {}  # Same immutable evidence already has real reranker results.
    return _bounded_rerank({**state, "deadline": min(state["deadline"], state["agent_deadline"])})


def _assess_agent(state: InvestigationState) -> dict:
    update = _assess(state)
    draft = state.get("agent_draft") or {}
    pool = state.get("pool_candidates") or []
    tier, _ = proposal_quality(None, pool, draft, state["settings"]["min_match_score"],
        state.get("source_tiers") or {}, state["history"].get("query_step", ""),
        state["history"].get("query_error", ""), _query_context(state), _routing_policy(state))
    review = (state.get("review_required", False) or not update["routing_assessment"]["sufficient"]
              or tier != "high" or monotonic() > state["agent_deadline"])
    cited = _agent_citations(draft) if draft else set()
    selected = [source for source in pool if source["key"] in cited]
    update.update(review_required=review, agent_draft={} if review else draft,
                  candidates=selected if not review else state.get("candidates", []))
    if review:
        update["retrieval_route"] = routing.event({**state, **update}, "agent_claims", "review_required",
            "Agent claims lack sufficient applicable, non-conflicting support or exceeded the deadline", monotonic())
    else:
        update["retrieval_route"] = routing.event({**state, **update}, "agent_claims", "validated",
            "Every proposed action has an applicable source and faithful supporting quotation", monotonic())
    return update


def _rerank(state: InvestigationState) -> dict:
    candidates = deepcopy(state.get("candidates") or [])
    if state.get("rule") or not candidates:
        return {"candidates": [], "pool_candidates": [], "rerank": [],
                "rerank_trace_id": None, "rerank_status": "not_needed"}
    query_step = state["history"].get("query_step", "")
    query_error = state["history"].get("query_error", "")
    llm_scores: dict[str, dict] = {}
    cache_info: dict = {"status": "disabled"}
    trace_id = None
    rerank_status = "disabled"
    if state["settings"]["llm_rerank_enabled"]:
        try:
            llm_scores, trace_id = rerank_candidates(
                state["issue"]["query_text"], candidates,
                state["settings"]["rerank_temperature"],
                state["settings"]["langfuse_enabled"], state["settings"], cache_info)
            rerank_status = "success"
            storage.audit("system", "rerank_cache_lookup", state["issue_key"],
                          {"investigation_id": state["id"], **cache_info})
        except Exception as exc:
            rerank_status = routing.failure_status(exc)
            cache_info.update(status="scoring_failed")
            storage.audit("system", "llm_rerank_failed", state["issue_key"],
                          {"investigation_id": state["id"], "error_type": type(exc).__name__})
    for candidate in candidates:
        key = candidate.get("key") or candidate.get("filename") or ""
        version = candidate.get("workaround_version") or {}
        votes = storage.feedback_stats(key, query_step, query_error, version.get("id", ""))
        total = votes["up"] + votes["down"]
        # Beta(2,2) prior shrinks sparse feedback toward a neutral 0.5.
        human_score = (votes["up"] + 2.0) / (total + 4.0)
        automated = storage.workaround_outcome_stats(version.get("id", "")) if version else {
            "succeeded": 0, "failed": 0}
        automated_total = automated["succeeded"] + automated["failed"]
        automated_score = ((automated["succeeded"] + 2.0) / (automated_total + 4.0)
                           if automated_total else 0.5)
        evidence_total = total + automated_total
        outcome_evidence_score = (
            (human_score * total + automated_score * automated_total) / evidence_total
            if evidence_total else 0.5)
        candidate.update({
            "llm_score": llm_scores.get(key, {}).get("llm_score", 0.5),
            "relevance_verified": key in llm_scores,
            "llm_reason": llm_scores.get(key, {}).get("llm_reason", "LLM score unavailable"),
            "human_score": human_score,
            "human_up": votes["up"], "human_down": votes["down"],
            "feedback_net": votes["up"] - votes["down"],
            "automated_successes": automated["succeeded"],
            "automated_failures": automated["failed"],
            "outcome_evidence_score": outcome_evidence_score,
        })
        candidate["composite_score"] = (
            state["settings"]["retrieval_weight"] * candidate["retrieval_score"]
            + state["settings"]["llm_rerank_weight"] * candidate["llm_score"]
            + state["settings"]["human_feedback_weight"] * outcome_evidence_score
        )
    candidates.sort(key=lambda item: (item["composite_score"], item.get("resolution_at", "")), reverse=True)
    candidates = prioritize_candidates(
        candidates, state["settings"]["min_match_score"], state.get("source_tiers") or {},
        query_step, query_error, _query_context(state), _routing_policy(state))
    ranked = [{"key": item.get("key") or item.get("filename") or "",
               "workaround_version_id": (item.get("workaround_version") or {}).get("id", ""),
               "retrieval_score": round(item["retrieval_score"], 4),
               "local_rank": item.get("local_rank"),
               "rovo_rank": item.get("rovo_rank"),
               "llm_score": round(item["llm_score"], 4),
               "human_score": round(item["human_score"], 4),
               "human_up": item["human_up"], "human_down": item["human_down"],
               "automated_successes": item["automated_successes"],
               "automated_failures": item["automated_failures"],
               "outcome_evidence_score": round(item["outcome_evidence_score"], 4),
               "composite_score": round(item["composite_score"], 4),
               "eligibility_tier": item["eligibility_tier"],
               "eligibility_reasons": item["eligibility_reasons"],
               "llm_reason": item["llm_reason"], "llm_cache": cache_info}
              for item in candidates]
    return {"candidates": candidates[:state["settings"]["jira_review_top_k"]],
            "pool_candidates": candidates, "rerank_status": rerank_status,
            "rerank": ranked, "rerank_trace_id": trace_id}


def _bounded_rerank(state: InvestigationState) -> dict:
    started = monotonic()
    try:
        result = routing.stage(state, state["settings"]["routing_rerank_timeout_seconds"],
                                _rerank, deepcopy(state), reserve=2)
    except Exception as exc:
        candidates = deepcopy(state.get("candidates") or [])
        for candidate in candidates:
            candidate["relevance_verified"] = False
        result = {"candidates": candidates, "pool_candidates": candidates, "rerank": [],
                  "rerank_status": routing.failure_status(exc), "rerank_trace_id": None}
    result["retrieval_route"] = routing.event(state, state["retrieval_phase"] + "_reranker",
        result["rerank_status"], "Real relevance assessment completed" if result["rerank_status"] == "success"
        else "Relevance assessment was not available; default scores cannot authorize a route", started)
    return result


def _draft_impl(state: InvestigationState) -> dict:
    try:
        result = specialist_registry.get("jira").invoke({
            "query_text": state["issue"]["query_text"],
            "candidates": state.get("candidates") or [], "rule": state.get("rule"),
            "temperature": state["settings"]["temperature"],
            "langfuse_enabled": state["settings"]["langfuse_enabled"]})
        return {"draft": result["draft"], "trace_id": result.get("trace_id"),
                "branch_drafts": result.get("branch_drafts", {}),
                "branch_agreement": result.get("branch_agreement", "single_branch")}
    except Exception as exc:
        # A failed model call cannot authorize an automatic fix.
        storage.audit("system", "jira_draft_failed", state["issue_key"],
                      {"reason": type(exc).__name__, "investigation_id": state["id"]})
        return {"draft": {"decline": True, "reason": "Drafting unavailable", "steps": []},
                "trace_id": None}


def _draft(state: InvestigationState) -> dict:
    try:
        return routing.stage(state, 20, _draft_impl, deepcopy(state), reserve=2)
    except Exception:
        return _review_draft(state)


def _review_candidates(state: InvestigationState) -> list[dict]:
    """Keep applicable, really reranked evidence separate from fallback health."""
    if state.get("issue_load_status") != "success":
        return []
    step, error = state["history"].get("query_step", ""), state["history"].get("query_error", "")
    context = _query_context(state)
    return [source for source in state.get("candidates", [])
            if same_failure(step, error, source, context)
            and routing_source_quality(source, step, error, context,
                state["settings"]["routing_min_llm_relevance"])[0] in {"high", "medium"}]


def _rejected_agent_actions(state: InvestigationState) -> list[str]:
    draft = state.get("agent_proposal", {}).get("draft") or {}
    return draft_support_reasons(state.get("pool_candidates", []), draft) if draft.get("steps") else []


def _review_draft(state: InvestigationState) -> dict:
    # No extra model call after a fallback failure. A complete short source can
    # still be shown faithfully as a historical lead, with publication blocked.
    if (not state.get("routing_assessment", {}).get("conflicting")
            and not _rejected_agent_actions(state)):
        for source in _review_candidates(state):
            evidence = _scrub(resolution_evidence(source.get("comment", "")))
            if not 12 <= len(evidence) <= 500:
                continue  # Never truncate prerequisites to fit an excerpt.
            draft = {"decline": False, "reason": "Historical lead for manual review", "steps": [
                {"text": evidence, "source_key": source["key"], "source_excerpt": evidence}]}
            if not draft_support_reasons([source], draft):
                return {"review_required": True, "draft": draft}
    return {"review_required": True, "draft": {"decline": True, "steps": [],
            "reason": "Evidence or investigation time is insufficient; manual review required"}}


def _writer_route(state: InvestigationState) -> str:
    if state.get("review_required") or monotonic() >= state["deadline"] - 2:
        return "review"
    return "agent" if state.get("agent_draft") else "jira"


def _use_agent_draft(state: InvestigationState) -> dict:
    return {"draft": state["agent_draft"], "branch_agreement": "validated_rovo_agent"}


def _gate(state: InvestigationState) -> dict:
    tier, reasons = proposal_quality(
        state.get("rule"), state.get("candidates") or [], state.get("draft"),
        state["settings"]["min_match_score"], state.get("source_tiers") or {},
        state["history"].get("query_step", ""),
        state["history"].get("query_error", ""), _query_context(state), _routing_policy(state))
    if (not state["history"].get("local_available", True)
        and state.get("candidates")
        and "Best raw historical match is below the posting threshold" in reasons):
        tier = "medium"
        reasons = ["Local similarity search was unavailable; Rovo lead requires manual review",
                   *reasons]
    if (state.get("candidates")
        and "rovo" in (state["candidates"][0].get("retrieval_origins") or [])
        and "local" not in (state["candidates"][0].get("retrieval_origins") or [])
        and "Best raw historical match is below the posting threshold" in reasons):
        tier = "medium"
        reasons = ["Rovo supplied the historical lead; manual verification is required",
                   *reasons]
    draft = state.get("draft") or {}
    review_sources = _review_candidates(state) if state.get("routing_applied") else []
    support_errors = draft_support_reasons(state.get("candidates", []), draft) if (
        draft.get("steps") and not state.get("rule")) else []
    rejected_agent = _rejected_agent_actions(state) if state.get("routing_applied") else []
    if (draft.get("decline") or not draft.get("steps") or len(draft.get("steps") or []) > 5):
        # A missing draft or exhausted fallback budget does not erase Medium
        # source evidence. It still cannot authorize an actionable comment.
        tier = "medium" if tier == "medium" and review_sources else "low"
        reasons = ["No concise, actionable draft is available for posting; manual review required",
                   *reasons]
    if support_errors or rejected_agent:
        tier = "low"
        reasons = [*support_errors, *rejected_agent, *reasons]
    if state.get("branch_agreement") == "different_procedures":
        tier = "low"
        reasons = ["Local and Rovo proposed different procedures; manual comparison is required",
                   *reasons]
    review_required = state.get("review_required", False)
    if state.get("routing_applied"):
        review_keys = {source["key"] for source in review_sources}
        if (state.get("routing_assessment", {}).get("conflicting")
                or (tier == "medium" and not draft.get("decline")
                    and not _agent_citations(draft).issubset(review_keys))):
            tier = "low"
        if tier != "high" or review_required:
            review_required = True
            reasons = ["Source validation or safety checks require manual review", *reasons]
        if tier == "low":
            draft = {"decline": True, "reason": "Evidence or safety validation requires manual review", "steps": []}
    candidates = prioritize_candidates(
        state.get("candidates") or [], state["settings"]["min_match_score"],
        state.get("source_tiers") or {}, state["history"].get("query_step", ""),
        state["history"].get("query_error", ""), _query_context(state), _routing_policy(state))
    if tier in {"high", "medium"} and not state.get("rule") and draft.get("steps"):
        # Comment attribution, version and feedback must identify the actual action source.
        primary = state["draft"]["steps"][0]["source_key"]
        candidates.sort(key=lambda candidate: (candidate.get("key") or candidate.get("filename")) != primary)
    decision = state.get("supervisor_result") or {}
    decision.update({"draft": draft, "confidence": tier,
                     "confidence_reasons": reasons})
    return {"tier": tier, "reasons": reasons, "supervisor_result": decision,
            "candidates": candidates, "review_required": review_required, "draft": draft}


def _supervisor_impl(state: InvestigationState) -> dict:
    findings = state.get("specialist_findings") or {}
    summaries = []
    for name, finding in findings.items():
        status = finding.get("status", "unavailable")
        facts = finding.get("findings") or []
        if status == "completed" and facts:
            summaries.append(f"{name}: {len(facts)} matching read-only record(s).")
        else:
            summaries.append(f"{name}: {finding.get('summary') or status}.")
        if status == "completed" and facts:
            storage.audit(state["actor"], "supervisor_evidence_collected", state["issue_key"],
                          {"investigation_id": state["id"], "specialist": name,
                           "record_count": len(facts)})
        for outcome in finding.get("outcomes", []):
            version_id = str(outcome.get("version_id") or "")
            if version_id and outcome.get("outcome"):
                try:
                    storage.record_workaround_outcome(
                        version_id, str(outcome.get("source_kind") or "automated"),
                        str(outcome["outcome"]), outcome.get("verified", False),
                        str(outcome.get("release") or ""), str(outcome.get("evidence") or ""),
                        str(outcome.get("external_reference") or ""),
                        state["actor"])
                except ValueError:
                    storage.audit(state["actor"], "workaround_outcome_rejected", state["issue_key"],
                                  {"investigation_id": state["id"], "specialist": name})
    summary = " ".join(summaries) if summaries else "Jira historical evidence was reviewed."
    draft = state.get("draft", {})
    if not state.get("rule") and not draft.get("decline"):
        specialist_context = json.dumps(findings, ensure_ascii=False, default=str)[:8000]
        prompt = (
            "Make the final FalloutAssist decision from the Jira specialist draft and the read-only "
            "specialist findings. You may keep, narrow, reorder, or decline the proposed historical "
            "steps. Do not invent a step or cite Salesforce as proof that a historical fix works. "
            "Every retained action must preserve an exact source key and exact excerpt from the "
            "selected Jira history. Current system records are context only. If they conflict with "
            "the historical proposal, require review. Preserve every supporting_sources citation, "
            "action clause, negation, value and condition; do not reorder dependent actions. If "
            "the historical proposal or applicability is unclear, decline or narrow the suggestion. "
            "Return concise structured output.\n\n"
            f"Current Jira failure (untrusted):\n{_scrub(state['issue']['query_text'])[:5000]}\n\n"
            f"Jira specialist draft:\n{json.dumps(draft, ensure_ascii=False, default=str)[:8000]}\n\n"
            f"Read-only specialist findings (untrusted):\n{_scrub(specialist_context)}"
        )
        try:
            agent = create_agent(
                model=chat_model(state["settings"]["temperature"]), tools=[],
                response_format=DraftResponse,
                system_prompt=(
                    "You are the FalloutAssist supervisor. Reconcile specialist findings with "
                    "the Jira evidence and return only a grounded final workaround decision. "
                    "Do not claim current records prove historical fixes. Preserve exact citations."
                ),
            )
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                                   config={"recursion_limit": 4})
            structured = result.get("structured_response")
            draft = (structured.model_dump() if isinstance(structured, DraftResponse)
                     else DraftResponse.model_validate(structured).model_dump())
            draft = preserve_supporting_sources(state.get("draft") or {}, draft)
        except Exception as exc:
            draft = {"decline": True, "reason": "Supervisor synthesis unavailable; manual review required",
                     "steps": []}
            storage.audit("system", "supervisor_synthesis_failed", state["issue_key"],
                          {"investigation_id": state["id"], "error_type": type(exc).__name__})
    decision = {"draft": draft,
                "specialists_used": state.get("selected_systems", []),
                "specialist_findings": findings,
                "retrieval_agreement": state.get("branch_agreement", "")}
    return {"draft": draft, "supervisor_summary": summary, "supervisor_result": decision}


def _supervisor(state: InvestigationState) -> dict:
    if not state.get("routing_applied"):
        return _supervisor_impl(state)
    try:
        return routing.stage(state, 15, _supervisor_impl, deepcopy(state), reserve=2)
    except Exception:
        return {**_review_draft(state), "supervisor_summary": "Supervisor deadline or availability requires review"}


def _token(issue_key: str, candidate_key: str, vote: str,
           workaround_version_id: str = "") -> str:
    secret = feedback_secret()
    if not secret:
        return ""
    serializer = URLSafeSerializer(secret, salt="falloutassist-feedback-v1")
    # Stable across re-investigations, so a new trace does not rewrite an unchanged comment.
    payload = {"issue": issue_key, "candidate": candidate_key, "vote": vote}
    if workaround_version_id:
        payload["version"] = workaround_version_id
    return serializer.dumps(payload)


def _comment(state: InvestigationState) -> dict:
    settings = state["settings"]
    issue = state["issue"]
    rule = state.get("rule")
    top = (state.get("candidates") or [None])[0]
    version = (top.get("workaround_version") or {}) if top and not rule else {}
    source_key = (f"RULE:{rule.get('id', '')}" if rule else
                  ((top.get("key") or top.get("filename") or "") if top else ""))
    source_url = (top.get("url") or "") if top and not rule else ""
    if version:
        release_label = ", ".join(version.get("releases", [])) or "release not recorded"
        version_line = f"Resolution version: v{version.get('version_number')} ({release_label})"
    elif top and top.get("resolution_at"):
        version_line = f"Resolution comment date: {str(top['resolution_at'])[:10]}"
    else:
        version_line = ""
    base = settings["public_base_url"].rstrip("/")
    format_token = _token(issue["key"], source_key, "format")
    format_line = (f"[Copy resolution format|{base}/resolution/{format_token}]"
                   if format_token else "Use the team's resolution format when closing this ticket.")
    decision = state.get("supervisor_result") or {}
    final_draft = decision.get("draft") or state.get("draft", {})
    final_tier = decision.get("confidence", state["tier"])
    retrieval_weight = float(settings.get("retrieval_weight", 0.25))
    rerank_weight = float(settings.get("llm_rerank_weight", 0.5))
    feedback_weight = float(settings.get("human_feedback_weight", 0.25))
    calculation = (
        "Past tickets are searched by meaning and exact words, then ranked using "
        f"retrieval ({retrieval_weight:.0%}), AI relevance ({rerank_weight:.0%}), and "
        f"feedback and verified outcomes ({feedback_weight:.0%}). FalloutAssist also "
        "checks that the failure matches and each proposed step is supported by the cited "
        "resolution. Confidence is a review label, not a probability of success."
    )
    feedback_line = ""
    if source_key and feedback_secret():
        up = _token(issue["key"], source_key, "up", version.get("id", ""))
        down = _token(issue["key"], source_key, "down", version.get("id", ""))
        feedback_line = f"[Helpful|{base}/feedback/{up}] | [Needs improvement|{base}/feedback/{down}]"
    if final_tier == "high":
        title = "Approved workaround" if rule else "Suggested workaround"
        lines = [f"*{title}*", "Overall confidence: High", "",
                 "*How this was calculated*", calculation, "",
                 "*Disclaimer*",
                 "This is historical guidance. Do not apply it blindly. Verify the exact order, customer, release, and prerequisites first.",
                 "", "*Fix starts*"]
        for index, step in enumerate(final_draft["steps"], 1):
            lines.append(f"- {_scrub(step['text'])[:1500]}")
        lines += ["*Fix ends*", "", f"Source: {source_key}" + (f" ({source_url})" if source_url else "")]
        if version_line:
            lines.append(version_line)
        lines.append("Verify the exact order and customer before applying any step, especially a destructive action.")
    elif final_tier == "medium" and final_draft and final_draft.get("steps"):
        lines = ["*Suggested workaround — medium confidence*", ""]
        for index, step in enumerate(final_draft["steps"], 1):
            lines.append(f"{index}. {_scrub(step['text'])[:1500]}")
        lines += ["", f"Source: {source_key}" + (f" ({source_url})" if source_url else "")]
        if version_line:
            lines.append(version_line)
        lines.append("Verify the order, customer, release, and prerequisites before applying this historical suggestion.")
    elif final_tier == "medium":
        lines = ["*No actionable workaround is available — manual review needed*", "",
                 "The available evidence does not support a concise action for this ticket.",
                 f"Review source: {source_key}" + (f" ({source_url})" if source_url else "")]
    else:
        lines = ["*No reliable workaround found yet*", "",
                 "The available historical evidence does not support a fix for this failure. Investigate manually and record the resolution when closing the ticket."]
    if final_tier == "medium" and final_draft and final_draft.get("steps"):
        lines = ["*Suggested workaround - medium confidence*", "Overall confidence: Medium", "",
                 "*How this was calculated*", calculation, "", "*Disclaimer*",
                 "This is a historical lead for review, not an approved fix. Do not apply it blindly; verify the exact order, customer, release, and prerequisites."]
        if state.get("review_required"):
            lines.append("Manual review required before use.")
        lines += ["", "*Fix starts*"]
        lines.extend(f"- {_scrub(step['text'])[:1500]}" for step in final_draft["steps"])
        lines += ["*Fix ends*", "", f"Source: {source_key}" + (f" ({source_url})" if source_url else "")]
        if version_line:
            lines.append(version_line)
        lines.append("Verify the order, customer, release, and prerequisites before applying this historical suggestion.")
    elif final_tier == "medium":
        lines = ["*No actionable workaround is available; manual review needed*",
                 "Overall confidence: Medium", "", "*How this was calculated*", calculation,
                 "", "*Disclaimer*",
                 "The available evidence does not support a concise action for this ticket. Do not apply an inferred fix.",
                 f"Review source: {source_key}" + (f" ({source_url})" if source_url else "")]
    elif final_tier == "low":
        lines = ["*No reliable workaround found yet*", "Overall confidence: Low", "",
                 "*How this was calculated*", calculation, "", "*Disclaimer*",
                 "The available historical evidence does not support a fix for this failure. Investigate manually and record the resolution when closing the ticket."]
    if final_tier == "high" and state.get("review_required"):
        lines.insert(2, "Manual review required before use.")
    if not rule:
        seen = set()
        matches = []
        for candidate in state.get("candidates") or []:
            key = candidate.get("key") or candidate.get("filename") or ""
            if not key or key in seen:
                continue
            seen.add(key)
            score = candidate.get("composite_score")
            valid_score = (isinstance(score, (int, float)) and not isinstance(score, bool)
                           and math.isfinite(score) and 0 <= score <= 1)
            label = f"{round(score, 4):.1%} match" if valid_score else "match score not recorded"
            matches.append(f"- {_scrub(str(key))[:100]}: {label}")
        if matches:
            lines += ["", "*Top matching past tickets*", *matches,
                      "Match scores combine retrieval, AI relevance, and outcome feedback using the configured weights; they are not probabilities that a fix will work.",
                      "These tickets were reviewed as candidates; their actions still require the confidence and applicability checks above."]
    finding = (state.get("specialist_findings") or {}).get("salesforce", {})
    records = finding.get("findings") or []
    if records:
        lines += ["", "*Supervisor evidence — Salesforce read-only checks*"]
        for record in records[:10]:
            details = {key: value for key, value in record.items()
                       if key in {"Name", "OrderNumber", "Status", "Type", "AccountNumber",
                                  "SerialNumber", "CaseNumber", "Subject", "EffectiveDate"}}
            if details:
                lines.append("• " + _scrub(json.dumps(details, ensure_ascii=False, default=str))[:400])
        lines.append("These current records provide context; they do not confirm that a historical workaround succeeded.")
    for name, finding in (state.get("specialist_findings") or {}).items():
        if name == "salesforce":
            continue
        records = finding.get("findings") or []
        if records:
            lines += ["", f"*Supervisor evidence — {name} read-only checks*"]
            for record in records[:10]:
                lines.append("• " + _scrub(json.dumps(record, ensure_ascii=False, default=str))[:400])
            lines.append("These live records provide context; they do not confirm that a historical workaround succeeded.")
    agreement = state.get("branch_agreement", "")
    if agreement == "independent_sources_agree":
        lines += ["", "Local and Rovo retrieval produced the same procedure from different Jira sources."]
    elif agreement == "same_ticket_both_paths":
        lines += ["", "Local and Rovo retrieval found the same Jira source; this is one supporting ticket, not two independent confirmations."]
    elif agreement == "different_procedures":
        lines += ["", "Local and Rovo evidence suggests different procedures; review both before applying either."]
    lines += ["", format_line, "",
              "_Auto-commented by FalloutAssist LangChain (8012) · FA-LANGCHAIN-8012_ "]
    if feedback_line:
        lines += ["", "*Feedback*", feedback_line]
    return {"comment_body": "\n".join(lines).strip()[:8000]}


def _source_details(state: InvestigationState) -> list[dict]:
    return [{"key": source["key"], "url": source.get("url", ""),
             "origins": source.get("retrieval_origins", ["local"])}
            for source in state.get("candidates", [])]


def _persist(state: InvestigationState) -> dict:
    issue = state["issue"]
    source_keys = [c.get("key") or c.get("filename") or "" for c in state.get("candidates") or []]
    if state.get("rule"):
        source_keys.insert(0, f"RULE:{state['rule'].get('id', '')}")
    source_data = {
        "id": state["id"], "issue_key": issue["key"], "issue_updated": issue["updated"],
        "tier": state["tier"], "status": "preview", "reasons": state["reasons"],
        "source_keys": source_keys, "query_step": state["history"].get("query_step", ""),
        "query_error": state["history"].get("query_error", ""),
        "settings": {**state["settings"],
                     "history_backend": state["history"].get("backend", "local"),
                     "retrieval_sources": {
                         "local_available": state["history"].get("local_available", True),
                         "rovo_available": state["history"].get("rovo_available", False),
                     },
                     "prompt_version": PROMPT_VERSION,
                     "reranker_trace_id": state.get("rerank_trace_id"),
                     "specialists": state.get("selected_systems", []),
                     "specialist_findings": state.get("specialist_findings", {}),
                     "supervisor_summary": state.get("supervisor_summary", ""),
                     "branch_drafts": state.get("branch_drafts", {}),
                     "branch_agreement": state.get("branch_agreement", ""),
                     "supervisor_result": state.get("supervisor_result", {})},
        "comment_body": state["comment_body"],
        "trace_id": state.get("trace_id"),
        "draft_trace_id": state.get("trace_id"),
        "reranker_trace_id": state.get("rerank_trace_id"),
        "rerank": state.get("rerank", []),
        "specialists": state.get("selected_systems", []),
        "specialist_findings": state.get("specialist_findings", {}),
        "supervisor_summary": state.get("supervisor_summary", ""),
        "branch_drafts": state.get("branch_drafts", {}),
        "branch_agreement": state.get("branch_agreement", ""),
        "supervisor_result": state.get("supervisor_result", {}),
    }
    source_data["settings"].update(retrieval_route=state.get("retrieval_route", []),
                                  review_required=state.get("review_required", state["tier"] == "low"),
                                  source_details=_source_details(state))
    storage.save_investigation(source_data, state["actor"], {
        "investigation_id": state["id"], "tier": state["tier"],
        "reasons": state["reasons"], "source_keys": source_keys,
        "trace_id": state.get("trace_id"),
        "reranker_trace_id": state.get("rerank_trace_id"),
        "prompt_version": PROMPT_VERSION,
        "rerank": state.get("rerank", []),
        "settings": state["settings"],
        "specialists": state.get("selected_systems", []),
        "specialist_statuses": {name: result.get("status", "unknown")
                                 for name, result in (state.get("specialist_findings") or {}).items()},
    })
    return {"status": "preview"}


def _publish_impl(state: InvestigationState) -> dict:
    settings = state["settings"]
    if not state["publish_requested"] or not settings["publish_enabled"] or settings["dry_run"]:
        return {"status": "preview"}
    if (state.get("tier") == "low" or state.get("review_required")
            or (state.get("deadline") is not None and monotonic() >= state["deadline"])):
        storage.update_investigation_status(state["id"], "preview")
        storage.audit(state["actor"], "jira_publish_skipped_low_confidence",
                      state["issue_key"], {"investigation_id": state["id"]})
        return {"status": "preview"}
    if not feedback_secret():
        raise RuntimeError("FEEDBACK_SECRET is required for live Jira publishing")
    if settings["public_base_url"].startswith("http://localhost"):
        raise RuntimeError("Set a reachable public_base_url before live publishing")
    issue = state["issue"]
    body = state["comment_body"]
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    prior = storage.publication(issue["key"])
    live_comment = _jira().existing_comment(issue["key"])
    if (prior and prior["body_hash"] == body_hash and live_comment
        and hashlib.sha256(live_comment["body"].encode("utf-8")).hexdigest() == body_hash):
        storage.save_publication(issue["key"], live_comment["id"], issue["updated"],
                                 body_hash, settings_hash, state["tier"], state["id"],
                                 state["actor"], "jira_comment_unchanged")
        storage.update_investigation_status(state["id"], "unchanged")
        return {"status": "unchanged", "comment_id": live_comment["id"]}
    # The local audit intent must commit before the external Jira write.
    storage.audit(state["actor"], "jira_publish_intent", issue["key"],
                  {"investigation_id": state["id"], "tier": state["tier"],
                   "body_hash": body_hash, "previous_comment_id": prior["comment_id"] if prior else ""})
    try:
        comment_id = _jira().publish_comment(issue["key"], body,
                                              prior["comment_id"] if prior else "")
    except Exception as exc:
        storage.update_investigation_status(state["id"], "publish_failed")
        storage.audit(state["actor"], "jira_publish_failed", issue["key"],
                      {"investigation_id": state["id"], "reason": type(exc).__name__})
        raise
    storage.save_publication(issue["key"], comment_id, issue["updated"], body_hash,
                             settings_hash, state["tier"], state["id"],
                             state["actor"], "jira_published")
    storage.update_investigation_status(state["id"], "published")
    return {"status": "published", "comment_id": comment_id}


def _publish(state: InvestigationState) -> dict:
    from .execution import deadline_context
    if not state.get("routing_applied"):
        return _publish_impl(state)
    # Side-effecting publication stays on the graph thread. A timeout cannot
    # leave a detached worker that later decides to publish a suggestion.
    token = deadline_context.set(min(state["deadline"], monotonic() + 10))
    try:
        return _publish_impl(state)
    except Exception as exc:
        storage.update_investigation_status(state["id"], "publish_failed")
        storage.audit(state["actor"], "jira_publication_needs_reconciliation", state["issue_key"],
                      {"investigation_id": state["id"], "error_type": type(exc).__name__})
        return {"status": "publish_failed", "review_required": True}
    finally:
        deadline_context.reset(token)


def _graph():
    builder = StateGraph(InvestigationState)
    builder.add_node("route", _route)
    builder.add_node("load_jira", _load)
    builder.add_node("local_retrieval", _local_retrieve)
    builder.add_node("local_reranker", _bounded_rerank)
    builder.add_node("assess_local", _assess)
    builder.add_node("rovo_mcp", _mcp_retrieve)
    builder.add_node("merge_history", _merge_history)
    builder.add_node("combined_reranker", _bounded_rerank)
    builder.add_node("assess_combined", _assess)
    builder.add_node("rovo_agent", _agent_investigate)
    builder.add_node("validate_agent_sources", _validate_agent_sources)
    builder.add_node("agent_reranker", _agent_rerank)
    builder.add_node("assess_agent", _assess_agent)
    builder.add_node("retrieval_ready", lambda state: {})
    builder.add_node("choose_writer", lambda state: {})
    builder.add_node("review_draft", _review_draft)
    builder.add_node("agent_draft", _use_agent_draft)
    builder.add_node("jira_specialist", _draft)
    builder.add_node("quality_gate", _gate)
    builder.add_node("supervisor", _supervisor)
    builder.add_node("compose_comment", _comment)
    builder.add_node("persist_audit", _persist)
    builder.add_node("jira_publisher", _publish)
    builder.add_edge(START, "route")
    builder.add_edge("route", "load_jira")
    builder.add_edge("load_jira", "local_retrieval")
    builder.add_conditional_edges("local_retrieval", _after_local_search,
                                  {"rerank": "local_reranker", "mcp": "rovo_mcp"})
    builder.add_edge("local_reranker", "assess_local")
    builder.add_conditional_edges("assess_local", _after_local,
                                  {"ready": "retrieval_ready", "mcp": "rovo_mcp"})
    builder.add_edge("rovo_mcp", "merge_history")
    builder.add_edge("merge_history", "combined_reranker")
    builder.add_edge("combined_reranker", "assess_combined")
    builder.add_conditional_edges("assess_combined", _after_combined,
                                  {"ready": "retrieval_ready", "agent": "rovo_agent"})
    builder.add_edge("rovo_agent", "validate_agent_sources")
    builder.add_edge("validate_agent_sources", "agent_reranker")
    builder.add_edge("agent_reranker", "assess_agent")
    builder.add_edge("assess_agent", "retrieval_ready")
    specialist_nodes = []
    # Graph construction happens at import, before lifespan initializes storage.
    # Runtime configuration is checked later by _route(), not during registration.
    for manifest in specialist_registry.manifests():
        name = manifest["name"]
        if name == "jira":
            continue
        node_name = f"specialist_{name}"
        def make_node(specialist_name: str):
            def invoke(state: InvestigationState) -> dict:
                return _run_specialist(state, specialist_name)
            return invoke
        builder.add_node(node_name, make_node(name))
        builder.add_edge("load_jira", node_name)
        specialist_nodes.append(node_name)
    builder.add_edge(["retrieval_ready", *specialist_nodes], "choose_writer")
    builder.add_conditional_edges("choose_writer", _writer_route,
                                  {"jira": "jira_specialist", "agent": "agent_draft", "review": "review_draft"})
    builder.add_edge("agent_draft", "supervisor")
    builder.add_edge("review_draft", "supervisor")
    builder.add_edge("jira_specialist", "supervisor")
    builder.add_edge("supervisor", "quality_gate")
    builder.add_edge("quality_gate", "compose_comment")
    builder.add_edge("compose_comment", "persist_audit")
    builder.add_edge("persist_audit", "jira_publisher")
    builder.add_edge("jira_publisher", END)
    return builder.compile()


GRAPH = _graph()


def investigate(issue_key: str, actor: str, publish: bool = False,
                local_only: bool = False) -> dict:
    item_id = uuid.uuid4().hex
    state: InvestigationState = {
        "id": item_id, "issue_key": issue_key, "actor": actor,
        "publish_requested": publish, "local_only": local_only,
        "settings": storage.settings(),
    }
    result = GRAPH.invoke(state, config={"recursion_limit": 40})
    return {
        "investigation_id": item_id,
        "issue_key": result["issue_key"],
        "tier": result["tier"], "reasons": result["reasons"],
        "status": result["status"], "comment_id": result.get("comment_id", ""),
        "comment_body": result["comment_body"],
        "sources": [c.get("key") or c.get("filename") for c in result.get("candidates") or []],
        "source_keys": ([f"RULE:{result['rule'].get('id', '')}"] if result.get("rule") else
                        [c.get("key") or c.get("filename") for c in result.get("candidates") or []]),
        "rerank": result.get("rerank", []),
        "trace_id": result.get("trace_id"),
        "reranker_trace_id": result.get("rerank_trace_id"),
        "specialists": result.get("selected_systems", []),
        "specialist_findings": result.get("specialist_findings", {}),
        "supervisor_summary": result.get("supervisor_summary", ""),
        "branch_drafts": result.get("branch_drafts", {}),
        "branch_agreement": result.get("branch_agreement", ""),
        "supervisor_result": result.get("supervisor_result", {}),
        "retrieval_route": result.get("retrieval_route", []),
        "review_required": result.get("review_required", result["tier"] == "low"),
        "source_details": _source_details(result),
    }

