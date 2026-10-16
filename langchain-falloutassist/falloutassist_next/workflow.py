"""Jira-first LangGraph investigation, quality gate, and audited publisher."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Annotated, TypedDict

from itsdangerous import URLSafeSerializer
from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph

from . import storage
from .agent import DraftResponse, PROMPT_VERSION, _scrub, chat_model
from .config import feedback_secret
from .plugins import JiraPlugin, registry
from .quality import proposal_quality, source_quality
from .reranker import rerank_candidates
from .specialist_registry import specialist_registry


def _merge_specialist_findings(left: dict, right: dict) -> dict:
    return {**(left or {}), **(right or {})}


class InvestigationState(TypedDict, total=False):
    id: str
    issue_key: str
    actor: str
    publish_requested: bool
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


def _jira() -> JiraPlugin:
    return registry.get("jira")  # type: ignore[return-value]


def _route(state: InvestigationState) -> dict:
    if "jira" not in {item["name"] for item in registry.catalog()}:
        raise RuntimeError("Jira plugin is not registered")
    systems = ["jira"]
    systems.extend(item["name"] for item in specialist_registry.catalog()
                   if item["name"] != "jira" and item["configured"])
    return {"selected_systems": systems}


def _load(state: InvestigationState) -> dict:
    _, issue = _jira().issue(state["issue_key"])
    return {"issue": issue, "issue_key": issue["key"]}


def _run_specialist(state: InvestigationState, name: str) -> dict:
    specialist = specialist_registry.get(name)
    if name not in state.get("selected_systems", []):
        result = {"specialist": name, "status": "skipped", "findings": [],
                  "summary": "Not selected by the router."}
    else:
        try:
            result = specialist.invoke({"query_text": state["issue"]["query_text"],
                                        "issue": state["issue"],
                                        "temperature": state["settings"]["temperature"],
                                        "langfuse_enabled": state["settings"]["langfuse_enabled"]})
        except Exception as exc:
            result = {"specialist": name, "status": "unavailable", "findings": [],
                      "summary": f"Specialist unavailable ({type(exc).__name__})."}
            storage.audit("system", "specialist_failed", state["issue_key"],
                          {"investigation_id": state["id"], "specialist": name,
                           "error_type": type(exc).__name__})
    return {"specialist_findings": {name: result}}


def _retrieve(state: InvestigationState) -> dict:
    settings = state["settings"]
    issue = state["issue"]
    excluded = {issue["key"], *storage.rejected_candidates(issue["key"])}
    try:
        history = _jira().search_history(
            issue["query_text"], settings["retrieval_top_k"],
            settings["min_match_score"], exclude_keys=excluded)
    except Exception as exc:
        storage.audit("system", "retrieval_failed", issue["key"],
                      {"investigation_id": state["id"], "error_type": type(exc).__name__})
        raise
    for source, error_type in history.get("retrieval_failures", {}).items():
        storage.audit(state["actor"], "retrieval_source_failed", issue["key"],
                      {"investigation_id": state["id"], "source": source,
                       "error_type": error_type, "continued_with_fallback": True})
    rule = _jira().approved_rule(issue["query_text"])
    if rule and f"RULE:{rule['id']}" in excluded:
        rule = None
    candidates = history["candidates"][:settings["rerank_pool_k"]]
    source_tiers = {}
    for candidate in candidates:
        key = candidate.get("key") or candidate.get("filename") or ""
        tier, _ = source_quality(candidate.get("type", "ticket"), candidate,
                                 candidate.get("comment", ""))
        source_tiers[key] = tier
        candidate["source_tier"] = tier
    return {"history": history, "rule": rule, "candidates": candidates,
            "source_tiers": source_tiers, "rerank": []}


def _rerank(state: InvestigationState) -> dict:
    candidates = state.get("candidates") or []
    if state.get("rule") or not candidates:
        return {"candidates": [], "rerank": [], "rerank_trace_id": None}
    query_step = state["history"].get("query_step", "")
    query_error = state["history"].get("query_error", "")
    llm_scores: dict[str, dict] = {}
    trace_id = None
    if state["settings"]["llm_rerank_enabled"]:
        try:
            llm_scores, trace_id = rerank_candidates(
                state["issue"]["query_text"], candidates,
                state["settings"]["rerank_temperature"],
                state["settings"]["langfuse_enabled"])
        except Exception as exc:
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
            "llm_reason": llm_scores.get(key, {}).get("llm_reason", "LLM score unavailable"),
            "human_score": human_score,
            "human_up": votes["up"], "human_down": votes["down"],
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
               "llm_reason": item["llm_reason"]}
              for item in candidates]
    return {"candidates": candidates[:state["settings"]["jira_review_top_k"]],
            "rerank": ranked, "rerank_trace_id": trace_id}


def _draft(state: InvestigationState) -> dict:
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


def _gate(state: InvestigationState) -> dict:
    tier, reasons = proposal_quality(
        state.get("rule"), state.get("candidates") or [], state.get("draft"),
        state["settings"]["min_match_score"], state.get("source_tiers") or {},
        state["history"].get("query_step", ""),
        state["history"].get("query_error", ""))
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
    if state.get("draft", {}).get("decline") and not state.get("rule"):
        tier = "medium" if tier == "high" else tier
        reasons = ["Supervisor declined to approve a workaround", *reasons]
    if state.get("branch_agreement") == "different_procedures":
        tier = "medium"
        reasons = ["Local and Rovo proposed different procedures; manual comparison is required", *reasons]
    top = (state.get("candidates") or [None])[0]
    if top and int(top.get("automated_failures", 0)) > int(top.get("automated_successes", 0)):
        tier = "medium"
        reasons = ["Verified automated runs failed more often than they succeeded for this workaround version",
                   *reasons]
    decision = state.get("supervisor_result") or {}
    decision.update({"draft": state.get("draft", {}), "confidence": tier,
                     "confidence_reasons": reasons})
    return {"tier": tier, "reasons": reasons, "supervisor_result": decision}


def _supervisor(state: InvestigationState) -> dict:
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
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            structured = result.get("structured_response")
            draft = (structured.model_dump() if isinstance(structured, DraftResponse)
                     else DraftResponse.model_validate(structured).model_dump())
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
    source_key = f"RULE:{rule.get('id', '')}" if rule else (
        top.get("key") or top.get("filename") or "") if top else ""
    source_url = (top.get("url") or "") if top and not rule else ""
    base = settings["public_base_url"].rstrip("/")
    format_token = _token(issue["key"], source_key, "format")
    format_line = (f"[Copy resolution format|{base}/resolution/{format_token}]"
                   if format_token else "Use the team's resolution format when closing this ticket.")
    feedback_lines = ""
    if source_key and feedback_secret() and state["tier"] == "high":
        up = _token(issue["key"], source_key, "up", version.get("id", ""))
        down = _token(issue["key"], source_key, "down", version.get("id", ""))
        feedback_lines = (f"\nWas this helpful? [Yes|{base}/feedback/{up}] | "
                          f"[No, improve it|{base}/feedback/{down}]\n")
    decision = state.get("supervisor_result") or {}
    final_draft = decision.get("draft") or state.get("draft", {})
    final_tier = decision.get("confidence", state["tier"])
    if final_tier == "high":
        title = "Approved workaround" if rule else "Suggested workaround"
        lines = [f"*{title}*", ""]
        for index, step in enumerate(final_draft["steps"], 1):
            lines.append(f"{index}. {_scrub(step['text'])[:1500]}")
        lines += ["", f"Source: {source_key}" + (f" ({source_url})" if source_url else "")]
        if version:
            release_label = ", ".join(version.get("releases", [])) or "release not recorded"
            lines.append(f"Workaround library version: v{version.get('version_number')} ({release_label})")
        if top and not rule:
            lines.append(f"Historical match score: {float(top.get('score') or 0):.2f} (not a probability of correctness)")
        lines += ["", "Verify the exact order and customer before applying any step, especially a destructive action."]
    elif final_tier == "medium":
        lines = ["*Historical lead — manual verification needed*", "",
                 "A related past resolution was found, but the evidence does not support an automatic fix for this ticket.",
                 f"Review source: {source_key}" + (f" ({source_url})" if source_url else "")]
    else:
        lines = ["*No reliable workaround found yet*", "",
                 "The available historical evidence does not support a fix for this failure. Investigate manually and record the resolution when closing the ticket."]
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
    lines += ["", format_line, feedback_lines,
              "_Auto-commented by FalloutAssist · FA-SUGGESTION_ "]
    return {"comment_body": "\n".join(lines).strip()[:8000]}


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


def _publish(state: InvestigationState) -> dict:
    settings = state["settings"]
    if not state["publish_requested"] or not settings["publish_enabled"] or settings["dry_run"]:
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


def _graph():
    builder = StateGraph(InvestigationState)
    builder.add_node("route", _route)
    builder.add_node("load_jira", _load)
    builder.add_node("retrieve_history", _retrieve)
    builder.add_node("composite_reranker", _rerank)
    builder.add_node("jira_specialist", _draft)
    builder.add_node("quality_gate", _gate)
    builder.add_node("supervisor", _supervisor)
    builder.add_node("compose_comment", _comment)
    builder.add_node("persist_audit", _persist)
    builder.add_node("jira_publisher", _publish)
    builder.add_edge(START, "route")
    builder.add_edge("route", "load_jira")
    builder.add_edge("load_jira", "retrieve_history")
    specialist_nodes = []
    for manifest in specialist_registry.catalog():
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
    builder.add_edge(["retrieve_history", *specialist_nodes], "composite_reranker")
    builder.add_edge("composite_reranker", "jira_specialist")
    builder.add_edge("jira_specialist", "supervisor")
    builder.add_edge("supervisor", "quality_gate")
    builder.add_edge("quality_gate", "compose_comment")
    builder.add_edge("compose_comment", "persist_audit")
    builder.add_edge("persist_audit", "jira_publisher")
    builder.add_edge("jira_publisher", END)
    return builder.compile()


GRAPH = _graph()


def investigate(issue_key: str, actor: str, publish: bool = False) -> dict:
    item_id = uuid.uuid4().hex
    state: InvestigationState = {
        "id": item_id, "issue_key": issue_key, "actor": actor,
        "publish_requested": publish, "settings": storage.settings(),
    }
    result = GRAPH.invoke(state)
    return {
        "investigation_id": item_id,
        "issue_key": result["issue_key"],
        "tier": result["tier"], "reasons": result["reasons"],
        "status": result["status"], "comment_id": result.get("comment_id", ""),
        "comment_body": result["comment_body"],
        "sources": [c.get("key") or c.get("filename") for c in result.get("candidates") or []],
        "rerank": result.get("rerank", []),
        "trace_id": result.get("trace_id"),
        "reranker_trace_id": result.get("rerank_trace_id"),
        "specialists": result.get("selected_systems", []),
        "specialist_findings": result.get("specialist_findings", {}),
        "supervisor_summary": result.get("supervisor_summary", ""),
        "branch_drafts": result.get("branch_drafts", {}),
        "branch_agreement": result.get("branch_agreement", ""),
        "supervisor_result": result.get("supervisor_result", {}),
    }

