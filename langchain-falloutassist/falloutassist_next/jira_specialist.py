"""Jira historical-resolution agent behind the common specialist contract."""

from __future__ import annotations

import re

from .agent import _scrub, draft_fix
from .specialists import SpecialistManifest


class JiraSuggestionSpecialist:
    manifest = SpecialistManifest(
        "jira", "Find and explain versioned Jira workarounds for matching fallout",
        False, ("search_history", "compare_resolution_versions", "draft_suggestion"),
    )

    def configured(self) -> bool:
        return True

    def invoke(self, context: dict) -> dict:
        rule = context.get("rule")
        if rule:
            answer = (rule.get("workaround_text") or "").strip()
            return {"specialist": "jira", "draft": {
                "decline": False, "reason": "Approved matching rule",
                "steps": [{"text": answer, "source_key": f"RULE:{rule.get('id', '')}",
                           "source_excerpt": answer}]}, "trace_id": None}
        candidates = context.get("candidates") or []
        if not candidates:
            return {"specialist": "jira", "draft": {
                "decline": True, "reason": "No strong historical match", "steps": []},
                "trace_id": None}
        origins = lambda candidate: candidate.get("retrieval_origins") or ["local"]
        branches = {
            "local": [item for item in candidates if "local" in origins(item)],
            "rovo": [item for item in candidates if "rovo" in origins(item)],
        }
        branch_drafts = {}
        traces = []
        for branch, items in branches.items():
            if not items:
                continue
            try:
                branch_draft, trace = self._draft_branch(context, items)
            except Exception as exc:
                branch_draft, trace = ({"decline": True,
                                         "reason": f"{branch.title()} drafting unavailable ({type(exc).__name__})",
                                         "steps": []}, None)
            branch_drafts[branch] = branch_draft
            if trace:
                traces.append(trace)
        local_steps = (branch_drafts.get("local") or {}).get("steps", [])
        rovo_steps = (branch_drafts.get("rovo") or {}).get("steps", [])
        if not branch_drafts:
            return {"specialist": "jira", "draft": {
                "decline": True, "reason": "No retriever supplied resolution evidence", "steps": []},
                "branch_drafts": {}, "branch_agreement": "none", "trace_id": None}
        if not local_steps and not rovo_steps:
            final = {"decline": True, "reason": "Both retrieval branches declined to propose a fix",
                     "steps": []}
            agreement = "none"
        elif not local_steps or not rovo_steps:
            final = dict(branch_drafts.get("local") or branch_drafts.get("rovo"))
            agreement = "single_branch"
        else:
            normalize = lambda text: " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
            rovo_by_text = {normalize(step.get("text", "")): step for step in rovo_steps}
            common = [step for step in local_steps
                      if normalize(step.get("text", "")) in rovo_by_text]
            local_procedures = {normalize(step.get("text", "")) for step in local_steps}
            rovo_procedures = {normalize(step.get("text", "")) for step in rovo_steps}
            if common and local_procedures == rovo_procedures:
                final_steps, used = [], set()
                for step in [*common, *local_steps, *rovo_steps]:
                    text_key = normalize(step.get("text", ""))
                    if not text_key or text_key in used:
                        continue
                    used.add(text_key)
                    combined = dict(step)
                    supporters = []
                    for branch_step in [*local_steps, *rovo_steps]:
                        if normalize(branch_step.get("text", "")) == text_key:
                            supporters.append({"source_key": branch_step.get("source_key", ""),
                                               "source_excerpt": branch_step.get("source_excerpt", "")})
                    combined["supporting_sources"] = supporters
                    final_steps.append(combined)
                rovo_by_text = {normalize(step.get("text", "")): step for step in rovo_steps}
                independent_support = any(
                    step.get("source_key") != rovo_by_text.get(normalize(step.get("text", "")), {}).get("source_key")
                    for step in common)
                agreement = "independent_sources_agree" if independent_support else "same_ticket_both_paths"
                final = {"decline": False, "reason": "Both retrieval branches support a matching procedure",
                         "steps": final_steps[:5], "branch_agreement": agreement}
            else:
                final_steps, seen = [], set()
                for step in [*local_steps, *rovo_steps]:
                    key = (normalize(step.get("text", "")), step.get("source_key", ""))
                    if key not in seen:
                        final_steps.append(step)
                        seen.add(key)
                final = {"decline": False,
                         "reason": "Local and Rovo retrieved different procedures; review both",
                         "steps": final_steps[:5], "branch_conflict": True}
                agreement = "different_procedures"
        return {"specialist": "jira", "draft": final, "branch_drafts": branch_drafts,
                "branch_agreement": agreement,
                "trace_id": traces[0] if traces else None}

    def _draft_branch(self, context: dict, candidates: list[dict]) -> tuple[dict, str | None]:
        top = candidates[0]
        body = (top.get("comment") or "").strip()
        key = top.get("key") or top.get("filename") or ""
        if "=== fix ===" in body.lower():
            fix = body[body.lower().index("=== fix ===") + len("=== fix ==="):].strip()
        else:
            fix = ""
        if 20 <= len(fix) <= 500:
            return ({"decline": False, "reason": "Structured historical fix",
                     "steps": [{"text": _scrub(fix), "source_key": key,
                                "source_excerpt": _scrub(fix)}]}, None)
        return draft_fix(context["query_text"], candidates, context["temperature"],
                         context["langfuse_enabled"])
