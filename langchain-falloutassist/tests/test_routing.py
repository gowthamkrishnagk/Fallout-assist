"""Exercise the compiled graph, real evidence gates and mocked external services."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from threading import Event
from time import monotonic
from unittest.mock import Mock, patch
from uuid import uuid4

from falloutassist_next import execution, storage, workflow
from falloutassist_next.config import DEFAULTS, validated_settings

STEP, ERROR = "Suspend insurance", "HTTP 409: account exists"
ACTION = "Update the enrollment ID in Salesforce"
BODY = ACTION + "; Retry the failed step."


def source(key="SAC-100", **changes):
    return {"key": key, "type": "ticket", "step": STEP, "error": ERROR, "summary": "Enrollment failed",
            "comment": BODY, "score": .95, "retrieval_score": 1.0,
            "url": "https://jira.example/browse/" + key, "retrieval_origins": ["local"], **changes}


def proposal(key="SAC-100", text=ACTION, quote=ACTION):
    return {"decline": False, "reason": "Source-supported action", "steps": [
        {"text": text, "source_key": key, "source_excerpt": quote}]}


class RoutingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        original_connect = storage.connect
        for target in (patch.object(storage, "database_backend", return_value="sqlite"),
                       patch.object(storage, "connect", side_effect=lambda: original_connect(Path(temporary.name) / "test.sqlite3"))):
            target.start()
            self.addCleanup(target.stop)
        storage.initialize()
        self.jira = Mock()
        self.jira.issue.return_value = ({}, {"key": "SAC-999", "updated": "2026-10-21",
            "query_text": f"Summary: Enrollment failed\nFailed Step: {STEP}\nError: {ERROR}"})
        self.jira.approved_rule.return_value = None
        self.jira.search_local_history.return_value = {"candidates": [source()], "query_step": STEP,
                                                        "query_error": ERROR, "backend": "local"}
        self.jira.search_rovo_candidates.return_value = {"candidates": [], "status": "success"}
        self.jira.existing_comment.return_value = None
        self.jira.publish_comment.return_value = "comment-1"
        self.jira._search.return_value = [{"key": "SAC-200"}]
        self.jira.fetch_resolution_candidate.side_effect = lambda key: source(key, score=0, retrieval_origins=["rovo_agent"])
        self.rerank = Mock(side_effect=lambda query, candidates, *args: (
            {c["key"]: {"llm_score": c.get("test_relevance", .95), "llm_reason": "Relevant resolution"}
             for c in candidates}, None))
        self.agent = Mock(return_value={"draft": {"decline": True, "reason": "Insufficient evidence", "steps": []}})
        def draft(state):
            if state.get("rule"):
                key, text = "RULE:" + state["rule"]["id"], state["rule"]["workaround_text"]
            else:
                key, text = state["candidates"][0]["key"], ACTION
            return {"draft": proposal(key, text=text, quote=text)}
        self.draft = Mock(side_effect=draft)
        self.supervisor = Mock(side_effect=lambda state: {"draft": state["draft"], "supervisor_summary": "Reviewed"})
        for target in (patch.object(workflow, "_jira", return_value=self.jira),
                       patch.object(workflow, "config_value", return_value="rovo"),
                       patch.object(workflow.specialist_registry, "catalog", return_value=[{"name": "jira", "configured": True}]),
                       patch.object(workflow, "rerank_candidates", self.rerank),
                       patch.object(workflow, "_draft_impl", self.draft),
                       patch.object(workflow, "_supervisor_impl", self.supervisor),
                       patch("falloutassist_next.rovo_a2a.investigate_evidence", self.agent),
                       patch.object(workflow, "feedback_secret", return_value="isolated-test-key")):
            target.start()
            self.addCleanup(target.stop)

    def run_case(self, publish=False, always_mcp=False, **changes):
        self.last_id = uuid4().hex
        state = {"id": self.last_id, "issue_key": "SAC-999", "actor": "tester",
                 "publish_requested": publish, "local_only": False,
                 "settings": {**DEFAULTS, "rovo_mcp_always_search": always_mcp,
                              "publish_enabled": publish, "dry_run": not publish,
                              "public_base_url": "https://falloutassist.example"}, **changes}
        return workflow.GRAPH.invoke(state, config={"recursion_limit": 40})

    def empty_local(self):
        self.jira.search_local_history.return_value["candidates"] = []

    def remote(self, candidates):
        self.jira.search_rovo_candidates.return_value = {"candidates": candidates, "status": "success"}

    def test_default_fetches_top_k_from_both_sources_and_reranks_once(self):
        self.assertTrue(DEFAULTS["rovo_mcp_always_search"])
        self.remote([source("SAC-200", retrieval_origins=["rovo"])])
        result = self.run_case(always_mcp=DEFAULTS["rovo_mcp_always_search"])
        self.assertEqual(result["tier"], "high")
        self.jira.search_local_history.assert_called_once()
        self.jira.search_rovo_candidates.assert_called_once()
        self.assertEqual(self.jira.search_local_history.call_args.args[1], DEFAULTS["retrieval_top_k"])
        self.assertEqual(self.jira.search_rovo_candidates.call_args.args[1], DEFAULTS["retrieval_top_k"])
        self.rerank.assert_called_once()
        self.agent.assert_not_called()
        self.assertEqual({c["key"] for c in result["pool_candidates"]}, {"SAC-100", "SAC-200"})
        events = {row["stage"]: row for row in result["retrieval_route"]}
        for stage in ("local_search", "rovo_mcp"):
            self.assertEqual(events[stage]["candidate_count"], 1)
            self.assertEqual(events[stage]["requested_top_k"], DEFAULTS["retrieval_top_k"])
        self.assertNotIn("local_reranker", events)

    def test_default_combines_and_deduplicates_even_with_strong_local_results(self):
        self.remote([source(retrieval_origins=["rovo"])])
        result = self.run_case(always_mcp=True)
        self.assertEqual(len(result["pool_candidates"]), 1)
        self.assertEqual(result["pool_candidates"][0]["retrieval_origins"], ["local", "rovo"])
        self.rerank.assert_called_once()
        self.agent.assert_not_called()

    def test_default_mcp_timeout_does_not_lower_strong_local_evidence(self):
        self.jira.search_rovo_candidates.side_effect = TimeoutError()
        result = self.run_case(always_mcp=True)
        self.assertEqual(result["tier"], "high")
        self.assertFalse(result["review_required"])
        self.assertEqual(result["search_status"]["mcp"], "timeout")
        self.agent.assert_not_called()

    def test_default_local_only_still_skips_mcp_and_agent(self):
        result = self.run_case(always_mcp=True, local_only=True)
        self.assertEqual(result["tier"], "high")
        self.jira.search_rovo_candidates.assert_not_called()
        self.agent.assert_not_called()

    def test_mcp_adapter_returns_at_most_top_k_usable_tickets(self):
        from falloutassist_next.plugins import JiraPlugin
        self.jira._search.return_value = [{"key": "SAC-100"}, {"key": "SAC-101"}]
        self.jira.fetch_resolution_candidate.side_effect = lambda key: source(key)
        with patch("falloutassist_next.rovo.semantic_issue_keys", return_value=["SAC-100", "SAC-101"]):
            result = JiraPlugin.search_rovo_candidates(self.jira, "failure", 1, set())
        self.assertEqual([c["key"] for c in result["candidates"]], ["SAC-100"])
        self.assertEqual(result["status"], "success")

    def test_medium_evidence_survives_agent_failures_and_stays_review_only(self):
        action = "Verify the account is active"
        self.jira.search_local_history.return_value["candidates"] = [source(comment=action)]
        for error, status in ((TimeoutError(), "timeout"), (RuntimeError("private-detail"), "error"),
                              (execution.StageUnavailable(), "unavailable")):
            with self.subTest(status=status):
                self.agent.side_effect = error
                result = self.run_case(publish=True, always_mcp=True)
                self.assertEqual(result["agent_status"], status)
                self.assertEqual(result["tier"], "medium")
                self.assertTrue(result["review_required"])
                self.assertIn(action, result["comment_body"])
                self.assertIn("Overall confidence: Medium", result["comment_body"])
                self.assertIn("Manual review required", result["comment_body"])
                self.assertNotIn("private-detail", str(result))
                self.jira.publish_comment.assert_not_called()
                from falloutassist_next.web import investigation
                saved = investigation(self.last_id, {"email": "tester"})
                self.assertEqual(saved["tier"], "medium")
                self.assertTrue(saved["review_required"])

    def test_medium_evidence_survives_mcp_and_agent_timeouts(self):
        self.jira.search_local_history.return_value["candidates"] = [source(comment="Verify the account is active")]
        self.jira.search_rovo_candidates.side_effect = TimeoutError()
        self.agent.side_effect = TimeoutError()
        result = self.run_case(publish=True, always_mcp=True)
        self.assertEqual(result["tier"], "medium")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["search_status"]["mcp"], "timeout")
        self.jira.publish_comment.assert_not_called()

    def test_medium_without_a_short_faithful_draft_keeps_its_evidence_rating(self):
        self.jira.search_local_history.return_value["candidates"] = [source(comment="Verify the account is active. " * 25)]
        self.agent.side_effect = TimeoutError()
        result = self.run_case(always_mcp=True)
        self.assertEqual(result["tier"], "medium")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["draft"]["steps"], [])
        self.assertIn("No actionable workaround", result["comment_body"])

    def test_medium_can_improve_to_high_with_better_supported_evidence(self):
        self.jira.search_local_history.return_value["candidates"] = [source(comment="Verify the account is active")]
        self.agent.side_effect = TimeoutError()
        self.assertEqual(self.run_case(always_mcp=True)["tier"], "medium")
        self.jira.search_local_history.return_value["candidates"] = [source()]
        self.agent.reset_mock()
        result = self.run_case(always_mcp=True)
        self.assertEqual(result["tier"], "high")
        self.assertFalse(result["review_required"])
        self.agent.assert_not_called()

    def test_medium_sources_do_not_allow_unsupported_supervisor_actions(self):
        self.jira.search_local_history.return_value["candidates"] = [source(comment="Verify the account is active")]
        self.supervisor.side_effect = lambda state: {"draft": proposal(text="Delete the customer account", quote="invented support")}
        result = self.run_case(publish=True, always_mcp=True)
        self.assertEqual(result["tier"], "low")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["draft"]["steps"], [])
        self.assertNotIn("Delete the customer account", result["comment_body"])
        self.jira.publish_comment.assert_not_called()

    def test_strong_local_skips_both_rovo_paths_and_publishes_only_after_gate(self):
        result = self.run_case(publish=True)
        self.assertEqual(result["tier"], "high")
        self.assertEqual(result["status"], "published")
        self.jira.search_rovo_candidates.assert_not_called()
        self.agent.assert_not_called()
        self.assertEqual(self.supervisor.call_count, 1)
        self.assertEqual(self.jira.publish_comment.call_count, 1)

    def test_technical_failure_is_distinct_from_empty_or_weak_matches(self):
        for outcome in ("empty", "weak", "error", "timeout", "unavailable"):
            with self.subTest(outcome=outcome):
                self.jira.search_local_history.side_effect = None
                self.jira.search_local_history.return_value["candidates"] = [source(test_relevance=.1)] if outcome == "weak" else []
                if outcome in {"error", "timeout", "unavailable"}:
                    self.jira.search_local_history.side_effect = {
                        "error": RuntimeError("credential=DO_NOT_EXPOSE"), "timeout": TimeoutError(),
                        "unavailable": execution.StageUnavailable()}[outcome]
                self.remote([source("SAC-200", score=0, retrieval_origins=["rovo"])])
                self.jira.search_rovo_candidates.reset_mock()
                result = self.run_case()
                self.jira.search_rovo_candidates.assert_called_once()
                self.assertEqual(result["tier"], "high")
                self.agent.assert_not_called()
                expected = outcome if outcome in {"error", "timeout", "unavailable"} else "success"
                self.assertEqual(result["search_status"]["local"], expected)
                self.assertNotIn("DO_NOT_EXPOSE", str(result))

    def test_combined_evidence_deduplicates_source_and_skips_agent(self):
        self.jira.search_local_history.return_value["candidates"] = [source(test_relevance=.1)]
        self.remote([source(test_relevance=.95, retrieval_origins=["rovo"])])
        result = self.run_case()
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["retrieval_origins"], ["local", "rovo"])
        self.assertEqual(result["tier"], "high")
        self.agent.assert_not_called()

    def test_weak_combined_evidence_invokes_agent_and_verifies_new_source(self):
        self.empty_local()
        self.agent.return_value = {"draft": proposal("SAC-200")}
        result = self.run_case(publish=True)
        self.agent.assert_called_once()
        self.jira.fetch_resolution_candidate.assert_called_once_with("SAC-200")
        self.assertEqual(result["tier"], "high")
        self.assertEqual(result["status"], "published")
        self.draft.assert_not_called()
        self.assertEqual(result["candidates"][0]["url"], "https://jira.example/browse/SAC-200")

    def test_agent_error_timeout_and_unavailable_end_in_review(self):
        self.empty_local()
        for error, expected in ((TimeoutError(), "timeout"), (RuntimeError("secret"), "error"),
                                (execution.StageUnavailable(), "unavailable")):
            with self.subTest(expected=expected):
                self.agent.side_effect = error
                result = self.run_case(publish=True)
                self.assertEqual(result["agent_status"], expected)
                self.assertTrue(result["review_required"])
                self.assertEqual(result["tier"], "low")
                self.jira.publish_comment.assert_not_called()

    def test_conflicting_procedures_still_require_review_after_agent(self):
        self.jira.search_local_history.return_value["candidates"] = [source(), source("SAC-101", comment="Reset the insurance account status in Salesforce")]
        self.agent.return_value = {"draft": proposal()}
        result = self.run_case(publish=True)
        self.agent.assert_called_once()
        self.assertTrue(result["routing_assessment"]["conflicting"])
        self.assertEqual(result["tier"], "low")
        self.jira.publish_comment.assert_not_called()

    def test_agent_cannot_invent_a_quote_or_change_source_conditions(self):
        self.empty_local()
        self.agent.return_value = {"draft": proposal("SAC-200", text="Delete the customer account", quote="Delete the customer account")}
        result = self.run_case(publish=True)
        self.assertEqual(result["tier"], "low")
        self.assertTrue(result["review_required"])
        self.assertTrue(result["supervisor_result"]["draft"]["decline"])
        self.assertEqual(result["supervisor_result"]["draft"]["steps"], [])
        self.jira.publish_comment.assert_not_called()

    def test_agent_unknown_project_or_current_ticket_cannot_be_a_source(self):
        self.empty_local()
        for key in ("OTHER-123", "SAC-999"):
            with self.subTest(key=key):
                self.agent.return_value = {"draft": proposal(key)}
                result = self.run_case(publish=True)
                self.assertTrue(result["review_required"])
                self.jira.fetch_resolution_candidate.assert_not_called()
                self.jira.publish_comment.assert_not_called()

    def test_mcp_timeout_keeps_useful_local_candidates(self):
        self.jira.search_local_history.return_value["candidates"] = [source(test_relevance=.3)]
        self.jira.search_rovo_candidates.side_effect = TimeoutError()
        result = self.run_case()
        self.assertEqual(result["pool_candidates"][0]["key"], "SAC-100")
        self.assertEqual(result["search_status"]["mcp"], "timeout")
        self.agent.assert_called_once()

    def test_failed_reranker_defaults_do_not_authorize_a_route(self):
        self.rerank.side_effect = TimeoutError()
        result = self.run_case(publish=True)
        self.jira.search_rovo_candidates.assert_called_once()
        self.agent.assert_called_once()
        self.assertEqual(result["tier"], "low")
        self.jira.publish_comment.assert_not_called()

    def test_local_only_preserves_explicit_opt_out(self):
        self.empty_local()
        result = self.run_case(local_only=True)
        self.assertTrue(result["review_required"])
        self.jira.search_rovo_candidates.assert_not_called()
        self.agent.assert_not_called()

    def test_approved_rule_does_not_need_rovo_or_a_model_relevance_score(self):
        self.empty_local()
        self.jira.approved_rule.return_value = {"id": "approved", "workaround_text": ACTION}
        result = self.run_case()
        self.assertEqual(result["tier"], "high")
        self.rerank.assert_not_called()
        self.jira.search_rovo_candidates.assert_not_called()

    def test_expired_budget_starts_no_external_operations(self):
        result = self.run_case(deadline=monotonic() - 1)
        self.assertTrue(result["review_required"])
        self.jira.issue.assert_not_called()
        self.jira.search_local_history.assert_not_called()
        self.jira.search_rovo_candidates.assert_not_called()
        self.agent.assert_not_called()

    def test_route_explanation_and_source_urls_survive_persistence(self):
        self.run_case()
        from falloutassist_next.web import investigation
        result = investigation(self.last_id, {"email": "tester"})
        self.assertEqual(result["settings"]["retrieval_route"][-1]["status"], "sufficient")
        self.assertEqual(result["settings"]["source_details"][0]["url"], "https://jira.example/browse/SAC-100")
        self.assertEqual(result["retrieval_route"], result["settings"]["retrieval_route"])

    def test_gate_clears_unsupported_supervisor_actions_even_after_strong_retrieval(self):
        self.supervisor.side_effect = lambda state: {"draft": proposal(text="Delete the customer account", quote="invented support")}
        result = self.run_case(publish=True)
        self.assertTrue(result["review_required"])
        self.assertEqual(result["draft"]["steps"], [])
        self.assertNotIn("Delete the customer account", result["comment_body"])
        self.jira.publish_comment.assert_not_called()

    def test_knowledge_package_preserves_route_flags_without_changing_its_graph(self):
        from falloutassist_next.knowledge_router import _assemble_node, _build_graph
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).isoformat()
        local = {"tier": "low", "review_required": True, "retrieval_route": [{"stage": "local_search", "status": "empty"}],
                 "source_details": [{"key": "SAC-100", "url": "https://jira.example/browse/SAC-100"}]}
        package = _assemble_node({"falloutassist": {"status": "completed", "elapsed_ms": 1, "result": local},
            "rovo_a2a": {"status": "unavailable", "elapsed_ms": 1}, "job_id": "test", "issue_key": "SAC-999",
            "job_started": monotonic(), "created_at": stamp, "started_at": stamp})["package"]
        self.assertTrue(package["proposals"]["falloutassist"]["review_required"])
        self.assertEqual(package["proposals"]["falloutassist"]["retrieval_route"], local["retrieval_route"])
        self.assertIn("rovo_a2a", _build_graph().get_graph().nodes)

    def test_configurable_two_source_rule_and_threshold_validation(self):
        self.empty_local()
        self.remote([source("SAC-100"), source("SAC-101")])
        result = self.run_case(settings={**DEFAULTS, "routing_min_sources": 2})
        self.assertEqual(result["tier"], "high")
        self.agent.assert_not_called()
        for key, value in (("routing_min_sources", 0), ("routing_min_llm_relevance", 1.1),
                           ("rovo_agent_budget_seconds", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validated_settings({key: value})


class DeadlineTests(unittest.TestCase):
    def test_abandoned_read_cannot_hold_up_caller(self):
        release = Event()
        started = monotonic()
        try:
            with self.assertRaises(TimeoutError):
                execution.call(lambda: release.wait(1), .02)
            self.assertLess(monotonic() - started, .25)
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
