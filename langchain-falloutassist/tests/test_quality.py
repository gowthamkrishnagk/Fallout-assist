"""Regression checks for suggestion evidence and identical Jira source policies.

All Jira/model/index operations are mocked; persistence uses a temporary database.
"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from falloutassist_next import plugins, quality, storage, workflow
from falloutassist_next.agent import DraftResponse
from falloutassist_next.config import DEFAULTS

STEP = "Suspend insurance"
ERROR = "HTTP 409: account exists"
ACTION = "Update the enrollment ID in Salesforce"
BODY = ACTION + "; Retry the failed step."


def source(key="SAC-100", **changes):
    return {"key": key, "type": "ticket", "step": STEP, "error": ERROR,
            "score": 0.95, "retrieval_score": 0.95, "comment": BODY,
            "url": "https://jira.invalid/browse/" + key, **changes}


def draft(key="SAC-100", text=ACTION, quote=ACTION, **changes):
    return {"decline": False, "steps": [{"source_key": key, "text": text,
                                          "source_excerpt": quote, **changes}]}


def assess(candidates, proposed, **context):
    return quality.proposal_quality(None, candidates, proposed, 0.7,
                                    {item["key"]: "high" for item in candidates},
                                    STEP, ERROR, context)


class EvidenceTests(unittest.TestCase):
    def test_shared_code_does_not_hide_opposite_or_unrelated_errors(self):
        for error in ("HTTP 409: account does not exist", "HTTP 409: invalid plan status",
                      "HTTP 500: account exists"):
            with self.subTest(error=error):
                self.assertFalse(quality.same_failure(STEP, ERROR, source(error=error)))

    def test_operation_and_order_context_conflicts_cannot_pass(self):
        self.assertFalse(quality.same_failure(STEP, ERROR, source(step="Resume insurance")))
        for name in ("order_type", "order_reason", "release"):
            with self.subTest(name=name):
                self.assertFalse(quality.same_failure(STEP, ERROR, source(**{name: "Old"}), {name: "New"}))
                self.assertFalse(quality.same_failure(STEP, ERROR, source(**{name: "Old"}), {}))
                self.assertTrue(quality.same_failure(STEP, ERROR, source(**{name: "Same"}), {name: "Same"}))

    def test_meaningful_numbers_are_preserved_but_labelled_instance_ids_can_differ(self):
        self.assertFalse(quality.same_failure(STEP, "Invalid product 12345678",
                                              source(error="Invalid product 87654321")))
        self.assertTrue(quality.same_failure(STEP, "Invalid status OrderId=12345678",
                                             source(error="Invalid status OrderId=87654321")))

    def test_numeric_signs_and_comparison_operators_are_preserved(self):
        for first, second in (("HTTP 409: status=-1", "HTTP 409: status=1"),
                              ("HTTP 409: count < 2", "HTTP 409: count > 2")):
            with self.subTest(first=first):
                self.assertFalse(quality.same_failure(STEP, first, source(error=second)))
        self.assertFalse(quality.step_is_supported("Set plan status to 1", "Set plan status to -1"))

    def test_known_order_type_aliases_do_not_create_false_conflicts(self):
        self.assertTrue(quality.same_failure(STEP, ERROR, source(order_type="Change order"),
                                             {"order_type": "Change"}))

    def test_supporting_excerpt_must_preserve_technical_case(self):
        self.assertFalse(quality.evidence_quote_is_present("Update productId in Salesforce",
                                                           "Update ProductID in Salesforce"))
        self.assertFalse(quality.step_is_supported("Update productId in Salesforce",
                                                   "Update ProductID in Salesforce"))

    def test_negation_conditions_objects_and_values_cannot_be_dropped_or_changed(self):
        pairs = [("Delete the account", "Do not delete the account"),
                 ("Retry the failed step", "Retry the failed step without proration"),
                 ("Delete the account", "Delete the test account"),
                 ("Update plan status to active", "Update plan status to suspended"),
                 ("Change status from active to suspended", "Change status from suspended to active"),
                 ("Update the plan status", "If the account is suspended, update the plan status"),
                 ("Delete the account", "Retry the Delete Account step")]
        for action, quote in pairs:
            with self.subTest(action=action, quote=quote):
                self.assertFalse(quality.step_is_supported(action, quote))

    def test_tense_articles_and_faithful_qualified_actions_are_supported(self):
        self.assertTrue(quality.step_is_supported(ACTION, "Updated the enrollment ID in Salesforce"))
        instruction = "Retry the failed step without proration"
        self.assertTrue(quality.step_is_supported(instruction, instruction))

    def test_cherry_picked_excerpt_cannot_hide_full_resolution_condition(self):
        candidate = source(comment="Do not delete the account. " + BODY)
        self.assertNotEqual(assess([candidate], draft())[0], "high")

    def test_action_must_be_in_solution_not_an_unrelated_table_field(self):
        body = "|Cause|" + ACTION + "|\n|Solution applied|Reset the enrollment status in Salesforce|"
        self.assertNotEqual(assess([source(comment=body)], draft())[0], "high")

    def test_every_primary_and_additional_source_must_match_failure(self):
        a, b = source(), source("SAC-101", error="HTTP 500: internal failure")
        self.assertNotEqual(assess([a, b], draft("SAC-101"))[0], "high")
        support = {"source_key": b["key"], "source_excerpt": ACTION}
        self.assertNotEqual(assess([a, b], draft(supporting_sources=[support]))[0], "high")

    def test_every_additional_source_must_support_the_same_action(self):
        b = source("SAC-101", comment="Delete the enrollment ID in Salesforce")
        support = {"source_key": b["key"], "source_excerpt": b["comment"]}
        self.assertNotEqual(assess([source(), b], draft(supporting_sources=[support]))[0], "high")

    def test_each_cited_source_needs_threshold_feedback_and_outcome_eligibility(self):
        for changes in ({"score": 0.6}, {"score": float("nan")}, {"feedback_net": -1},
                        {"automated_failures": 2, "automated_successes": 1}):
            with self.subTest(changes=changes):
                self.assertNotEqual(assess([source(**changes)], draft())[0], "high")

    def test_invalid_first_candidate_does_not_block_valid_alternative(self):
        bad, good = source(error="HTTP 500: internal failure"), source("SAC-101")
        self.assertEqual(assess([bad, good], draft(good["key"]))[0], "high")

    def test_stale_high_tier_cannot_override_a_placeholder_resolution(self):
        self.assertNotEqual(assess([source(comment="1. Completed")], draft())[0], "high")

    def test_action_sequence_must_follow_documented_order(self):
        proposed = draft(text="Retry the failed step", quote="Retry the failed step")
        proposed["steps"].append(draft()["steps"][0])
        self.assertNotEqual(assess([source()], proposed)[0], "high")
        self.assertFalse(quality.step_is_supported("Retry the failed step; " + ACTION, BODY))

    def test_retry_cannot_omit_the_documented_correction_prerequisite(self):
        self.assertNotEqual(assess([source()], draft(text="Retry the failed step",
                                                     quote="Retry the failed step"))[0], "high")

    def test_supervisor_cannot_silently_remove_retained_action_support(self):
        support = {"source_key": "SAC-101", "source_excerpt": ACTION}
        revised = quality.preserve_supporting_sources(draft(supporting_sources=[support]), draft())
        self.assertEqual(revised["steps"][0]["supporting_sources"], [support])
        candidates = [source(), source("SAC-101", error="HTTP 500: internal failure")]
        self.assertNotEqual(assess(candidates, revised)[0], "high")

    def test_supervisor_schema_retains_additional_source_citations(self):
        proposed = {**draft(supporting_sources=[{"source_key": "SAC-101", "source_excerpt": ACTION}]),
                    "reason": "Historical evidence"}
        self.assertEqual(DraftResponse.model_validate(proposed).model_dump()["steps"][0]["supporting_sources"],
                         proposed["steps"][0]["supporting_sources"])

    def test_same_ticket_found_by_both_paths_can_support_a_faithful_procedure(self):
        proposed = draft(text=BODY, quote=BODY,
                         supporting_sources=[{"source_key": "SAC-100", "source_excerpt": BODY}])
        self.assertEqual(assess([source()], proposed)[0], "high")

    def test_approved_rule_draft_must_preserve_approved_action(self):
        rule = {"id": "one", "workaround_text": ACTION}
        self.assertEqual(quality.proposal_quality(rule, [], draft("RULE:one"), .7, {})[0], "high")
        self.assertNotEqual(quality.proposal_quality(rule, [], draft("RULE:one", text="Delete account"), .7, {})[0], "high")


class SourceQualityTests(unittest.TestCase):
    def test_administrative_numbering_cannot_promote_placeholder_solution(self):
        body = "1. Check ticket closure status\n|Solution applied|Completed the step|"
        self.assertEqual(quality.source_quality("ticket", source(), body)[0], "low")

    def test_clear_table_or_plain_solution_does_not_need_numbering(self):
        for body in (BODY, "|Cause|Duplicate enrollment|\n|Solution applied|" + BODY + "|"):
            with self.subTest(body=body):
                tier, reasons = quality.source_quality("ticket", source(), body)
                self.assertEqual(tier, "high")
                self.assertIn("Automatically assessed", reasons[0])
                self.assertNotIn("reviewed", " ".join(reasons))

    def test_multiline_solution_and_customer_action_are_retained(self):
        body = "|Solution applied|1. " + ACTION + "\n2. Retry the failed step|\n|Customer action|Verify successful completion|"
        self.assertIn("2. Retry", quality.resolution_text(body))
        self.assertIn("Verify successful completion", quality.resolution_evidence(body))
        self.assertEqual(quality.source_quality("ticket", source(), body)[0], "high")

    def test_empty_solution_does_not_fall_back_to_other_actions(self):
        body = "|Solution applied| |\n|Customer action|Create a new ticket|"
        self.assertEqual(quality.resolution_text(body), "")
        self.assertEqual(quality.source_quality("ticket", source(), body)[0], "low")

    def test_verification_only_is_not_a_high_quality_corrective_procedure(self):
        self.assertNotEqual(quality.source_quality("ticket", source(), "Verify the account is active")[0], "high")

    def test_structured_fix_markers_are_not_part_of_the_action_evidence(self):
        body = "=== FIX ===\n1. " + ACTION + "\n2. Retry the failed step\n=== END ==="
        self.assertEqual(quality.source_quality("ticket", source(), body)[0], "high")
        self.assertNotIn("===", quality.resolution_text(body))


class WorkflowSelectionTests(unittest.TestCase):
    def test_eligible_candidate_is_selected_before_review_pool_is_cut(self):
        bad, good = source(error="HTTP 500: internal failure"), source("SAC-101", retrieval_score=.7)
        state = {"candidates": [bad, good], "source_tiers": {bad["key"]: "high", good["key"]: "high"},
                 "history": {"query_step": STEP, "query_error": ERROR},
                 "settings": {**DEFAULTS, "llm_rerank_enabled": False, "jira_review_top_k": 1},
                 "issue": {"query_text": f"Failed Step: {STEP}\nError: {ERROR}"}}
        with patch.object(storage, "feedback_stats", return_value={"up": 0, "down": 0}):
            result = workflow._rerank(state)
        self.assertEqual([item["key"] for item in result["candidates"]], [good["key"]])
        self.assertLess(result["rerank"][0]["composite_score"], result["rerank"][1]["composite_score"])
        self.assertEqual(result["rerank"][0]["eligibility_tier"], "high")

    def test_comment_attributes_the_actual_action_source(self):
        a, b = source(), source("SAC-101", workaround_version={
            "id": "version-2", "version_number": 2, "releases": ["R3"]})
        state = {"candidates": [a, b], "source_tiers": {a["key"]: "high", b["key"]: "high"},
                 "history": {"query_step": STEP, "query_error": ERROR}, "draft": draft(b["key"]),
                 "settings": dict(DEFAULTS),
                 "issue": {"key": "SAC-200", "query_text": f"Failed Step: {STEP}\nError: {ERROR}"}}
        state.update(workflow._gate(state))
        self.assertEqual(state["tier"], "high")
        with patch.object(workflow, "feedback_secret", return_value=""):
            body = workflow._comment(state)["comment_body"]
        self.assertIn("Source: SAC-101 (", body)
        self.assertIn("Resolution version: v2 (R3)", body)
        self.assertNotIn("Source: SAC-100 (", body)

    def test_medium_suggestion_also_shows_resolution_version(self):
        candidate = source(workaround_version={
            "id": "version-3", "version_number": 3, "releases": []})
        state = {"candidates": [candidate], "settings": dict(DEFAULTS),
                 "issue": {"key": "SAC-200"}, "tier": "medium",
                 "review_required": True, "draft": draft(),
                 "supervisor_result": {"confidence": "medium", "draft": draft()}}
        with patch.object(workflow, "feedback_secret", return_value=""):
            body = workflow._comment(state)["comment_body"]
        self.assertIn("Resolution version: v3 (release not recorded)", body)


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        connect = storage.connect
        path = Path(directory.name) / "quality.sqlite3"
        for mock in (patch.object(storage, "database_backend", return_value="sqlite"),
                     patch.object(storage, "connect", side_effect=lambda: connect(path)),
                     patch.object(plugins, "required_env", return_value="project=SAC"),
                     patch.object(plugins.JiraPlugin, "base_url", return_value="https://jira.invalid")):
            mock.start()
            self.addCleanup(mock.stop)
        storage.initialize()
        self.plugin = plugins.JiraPlugin()
        self.good = {"id": "good", "body": "|Solution applied|" + BODY + "|\n|Order Type Failed|Change order|",
                     "created": "2026-10-01T10:00:00Z", "author": {"accountId": "engineer"}}
        self.issue = {"key": "SAC-100", "fields": {
            "summary": "Failure", "description": f"Failed Step: {STEP}\nError: {ERROR}",
            "updated": "2026-10-02T10:00:00Z", "resolutiondate": "2026-10-01T12:00:00Z",
            "comment": {"comments": [self.good], "total": 1}}}

    def ingest(self):
        with patch.object(self.plugin, "_iter_search", return_value=iter([copy.deepcopy(self.issue)])), \
                patch.object(plugins, "HistoryIndex") as index:
            result = self.plugin.ingest()
        return result, index

    def test_both_paths_choose_same_fix_and_ignore_activity_placeholder_and_late_edits(self):
        rejected = ["This will now be automatically closed. Create a new ticket.",
                    "Update from Salesforce: PR_Error_Description__c: Update plan status",
                    "|Solution applied|Completed the step|", "FA-SUGGESTION " + BODY]
        comments = [self.good, *[{"id": str(i), "body": body, "created": "2026-10-02T10:00:00Z"}
                                for i, body in enumerate(rejected)],
                    {"id": "late", "body": "Delete the account in Salesforce", "created": "2026-10-01T10:00:00Z",
                     "updated": "2026-10-20T10:00:00Z"}]
        self.issue["fields"]["comment"] = {"comments": comments, "total": len(comments)}
        self.ingest()
        with patch.object(self.plugin, "_request", return_value=self.issue):
            rovo = self.plugin.fetch_resolution_candidate("SAC-100")
        local = storage.source_document("SAC-100")
        self.assertEqual(rovo["comment"], local["body"])
        self.assertEqual(rovo["workaround_text"], BODY)
        self.assertEqual(rovo["order_type"], "Change order")
        self.assertEqual(rovo["order_type"], local["metadata"]["order_type"])
        self.assertEqual(rovo["source_tier"], storage.source_tier("ticket", "SAC-100"))

    def test_assignee_free_form_resolution_is_used_when_no_labelled_resolution_exists(self):
        fields = copy.deepcopy(self.issue["fields"])
        fields["assignee"] = {"accountId": "resolver"}
        fields["status"] = {"name": "Resolved"}
        fields["comment"] = {"comments": [
            {"id": "other", "body": BODY, "author": {"accountId": "other"},
             "created": "2026-10-01T10:00:00Z"},
            {"id": "assignee", "body": "Update the enrollment ID in Salesforce; retry the failed step.",
             "author": {"accountId": "resolver"}, "created": "2026-10-01T11:00:00Z"},
        ]}
        selected = plugins._select_resolution_comment(
            fields, fields["comment"]["comments"], STEP, ERROR)
        self.assertEqual(selected["comment_id"], "assignee")
        self.assertEqual(selected["selection_reason"], "assignee_authored_fallback")
        self.assertEqual(selected["tier"], "high")

    def test_cancelled_ticket_resolution_comment_is_excluded(self):
        issue = copy.deepcopy(self.issue)
        issue["fields"]["status"] = {"name": "Cancelled"}
        with patch.object(self.plugin, "_request", return_value=issue):
            self.assertIsNone(self.plugin.fetch_resolution_candidate("SAC-100"))
        selected = plugins._select_resolution_comment(
            issue["fields"], [self.good], STEP, ERROR)
        self.assertEqual(selected["selection_reason"], "excluded_cancelled_ticket")

    def test_newer_strong_assignee_comment_supersedes_older_labelled_resolution(self):
        fields = copy.deepcopy(self.issue["fields"])
        fields["assignee"] = {"accountId": "resolver"}
        fields["status"] = {"name": "Closed"}
        comments = [
            {"id": "assignee", "body": BODY, "author": {"accountId": "resolver"},
             "created": "2026-10-01T11:00:00Z"},
            {"id": "labelled", "body": "|Solution applied|" + BODY + "|",
             "author": {"accountId": "other"}, "created": "2026-10-01T10:00:00Z"},
        ]
        selected = plugins._select_resolution_comment(fields, comments, STEP, ERROR)
        self.assertEqual(selected["comment_id"], "assignee")
        self.assertEqual(selected["selection_reason"], "assignee_authored_fallback")
        self.assertEqual(selected["tier"], "high")

    def test_newer_strong_resolution_creates_current_version_and_supersedes_old(self):
        self.ingest()
        original = self.issue["fields"]["comment"]["comments"][0]
        updated = {**original, "id": "new-resolution",
                   "body": "|Solution applied|Reset the enrollment ID in Salesforce; Retry the failed step.|",
                   "created": "2026-10-03T10:00:00Z"}
        self.issue["fields"]["comment"] = {"comments": [original, updated], "total": 2}
        self.issue["fields"]["updated"] = "2026-10-04T10:00:00Z"
        self.ingest()
        rows = storage.workaround_rows(step=STEP, error=ERROR)
        versions = rows[0]["versions"]
        self.assertEqual(len(versions), 2)
        self.assertEqual({version["status"] for version in versions}, {"current", "superseded"})
        current = next(version for version in versions if version["status"] == "current")
        self.assertIn("Reset the enrollment ID", current["body"])

    def test_unchanged_issue_is_reassessed_when_policy_version_changes(self):
        self.ingest()
        old = storage.source_document("SAC-100")
        old["metadata"].pop("source_policy_version")
        old["content_hash"] = "pre-policy-content-hash"
        storage.upsert_document(old, "high", [])
        result, index = self.ingest()
        self.assertEqual(result["skipped"], 0)
        index.return_value.upsert_many.assert_called_once()
        self.assertEqual(storage.source_document("SAC-100")["metadata"]["source_policy_version"],
                         quality.SOURCE_POLICY_VERSION)

    def test_ingestion_counts_distinguish_new_unchanged_and_updated_tickets(self):
        first, _ = self.ingest()
        self.assertEqual((first['new_documents'], first['updated_documents'], first['removed_documents']), (1, 0, 0))
        again, _ = self.ingest()
        self.assertEqual((again['new_documents'], again['updated_documents'], again['skipped']), (0, 0, 1))
        self.issue['fields']['updated'] = '2026-10-03T10:00:00Z'
        self.issue['fields']['comment']['comments'][0]['body'] = '|Solution applied|Reset the enrollment status in Salesforce; Retry the failed step|'
        updated, _ = self.ingest()
        self.assertEqual((updated['new_documents'], updated['updated_documents'], updated['removed_documents']), (0, 1, 0))
        self.assertEqual(storage.latest_ingestion()['updated_documents'], 1)
        self.assertEqual(self.ingest()[0]["skipped"], 1)

    def test_a_successful_rovo_rejection_cannot_reintroduce_stale_local_evidence(self):
        local = {"candidates": [source()], "query_step": STEP, "query_error": ERROR}
        with patch.object(plugins, "config_value", return_value="rovo"), \
                patch.object(plugins.HistoryIndex, "search", return_value=local), \
                patch("falloutassist_next.rovo.semantic_issue_keys", return_value=["SAC-100"]), \
                patch.object(self.plugin, "fetch_resolution_candidate", return_value=None):
            self.assertEqual(self.plugin.search_history("query", 5, .7)["candidates"], [])

    def test_rovo_network_failure_still_allows_local_evidence_fallback(self):
        local = {"candidates": [source()], "query_step": STEP, "query_error": ERROR}
        with patch.object(plugins, "config_value", return_value="rovo"), \
                patch.object(plugins.HistoryIndex, "search", return_value=local), \
                patch("falloutassist_next.rovo.semantic_issue_keys", return_value=["SAC-100"]), \
                patch.object(self.plugin, "fetch_resolution_candidate", side_effect=TimeoutError):
            result = self.plugin.search_history("query", 5, .7)
        self.assertEqual(result["candidates"][0]["key"], "SAC-100")
        self.assertIn("rovo_fetch", result["retrieval_failures"])


if __name__ == "__main__":
    unittest.main()
