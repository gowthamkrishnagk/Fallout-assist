"""Rovo-only search scope, real issue identity, and resolution selection."""
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from falloutassist_next import rovo_history as history

STEP = "Cancel Addon WITH PRORATION in Aria"
ERROR = "14047 | Invalid client plan instance identifier"
ISSUE = {"key": "SAC-1", "step": STEP, "error": ERROR,
         "query_text": "Order: private-order-id\nFailed Step: " + STEP + "\nError Description: " + ERROR}


def result(payload):
    return SimpleNamespace(structuredContent=payload, content=[SimpleNamespace(text=json.dumps(payload))],
                           isError=False)


def source(key, step=STEP, error=ERROR):
    return {"key": key, "step": step, "error": error, "source_tier": "high",
            "comment": "Update the correct Insurance OLI Id and retry.",
            "url": "https://jira.example/browse/" + key}


class RovoHistoryTests(unittest.TestCase):
    def test_dashboards_and_references_are_never_issue_matches(self):
        dashboards = {"results": [{"type": "dashboard", "title": "5G QA-70905 PI-4",
                                    "url": "https://jira.example/jira/dashboards/123"}]}
        self.assertEqual(history.issue_keys(result(dashboards)), [])
        mixed = {"results": dashboards["results"] + [{"type": "issue", "issueKey": "SAC-2",
                    "snippet": "Compare QA-70905", "url": "https://jira.example/browse/SAC-2"}]}
        self.assertEqual(history.issue_keys(result(mixed)), ["SAC-2"])
        plain = SimpleNamespace(structuredContent=None,
                                content=[SimpleNamespace(text="QA-70905 dashboard, not a ticket result")])
        self.assertEqual(history.issue_keys(plain), [])

    def test_query_uses_failure_signature_and_jql_enforces_requested_scope(self):
        query = history.semantic_query(ISSUE)
        self.assertIn(STEP, query)
        self.assertIn(ERROR, query)
        self.assertNotIn("private-order-id", query)
        jql = history.exact_jql(ISSUE)
        self.assertIn("project = SAC AND status in (Resolved, Closed)", jql)
        self.assertIn("key != SAC-1", jql)
        self.assertIn('text ~ "14047"', jql)
        self.assertNotIn("WITHOUT PRORATION", jql)

    def test_query_and_error_match_ignore_per_ticket_phone_and_salesforce_ids(self):
        current = {"key": "SAC-100", "step": "Nokia Modify(Change MSISDN) Line Prepaid",
                   "error": "Mobile Voice Service 13402048777 does not exist for customer 001Pl00001ur69o."}
        other_customer = "Mobile Voice Service 19392297752 does not exist for customer 001Pl00001rmxpz."
        query = history.semantic_query(current)
        jql = history.exact_jql(current)
        self.assertIn("Mobile Voice Service does not exist for customer", query)
        self.assertNotIn("13402048777", query)
        self.assertNotIn("001Pl00001ur69o", query)
        self.assertIn('summary ~ "Nokia Modify(Change MSISDN) Line Prepaid"', jql)
        self.assertIn('text ~ "Mobile Voice Service does not exist"', jql)
        self.assertNotIn("13402048777", jql)
        self.assertNotIn("001Pl00001ur69o", jql)
        self.assertEqual(history._normalized(history.stable_error(current["error"])),
                         history._normalized(history.stable_error(other_customer)))

        jira = MagicMock()
        jira._search.return_value = [{"key": "SAC-101"}, {"key": "SAC-102"}]
        jira.fetch_resolution_candidate.side_effect = [
            source("SAC-101", step="Nokia Modify(Change MSISDN) Line PortIn", error=other_customer),
            source("SAC-102", step=current["step"], error=other_customer),
        ]
        details = {"mode": "manual_keys"}
        selected = history.select_sources(jira, current, ["SAC-101", "SAC-102"], details)
        self.assertEqual([item["key"] for item in selected], ["SAC-102"])
        self.assertEqual(details["rejected"]["different_step"], 1)

    def test_short_error_codes_are_not_removed_as_customer_ids(self):
        self.assertEqual(history.stable_error("14047 | Invalid client plan instance identifier"),
                         "14047 | Invalid client plan instance identifier")

    def test_selection_checks_comments_beyond_first_five_keys(self):
        jira = MagicMock()
        keys = [f"SAC-{n}" for n in range(2, 12)]
        jira._search.return_value = [{"key": key} for key in keys]
        jira.fetch_resolution_candidate.side_effect = lambda key: None if int(key[4:]) < 7 else source(key)
        details = {"mode": "manual_keys"}
        selected = history.select_sources(jira, ISSUE, keys, details)
        self.assertEqual([s["key"] for s in selected], keys[5:])
        self.assertEqual(details["rejected"]["no_resolution"], 5)
        self.assertEqual(details["selected_sources"], 5)

    def test_with_without_proration_change_plan_and_other_errors_do_not_mix(self):
        jira = MagicMock()
        candidates = {"SAC-2": source("SAC-2", step=STEP.replace("WITH", "WITHOUT")),
                      "SAC-3": source("SAC-3", step=STEP + " (Change Plan)"),
                      "SAC-4": source("SAC-4", error="14048 | Different error"),
                      "SAC-5": source("SAC-5")}
        jira._search.return_value = [{"key": key} for key in candidates]
        jira.fetch_resolution_candidate.side_effect = candidates.get
        details = {"mode": "manual_keys"}
        selected = history.select_sources(jira, ISSUE, list(candidates), details)
        self.assertEqual([s["key"] for s in selected], ["SAC-3", "SAC-5"])
        self.assertEqual(details["rejected"]["different_step"], 1)
        self.assertEqual(details["rejected"]["different_error"], 1)

    def test_scope_is_validated_before_hydrating_or_accepting_manual_keys(self):
        jira = MagicMock()
        # SAC-2 is open and the Jira scope query excludes it.
        jira._search.return_value = [{"key": "SAC-3"}]
        jira.fetch_resolution_candidate.return_value = source("SAC-3")
        details = {"mode": "manual_keys"}
        selected = history.select_sources(jira, ISSUE, ["QA-2", "SAC-1", "SAC-2", "SAC-3"], details)
        self.assertEqual([s["key"] for s in selected], ["SAC-3"])
        jira.fetch_resolution_candidate.assert_called_once_with("SAC-3")
        self.assertIn("project = SAC AND status in (Resolved, Closed)", jira._search.call_args.args[0])

    def test_rovo_preview_uses_both_mcp_tools_and_filters_unrelated_projects(self):
        session = MagicMock()
        session.initialize = AsyncMock()
        session.call_tool = AsyncMock(side_effect=[result({"issues": [{"key": "SAC-2"}]}),
            result({"results": [{"type": "issue", "issueKey": "QA-123"},
                                 {"type": "issue", "issueKey": "SAC-3"}]})])

        @asynccontextmanager
        async def transport(*args, **kwargs):
            yield (None, None, None)

        @asynccontextmanager
        async def client(*args, **kwargs):
            yield session

        with patch.object(history, "streamablehttp_client", transport), \
             patch.object(history, "ClientSession", client), \
             patch.object(history, "required_env", return_value="test-site"), \
             patch.object(history, "_auth_header", return_value="Basic test-only"):
            found = history.search(ISSUE)
        self.assertEqual(found["keys"], ["SAC-2", "SAC-3"])
        calls = session.call_tool.call_args_list
        self.assertEqual([call.args[0] for call in calls], ["searchJiraIssuesUsingJql", "search"])
        self.assertEqual(calls[1].args[1]["limit"], 30)
        self.assertNotIn("private-order-id", calls[1].args[1]["query"])

    def test_search_merges_scoped_jira_jql_results_with_rovo_candidates(self):
        jira = MagicMock()
        jira._search.return_value = [{"key": "SAC-4"}, {"key": "SAC-2"}]
        mcp_result = {"keys": ["SAC-2", "SAC-3"], "retrieval": {"warnings": []}}
        with patch.object(history, "_search", new=AsyncMock(return_value=mcp_result)):
            found = history.search(ISSUE, jira=jira)
        self.assertEqual(found["keys"], ["SAC-4", "SAC-2", "SAC-3"])
        self.assertEqual(found["retrieval"]["jira_exact_candidates"], 2)
        jira._search.assert_called_once_with(history.exact_jql(ISSUE), "status", history.SEARCH_LIMIT)


if __name__ == "__main__":
    unittest.main()
