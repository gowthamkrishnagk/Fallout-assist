"""Structured fallback transport compatibility without live Atlassian calls."""

import json
import unittest
from time import monotonic
from unittest.mock import Mock, patch

import requests

from falloutassist_next import rovo, rovo_a2a
from falloutassist_next.retrieval_routing import failure_status


class AgentTransportTests(unittest.TestCase):
    def setUp(self):
        self.response = Mock(status_code=200, headers={"Content-Type": "text/event-stream"})
        self.draft = {"decline": False, "reason": "Evidence", "steps": [
            {"text": "Update the enrollment ID in Salesforce", "source_key": "SAC-100",
             "source_excerpt": "Update the enrollment ID in Salesforce"}]}
        self.issue = {"key": "SAC-999", "query_text": "Step: Suspend insurance\nError: account exists"}
        self.evidence = [{"key": "SAC-100", "url": "https://jira.example/browse/SAC-100",
                          "comment": "Update the enrollment ID in Salesforce"}]
        self.release = Mock()
        self.stream = Mock(return_value={"task_id": "task-1", "text": json.dumps(self.draft)})
        self.http = Mock(return_value=self.response)
        for target in (patch.object(rovo_a2a.storage, "integration_secret", return_value={"connected": True}),
                       patch.object(rovo_a2a.storage, "get_state", return_value="generation"),
                       patch.object(rovo_a2a.storage, "claim_lease", return_value=True),
                       patch.object(rovo_a2a.storage, "release_lease", self.release),
                       patch.object(rovo_a2a.storage, "save_integration_secret"),
                       patch.object(rovo_a2a, "_recover_preview_lease"),
                       patch.object(rovo_a2a, "_access", return_value=("https://a2a.atlassian.com/test", "secret-test-token")),
                       patch.object(rovo_a2a, "_stream", self.stream),
                       patch.object(rovo_a2a, "_http", self.http)):
            target.start()
            self.addCleanup(target.stop)

    def test_structured_fallback_supplies_ticket_and_evidence_without_saving_preview(self):
        result = rovo_a2a.investigate_evidence(self.issue, self.evidence, monotonic() + 1)
        self.assertEqual(result["draft"]["steps"][0]["source_key"], "SAC-100")
        self.assertNotIn("secret-test-token", str(result))
        prompt = self.http.call_args.kwargs["json"]["params"]["message"]["parts"][0]["text"]
        self.assertIn("Read-only", prompt)
        self.assertIn("SAC-100", prompt)
        self.assertIn("at most five new Jira keys", prompt)
        rovo_a2a.storage.save_integration_secret.assert_not_called()
        self.response.close.assert_called_once()
        self.release.assert_called_once()

    def test_free_form_or_timed_out_output_is_rejected_and_lease_is_released(self):
        for response in ("Here is a possible fix without structured citations", None):
            with self.subTest(response=response):
                self.release.reset_mock()
                self.stream.side_effect = rovo_a2a.A2ATimeout() if response is None else None
                self.stream.return_value = {"text": response}
                with self.assertRaises((ValueError, rovo_a2a.A2AError)):
                    rovo_a2a.investigate_evidence(self.issue, self.evidence, monotonic() + 1)
                self.release.assert_called_once()

    def test_missing_oauth_is_unavailable_and_does_not_contact_agent(self):
        with patch.object(rovo_a2a.storage, "integration_secret", return_value=None):
            with self.assertRaises(rovo_a2a.A2ABusyError) as caught:
                rovo_a2a.investigate_evidence(self.issue, [], monotonic() + 1)
        self.assertEqual(failure_status(caught.exception), "unavailable")
        self.http.assert_not_called()

    def test_connection_change_invalidates_completed_proposal(self):
        with patch.object(rovo_a2a.storage, "get_state", side_effect=["old", "new"]):
            with self.assertRaises(rovo_a2a.A2AError):
                rovo_a2a.investigate_evidence(self.issue, self.evidence, monotonic() + 1)
        self.release.assert_called_once()

    def test_mcp_typed_failures_preserve_timeout_vs_unavailable(self):
        import httpx
        self.assertEqual(failure_status(rovo._search_error(httpx.ReadTimeout("test"))), "timeout")
        self.assertEqual(failure_status(rovo._search_error(httpx.ConnectError("test"))), "unavailable")


if __name__ == "__main__":
    unittest.main()
