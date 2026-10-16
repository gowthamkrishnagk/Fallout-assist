"""Evidence, version isolation and cache checks; all model calls are mocked."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from falloutassist_next import storage, synthesis

FIX = ("Update the addon OLI ID in Salesforce to the correct client plan instance "
       "identifier; Retry the Cancel Addon WITHOUT PRORATION step in Aria")
COMMENT = ("| Field | Details |\n| Solution applied | " + FIX + " |\n"
           "| Customer action | Retry the order cancellation step and verify successful completion |")


def version():
    return {"id": "version-one", "body": FIX, "order_reason": "Change Plan",
            "failed_step": "Cancel Installments in Aria (postpaid Change Plan)",
            "error_description": "14047 | Invalid client plan instance identifier",
            "source_keys_json": '["SAC-249749"]', "releases_json": "[]",
            "source_comments": [{"source_key": "SAC-249749", "body": COMMENT,
                                 "url": "https://lla.atlassian.net/browse/SAC-249749"}]}


def draft(excerpt=FIX, source_key="SAC-249749"):
    return synthesis.ProcedureSynthesis(
        actionable=True, summary="Correct the addon OLI ID, then retry the documented Aria step.",
        steps=[synthesis.SynthesisStep(text="Update the addon OLI ID in Salesforce.",
                                       source_key=source_key, source_excerpt=excerpt)],
        applicability_note="The failed step is Cancel Installments; the source says Cancel Addon WITHOUT PRORATION.",
        missing_information=["The correct client plan instance identifier is not provided."],
    )


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self):
        return "isolated-scripted-chat"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class GroundingTests(unittest.TestCase):
    def test_multiple_structured_responses_can_be_repaired_once(self):
        item = draft().model_dump()
        duplicate = AIMessage(content="", tool_calls=[
            {"name": "ProcedureSynthesis", "args": item, "id": "one"},
            {"name": "ProcedureSynthesis", "args": item, "id": "two"}])
        single = AIMessage(content="", tool_calls=[
            {"name": "ProcedureSynthesis", "args": item, "id": "corrected"}])
        model = ScriptedChatModel(responses=[duplicate, single])
        with patch.object(synthesis, "chat_model", return_value=model):
            result = synthesis._structured_call(synthesis.ProcedureSynthesis, "test", {},
                                                {"temperature": 0.1}, [])
        self.assertEqual(result.model_dump(), item)
        self.assertEqual(model.calls, 2)

    def test_repeated_multiple_responses_do_not_retry_indefinitely(self):
        item = draft().model_dump()
        duplicate = AIMessage(content="", tool_calls=[
            {"name": "ProcedureSynthesis", "args": item, "id": "one"},
            {"name": "ProcedureSynthesis", "args": item, "id": "two"}])
        model = ScriptedChatModel(responses=[duplicate, duplicate, duplicate])
        with patch.object(synthesis, "chat_model", return_value=model):
            with self.assertRaises(synthesis.MultipleStructuredOutputsError):
                synthesis._structured_call(synthesis.ProcedureSynthesis, "test", {},
                                           {"temperature": 0.1}, [])
        self.assertEqual(model.calls, 2)

    def test_full_comment_preserves_customer_actions(self):
        evidence = synthesis.source_evidence(version())
        self.assertIn("verify successful completion", evidence[0]["body"])

    def test_old_version_cannot_use_newer_different_procedure(self):
        old = version()
        old["source_comments"][0]["body"] = "| Solution applied | Delete the subscription |"
        self.assertEqual(synthesis.source_evidence(old)[0]["body"], FIX)

    def test_unknown_source_or_fabricated_quote_is_rejected_before_review(self):
        for candidate in (draft("Delete all records"), draft(source_key="SAC-UNKNOWN")):
            with self.subTest(candidate=candidate), patch.object(
                    synthesis, "_structured_call", return_value=candidate) as model:
                with self.assertRaises(ValueError):
                    synthesis.synthesize(version(), {"temperature": 0.1})
                self.assertEqual(model.call_count, 1)

    def test_semantic_review_rejects_unsupported_action_despite_valid_quote(self):
        unsafe = draft()
        unsafe.steps[0].text = "Delete the subscription and recreate it."
        with patch.object(synthesis, "_structured_call", side_effect=[
            unsafe, synthesis.GroundingReview(supported=False, reason="Source does not document deletion")]):
            with self.assertRaises(ValueError):
                synthesis.synthesize(version(), {"temperature": 0.1})

    def test_model_receives_error_code_and_full_evidence(self):
        with patch.object(synthesis, "_structured_call", side_effect=[
            draft(), synthesis.GroundingReview(supported=True, reason="Supported")]) as model:
            result, _ = synthesis.synthesize(version(), {"temperature": 0.1})
        payload = model.call_args_list[0].args[2]
        self.assertTrue(payload["context"]["error_description"].startswith("14047"))
        self.assertIn("verify successful completion", payload["resolutions"]["SAC-249749"])
        self.assertEqual(result["status"], "ready")

    def test_step_name_difference_is_flagged_even_if_writer_omits_it(self):
        incomplete = draft()
        incomplete.applicability_note = ""
        with patch.object(synthesis, "_structured_call", side_effect=[
            incomplete, synthesis.GroundingReview(supported=True, reason="Supported")]):
            result, _ = synthesis.synthesize(version(), {"temperature": 0.1})
        note = result["synthesis"]["applicability_note"]
        self.assertIn("Cancel Installments in Aria (postpaid Change Plan)", note)
        self.assertIn("Cancel Addon WITHOUT PRORATION", note)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        original_connect = storage.connect
        path = Path(self.directory.name) / "isolated.sqlite3"
        self.patches = [patch.object(storage, "database_backend", return_value="sqlite"),
                        patch.object(storage, "connect", side_effect=lambda: original_connect(path)),
                        patch.object(synthesis, "config_value", return_value="test-model")]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        storage.initialize()

    def test_cache_reuses_same_evidence_and_invalidates_changed_procedure_or_model(self):
        with patch.object(synthesis, "synthesize", return_value=(
                {"status": "ready", "synthesis": draft().model_dump()}, None)) as model:
            original = version()
            self.assertFalse(synthesis.prepare_synthesis(original, "tester")["cached"])
            self.assertTrue(synthesis.prepare_synthesis(copy.deepcopy(original), "tester")["cached"])
            self.assertEqual(model.call_count, 1)
            # Ingestion timestamps alone must not incur another model call.
            original["updated_at"] = "tomorrow"
            self.assertTrue(synthesis.prepare_synthesis(original, "tester")["cached"])
            original["source_comments"][0]["body"] += "\n| Release | R2 |"
            self.assertFalse(synthesis.prepare_synthesis(original, "tester")["cached"])
            with patch.object(synthesis, "config_value", return_value="different-model"):
                self.assertFalse(synthesis.prepare_synthesis(original, "tester")["cached"])
            self.assertEqual(model.call_count, 3)

    def test_failure_does_not_cache_success_and_can_be_retried(self):
        with patch.object(synthesis, "synthesize", side_effect=RuntimeError("provider unavailable")):
            result = synthesis.prepare_synthesis(version(), "tester")
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(storage.lease_active("synthesis:version-one"))
        self.assertEqual(storage.get_state("workaround_synthesis:version-one"), "")
        with patch.object(synthesis, "synthesize", return_value=(
                {"status": "ready", "synthesis": draft().model_dump()}, None)):
            self.assertEqual(synthesis.prepare_synthesis(version(), "tester")["status"], "ready")

    def test_read_only_cache_lookup_never_starts_generation(self):
        with patch.object(synthesis, "synthesize") as model:
            self.assertIsNone(synthesis.cached_synthesis(version()))
            model.assert_not_called()
        with patch.object(synthesis, "synthesize", return_value=(
                {"status": "ready", "synthesis": draft().model_dump()}, None)):
            synthesis.prepare_synthesis(version(), "tester")
        with patch.object(synthesis, "synthesize") as model:
            self.assertTrue(synthesis.cached_synthesis(version())["cached"])
            changed = version()
            changed["source_comments"][0]["body"] += "\n| Release | R2 |"
            self.assertIsNone(synthesis.cached_synthesis(changed))
            model.assert_not_called()

    def test_original_comment_endpoint_does_not_wait_for_ai(self):
        item = {"source_key": "SAC-249749", "source_type": "ticket", "issue_updated": "2026-08-19",
                "summary": "Change plan", "step": version()["failed_step"],
                "error": version()["error_description"], "body": COMMENT, "url": "https://example.invalid",
                "content_hash": "sample", "order_reason": "Change Plan", "workaround_text": FIX}
        storage.upsert_document(item, "medium", [])
        saved = storage.record_workaround(item)
        with patch.dict(os.environ, {"SESSION_SECRET": "isolated-test-session-key"}):
            from fastapi.testclient import TestClient
            from falloutassist_next import web
        web.app.dependency_overrides[web._admin] = lambda: {"email": "tester"}
        try:
            with TestClient(web.app) as client, patch.object(synthesis, "prepare_synthesis") as generate:
                response = client.get(f"/api/workaround-versions/{saved['id']}")
                self.assertEqual(response.status_code, 200)
                self.assertIn("Customer action", response.json()["original_comments"][0]["body"])
                self.assertIsNone(response.json()["saved_synthesis"])
                generate.assert_not_called()
                self.assertEqual(client.get("/api/workaround-versions/missing").status_code, 404)
        finally:
            web.app.dependency_overrides.clear()

    def test_activity_logs_and_old_assistant_comments_never_call_llm(self):
        for text in ("Update from Salesforce: OrderStepId: a5o123 Execution Log: FATAL",
                     "Auto-suggested by FalloutAssist FA-SUGGESTION",
                     "This ticket will now be automatically closed"):
            with self.subTest(text=text), patch.object(synthesis, "synthesize") as model:
                item = version()
                item["body"] = text
                self.assertEqual(synthesis.prepare_synthesis(item, "tester")["status"], "no_action")
                model.assert_not_called()

    def test_duplicate_generation_is_bounded_by_lease(self):
        storage.claim_lease("synthesis:version-one", "another-request", 300)
        with patch.object(synthesis, "synthesize") as model:
            self.assertEqual(synthesis.prepare_synthesis(version(), "tester")["status"], "busy")
            model.assert_not_called()

    def test_version_lookup_reads_full_original_comment(self):
        item = {"source_key": "SAC-249749", "source_type": "ticket", "issue_updated": "2026-08-19",
                "summary": "Change plan", "step": version()["failed_step"],
                "error": version()["error_description"], "body": COMMENT, "url": "https://example.invalid",
                "content_hash": "sample", "order_reason": "Change Plan", "workaround_text": FIX}
        storage.upsert_document(item, "medium", [])
        saved = storage.record_workaround(item)
        record = storage.workaround_version(saved["id"])
        self.assertEqual(record["body"], FIX)
        self.assertEqual(record["source_comments"][0]["body"], COMMENT)
        self.assertIsNone(storage.workaround_version("missing"))

    def test_endpoint_requires_csrf_and_returns_404_for_missing_version(self):
        with patch.dict(os.environ, {"SESSION_SECRET": "isolated-test-session-key"}):
            from fastapi.testclient import TestClient
            from falloutassist_next import web
        web.app.dependency_overrides[web._admin] = lambda: {"email": "tester"}
        try:
            with TestClient(web.app) as client:
                csrf = client.get("/api/auth/me").json()["csrf"]
                url = "/api/workaround-versions/missing/synthesis"
                self.assertEqual(client.post(url, json={}).status_code, 403)
                self.assertEqual(client.post(url, json={}, headers={"x-csrf-token": csrf}).status_code, 404)
        finally:
            web.app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
