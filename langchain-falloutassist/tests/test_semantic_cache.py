"""Cache correctness and workflow checks using temporary storage and mocked models."""

import copy
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from falloutassist_next import reranker, semantic_cache, storage
from falloutassist_next.config import DEFAULTS, validated_settings

QUERY = ("Summary: Customer provisioning failed\nFailed Step: Activate Account\n"
         "Error: 5062 | Invalid plan status\nOrder Type: Change\nOrder Reason: Upgrade\nRelease: R1")
CANDIDATES = [{"key": "SAC-123", "summary": "Account activation", "step": "Activate Account",
               "error": "5062 | Invalid plan status", "order_type": "Change",
               "order_reason": "Upgrade", "comment": "Update the plan status and retry activation.",
               "source_tier": "medium", "retrieval_score": 0.9}]
RESPONSE = {"candidates": [{"source_key": "SAC-123", "relevance": 0.85,
                            "reason": "Same failure and applicable corrective action."}]}


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        original_connect = storage.connect
        path = Path(directory.name) / "cache.sqlite3"
        self.config = {key: key for key in (
            "EMBEDDING_MODEL", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION")}
        patches = [patch.object(storage, "database_backend", return_value="sqlite"),
                   patch.object(storage, "connect", side_effect=lambda: original_connect(path)),
                   patch.object(semantic_cache, "config_value", side_effect=self.config.__getitem__)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.embed = patch.object(semantic_cache, "embed_query", return_value=[1.0, 0.0]).start()
        self.addCleanup(patch.stopall)
        self.model = patch.object(reranker, "chat_model").start()
        self.invoke = self.model.return_value.with_structured_output.return_value.invoke
        self.invoke.return_value = RESPONSE
        self.settings = dict(DEFAULTS)
        self.settings["semantic_cache_backend"] = "database"
        storage.initialize()
        storage.update_settings({"semantic_cache_backend": "database"}, "tester")

    def rank(self, query=QUERY, candidates=None):
        info = {}
        result = reranker.rerank_candidates(query, copy.deepcopy(
            CANDIDATES if candidates is None else candidates), 0.0, False, self.settings, info)
        return result, info

    def count(self):
        with storage.connect() as db:
            return db.execute("SELECT COUNT(*) AS n FROM semantic_cache").fetchone()["n"]


class SemanticCacheTests(CacheTestCase):
    def test_exact_hit_survives_reinitialization_without_embedding_or_model_call(self):
        first, info = self.rank()
        self.assertEqual(info["status"], "miss")
        storage.initialize()
        self.embed.reset_mock()
        second, info = self.rank()
        self.assertEqual(first, second)
        self.assertEqual(info["status"], "exact_hit")
        self.assertEqual(self.invoke.call_count, 1)
        self.embed.assert_not_called()

    def test_paraphrase_reuses_scores_and_respects_threshold(self):
        self.rank()
        _, info = self.rank(QUERY.replace("Customer provisioning failed", "Client provisioning failure"))
        self.assertEqual(info["status"], "semantic_hit")
        self.assertEqual(self.invoke.call_count, 1)
        self.embed.return_value = [0.0, 1.0]
        _, info = self.rank(QUERY.replace("Customer provisioning failed", "Account could not provision"))
        self.assertEqual(info["status"], "miss")
        self.assertEqual(self.invoke.call_count, 2)

    def test_changed_failure_context_numbers_and_negation_cannot_hit(self):
        self.rank()
        changes = [("Activate Account", "Suspend Account"), ("5062", "5063"),
                   ("Order Type: Change", "Order Type: Sales"), ("Upgrade", "Downgrade"),
                   ("R1", "R2"), ("Invalid plan status", "Valid plan status"),
                   ("Customer provisioning failed", "Customer provisioning not failed"),
                   ("Customer provisioning failed", "Customer provisioning failed HTTP 503")]
        for before, after in changes:
            with self.subTest(after=after):
                _, info = self.rank(QUERY.replace(before, after))
                self.assertEqual(info["status"], "miss")
        self.assertEqual(self.invoke.call_count, 1 + len(changes))

    def test_unknown_order_context_only_allows_exact_reuse(self):
        incomplete = QUERY.replace("Order Type: Change\n", "")
        self.rank(incomplete)
        _, info = self.rank(incomplete)
        self.assertEqual(info["status"], "exact_hit")
        _, info = self.rank(incomplete.replace("Customer", "Client"))
        self.assertEqual(info["status"], "miss")
        self.assertFalse(info["semantic_eligible"])
        self.embed.assert_not_called()

    def test_complete_evidence_version_quality_and_model_invalidate(self):
        self.rank()
        for field, value in [("comment", CANDIDATES[0]["comment"] + "x" * 1700),
                             ("source_tier", "low"), ("release", "R2"),
                             ("workaround_version", {"id": "new", "version_number": 2})]:
            with self.subTest(field=field):
                changed = copy.deepcopy(CANDIDATES)
                changed[0][field] = value
                self.assertEqual(self.rank(candidates=changed)[1]["status"], "miss")
        # Even an edit beyond the model's 1500-character resolution excerpt invalidates.
        long = copy.deepcopy(CANDIDATES)
        long[0]["comment"] = "x" * 1700
        self.rank(candidates=long)
        long[0]["comment"] += " changed"
        self.assertEqual(self.rank(candidates=long)[1]["status"], "miss")
        self.config["AZURE_OPENAI_CHAT_DEPLOYMENT"] = "new-deployment"
        self.assertEqual(self.rank()[1]["status"], "miss")
        storage.set_state("connection_version", "new-credentials")
        self.assertEqual(self.rank()[1]["status"], "miss")
        with patch.object(reranker, "PROMPT_VERSION", "new-prompt"):
            self.assertEqual(self.rank()[1]["status"], "miss")
        self.config["EMBEDDING_MODEL"] = "different-embeddings"
        self.assertEqual(self.rank()[1]["status"], "miss")

    def test_expiry_and_reduced_lifetime_apply_on_read(self):
        self.rank()
        with storage.connect() as db:
            db.execute("UPDATE semantic_cache SET created_at=?", (
                (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),))
        self.settings["semantic_cache_ttl_minutes"] = 1
        self.assertEqual(self.rank()[1]["status"], "miss")
        with storage.connect() as db:
            db.execute("UPDATE semantic_cache SET expires_at=?", (
                (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),))
        self.assertEqual(self.rank()[1]["status"], "miss")
        self.assertEqual(self.count(), 1)

    def test_bounded_storage_clear_and_disabled_cache(self):
        self.settings["semantic_cache_max_entries"] = 2
        for count in range(4):
            self.rank(QUERY + f"\nHTTP {count}")
        self.assertEqual(self.count(), 2)
        storage.clear_semantic_cache("tester")
        self.assertEqual(self.count(), 0)
        self.assertEqual(storage.recent_audit(1)[0]["action"], "semantic_cache_cleared")
        self.settings["semantic_cache_enabled"] = False
        self.assertEqual(self.rank()[1]["status"], "disabled")
        self.assertEqual(self.rank()[1]["status"], "disabled")
        self.assertEqual(self.count(), 0)

    def test_corrupt_result_is_a_miss_and_failed_model_result_is_never_cached(self):
        self.rank()
        for value in ('{"foreign-ticket":{"llm_score":1,"llm_reason":"wrong"}}',
                      '{"SAC-123":{"llm_score":NaN,"llm_reason":"invalid"}}', 'not json'):
            with storage.connect() as db:
                db.execute("UPDATE semantic_cache SET result_json=?", (value,))
            self.assertEqual(self.rank()[1]["status"], "miss")
        storage.clear_semantic_cache("tester")
        self.invoke.return_value = {"candidates": []}
        with self.assertRaises(ValueError):
            self.rank()
        self.assertEqual(self.count(), 0)
        self.invoke.side_effect = RuntimeError("model down")
        with self.assertRaises(RuntimeError):
            self.rank()
        self.assertEqual(self.count(), 0)

    def test_cache_io_and_embedding_failures_still_use_model(self):
        with patch.object(storage, "semantic_cache_entries", side_effect=RuntimeError("db down")):
            self.assertEqual(self.rank()[1]["status"], "unavailable")
        self.embed.side_effect = RuntimeError("embeddings down")
        _, info = self.rank()
        self.assertEqual(info["embedding_error"], "RuntimeError")
        self.assertEqual(self.invoke.call_count, 2)
        storage.clear_semantic_cache("tester")
        with patch.object(storage, "put_semantic_cache", side_effect=RuntimeError("db down")):
            _, info = self.rank()
            self.assertEqual(info["write_error"], "RuntimeError")
        self.assertEqual(self.count(), 0)

    def test_cache_hit_recomputes_feedback_composite_and_records_metadata(self):
        from falloutassist_next import workflow
        state = {"candidates": copy.deepcopy(CANDIDATES), "history": {"query_step": "Activate Account",
                 "query_error": "5062 | Invalid plan status"}, "settings": self.settings,
                 "issue": {"query_text": QUERY}, "issue_key": "SAC-456", "id": "review"}
        with patch.object(storage, "feedback_stats", return_value={"up": 0, "down": 0}):
            first = workflow._rerank(copy.deepcopy(state))
        with patch.object(storage, "feedback_stats", return_value={"up": 10, "down": 0}) as feedback:
            second = workflow._rerank(copy.deepcopy(state))
        feedback.assert_called_once()
        self.assertEqual(self.invoke.call_count, 1)
        self.assertEqual(second["rerank"][0]["llm_cache"]["status"], "exact_hit")
        self.assertGreater(second["rerank"][0]["composite_score"], first["rerank"][0]["composite_score"])

    def test_clear_endpoint_is_open_but_requires_csrf(self):
        with patch.dict(os.environ, {"SESSION_SECRET": "isolated-test-session-key"}):
            from fastapi.testclient import TestClient
            from falloutassist_next import web
        try:
            with TestClient(web.app) as client:
                self.assertEqual(client.post("/api/semantic-cache/clear").status_code, 403)
                web.app.dependency_overrides[web._admin] = lambda: {"email": "tester"}
                self.rank()
                self.assertEqual(client.post("/api/semantic-cache/clear").status_code, 403)
                self.assertEqual(self.count(), 1)
                token = client.get("/api/session").json()["csrf"]
                self.assertEqual(client.post("/api/semantic-cache/clear", headers={
                    "x-csrf-token": token}).status_code, 200)
                self.assertEqual(self.count(), 0)
        finally:
            web.app.dependency_overrides.clear()


class SettingsTests(unittest.TestCase):
    def test_cache_setting_boundaries(self):
        for key, value in [("semantic_cache_threshold", 0.89), ("semantic_cache_threshold", float('nan')),
                           ("semantic_cache_threshold", True), ("semantic_cache_enabled", "true"),
                           ("semantic_cache_ttl_minutes", 0), ("semantic_cache_max_entries", 10001),
                           ("semantic_cache_backend", "unknown")]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validated_settings({key: value})
        self.assertEqual(validated_settings({})["semantic_cache_threshold"], 0.97)


if __name__ == "__main__":
    unittest.main()
