"""Recall, token coverage, source identity and bounded remote-fallback regressions."""

import json
import re
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter
from unittest.mock import patch

from langchain_core.documents import Document

from falloutassist_next import history, plugins, retrieval_pool, storage
from falloutassist_next.embedding_text import EmbeddingWindow
from falloutassist_next.retrieval_evaluation import evaluate, load_cases


class CharacterTokenizer:
    """Small deterministic tokenizer: punctuation and technical identifiers cost tokens too."""
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(match.start(), match.end()) for match in re.finditer(r"\S", text)]}

    def encode(self, text, **kwargs):
        return [1] * (len(re.findall(r"\S", text)) + (2 if kwargs.get("add_special_tokens") else 0))


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.window = EmbeddingWindow(CharacterTokenizer(), 128)

    def test_every_vector_fits_and_the_final_instruction_is_represented(self):
        text = "Update ProductID=old-plan only if account is suspended.\n" * 50 + "TAIL_FIX"
        pieces = self.window.passages(text, "Failed Step: Cancel addon")
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(self.window.count(piece) <= 128 for piece in pieces))
        self.assertIn("TAIL_FIX", pieces[-1])
        self.assertTrue(any("ProductID=old-plan" in piece for piece in pieces))

    def test_upsert_preserves_sources_and_embeds_resolution_tail(self):
        item = {"source_key": "DEMO-1", "source_type": "ticket", "step": "Cancel addon",
                "error": "14047 | Invalid plan", "summary": "Cancellation failed",
                "body": "Check the account and its plan. " * 40 + "TAIL_FIX"}
        with patch.object(history, "_embedding_window", return_value=self.window), \
                patch.object(history, "_store") as store:
            history.HistoryIndex().upsert_many([item])
        vectors = [doc for call in store.return_value.add_documents.call_args_list for doc in call.args[0]]
        self.assertTrue(all(doc.metadata["source_key"] == "DEMO-1" for doc in vectors))
        self.assertTrue(all(self.window.count(doc.page_content) <= 128 for doc in vectors))
        self.assertTrue(any("TAIL_FIX" in doc.page_content for doc in vectors))
        store.return_value.delete.assert_called_once_with(where={"source_key": {"$in": ["DEMO-1"]}})

    def test_delete_removes_every_vector_for_a_source(self):
        with patch.object(history, "_store") as store:
            history.HistoryIndex().delete_many(["DEMO-1"])
        store.return_value.delete.assert_called_once_with(where={"source_key": {"$in": ["DEMO-1"]}})


class StoredRetrievalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        original_connect = storage.connect
        for target in (patch.object(storage, "database_backend", return_value="sqlite"),
                       patch.object(storage, "connect", side_effect=lambda: original_connect(self.directory / "test.sqlite3"))):
            target.start()
            self.addCleanup(target.stop)
        storage.initialize()

    def add_source(self, key, error="14047", body="Update the plan identifier and retry the failed step."):
        storage.upsert_document({"source_key": key, "source_type": "ticket", "summary": "Case",
                                 "step": "Cancel addon", "error": error, "body": body,
                                 "url": "https://example.invalid/" + key, "content_hash": key,
                                 "issue_updated": "2026-01-01", "metadata": {}}, "medium", [])

    def test_long_step_cannot_push_error_code_out_of_keyword_query(self):
        self.add_source("DEMO-1")
        query = " ".join(f"irrelevant{i}" for i in range(50)) + " 14047"
        self.assertEqual(storage.lexical_keys(query, 5, error="14047"), ["DEMO-1"])

    def test_repeated_words_do_not_exhaust_unique_term_budget(self):
        self.add_source("DEMO-1", error="specialfailure")
        self.assertEqual(storage.lexical_keys("cancel " * 40 + "specialfailure", 5), ["DEMO-1"])

    def test_error_field_has_more_weight_than_same_word_in_body(self):
        self.add_source("DEMO-1", error="14047", body="Ordinary resolution text")
        self.add_source("DEMO-2", error="Other", body="Ordinary resolution 14047")
        self.assertEqual(storage.lexical_keys("14047", 5, error="14047")[0], "DEMO-1")

    def test_duplicate_vector_passages_cannot_vote_twice_in_rrf(self):
        self.add_source("DEMO-1")
        self.add_source("DEMO-2")
        first = Document(page_content="", metadata={"source_key": "DEMO-1"})
        second = Document(page_content="", metadata={"source_key": "DEMO-2"})
        with patch.object(history, "_store") as store, patch.object(history, "_embedding_window"), \
                patch.object(storage, "lexical_keys", return_value=["DEMO-2", "DEMO-1"]):
            store.return_value.similarity_search_with_score.return_value = [(first, .1), (first, .2), (second, .15)]
            result = history.HistoryIndex().search("Step: Cancel addon\nError: 14047", 5, .7, set())
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual({row["retrieval_score"] for row in result["candidates"]}, {1.0})

    def test_different_explicit_error_codes_still_get_rejected(self):
        self.add_source("DEMO-1", error="5062")
        with patch.object(history, "_store") as store, patch.object(history, "_embedding_window"):
            store.return_value.similarity_search_with_score.return_value = [
                (Document(page_content="", metadata={"source_key": "DEMO-1"}), .01)]
            result = history.HistoryIndex().search("Step: Cancel addon\nError: 14047", 5, .7, set())
        self.assertEqual(result["candidates"], [])

    def local_result(self):
        return {"candidates": [], "query_step": "Cancel addon", "query_error": "14047"}

    def test_slow_mcp_returns_local_results_without_waiting_for_background_work(self):
        pending = Future()
        pending.set_running_or_notify_cancel()
        with patch.object(plugins, "config_value", return_value="rovo"), \
                patch.object(storage, "settings", return_value={"rovo_retrieval_budget_seconds": .02}), \
                patch.object(plugins.HistoryIndex, "search", return_value=self.local_result()), \
                patch.object(retrieval_pool, "submit", return_value=pending):
            started = perf_counter()
            result = plugins.JiraPlugin().search_history("query", 5, .7)
            elapsed = perf_counter() - started
        pending.set_result([])
        self.assertLess(elapsed, .25)
        self.assertEqual(result["backend"], "local")
        self.assertEqual(result["retrieval_failures"]["rovo"], "TimeoutError")

    def test_slow_evidence_fetch_uses_the_same_remote_deadline(self):
        search, pending = Future(), Future()
        search.set_result(["DEMO-1"])
        pending.set_running_or_notify_cancel()
        with patch.object(plugins, "config_value", return_value="rovo"), \
                patch.object(storage, "settings", return_value={"rovo_retrieval_budget_seconds": .02}), \
                patch.object(plugins.HistoryIndex, "search", return_value=self.local_result()), \
                patch.object(retrieval_pool, "submit", side_effect=[search, pending]):
            result = plugins.JiraPlugin().search_history("query", 5, .7)
        pending.set_result(None)
        self.assertIn("TimeoutError", result["retrieval_failures"]["rovo_fetch"])
        self.assertEqual(result["candidates"], [])


class EvaluationTests(unittest.TestCase):
    def test_reports_recall_hit_rate_and_reciprocal_rank(self):
        cases = [{"id": "one", "query_text": "a", "verified": True,
                  "issue_key": "DEMO-current", "expected_source_keys": ["DEMO-1", "DEMO-2"]},
                 {"id": "two", "query_text": "b", "verified": True, "expected_source_keys": ["DEMO-3"]}]
        def search(text, top_k, min_score, excluded):
            if text == "a":
                self.assertIn("DEMO-current", excluded)
                return {"candidates": [{"key": "DEMO-other"}, {"key": "DEMO-1"}]}
            return {"candidates": []}
        result = evaluate(cases, search, 5)
        self.assertEqual(result["hit_rate_at_k"], .5)
        self.assertEqual(result["recall_at_k"], .25)
        self.assertEqual(result["mrr_at_k"], .25)
        self.assertNotIn("query_text", str(result))

    def test_unverified_or_self_matching_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            for verified, key in ((False, "DEMO-1"), (True, "DEMO-current")):
                path.write_text(json.dumps({"id": "one", "query_text": "a", "verified": verified,
                                            "issue_key": "DEMO-current", "expected_source_keys": [key]}))
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_remote_slots_are_released_only_when_background_work_finishes(self):
        pending = Future()
        pending.set_running_or_notify_cancel()
        with patch.object(retrieval_pool, "_slots", BoundedSemaphore(1)), \
                patch.object(retrieval_pool._pool, "submit", return_value=pending):
            retrieval_pool.submit(lambda: None)
            with self.assertRaises(retrieval_pool.RetrievalBusy):
                retrieval_pool.submit(lambda: None)
            pending.set_result(None)
            self.assertTrue(retrieval_pool._slots.acquire(blocking=False))


if __name__ == "__main__":
    unittest.main()
