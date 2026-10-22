"""Issue-key filtering for the local workaround library."""

import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from falloutassist_next import storage
from falloutassist_next import plugins


class WorkaroundIssueSearchTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test-tmp"
        test_root.mkdir(exist_ok=True)
        path = test_root / f"workaround-search-{uuid4().hex}.sqlite3"
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        original_connect = storage.connect
        self.backend = patch.object(storage, "database_backend", return_value="sqlite")
        self.connection = patch.object(storage, "connect",
                                        side_effect=lambda: original_connect(path))
        self.backend.start()
        self.connection.start()
        self.addCleanup(self.connection.stop)
        self.addCleanup(self.backend.stop)
        storage.initialize()

    def _add_workaround(self, source_key, reason):
        item = {
            "source_key": source_key,
            "source_type": "ticket",
            "issue_updated": "2026-10-01",
            "summary": reason,
            "step": "Retry the activation step",
            "error": "Service activation failed",
            "body": "Apply the documented recovery procedure and verify the service.",
            "workaround_text": "Apply the documented recovery procedure and verify the service.",
            "url": f"https://example.invalid/browse/{source_key}",
            "content_hash": source_key,
            "order_reason": reason,
        }
        storage.upsert_document(item, "medium", [])
        self.assertIsNotNone(storage.record_workaround(item))

    def test_issue_key_matches_only_exact_local_source_key(self):
        self._add_workaround("SAC-12345", "Change plan")
        self._add_workaround("SAC-123456", "New activation")

        rows = storage.workaround_rows(issue_key="sac-12345")

        self.assertEqual([row["order_reason"] for row in rows], ["Change plan"])
        self.assertEqual(rows[0]["versions"][0]["source_keys"], ["SAC-12345"])

    def test_issue_preview_can_force_local_semantic_search_when_rovo_is_configured(self):
        expected = {"candidates": [], "strong": [], "context": [], "best_score": 0.0}
        with patch.object(plugins, "config_value", return_value="rovo"), \
             patch.object(plugins.HistoryIndex, "search", return_value=expected) as local_search, \
             patch("falloutassist_next.rovo.semantic_issue_keys") as rovo_search:
            result = plugins.JiraPlugin().search_history(
                "Step: Cancel add-on\nError: invalid plan", 5, 0.4,
                exclude_keys={"SAC-260434"}, local_only=True)

        self.assertEqual(result["backend"], "local")
        self.assertEqual(result["candidates"], [])
        local_search.assert_called_once_with(
            "Step: Cancel add-on\nError: invalid plan", 5, 0.4, {"SAC-260434"})
        rovo_search.assert_not_called()

    def test_issue_key_can_be_combined_with_text_filters(self):
        self._add_workaround("SAC-12345", "Change plan")
        self._add_workaround("SAC-12346", "Change plan retry")

        rows = storage.workaround_rows(order_reason="change", issue_key="SAC-12346")

        self.assertEqual([row["order_reason"] for row in rows], ["Change plan retry"])


if __name__ == "__main__":
    unittest.main()
