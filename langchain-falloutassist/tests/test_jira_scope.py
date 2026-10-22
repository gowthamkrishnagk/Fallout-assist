"""Accurate persisted-ticket counts and read-only JQL previews."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from falloutassist_next import storage


class JiraScopeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        original = storage.connect
        path = Path(temporary.name) / "scope.sqlite3"
        for item in (patch.object(storage, "database_backend", return_value="sqlite"),
                     patch.object(storage, "connect", side_effect=lambda: original(path)),
                     patch.dict(os.environ, {"SESSION_SECRET": "isolated-jql-preview-key"})):
            item.start()
            self.addCleanup(item.stop)
        storage.initialize()

    def test_ingestion_counts_exclude_uploads_and_ratings_without_stored_issues(self):
        for key, kind, tier in (("SAC-100", "ticket", "high"), ("SAC-101", "ticket", "low"),
                                 ("DOC:sample:1", "doc", "high")):
            storage.upsert_document({"source_key": key, "source_type": kind, "issue_updated": "today",
                                     "summary": "Example", "step": "Step", "error": "Error",
                                     "body": "Update the enrollment ID in Salesforce", "url": "",
                                     "content_hash": key, "metadata": {}}, tier, [])
        storage.upsert_source("ticket", "SAC-999", "high", [], "Rating only", "Step", "Error")
        self.assertEqual(storage.jira_ingestion_counts(), {"stored": 2, "searchable": 1, "excluded": 1})

    def test_last_ingestion_is_not_hidden_by_later_unrelated_activity(self):
        storage.audit("tester", "ingest_completed", "jira", {"fetched": 10, "indexed": 6, "skipped": 4})
        storage.audit("tester", "document_uploaded", "document", {})
        self.assertEqual(storage.latest_ingestion()["fetched"], 10)

    def test_query_preview_returns_the_exact_draft_without_saving_it(self):
        from fastapi.testclient import TestClient
        from falloutassist_next import web
        query = 'project = SAC AND statusCategory = Done ORDER BY updated DESC'
        with TestClient(web.app) as client:
            web.app.dependency_overrides[web._admin] = lambda: {"email": "tester"}
            try:
                self.assertEqual(client.post('/api/connections/test-jql', json={"kind": "history", "jql": query}).status_code, 403)
                csrf = {"x-csrf-token": client.get('/api/session').json()["csrf"]}
                with patch.object(web.JiraPlugin, 'test_jql', return_value={"total": None,
                    "sample_keys": ["SAC-100"], "has_more": True}) as preview:
                    response = client.post('/api/connections/test-jql', json={"kind": "history", "jql": query}, headers=csrf)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["jql"], query)
                preview.assert_called_once_with(query)
                self.assertIsNone(storage.connection_value("JIRA_INGEST_JQL"))
                self.assertEqual(storage.jira_ingestion_counts()["stored"], 0)
            finally:
                web.app.dependency_overrides.clear()


if __name__ == '__main__':
    unittest.main()
