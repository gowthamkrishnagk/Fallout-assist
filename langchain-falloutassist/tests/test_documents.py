"""Upload, extraction, retrieval, isolation and API checks without external calls."""

import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from falloutassist_next import documents, plugins, storage
from falloutassist_next.history import HistoryIndex

TEXT = b"Failed Step: Suspend insurance\nError: HTTP 409: account exists\nSolution applied: Update the enrollment ID in Salesforce; Retry the failed step."


class ExtractionTests(unittest.TestCase):
    def test_text_and_markdown_preserve_source_text(self):
        for name in ("procedure.txt", "procedure.md"):
            self.assertEqual(documents.extract_text(name, TEXT)[0], TEXT.decode())

    def test_word_tables_and_paragraphs_are_extracted_in_order(self):
        xml = b'''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
          <w:p><w:r><w:t>Enrollment recovery instructions</w:t></w:r></w:p>
          <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Solution applied</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:r><w:t>Update the enrollment ID in Salesforce</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
          </w:body></w:document>'''
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("word/document.xml", xml)
        text, _ = documents.extract_text("procedure.docx", content.getvalue())
        self.assertEqual(text.count("Update the enrollment ID"), 1)
        self.assertIn("|Solution applied|Update the enrollment ID in Salesforce|", text)

    def test_pdf_text_is_extracted_and_image_only_pdf_is_rejected(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        writer = PdfWriter()
        page = writer.add_blank_page(400, 400)
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 10 100 Td (Update the enrollment ID in Salesforce) Tj ET")
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                                NameObject("/Subtype"): NameObject("/Type1"),
                                                NameObject("/BaseFont"): NameObject("/Helvetica")})})})
        page[NameObject("/Contents")] = content
        buffer = io.BytesIO()
        writer.write(buffer)
        self.assertIn("Update the enrollment ID", documents.extract_text("procedure.pdf", buffer.getvalue())[0])
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        buffer = io.BytesIO()
        writer.write(buffer)
        with self.assertRaisesRegex(ValueError, "No usable text"):
            documents.extract_text("scan.pdf", buffer.getvalue())

    def test_invalid_unsupported_and_oversized_files_are_rejected(self):
        for filename, data in (("a.exe", TEXT), ("a.txt", b""), ("a.pdf", b"invalid"),
                               ("a.docx", b"invalid"), ("a.txt", b"\xff" * 30)):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                documents.extract_text(filename, data)
        with patch.object(documents, "MAX_FILE_BYTES", 10), self.assertRaises(ValueError):
            documents.extract_text("a.txt", TEXT)


class DocumentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        original_connect = storage.connect
        path = Path(temporary.name) / "documents.sqlite3"
        for item in (patch.object(storage, "database_backend", return_value="sqlite"),
                     patch.object(storage, "connect", side_effect=lambda: original_connect(path)),
                     patch.object(documents, "HistoryIndex")):
            value = item.start()
            self.addCleanup(item.stop)
            if item.attribute == "HistoryIndex":
                self.index = value.return_value
        storage.initialize()

    def add(self, **changes):
        return documents.add_document(changes.get("filename", "enrollment.txt"), changes.get("content", TEXT),
                                      "Enrollment recovery", "sop", {}, "tester")

    def test_upload_is_persisted_indexed_and_original_is_downloadable(self):
        item = self.add(filename="../../enrollment.txt")
        self.assertEqual(item["filename"], "enrollment.txt")
        self.assertEqual(item["status"], "ready")
        self.assertEqual(item["searchable_chunks"], 1)
        self.assertEqual(item["context"]["step"], "Suspend insurance")
        self.assertNotIn("file_base64", item)
        self.assertNotIn("text_content", documents.list_documents()[0])
        key = storage.all_document_keys("doc")[0]
        self.assertEqual(storage.source_document(key)["source_type"], "doc")
        self.index.upsert_many.assert_called_once()

    def test_index_failure_keeps_original_for_retry_without_searchable_partial_sources(self):
        self.index.upsert_many.side_effect = RuntimeError("index unavailable")
        item = self.add()
        self.assertEqual(item["status"], "failed")
        self.assertEqual(storage.all_document_keys("doc"), [])
        self.assertTrue(documents.document(item["id"], True)["file_base64"])
        self.index.upsert_many.side_effect = None
        self.assertEqual(documents.retry_index(item["id"], "tester")["status"], "ready")

    def test_document_changes_respect_the_ingestion_lease(self):
        self.assertTrue(storage.claim_lease("jira_ingest", "other-process", 60))
        with self.assertRaises(documents.IndexBusy):
            self.add()
        self.assertEqual(documents.list_documents(), [])

    def test_delete_removes_source_rows_and_original_and_invalidates_corpus(self):
        item = self.add()
        prior = storage.get_state("corpus_version")
        self.assertTrue(documents.delete_document(item["id"], "tester"))
        self.assertIsNone(documents.document(item["id"]))
        self.assertEqual(storage.all_document_keys("doc"), [])
        self.assertNotEqual(prior, storage.get_state("corpus_version"))
        self.index.delete_many.assert_called_once()

    def test_jira_refresh_and_full_rebuild_retain_uploaded_files_and_sources(self):
        item = self.add()
        for full in (False, True):
            with self.subTest(full=full), patch.object(plugins, "required_env", return_value="project=SAC"), \
                    patch.object(plugins.JiraPlugin, "_iter_search", return_value=iter([])), \
                    patch.object(plugins, "HistoryIndex") as index:
                plugins.JiraPlugin().ingest(full=full)
            self.assertEqual(len(storage.all_document_keys("doc")), 1)
            self.assertIsNotNone(documents.document(item["id"]))
            if full:
                index.return_value.upsert_many.assert_any_call(self.index.upsert_many.call_args.args[0])

    def test_uploaded_passages_are_returned_by_local_search_with_document_citations(self):
        item = self.add()
        key = storage.all_document_keys("doc")[0]
        from langchain_core.documents import Document
        with patch("falloutassist_next.history._store") as vector, \
                patch("falloutassist_next.history._embedding_window"):
            vector.return_value.similarity_search_with_score.return_value = [
                (Document(page_content="", metadata={"source_key": key}), .05)]
            result = HistoryIndex().search(TEXT.decode(), 5, .7, set())
        source = result["candidates"][0]
        self.assertEqual(source["type"], "doc")
        self.assertEqual(source["filename"], "enrollment.txt")
        self.assertIn(f"/api/documents/{item['id']}/file", source["url"])

    def test_long_document_passages_require_review_of_the_complete_procedure(self):
        content = TEXT + b"\n" + b"Update the enrollment ID in Salesforce.\n" * 200
        self.add(content=content)
        self.assertGreater(len(storage.all_document_keys("doc")), 1)
        self.assertTrue(all(storage.source_tier("doc", key) != "high"
                            for key in storage.all_document_keys("doc")))

    def test_document_api_is_open_but_mutations_require_csrf(self):
        with patch.dict(os.environ, {"SESSION_SECRET": "isolated-document-test-key"}):
            from fastapi.testclient import TestClient
            from falloutassist_next import web
            with TestClient(web.app) as client:
                self.assertEqual(client.get("/api/documents").status_code, 200)
                web.app.dependency_overrides[web._admin] = lambda: {"email": "tester"}
                try:
                    self.assertEqual(client.post("/api/documents", files={"file": ("sop.txt", TEXT)}).status_code, 403)
                    csrf = {"x-csrf-token": client.get("/api/session").json()["csrf"]}
                    response = client.post("/api/documents", files={"file": ("sop.txt", TEXT)},
                                           data={"kind": "sop", "title": "Test SOP"}, headers=csrf)
                    self.assertEqual(response.status_code, 200, response.text)
                    identifier = response.json()["id"]
                    self.assertEqual(client.get(f"/api/documents/{identifier}/file").content, TEXT)
                    self.assertEqual(client.delete(f"/api/documents/{identifier}").status_code, 403)
                    self.assertEqual(client.delete(f"/api/documents/{identifier}", headers=csrf).status_code, 200)
                    self.assertEqual(client.get(f"/api/documents/{identifier}").status_code, 404)
                finally:
                    web.app.dependency_overrides.clear()

    def test_dashboard_shows_configured_deployment_without_connection_secrets(self):
        from falloutassist_next import web
        with patch.object(web, "config_value", side_effect=lambda key: {
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "team-model-deployment", "EMBEDDING_MODEL": "embedding-model"}[key]), \
                patch.object(web.specialist_registry, "catalog", return_value=[]):
            result = web.dashboard({"email": "tester"})
        self.assertEqual(result["models"], {"chat_deployment": "team-model-deployment", "embedding": "embedding-model"})
        self.assertNotIn("API_KEY", str(result))


if __name__ == "__main__":
    unittest.main()
