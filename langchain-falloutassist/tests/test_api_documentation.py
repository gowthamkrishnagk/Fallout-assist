"""Swagger/service access checks with isolated storage and no external operations."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from falloutassist_next import storage, web
from falloutassist_next.config import DEFAULTS, CONNECTION_DEFAULTS

API_KEY = 'isolated-falloutassist-api-key-for-tests-only'


class FalloutAssistApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        connect = storage.connect
        path = Path(temporary.name) / 'api.sqlite3'
        for target in (patch.object(storage, 'database_backend', return_value='sqlite'),
                       patch.object(storage, 'connect', side_effect=lambda: connect(path)),
                       patch.dict(os.environ, {'SESSION_SECRET': 'isolated-api-documentation-secret'})):
            target.start()
            self.addCleanup(target.stop)
        storage.initialize()
        storage.update_connection_config({'KNOWLEDGE_ROUTER_API_KEY': API_KEY}, 'tester')
        self.client = TestClient(web.app)
        self.addCleanup(self.client.close)
        self.headers = {'Authorization': 'Bearer ' + API_KEY}

    def test_swagger_groups_main_and_management_apis_and_documents_all_settings(self):
        response = self.client.get('/openapi.json')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        from fastapi.openapi.models import OpenAPI
        OpenAPI.model_validate(data)
        self.assertEqual(data['info']['title'], 'FalloutAssist API')
        self.assertEqual(data['paths']['/api/preview']['post']['tags'], ['Main APIs'])
        self.assertEqual(data['paths']['/api/ingestion/status']['get']['tags'], ['Ingestion APIs'])
        self.assertEqual(data['paths']['/api/settings']['put']['tags'], ['Configuration APIs'])
        self.assertEqual(data['paths']['/api/documents']['post']['tags'], ['Documents and SOP APIs'])
        self.assertEqual(data['paths']['/api/knowledge/jobs']['post']['security'], [{'FalloutAssistApiKey': []}])
        schemas = data['components']['schemas']
        self.assertEqual(set(schemas['FalloutAssistSettings']['properties']), set(DEFAULTS))
        self.assertEqual(set(schemas['FalloutAssistConnectionsUpdate']['properties']), set(CONNECTION_DEFAULTS))
        self.assertTrue(schemas['FalloutAssistSettings']['properties']['rovo_mcp_always_search']['default'])
        self.assertTrue(schemas['FalloutAssistConnectionsUpdate']['properties']['KNOWLEDGE_ROUTER_API_KEY']['writeOnly'])
        self.assertNotIn(API_KEY, response.text)
        self.assertEqual(self.client.get('/docs').status_code, 200)
        self.assertEqual(self.client.get('/redoc').status_code, 200)

    def test_service_key_updates_retrieval_and_scheduler_settings_without_browser_csrf(self):
        changes = {'retrieval_top_k': 30, 'rerank_pool_k': 20, 'jira_review_top_k': 5,
                   'rovo_mcp_always_search': True, 'ingest_interval_minutes': 15, 'scan_interval_minutes': 0}
        result = self.client.put('/api/settings', json=changes, headers=self.headers)
        self.assertEqual(result.status_code, 200, result.text)
        for key, value in changes.items():
            self.assertEqual(storage.settings()[key], value)
        self.assertEqual(self.client.put('/api/settings', json=changes).status_code, 403)
        self.assertEqual(self.client.put('/api/settings', json=changes,
            headers={'Authorization': 'Bearer invalid'}).status_code, 401)
        invalid = self.client.put('/api/settings', json={'retrieval_top_k': 1}, headers=self.headers)
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(storage.settings()['retrieval_top_k'], 30)

    def test_service_can_update_jql_and_read_masked_connections(self):
        jql = 'project = SAC AND status in (Resolved, Closed)'
        result = self.client.put('/api/connections', json={'JIRA_INGEST_JQL': jql}, headers=self.headers)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['values']['JIRA_INGEST_JQL'], jql)
        self.assertTrue(result.json()['secrets']['KNOWLEDGE_ROUTER_API_KEY'])
        self.assertNotIn(API_KEY, result.text)

    def test_saving_a_replacement_key_revokes_the_previous_key(self):
        replacement = 'replacement-falloutassist-key-for-tests-only'
        self.assertEqual(self.client.put('/api/connections', json={'KNOWLEDGE_ROUTER_API_KEY': replacement},
                                         headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get('/api/settings', headers=self.headers).status_code, 401)
        self.assertEqual(self.client.get('/api/settings', headers={'Authorization': 'Bearer ' + replacement}).status_code, 200)

    def test_status_reports_persisted_totals_and_latest_new_updated_counts(self):
        for key, kind, tier in (('SAC-100', 'ticket', 'high'), ('SAC-101', 'ticket', 'low'), ('DOC:sample:1', 'doc', 'high')):
            storage.upsert_document({'source_key': key, 'source_type': kind, 'issue_updated': 'today',
                'summary': 'Example', 'step': 'Step', 'error': 'Error', 'body': 'Update the enrollment ID in Salesforce',
                'url': '', 'content_hash': key, 'metadata': {}}, tier, [])
        storage.audit('tester', 'ingest_completed', 'jira', {'fetched': 3, 'new_documents': 2,
            'updated_documents': 1, 'removed_documents': 0, 'skipped': 0})
        response = self.client.get('/api/ingestion/status', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['totals'], {'stored': 2, 'searchable': 1, 'excluded': 1})
        self.assertEqual(response.json()['last_completed_run']['new_documents'], 2)
        self.assertEqual(response.json()['last_completed_run']['updated_documents'], 1)
        self.assertIn('retrieval_top_k', response.json()['configuration']['settings'])

    def test_service_ingestion_passes_explicit_mode_without_a_live_jira_call(self):
        with patch.object(web, 'ingest_once', return_value={'ok': True, 'new_documents': 2}) as ingest:
            response = self.client.post('/api/ingest', json={'full': False}, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            ingest.assert_called_once_with('falloutassist_api', full=False)

    def test_knowledge_job_uses_required_bearer_key_and_preserves_async_contract(self):
        self.assertEqual(self.client.post('/api/knowledge/jobs', json={'issue_key': 'SAC-100'}).status_code, 401)
        with patch.object(web.knowledge_router, 'submit', return_value={
            'job_id': 'a' * 32, 'issue_key': 'SAC-100', 'status': 'queued', 'deduplicated': False}) as submit:
            response = self.client.post('/api/knowledge/jobs', json={'issue_key': 'sac-100'}, headers=self.headers)
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()['status_url'], '/api/knowledge/jobs/' + 'a' * 32)
            submit.assert_called_once_with('SAC-100', [], 'knowledge_agent')


if __name__ == '__main__':
    unittest.main()
