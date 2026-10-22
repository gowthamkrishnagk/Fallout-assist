"""OAuth credential handling and complete-only A2A preview responses."""
import base64
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit, parse_qs
import requests

from falloutassist_next import storage, rovo_a2a as a2a

DISCOVERY = {"endpoint":"https://a2a.atlassian.com/v1/rovo", "authorize":"https://auth.atlassian.com/authorize",
             "token":"https://auth.atlassian.com/oauth/token", "register":"https://auth.atlassian.com/client/dcr/register",
             "scopes":sorted(a2a.SCOPES), "name":"Atlassian Rovo"}
ADMIN = {"id":"admin-id", "email":"admin@example.test", "role":"admin"}


class HistoricalKeyTests(unittest.TestCase):
    def test_mcp_failure_is_actionable_and_manual_keys_bypass_search(self):
        issue = {"key":"SAC-1", "query_text":"example issue"}
        with patch("falloutassist_next.rovo_history.search", side_effect=RuntimeError("401")) as search:
            with self.assertRaises(a2a.A2AError) as raised:
                a2a._historical_keys(issue, [])
        self.assertIn("Rovo MCP historical search", str(raised.exception))
        search.assert_called_once()
        with patch("falloutassist_next.rovo_history.search") as search:
            self.assertEqual(a2a._historical_keys(issue, ["SAC-2"]), ["SAC-2"])
        search.assert_not_called()

    def test_candidate_pool_is_kept_until_usable_comments_are_selected(self):
        issue = {"key":"SAC-1", "query_text":"example issue"}
        keys = [f"SAC-{n}" for n in range(2, 12)]
        details = {}
        with patch("falloutassist_next.rovo_history.search", return_value={
                "keys": keys, "retrieval": {"project": "SAC"}}):
            self.assertEqual(a2a._historical_keys(issue, [], details), keys)
        self.assertEqual(details["project"], "SAC")


class Response:
    def __init__(self, value=None, status=200, events=None):
        self.value, self.status_code = value or {}, status
        self.events = events or []
        self.headers = {"Content-Type":"text/event-stream"}
    def json(self):
        return self.value
    def iter_lines(self, **kwargs):
        yield ': heartbeat'
        yield ''
        for event in self.events:
            yield 'data: '+json.dumps(event)
            yield ''
    def close(self):
        pass


class A2ATests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        original=storage.connect
        path=Path(temporary.name)/'a2a.sqlite3'
        for item in (patch.object(storage,'database_backend',return_value='sqlite'),
                     patch.object(storage,'connect',side_effect=lambda:original(path)),
                     patch.dict(os.environ,{'SESSION_SECRET':'isolated-rovo-oauth-test-secret'})):
            item.start()
            self.addCleanup(item.stop)
        storage.initialize()

    def start(self):
        with patch.object(a2a,'discover',return_value=copy.deepcopy(DISCOVERY)), \
             patch.object(a2a,'_http',return_value=Response({'client_id':'dynamic-client'})) as http:
            result=a2a.start('http://localhost:18012/',ADMIN['id'],'browser-binding')
        params=parse_qs(urlsplit(result['authorization_url']).query)
        return result,params,http

    def test_jira_read_errors_are_actionable_and_do_not_expose_upstream_body(self):
        response = requests.Response()
        response.status_code = 404
        response._content = b"private Jira response"
        error = requests.HTTPError("404 from Jira", response=response)
        message = str(a2a._jira_read_error("SAC-260381", error))
        self.assertIn("SAC-260381", message)
        self.assertIn("not found or is not accessible", message)
        self.assertNotIn("private Jira response", message)

    def test_dead_preview_owner_is_recovered_without_removing_completed_preview(self):
        holder=f'rovo:{a2a._HOST}:99999999:test'
        storage.claim_lease('rovo_a2a_preview',holder,360)
        storage.save_integration_secret('rovo_a2a:last_preview',{'text':'saved resolution'})
        with patch.object(a2a,'_process_alive',return_value=False):
            state=a2a.status()
        self.assertFalse(state['preview_running'])
        self.assertEqual(state['last_preview']['text'],'saved resolution')

    def test_live_or_foreign_preview_owner_is_never_cleared(self):
        for host in (a2a._HOST,'another-host'):
            with self.subTest(host=host):
                holder=f'rovo:{host}:123:test'
                storage.claim_lease('rovo_a2a_preview',holder,360)
                with patch.object(a2a,'_process_alive',return_value=True):
                    self.assertTrue(a2a.status()['preview_running'])
                storage.release_lease('rovo_a2a_preview',holder)

    def test_duplicate_preview_is_http_conflict_and_does_not_start_work(self):
        from fastapi.testclient import TestClient
        from falloutassist_next import web
        holder=f'rovo:{a2a._HOST}:123:test'
        storage.claim_lease('rovo_a2a_preview',holder,360)
        with TestClient(web.app) as client:
            web.app.dependency_overrides[web._admin]=lambda:ADMIN
            try:
                csrf=client.get('/api/session').json()['csrf']
                with patch.object(a2a,'_process_alive',return_value=True), \
                     patch.object(a2a,'_access',return_value=(DISCOVERY['endpoint'],'test-access')), \
                     patch('falloutassist_next.plugins.JiraPlugin') as jira:
                    response=client.post('/api/rovo-a2a/preview',json={'issue_key':'SAC-1'},
                                         headers={'x-csrf-token':csrf})
                self.assertEqual(response.status_code,409,response.text)
                jira.assert_not_called()
            finally:
                web.app.dependency_overrides.clear()

    def connect(self):
        _,params,_=self.start()
        with patch.object(a2a,'_http',return_value=Response({'access_token':'sensitive-access',
                'refresh_token':'sensitive-refresh','expires_in':3600})):
            a2a.finish(params['state'][0],'authorization-code',ADMIN,'browser-binding')
        return params

    def test_discovery_uses_card_dcr_and_rejects_untrusted_endpoint(self):
        card={'url':DISCOVERY['endpoint'],'protocolVersion':'0.3.0',
              'capabilities':{'streaming':True,'extensions':[{'uri':'https://cloud.google.com/marketplace/docs/partners/ai-agents/setup-dcr',
                              'params':{'target_url':DISCOVERY['register']}}]},
              'securitySchemes':{'oauth2':{'flows':{'authorizationCode':{
                  'authorizationUrl':DISCOVERY['authorize'],'tokenUrl':DISCOVERY['token']}}}},
              'security':[{'oauth2':sorted(a2a.SCOPES)}]}
        with patch.object(a2a,'_http',return_value=Response(card)):
            self.assertEqual(a2a.discover()['register'],DISCOVERY['register'])
            card['url']='https://attacker.invalid/steal'
            with self.assertRaises(a2a.A2AError):
                a2a.discover()

    def test_oauth_state_is_encrypted_pkce_bound_and_single_use(self):
        result,params,http=self.start()
        self.assertEqual(http.call_args.kwargs['json']['token_endpoint_auth_method'],'none')
        self.assertEqual(result['redirect_uri'],'http://localhost:18012/api/rovo-a2a/callback')
        state=params['state'][0]
        pending=storage.integration_secret('rovo_a2a:pending:'+hashlib.sha256(state.encode()).hexdigest())
        expected=base64.urlsafe_b64encode(hashlib.sha256(pending['verifier'].encode()).digest()).rstrip(b'=').decode()
        self.assertEqual(params['code_challenge'],[expected])
        with storage.connect() as db:
            raw=' '.join(r['payload'] for r in db.execute('SELECT payload FROM integration_secrets'))
        self.assertNotIn(pending['verifier'],raw)
        self.assertNotIn('browser-binding',raw)
        with patch.object(a2a,'_http',return_value=Response({'access_token':'secret-access',
                'refresh_token':'secret-refresh','expires_in':3600})) as exchange:
            a2a.finish(state,'auth-code',ADMIN,'browser-binding')
            self.assertEqual(exchange.call_args.kwargs['data']['code_verifier'],pending['verifier'])
            with self.assertRaises(a2a.A2AError):
                a2a.finish(state,'auth-code',ADMIN,'browser-binding')
            self.assertEqual(exchange.call_count,1)
        public=json.dumps(a2a.status())+json.dumps(storage.connection_snapshot())
        self.assertNotIn('secret-access',public)
        self.assertNotIn('secret-refresh',public)

    def test_wrong_browser_expired_or_disconnected_flow_cannot_exchange_code(self):
        for mode in ('wrong-browser','expired','disconnected'):
            with self.subTest(mode=mode):
                _,params,_=self.start()
                state=params['state'][0]
                key='rovo_a2a:pending:'+hashlib.sha256(state.encode()).hexdigest()
                if mode=='expired':
                    p=storage.integration_secret(key); p['created']=time.time()-601
                    storage.save_integration_secret(key,p)
                if mode=='disconnected':
                    a2a.disconnect(ADMIN['email'])
                with patch.object(a2a,'_http') as http, self.assertRaises(a2a.A2AError):
                    a2a.finish(state,'code',ADMIN,'wrong' if mode=='wrong-browser' else 'browser-binding')
                http.assert_not_called()

    def test_refresh_rotates_encrypted_credentials_and_disconnect_removes_them(self):
        self.connect()
        c=storage.integration_secret(a2a.KEY); c['expires_at']=0
        storage.save_integration_secret(a2a.KEY,c)
        with patch.object(a2a,'_http',return_value=Response({'access_token':'new-access',
                'refresh_token':'rotated-refresh','expires_in':3600})) as http:
            self.assertEqual(a2a._access()[1],'new-access')
            self.assertEqual(http.call_args.kwargs['data']['refresh_token'],'sensitive-refresh')
        self.assertEqual(storage.integration_secret(a2a.KEY)['refresh_token'],'rotated-refresh')
        a2a.disconnect(ADMIN['email'])
        self.assertFalse(a2a.status()['connected'])
        with self.assertRaises(a2a.A2AError):
            a2a._access()

    def test_callback_origin_accepts_https_or_loopback_only(self):
        for url in ('http://192.0.2.69:8012','https://user:pass@safe.test','https://safe.test/?next=evil'):
            with self.assertRaises(a2a.A2AError):
                a2a.callback_url(url)
        self.assertEqual(a2a.callback_url('https://app.example/'),'https://app.example/api/rovo-a2a/callback')

    def test_stream_accepts_completed_artifacts_and_appends_chunks(self):
        rid='test-id'
        events=[{'id':rid,'result':{'kind':'artifact-update','taskId':'task-1',
            'artifact':{'artifactId':'answer','parts':[{'kind':'text','text':'Update enrollment'}]}}},
            {'id':rid,'result':{'kind':'artifact-update','taskId':'task-1','append':True,
            'artifact':{'artifactId':'answer','parts':[{'kind':'text','text':', then retry.'}]}}},
            {'id':rid,'result':{'kind':'status-update','taskId':'task-1','status':{'state':'completed'}}}]
        result=a2a._stream(Response(events=events),rid)
        self.assertEqual(result['text'],'Update enrollment, then retry.')
        self.assertEqual(result['task_id'],'task-1')

    def test_rovo_token_chunks_are_combined_even_when_append_flag_is_false(self):
        rid='chunk-test'
        events=[{'id':rid,'result':{'kind':'artifact-update','taskId':'task-1',
            'append':False,'lastChunk':False,'artifact':{'artifactId':'answer',
            'parts':[{'kind':'text','text':text}]}}} for text in ('**Root', ' Cause**\n', 'Invalid identifier.')]
        events.append({'id':rid,'result':{'kind':'status-update','taskId':'task-1',
                       'status':{'state':'completed'},'final':True}})
        self.assertEqual(a2a._stream(Response(events=events),rid)['text'],
                         '**Root Cause**\nInvalid identifier.')

    def test_non_chunk_artifact_replacements_do_not_duplicate_old_text(self):
        rid='replacement-test'
        events=[{'id':rid,'result':{'kind':'artifact-update','artifact':{
            'artifactId':'answer','parts':[{'kind':'text','text':text}]}}}
            for text in ('draft text', 'final resolution')]
        events.append({'id':rid,'result':{'kind':'status-update','status':{'state':'completed'}}})
        self.assertEqual(a2a._stream(Response(events=events),rid)['text'],'final resolution')

    def test_completed_markdown_fragment_is_not_a_valid_resolution(self):
        rid='fragment-test'
        events=[{'id':rid,'result':{'kind':'artifact-update','artifact':{
            'artifactId':'answer','parts':[{'kind':'text','text':'**'}]}}},
            {'id':rid,'result':{'kind':'status-update','status':{'state':'completed'}}}]
        with self.assertRaises(a2a.A2AError):
            a2a._stream(Response(events=events),rid)

    def test_stream_rejects_partial_conflicting_request_ids_and_org_errors(self):
        for events in ([{'id':'request','result':{'kind':'artifact-update','artifact':{
                'artifactId':'a','parts':[{'kind':'text','text':'Unfinished procedure'}]}}}],
                [{'id':'other-request','result':{'kind':'message','parts':[{'kind':'text','text':'fix'}]}}],
                [{'id':'request','result':{'kind':'status-update','status':{'state':'input-required'}}}]):
            with self.assertRaises(a2a.A2AError):
                a2a._stream(Response(events=events),'request')
        with self.assertRaisesRegex(a2a.A2AError,'Agent2Agent'):
            a2a._stream(Response(events=[{'id':'request','error':{'code':-32007,'message':'unsafe detail'}}]),'request')

    def test_preview_returns_rovo_output_without_posting_and_records_engine(self):
        self.connect()
        issue={'key':'SAC-200','query_text':'Failed Step: Activate\nError: duplicate enrollment'}
        source={'key':'SAC-100','step':'Activate','error':'duplicate enrollment','comment':'Correct enrollment ID, then retry.',
                'url':'https://jira.example/browse/SAC-100'}
        def http(method,url,**kwargs):
            self.assertEqual(kwargs['json']['method'],'message/stream')
            prompt=kwargs['json']['params']['message']['parts'][0]['text']
            self.assertIn('Do not create',prompt)
            rid=kwargs['json']['id']
            return Response(events=[{'id':rid,'result':{'kind':'message','parts':[{
                'kind':'text','text':'=== FIX ===\n1. Correct enrollment ID, then retry.\n=== END ==='}]}}])
        with patch('falloutassist_next.plugins.JiraPlugin') as jira, patch.object(a2a,'_http',side_effect=http):
            jira.return_value.issue.return_value=({},issue)
            jira.return_value._search.return_value=[{'key':'SAC-100'}]
            jira.return_value.fetch_resolution_candidate.return_value=source
            result=a2a.resolution('SAC-200',['SAC-100'],ADMIN['email'])
            jira.return_value.publish_comment.assert_not_called()
        self.assertEqual(result['engine'],'Atlassian Rovo A2A')
        self.assertTrue(result['review_required'])
        self.assertEqual(a2a.status()['last_preview']['issue_key'],'SAC-200')

    def test_admin_api_csrf_and_callback_logs(self):
        from falloutassist_next import web
        from fastapi.testclient import TestClient
        with TestClient(web.app,base_url='http://localhost:18012') as client:
            self.assertEqual(client.get('/api/rovo-a2a').status_code,200)
            web.app.dependency_overrides[web._admin]=lambda: ADMIN
            try:
                for path in ('connect','disconnect','preview'):
                    self.assertEqual(client.post('/api/rovo-a2a/'+path,json={}).status_code,403)
                csrf=client.get('/api/session').json()['csrf']
                with patch.object(a2a,'discover',return_value=copy.deepcopy(DISCOVERY)), \
                        patch.object(a2a,'_http',return_value=Response({'client_id':'client'})):
                    start=client.post('/api/rovo-a2a/connect',json={},headers={'x-csrf-token':csrf})
                self.assertEqual(start.status_code,200,start.text)
                state=parse_qs(urlsplit(start.json()['authorization_url']).query)['state'][0]
                with patch.object(a2a,'_http',return_value=Response({'access_token':'access','refresh_token':'refresh','expires_in':3600})):
                    response=client.get('/api/rovo-a2a/callback',params={'state':state,'code':'code'},follow_redirects=False)
                self.assertEqual(response.status_code,303,response.text)
                self.assertEqual(response.headers['referrer-policy'],'no-referrer')
                self.assertTrue(client.get('/api/rovo-a2a').json()['connected'])
            finally:
                web.app.dependency_overrides.clear()
        record=logging.LogRecord('uvicorn.access',logging.INFO,'',0,'%s - %s %s HTTP/%s %d',
            ('client','GET','/api/rovo-a2a/callback?code=secret&state=secret','1.1',303),None)
        web._OAuthLogFilter().filter(record)
        self.assertNotIn('secret',record.getMessage())


if __name__=='__main__':
    unittest.main()
