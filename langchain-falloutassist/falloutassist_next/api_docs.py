"""Swagger descriptions and schemas for the existing dashboard/service API."""
from fastapi.openapi.utils import get_openapi

from .config import DEFAULTS, CONNECTION_DEFAULTS, SECRET_KEYS


TAGS = [
    {"name": "Main APIs", "description": "Request a workaround, submit/poll agent jobs, or run an explicitly requested new-ticket scan."},
    {"name": "Ingestion APIs", "description": "Other important APIs: stored Jira totals, new/updated ingestion counts, history refresh/rebuild and JQL previews."},
    {"name": "Configuration APIs", "description": "Other important APIs: all retrieval, top-K, reranking, cache, scheduler, publishing and connection settings."},
    {"name": "Documents and SOP APIs", "description": "Upload, inspect, index or remove SOP/reference documents."},
    {"name": "Rovo Agent APIs", "description": "Rovo A2A connection status, browser OAuth and separate resolution previews."},
    {"name": "Supporting APIs", "description": "Evidence, saved reviews, workarounds, rules, feedback, audit and browser session support."},
]


def _object(properties=None, required=None, **extra):
    return {"type": "object", "properties": properties or {},
            **({"required": required} if required else {}), **extra}


def _schemas():
    settings = {}
    descriptions = {
        "retrieval_top_k": "Usable tickets requested from each retrieval source. Either can return fewer; MCP discovery is capped at 30 candidates.",
        "rerank_pool_k": "Bounded deduplicated comparison pool; must not exceed retrieval_top_k.",
        "jira_review_top_k": "Final drafting sources; must not exceed rerank_pool_k.",
        "rovo_mcp_always_search": "Default true: local top K plus MCP top K, then one combined rerank. False restores local-first MCP fallback.",
        "routing_min_sources": "Applicable, actionable, traceable sources needed to stop investigation fallback.",
        "routing_min_llm_relevance": "Real AI relevance threshold used with applicability and evidence checks, not a success probability.",
        "ingest_interval_minutes": "History refresh interval in minutes. Zero disables automatic ingestion.",
        "scan_interval_minutes": "New-ticket scan interval in minutes. Zero disables automatic scans.",
        "publish_enabled": "Allows Jira posting during scans only when dry_run is false and evidence/review checks pass.",
        "dry_run": "True saves scan previews without posting. Manual previews and knowledge jobs never post.",
    }
    for key, value in DEFAULTS.items():
        kind = "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "number" if isinstance(value, float) else "string"
        settings[key] = {"type": kind, "default": value, "description": descriptions.get(key, key.replace('_', ' ').capitalize())}
    source = _object({"key": {"type": "string"}, "url": {"type": "string"},
                      "origins": {"type": "array", "items": {"type": "string"}}})
    draft = _object({"decline": {"type": "boolean"}, "reason": {"type": "string"},
        "steps": {"type": "array", "items": _object({"text": {"type": "string"},
            "source_key": {"type": "string"}, "source_excerpt": {"type": "string"},
            "supporting_sources": {"type": "array", "items": _object({
                "source_key": {"type": "string"}, "source_excerpt": {"type": "string"}})}})}})
    route = _object({"stage": {"type": "string"}, "status": {"type": "string"},
        "reason": {"type": "string"}, "elapsed_ms": {"type": "integer"},
        "candidate_count": {"type": "integer"}, "requested_top_k": {"type": "integer"}})
    proposal = _object({"status": {"type": "string"}, "confidence": {"type": "string", "enum": ["", "low", "medium", "high"]},
        "review_required": {"type": "boolean", "description": "If true, the consuming service must obtain human review before acting."},
        "reasoning": {"type": "array", "items": {"type": "string"}},
        "final_suggestion": draft, "source_keys": {"type": "array", "items": {"type": "string"}},
        "source_details": {"type": "array", "items": source}, "retrieval_route": {"type": "array", "items": route}})
    package = _object({"schema_version": {"type": "string"}, "package_id": {"type": "string"},
        "generated_at": {"type": "string"}, "issue": {"type": "object"},
        "status": {"type": "string", "enum": ["ready", "partial", "failed"]},
        "proposals": _object({"falloutassist": proposal, "rovo_a2a": _object({
            "status": {"type": "string"}, "suggestion": _object({"text": {"type": "string"},
                "review_required": {"type": "boolean"}, "task_id": {"type": "string"}, "engine": {"type": "string"}}),
            "sources": {"type": "array", "items": source}, "retrieval": {"type": "object"},
            "knowledge_evidence": {"type": "array", "items": {"type": "object"}}})}),
        "execution_guidance": _object({"may_execute": {"type": "boolean", "const": False},
            "required_checks": {"type": "array", "items": {"type": "string"}}, "note": {"type": "string"}})})
    run = _object({name: {"type": "integer", "minimum": 0} for name in (
        "fetched", "indexed", "changed_documents", "new_documents", "updated_documents", "removed_documents", "skipped")})
    run["properties"].update(ok={"type": "boolean"}, busy={"type": "boolean"},
        full_rebuild={"type": "boolean"}, created_at={"type": "string"}, quality_counts={"type": "object"})
    return {
        "FalloutAssistSettings": _object(settings, description="All runtime settings. PUT accepts a partial object. Weights must sum to 1 and drafting/pool K must fit retrieval K.", additionalProperties=False),
        "FalloutAssistConnectionsUpdate": _object({key: {"type": "string",
            **({"writeOnly": True, "format": "password"} if key in SECRET_KEYS else {}),
            "description": ("Legacy single FalloutAssist key; prefer POST /api/falloutassist-api-keys to create named keys." if key == "KNOWLEDGE_ROUTER_API_KEY" else
                            "Secret: omit to preserve; empty string clears it." if key in SECRET_KEYS else key.replace('_', ' ').capitalize())}
            for key in CONNECTION_DEFAULTS}, additionalProperties=False),
        "FalloutAssistConnections": _object({"values": {"type": "object", "description": "Non-secret connection configuration."},
            "secrets": {"type": "object", "additionalProperties": {"type": "boolean"}, "description": "Whether each secret is configured. Secret values are never returned."}}),
        "FalloutAssistJobRequest": _object({"issue_key": {"type": "string", "examples": ["SAC-12345"]},
            "source_keys": {"type": "array", "items": {"type": "string"}, "maxItems": 5, "default": [],
                            "description": "Optional historical keys for the separate Rovo proposal."}}, ["issue_key"]),
        "FalloutAssistJobAccepted": _object({"job_id": {"type": "string"}, "issue_key": {"type": "string"},
            "status": {"type": "string", "enum": ["queued", "running"]}, "deduplicated": {"type": "boolean"},
            "status_url": {"type": "string", "description": "Poll this relative URL with the same Bearer key."}}),
        "FalloutAssistJobStatus": _object({"job_id": {"type": "string"}, "issue_key": {"type": "string"},
            "status": {"type": "string", "enum": ["queued", "running", "ready", "partial", "failed"]},
            "progress": {"type": ["object", "null"]}, "created_at": {"type": "string"},
            "updated_at": {"type": "string"}, "result": {"anyOf": [package, {"type": "null"}]},
            "error": {"type": ["string", "null"]}}),
        "FalloutAssistPreviewRequest": _object({"issue_key": {"type": "string"},
            "local_only": {"type": "boolean", "default": False}}, ["issue_key"]),
        "FalloutAssistPreview": _object({"issue_key": {"type": "string"},
            "tier": {"type": "string", "enum": ["low", "medium", "high"]},
            "review_required": {"type": "boolean"}, "status": {"type": "string"},
            "reasons": {"type": "array", "items": {"type": "string"}},
            "comment_body": {"type": "string", "description": "Preview text only; no Jira comment was posted."},
            "source_keys": {"type": "array", "items": {"type": "string"}},
            "source_details": {"type": "array", "items": source},
            "retrieval_route": {"type": "array", "items": route},
            "supervisor_result": _object({"draft": draft, "confidence": {"type": "string"}})}),
        "FalloutAssistIngestRequest": _object({"full": {"type": "boolean", "default": False,
            "description": "False performs incremental refresh; true rebuilds vectors. Both use the saved authoritative history JQL."}}),
        "FalloutAssistIngestResult": run,
        "FalloutAssistIngestionStatus": _object({"running": {"type": "boolean"},
            "totals": _object({key: {"type": "integer"} for key in ("stored", "searchable", "excluded")}),
            "last_completed_run": {"anyOf": [run, {"type": "null"}], "description": "Latest successful run. Older runs may lack new/updated/removed counts; missing does not mean zero."},
            "configuration": _object({"history_jql": {"type": "string"}, "inflow_jql": {"type": "string"},
                                      "settings": {"$ref": "#/components/schemas/FalloutAssistSettings"}})}),
        "FalloutAssistJqlPreviewRequest": _object({"kind": {"type": "string", "enum": ["history", "inflow"]},
                                                   "jql": {"type": "string", "maxLength": 4000}}, ["kind", "jql"]),
        "FalloutAssistApiKeyCreate": _object({"name": {"type": "string", "minLength": 1,
            "maxLength": 64, "pattern": "^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$"}}, ["name"]),
        "FalloutAssistApiKeyRecord": _object({"id": {"type": "string"}, "name": {"type": "string"},
            "prefix": {"type": "string"}, "created_at": {"type": "string"},
            "last_used_at": {"type": ["string", "null"]}, "revoked_at": {"type": ["string", "null"]},
            "active": {"type": "boolean"}}),
        "FalloutAssistApiKeyCreated": _object({"key": {"type": "string", "writeOnly": True,
            "description": "Plaintext key returned once at creation."},
            "record": {"$ref": "#/components/schemas/FalloutAssistApiKeyRecord"}}),
        "FalloutAssistApiKeyList": _object({"created_count": {"type": "integer"},
            "active_count": {"type": "integer"}, "keys": {"type": "array",
            "items": {"$ref": "#/components/schemas/FalloutAssistApiKeyRecord"}}}),
        "FalloutAssistApiKeyRevoked": _object({"ok": {"type": "boolean"}}),
    }


OPERATIONS = {
    ("/api/preview", "post"): ("Preview a workaround", "Runs the configured retrieval graph and final evidence gate. Never posts a Jira comment.", "FalloutAssistPreviewRequest", "FalloutAssistPreview", {"issue_key": "SAC-12345"}),
    ("/api/knowledge/jobs", "post"): ("Submit a FalloutAssist agent job", "Returns immediately. The job supplies separate FalloutAssist/Rovo proposals with citations; it never executes fixes or posts comments. Queue capacity errors return 429 with Retry-After.", "FalloutAssistJobRequest", "FalloutAssistJobAccepted", {"issue_key": "SAC-12345"}),
    ("/api/knowledge/jobs/{job_id}", "get"): ("Poll a FalloutAssist agent job", "Poll every 2–5 seconds with a bounded client deadline. Partial results may be usable while Rovo is pending. Inspect each proposal's review_required and execution_guidance; ready never means approved for execution.", None, "FalloutAssistJobStatus", None),
    ("/api/ingestion/status", "get"): ("Get ingestion totals and latest-run changes", "Stored/searchable/excluded Jira totals exclude SOP passages. Latest completed run reports new, content-updated, removed and skipped tickets. Older records can omit the new counters.", None, "FalloutAssistIngestionStatus", None),
    ("/api/ingest", "post"): ("Refresh or rebuild Jira history", "Synchronous: can take minutes. full=false refreshes incrementally; full=true rebuilds vectors and restores uploaded-document indexing. Uses saved history JQL; existing mass-prune safeguards apply. A concurrent ingest returns 200 with ok=false,busy=true. Does not post Jira comments.", "FalloutAssistIngestRequest", "FalloutAssistIngestResult", {"full": False}),
    ("/api/settings", "get"): ("Get all runtime configuration", "All top-K, relevance, time-budget, cache, scheduler and publishing values.", None, "FalloutAssistSettings", None),
    ("/api/settings", "put"): ("Update runtime configuration", "Partial update; unspecified values stay unchanged. Weights must sum to 1; jira_review_top_k <= rerank_pool_k <= retrieval_top_k. Schedule intervals use minutes; zero disables that task. Changes apply to subsequent work.", "FalloutAssistSettings", "FalloutAssistSettings", {"retrieval_top_k": 20, "rerank_pool_k": 20, "jira_review_top_k": 5, "rovo_mcp_always_search": True}),
    ("/api/connections", "get"): ("Get connections and saved JQL", "Returns non-secret values and configured-secret flags. It never returns saved API keys, tokens or passwords.", None, "FalloutAssistConnections", None),
    ("/api/connections", "put"): ("Update connections or JQL", "Partial update. Omit secrets to preserve them; an empty secret clears it. Named FalloutAssist API keys are managed through /api/falloutassist-api-keys. History/inflow JQL is saved here, not in runtime settings.", "FalloutAssistConnectionsUpdate", "FalloutAssistConnections", {"JIRA_INGEST_JQL": "project = SAC AND status in (Resolved, Closed) ORDER BY updated DESC"}),
    ("/api/connections/test-jql", "post"): ("Preview a JQL filter without saving", "Read-only Jira query preview. Does not start ingestion or change the saved JQL.", "FalloutAssistJqlPreviewRequest", None, {"kind": "history", "jql": "project = SAC AND status in (Resolved, Closed) ORDER BY updated DESC"}),
    ("/api/falloutassist-api-keys", "get"): ("List FalloutAssist API key records", "Returns created and active counts plus each key's name, prefix, creation time, last use and revocation status. Key values and verifiers are never returned.", None, "FalloutAssistApiKeyList", None),
    ("/api/falloutassist-api-keys", "post"): ("Create a named FalloutAssist API key", "Creates and saves a key immediately. The plaintext key is returned once for copying; only its verifier and metadata are stored. It remains active until revoked.", "FalloutAssistApiKeyCreate", "FalloutAssistApiKeyCreated", {"name": "Deep Agent"}),
    ("/api/falloutassist-api-keys/{key_id}", "delete"): ("Revoke a FalloutAssist API key", "Revokes the named key immediately while retaining its metadata in the key list.", None, "FalloutAssistApiKeyRevoked", None),
    ("/api/scan", "post"): ("Review new tickets now", "May post Jira comments when saved publishing settings permit it and final evidence/review checks pass. Use /api/preview or knowledge jobs for read-only integration.", None, None, None),
}


def _tag(path):
    if path.startswith('/api/knowledge/jobs') or path in {'/api/preview', '/api/scan'}:
        return 'Main APIs'
    if path in {'/api/dashboard', '/api/ingest', '/api/ingestion/status', '/api/connections/test-jql'}:
        return 'Ingestion APIs'
    if path in {'/api/settings', '/api/connections', '/api/semantic-cache/clear'}:
        return 'Configuration APIs'
    if path.startswith('/api/falloutassist-api-keys'):
        return 'Configuration APIs'
    if path.startswith('/api/documents'):
        return 'Documents and SOP APIs'
    if path.startswith('/api/rovo-a2a'):
        return 'Rovo Agent APIs'
    return 'Supporting APIs'


def configure_api_docs(app):
    """Document actual routes without changing runtime request/response contracts."""
    def openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title='FalloutAssist API', version='1.0.0', routes=app.routes, tags=TAGS,
                    description='FalloutAssist workaround and management APIs. Under Configurations → FalloutAssist API, name and generate a key, then copy it once. In Swagger Authorize, enter the key alone. Service requests send Authorization: Bearer <key>. Browser dashboard access remains open in this deployment; browser mutations require a session CSRF token. The service key authorizes management operations as well as agent jobs. Current deployment TLS trust is configured separately in each API client.')
        components = schema.setdefault('components', {})
        schema['servers'] = [{'url': '/', 'description': 'The current FalloutAssist deployment origin.'}]
        components.setdefault('schemas', {}).update(_schemas())
        components.setdefault('securitySchemes', {}).update({
            'FalloutAssistApiKey': {'type': 'http', 'scheme': 'bearer',
                'description': 'FalloutAssist API key. Enter only the generated key, without the Bearer prefix.'},
            'BrowserCSRF': {'type': 'apiKey', 'in': 'header', 'name': 'X-CSRF-Token',
                'description': 'Dashboard-only alternative: call /api/session in the same browser, then use its CSRF token and session cookie.'}})
        for path, methods in schema['paths'].items():
            for method, operation in methods.items():
                if method not in {'get', 'post', 'put', 'patch', 'delete'}:
                    continue
                operation['tags'] = [_tag(path)]
                if path.startswith('/api/knowledge/jobs'):
                    operation['security'] = [{'FalloutAssistApiKey': []}]
                elif path.startswith('/api/') and path != '/api/session':
                    operation['security'] = ([{}, {'FalloutAssistApiKey': []}] if method == 'get' else
                                             [{'FalloutAssistApiKey': []}, {'BrowserCSRF': []}])
                details = OPERATIONS.get((path, method))
                if details:
                    summary, description, request, response, example = details
                    operation.update(summary=summary, description=description)
                    if request:
                        content = {'schema': {'$ref': '#/components/schemas/' + request}}
                        if example is not None:
                            content['example'] = example
                        operation['requestBody'] = {'required': True, 'content': {'application/json': content}}
                    if response:
                        code = ('202' if (path, method) == ('/api/knowledge/jobs', 'post') else
                                '201' if (path, method) == ('/api/falloutassist-api-keys', 'post') else '200')
                        operation['responses'][code] = {'description': 'Accepted; poll status_url' if code == '202' else 'Created; plaintext key is returned once' if code == '201' else 'Successful response',
                            'content': {'application/json': {'schema': {'$ref': '#/components/schemas/' + response}}}}
                if path.startswith('/api/') and path != '/api/session':
                    for code, message in {'400': 'Invalid configuration, issue key or operation input.',
                                          '401': 'Missing required key, or a supplied API key is invalid.',
                                          '403': 'Browser mutation is missing a valid session CSRF token.'}.items():
                        if (method == 'get' and code in {'400', '403'}) or (path.startswith('/api/knowledge/jobs') and code == '403'):
                            continue
                        operation['responses'].setdefault(code, {'description': message})
                if path == '/api/knowledge/jobs' and method == 'post':
                    operation['responses']['429'] = {'description': 'Job queue full. Retry later.',
                        'headers': {'Retry-After': {'schema': {'type': 'string'}, 'description': 'Suggested retry delay in seconds.'}}}
                if path == '/api/knowledge/jobs/{job_id}':
                    operation['responses']['404'] = {'description': 'Unknown or invalid job ID.'}
                if path == '/api/rovo-a2a/connect':
                    operation['security'] = [{'BrowserCSRF': []}]
                    operation['description'] = 'Browser-bound OAuth. Start from the dashboard and complete consent in the same browser. The FalloutAssist service key does not replace Atlassian OAuth or organization approval.'
                if path == '/api/rovo-a2a/callback':
                    operation['security'] = []
                    operation['description'] = 'Atlassian browser redirect only. Requires the session that started sign-in; do not call this endpoint manually.'
        app.openapi_schema = schema
        return schema
    app.openapi = openapi
