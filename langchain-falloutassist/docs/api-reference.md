# FalloutAssist API

## Swagger and authentication

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- Live OpenAPI specification: `/openapi.json`
- Exported specification: [openapi.json](openapi.json)
- Current deployment origin: `https://<your-app-host>:8012`

Open **Configurations > FalloutAssist API**, enter a name such as `Deep Agent`,
then select **Generate key**. The key is saved immediately and shown once. Select
**Copy key** to copy and hide it. The list shows the created and active counts,
name, short prefix, creation time, last use and revocation state. Multiple named
keys can stay active at once; revoke one from the list when access should end.
The full key is never returned again. FalloutAssist stores a one-way verifier
and the key metadata, not a readable copy of the secret.

Send `Authorization: Bearer <key>` on service requests. In Swagger, select
**Authorize > FalloutAssistApiKey** and enter the key alone, without `Bearer`.
The key authorizes configuration and ingestion management as well as agent jobs.
Configure the agent service's HTTPS trust with the app's CA certificate.

The existing dashboard remains open in this deployment. Browser mutations still
need the session cookie and `X-CSRF-Token` obtained from `/api/session`. A valid
service Bearer key is an alternative to browser CSRF; an invalid supplied key is
rejected. Agent-job endpoints always require the API key.

Client examples below use `FALLOUTASSIST_URL`, `FALLOUTASSIST_API_KEY` and
`FALLOUTASSIST_CA_BUNDLE` as client environment variables. The server's existing
environment/configuration identifier `KNOWLEDGE_ROUTER_API_KEY` is retained for
backward compatibility; its user-facing name is **FalloutAssist API key**.

## Main APIs

| Method | Endpoint | Purpose | External effect |
| --- | --- | --- | --- |
| POST | `/api/preview` | Synchronous workaround preview for a Jira key | No Jira posting |
| POST | `/api/knowledge/jobs` | Submit a queued job for another agent | No posting or remediation |
| GET | `/api/knowledge/jobs/{job_id}` | Poll proposals, citations and review status | Read only |
| POST | `/api/scan` | Review tickets selected by the saved inflow JQL | Can post comments when publishing is enabled and all gates pass |

Preview request:

```json
{"issue_key":"SAC-12345","local_only":false}
```

The default graph retrieves local top K and Rovo MCP top K, deduplicates/fuses
ranks, and reranks the bounded comparison pool once. Insufficient combined
evidence invokes the Rovo agent, followed by independent source checks and the
final evidence gate. `local_only:true` explicitly skips both Rovo integrations.
The saved `rovo_mcp_always_search:false` setting restores local-first fallback.

Knowledge job request:

```bash
curl --cacert "$FALLOUTASSIST_CA_BUNDLE" \
  -H "Authorization: Bearer $FALLOUTASSIST_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"issue_key":"SAC-12345"}' \
  "$FALLOUTASSIST_URL/api/knowledge/jobs"
```

An accepted job returns HTTP 202:

```json
{
  "job_id":"0123456789abcdef0123456789abcdef",
  "issue_key":"SAC-12345",
  "status":"queued",
  "deduplicated":false,
  "status_url":"/api/knowledge/jobs/0123456789abcdef0123456789abcdef"
}
```

Poll the returned `status_url` using the same key every 2–5 seconds, with a bounded
overall client deadline. Job states are `queued`, `running`, `ready`, `partial`
and `failed`. A completed FalloutAssist proposal can be available while the
independent Rovo proposal is pending. A Rovo cache miss schedules its background
refresh; poll the same job to see the updated proposal. The queued job accepts
optional `source_keys` containing at most five historical Jira keys.

Important response paths:

| Path inside the polling response | Meaning |
| --- | --- |
| `result.proposals.falloutassist.confidence` | High, Medium or Low evidence confidence |
| `result.proposals.falloutassist.review_required` | Human review is required before acting |
| `result.proposals.falloutassist.final_suggestion.steps` | Structured steps with exact source keys and excerpts |
| `result.proposals.falloutassist.retrieval_route` | Actual searches, statuses, reasons, counts and timings |
| `result.proposals.falloutassist.source_details` | Traceable source IDs/URLs and retrieval origins |
| `result.proposals.rovo_a2a` | Separate Rovo proposal, source evidence and availability |
| `result.execution_guidance.may_execute` | Always false: this package does not authorize remediation |

Medium evidence is not downgraded merely because a Rovo fallback failed. Medium
routed proposals remain review-only. Missing, unverified, contradictory or
unsupported evidence can produce Low. A `ready` job is a completed package, not
approval to execute its historical workaround.

## Other important APIs: ingestion

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/ingestion/status` | Jira totals, latest completed-run changes and current configuration |
| GET | `/api/dashboard` | Broader dashboard counts, models, corpus version, quality and integration status |
| POST | `/api/ingest` | Incremental refresh or full vector rebuild using saved history JQL |
| POST | `/api/connections/test-jql` | Preview a draft JQL filter without saving or ingesting |

Check totals first:

```bash
curl --cacert "$FALLOUTASSIST_CA_BUNDLE" \
  -H "Authorization: Bearer $FALLOUTASSIST_API_KEY" \
  "$FALLOUTASSIST_URL/api/ingestion/status"
```

| Status field | Meaning |
| --- | --- |
| `running` | An ingestion lease is active |
| `totals.stored` | Persisted Jira tickets, including excluded low-quality records |
| `totals.searchable` | High/Medium Jira sources eligible for retrieval |
| `totals.excluded` | Stored Jira records excluded from retrieval |
| `last_completed_run.new_documents` | Newly stored ticket keys in the latest successful run |
| `last_completed_run.updated_documents` | Existing tickets whose stored content/quality changed |
| `last_completed_run.removed_documents` | Tickets removed because they no longer match the authoritative history scan |
| `last_completed_run.skipped` | Unchanged tickets skipped during refresh |
| `last_completed_run.fetched` | Tickets returned by the history scan |
| `configuration.history_jql` / `inflow_jql` | Saved history and new-ticket filters |
| `configuration.settings` | Complete runtime settings, including top K and schedules |

Totals count Jira tickets, not uploaded SOP passages or individual embedding
vectors. `last_completed_run` is null before a successful ingest. Older audit
records can omit the new counters; missing does not mean zero. Failed/in-progress
runs do not replace the latest completed-run record. `indexed` is a retained
legacy aggregate and should not be interpreted as an embedding-vector count.

Incremental refresh body: `{"full":false}`. Full rebuild body: `{"full":true}`.
Both use the saved history JQL; incremental refresh is not a request to ingest
one arbitrary issue. The history scan is authoritative, and existing mass-prune
safeguards still apply. Uploaded documents are restored to the vector index
during a full rebuild.

Ingestion is synchronous and may take minutes. Use an appropriate HTTP timeout;
do not blindly retry an uncertain running request. A concurrent request returns
HTTP 200 with `{"ok":false,"busy":true}`. Poll `/api/ingestion/status` instead.

## Other important APIs: configuration

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/settings` | Read every runtime setting |
| PUT | `/api/settings` | Update any supported subset of runtime settings |
| GET | `/api/connections` | Read non-secret connection values and secret-configured flags |
| PUT | `/api/connections` | Update saved JQL, connections and model configuration |
| GET | `/api/falloutassist-api-keys` | List created/active counts and safe key metadata |
| POST | `/api/falloutassist-api-keys` | Create and immediately activate a named key; returns plaintext once |
| DELETE | `/api/falloutassist-api-keys/{key_id}` | Revoke a key and retain its metadata |
| POST | `/api/semantic-cache/clear` | Invalidate saved AI match scores |

Example top-K, routing and scheduler update:

```json
{
  "retrieval_top_k":30,
  "rerank_pool_k":20,
  "jira_review_top_k":5,
  "rovo_mcp_always_search":true,
  "routing_min_sources":1,
  "routing_min_llm_relevance":0.75,
  "ingest_interval_minutes":60,
  "scan_interval_minutes":0
}
```

Send that object with PUT `/api/settings`. Unspecified settings stay unchanged.
`jira_review_top_k <= rerank_pool_k <= retrieval_top_k` must hold. Retrieval, AI
and feedback weights must sum to 1. Schedule intervals are minutes; zero disables
the task. The worker notices schedule changes within its five-second loop.
The Swagger `FalloutAssistSettings` schema lists **all** supported fields,
including time budgets, cache, confidence, temperature and publishing controls.

JQL is connection configuration; send PUT `/api/connections`, for example:

```json
{
  "JIRA_INGEST_JQL":"project = SAC AND status in (Resolved, Closed) ORDER BY updated DESC",
  "JIRA_INFLOW_JQL":"project = SAC AND statusCategory != Done ORDER BY created DESC"
}
```

Use POST `/api/connections/test-jql` with `{"kind":"history","jql":"..."}`
before saving. It returns the exact draft and matching samples without modifying
configuration. Editing filters does not itself start a manual ingestion.

Connection secrets are write-only. Omit a secret to preserve it; an empty value
clears it. Named FalloutAssist API keys use a verifier so the application can
authenticate them without retaining a readable copy. Other saved connection
secrets are encrypted in application storage.

## Supporting APIs

Swagger groups these separately from the main and management operations:

- **Documents and SOP APIs:** `/api/documents`, file upload/download, retry indexing and deletion.
- **Rovo Agent APIs:** `/api/rovo-a2a`, browser connect/callback/disconnect and separate preview.
- **Supporting APIs:** `/api/sources`, `/api/workarounds`, version evidence/synthesis/outcomes,
  saved `/api/investigations`, `/api/rules`, `/api/feedback`, `/api/audit` and audit verification.

Rovo OAuth is browser-bound. Start it through the dashboard and complete consent
in the same browser. The FalloutAssist API key does not replace Atlassian OAuth
or organization domain approval. File upload uses multipart form data; all form
fields are listed by Swagger. Document/source deletion and rule approval are
explicit write operations.

## Error handling

| Status | Meaning |
| --- | --- |
| 400 | Invalid issue key, JQL, configuration, bounds or operation input |
| 401 | A required service key is missing, or the supplied key is invalid |
| 403 | A browser mutation lacks a valid session CSRF token |
| 404 | Requested job, document, source or version does not exist |
| 409 | A document/index or Rovo operation is already running |
| 413 | Uploaded document exceeds the size limit |
| 422 | Malformed JSON/request types or missing framework-required fields |
| 429 | Knowledge queue is full; honor `Retry-After` |
| 5xx | Upstream or server failure; inspect the operation result/audit before retrying writes |

An HTTP 200 response does not prove a workaround is safe. Inspect confidence,
`review_required`, each source and operation-specific statuses such as
`busy`, `partial`, `timeout`, or `unavailable`.
