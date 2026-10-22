# FalloutAssist: LangGraph supervisor with Jira and Salesforce specialists

Implementation reviewed on **2026-10-20**. This document describes this checkout;
inspect the running process, saved settings and deployed revision before drawing
conclusions about another environment. See the [SOP](SOP.md) for commands,
troubleshooting, deployment and the coding-assistant handoff.

## Start here

The current local target is **http://127.0.0.1:8012**. Run from this directory:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8012 --workers 1 --access-log
```

Use this directory's virtual environment. Both this `app.py` and the repository's
root `app.py` import `falloutassist_next.web:app`; root `legacy_app.py` retains the
earlier implementation. The port is a launch option, not an application version.
The configured `public_base_url` is independent and still defaults to
`http://localhost:8010` in code; set it to the intended reachable URL before using
feedback links or live publishing.

| Read next | What it explains |
| --- | --- |
| [System overview](#system-overview) | Three separate request flows and storage responsibilities |
| [Flow](#flow) | Main FalloutAssist investigation and publish gate |
| [Downstream API](#downstream-knowledge-routing-api) | Asynchronous knowledge packages for another agent |
| [Storage](#what-is-stored) | Relational state, Chroma and evidence quality |
| [Setup](#setup) | Authentication and connection configuration |
| [Retrieval](#rovo-retrieval-and-composite-reranking) | Local/Rovo fusion and evidence-based reranking |
| [Caches](#semantic-cache-for-ai-match-scores) | What Redis caches and what it does not cache |
| [Source map](#source-map-for-a-coding-assistant) | Files to inspect for a particular change |
| [Limits](#implemented-limits-and-open-constraints) | Memory, latency and deployment boundaries |

## System overview

FalloutAssist supplies historical knowledge and proposed procedures. It does not
perform remediation in Aria, Salesforce or other downstream systems.

```mermaid
flowchart TD
    J[Jira REST: current tickets and resolution comments]
    I[Ingestion and source-quality assessment]
    DB[(PostgreSQL or explicit SQLite)]
    E[Hugging Face embeddings]
    C[(Persistent Chroma index)]
    M[Rovo MCP: matching issue keys]
    F[FalloutAssist LangGraph investigation]
    L[Azure OpenAI: reranking and grounded drafting]
    R[(Optional Redis / RedisVL match scores)]
    A[Rovo A2A: its own cited synthesis]
    P[Preview, audit and optional scan comment]
    K[Knowledge API: separate FA and Rovo proposals]
    D[Future action agent: verifies and executes with its own tools]
    J --> I --> DB
    I --> E --> C
    DB --> F
    C --> F
    J --> F
    M --> F
    F <--> L
    L <--> R
    F --> P
    J --> A
    M --> A
    A <--> DB
    F --> K
    A --> K
    K --> D
```

| Path | Retrieval and generation | External effect |
| --- | --- | --- |
| FalloutAssist preview / scan | `local`: local only. `rovo`: local top K + MCP top K by default, deduplicated and reranked together; insufficient combined evidence escalates to a source-validated A2A investigation | Manual preview does not post. A scan can publish through the final gated node |
| Rovo agent preview | Jira REST loads the current ticket; MCP discovers history; Jira REST fetches validated evidence; Rovo A2A generates its own proposal, unless cached | Saves a preview; does not post Jira comments |
| Knowledge job | LangGraph runs the FalloutAssist branch alongside a Rovo cache lookup; a cold Rovo miss schedules background refresh | Returns separate proposals and evidence; does not publish or execute fixes |

The diagram abbreviates cache use: Rovo history and synthesis caches are encrypted
records in the application database, not Redis. MCP contributes retrieval keys;
A2A contributes Rovo's own generated resolution. They are not interchangeable.

This application runs in its own port and database configuration. The LangGraph
supervisor coordinates the Jira historical-resolution specialist and an optional
read-only Salesforce specialist through a shared specialist registry. Aria,
Matrixx and Nokia can be added by implementing that contract and registering
their specialist.

## Flow

`Jira inflow -> load Jira context -> local semantic/lexical top K + MCP top K
-> merge/deduplicate/RRF/rerank/assess
-> sufficient: draft; insufficient: Rovo agent -> verify sources and claims
-> join read-only specialists -> supervisor -> evidence gate -> compose -> audit -> optional publish`

The graph uses explicit conditional edges for retrieval mode and evidence quality. See
[Retrieval routing](docs/retrieval-routing.md) for the compiled graph map, decision
rule, time budgets, review-only behavior and regression checks. The Rovo agent is skipped
when combined evidence passes the real reranker and conflict checks. Settings can
turn off `rovo_mcp_always_search` to restore local-first MCP fallback. Medium
evidence remains Medium during fallback errors, with manual review separate from
the confidence rating; better supported evidence can increase it to High.
The separate Rovo preview and knowledge-proposal branch remain
independent; explicit local-only previews never call either Rovo integration.

The Jira specialist is a LangChain `create_agent` with one bounded tool for reading
selected historical resolutions. The Salesforce specialist is a LangChain agent
whose tool can only query allowlisted objects, fields, and lookup columns. The
graph owns routing, parallel investigation, evidence synthesis, and the publish
gate. Specialists cannot post Jira comments or write to Salesforce. Approved Jira
rules are matched deterministically and require admin review before they can
produce a high quality suggestion.

## Downstream knowledge routing API

See the grouped [FalloutAssist API guide](docs/api-reference.md), live Swagger
at `/docs`, and the downloadable schema at `/openapi.json`. The guide includes
ingestion totals/new/updated counts, incremental/full ingestion, all settings,
top K, schedules, JQL updates, service authentication and supporting APIs.

The read-only `POST /api/knowledge/jobs` endpoint accepts a Jira `issue_key` and
optional `source_keys` (up to five). It returns HTTP 202 with a `job_id` and
`status_url`. The FalloutAssist suggestion and a reusable Rovo knowledge lookup
run concurrently. A cache hit returns the stored Rovo synthesis with its source
citations. On a cache miss, the API returns the FalloutAssist result as `partial`
and schedules Rovo discovery and synthesis in a separate background worker; poll
the same `status_url` to see the Rovo proposal once it has been stored. Successful
Rovo syntheses are encrypted and reused for 24 hours for matching site, failure
step, error, order type, order reason, and release. Manual `source_keys` continue
to use the direct Rovo path. The package includes source citations, evidence,
confidence, retrieval metadata, and checks the downstream agent must perform.
FalloutAssist does not execute the suggested actions. Knowledge jobs use a
bounded persistent single-worker queue; Rovo refreshes use their own bounded
persistent queue and resume after application startup. The FalloutAssist branch
still follows `JIRA_HISTORY_BACKEND`; when it is set to `rovo`, that branch also
uses MCP retrieval and its own response time can include Atlassian latency.

For service-to-service access, open **Configurations > FalloutAssist API**,
enter a descriptive key name, then select **Generate key**. The app stores the
key immediately and displays it once; **Copy key** copies and hides it. The
screen lists the created/active totals, key names, short prefixes, creation time,
last use and revocation state. Multiple named keys can remain active, and each
can be revoked independently. The full key is never shown again; the app stores
a one-way verifier and metadata. Alternatively, a legacy
`KNOWLEDGE_ROUTER_API_KEY` remains accepted. Send `Authorization: Bearer <key>`
on service requests. The dashboard has no account login; agent job endpoints
require a bearer key. The
queue accepts at most ten waiting or running jobs; duplicate requests for the
same issue and source keys share the existing queued job.

## What is stored

- PostgreSQL (primary) or SQLite (explicit local fallback): settings, encrypted
  connection secrets,
  ingested resolution text, source quality tiers, versioned workarounds and release labels,
  approved rules, investigations, publication IDs, feedback, scan state, and a
  hash-linked application audit log.
- Chroma via LangChain: an independent persistent vector index of the ingested
  resolutions. PostgreSQL full-text search or SQLite FTS5 supplies lexical recall.
- Langfuse, if enabled: model and tool traces. The selected application database
  remains the application action audit; Langfuse is optional.

Source quality `high` / `medium` / `low` means how complete an ingested
resolution is. An investigation's tier is a separate decision about whether a
new Jira ticket has enough grounded evidence for a suggested workaround. A
similarity score is not a probability that a fix is correct.

Source quality is assessed automatically from the actual `Solution applied`,
`Resolution`, `Workaround`, or `Fix` field (including multiline values), with plain
resolution text as a fallback. Numbering does not establish quality, and high
source quality does not mean a human approved it. Explicit approval belongs to
admin-reviewed rules. Placeholder solutions, closure notices, Salesforce activity
snapshots and FalloutAssist comments cannot supply corrective evidence.

Local ingestion and Rovo hydration use one comment-selection policy, including
the same post-resolution age limit. Incremental ingestion reassesses unchanged
issues when the source-policy version changes; a destructive full rebuild is not
required to apply this policy. Existing saved investigations remain historical.

Before the drafting shortlist is cut, eligible sources are prioritized within the
retrieved/reranked pool. The final gate checks every action and every supporting
source against the failed operation, error message/code, known order context and
release. Shared error codes alone are insufficient. Exact resolution excerpts,
faithful action clauses, negation, technical values, conditions and prerequisite
order must be preserved. Ambiguous paraphrases and missing applicability context
produce a manual-review lead. Comments and feedback identify the actual primary
action source rather than an unrelated first retrieval result.

## Model display and document/SOP uploads

The Jira scope panel shows the current number of stored Jira tickets, the number
eligible for search, and the number retained only for review. Uploaded passages
are excluded from these ticket counts. Last-ingestion time and checked, changed
and unchanged counts describe the last completed run; running status refreshes
automatically.

Expand **Preview or edit saved JQL filters** to see the exact history and new-ticket
queries. Preview performs a read-only Jira search and shows sample ticket links
and a total when Jira provides one. Edit opens the corresponding Jira setup field;
unsaved drafts survive tab changes. Previewing a draft does not save it or ingest
tickets. **Save ticket filters** applies edits to subsequent syncs and reviews.

The dashboard displays the configured Azure chat deployment and embedding model
above every tab. Azure deployment names can be custom aliases; this display does
not infer a different underlying model name. Changes saved in Connections refresh
the model display immediately.

Use **Documents & SOPs** to upload PDF, DOCX, UTF-8 TXT or Markdown files up to
10 MB. PDFs must contain selectable text (OCR is not performed). Extracted text is
limited to 200,000 characters and PDFs to 100 pages. Add an optional title, document
type, failed step, error, order type/reason and release. Recognized structured
fields in the document fill missing applicability fields.

Original files and extracted text are retained in the application database.
Usable passages enter local semantic/lexical retrieval alongside Jira evidence;
uploads do not bypass the existing applicability gate. Long documents are split
into cited passages and require review of the complete procedure. The tab shows
searchable passage counts, extracted text, original downloads and removal controls.
An indexing failure retains the file and offers **Retry indexing**, without
leaving partially searchable evidence. Jira syncs preserve uploads, and full Jira
rebuilds restore their vectors. Upload/index/delete operations share the ingestion
lease to avoid racing a rebuild. API reads are available to anyone who can reach
the dashboard; mutations also require a CSRF token.

Endpoints: `GET/POST /api/documents`, `GET/DELETE /api/documents/{id}`,
`GET /api/documents/{id}/file`, and `POST /api/documents/{id}/index`.
Uploads use multipart form data with `file`, `title`, `kind` (`document` or `sop`),
`step`, `error`, `order_type`, `order_reason` and `release`. Install the updated
requirements before starting the app.

## Readable workaround procedures

Expanding a version in **Find a workaround** shows its stored resolution text
immediately. The `GET /api/workaround-versions/{version_id}` endpoint
loads full original comments and a current saved rewrite without invoking the LLM.
When no valid saved rewrite exists, the UI separately calls the
CSRF-protected `POST /api/workaround-versions/{version_id}/synthesis` endpoint.
The original remains readable while AI works, with an elapsed-time indicator.
Concurrent views share pending browser requests; reopening a saved procedure does
not start model work. It uses the saved
Azure chat connection and response temperature to rewrite the exact version as a
summary, numbered actions, applicability notes and missing details. Full stored
Jira resolution comments supplement the extracted solution only when that
solution matches the same version; newer, different fixes cannot leak into old
versions. Original comments and per-step supporting quotes remain visible.

A LangChain structured-output call writes the procedure, exact source quotes are
checked in code, and a second model call reviews its factual support. Salesforce
snapshots, assistant comments and automatic closures cannot become synthesized
procedures. Unavailable or ungrounded rewrites show a retry state with the source
still accessible. This feature does not execute fixes, change confidence or post
Jira comments. Generation and failures are audited, with optional Langfuse traces.

Successful rewrites are cached in application storage (PostgreSQL, or the explicit
SQLite fallback). Changes to evidence, version, release context, model deployment,
temperature or synthesis prompt invalidate the cache. Generation occurs only for
expanded versions; a shared lease prevents duplicate concurrent model calls.
No re-ingestion is needed when a matching full comment is already stored.
An accidental multiple-response model output gets at most one repair attempt per
model call, while evidence checks and semantic review still apply.

Run the isolated synthesis checks from this directory with
`python -m unittest discover -s tests -v`. Model calls are mocked and temporary
databases are used; application data and Jira publishing are not touched.

## Setup

### Rovo's own resolution via A2A

The **Rovo agent** tab connects to Atlassian Rovo's A2A 0.3 gateway. This is separate
from the existing Rovo MCP semantic search. The app discovers the agent card and its
advertised client-registration endpoint, registers a public OAuth client, and uses
PKCE to connect an administrator's Atlassian account. OAuth state and credentials
are encrypted in `integration_secrets` with the existing session-derived encryption
key; preserve `SESSION_SECRET` during upgrades and database migration.

1. An organization administrator enables **Atlassian Administration > Rovo >
   Agent2Agent > Enable A2A connections**. Rovo and AI access must be enabled for the
   account/site as well.
2. Open FalloutAssist using HTTPS, or establish a localhost tunnel from the workstation:
   `ssh -N -L 18012:127.0.0.1:8012 prodapt-server`. Keep it running and open
   `http://127.0.0.1:18012`. FalloutAssist has no sign-in screen; complete Atlassian
   OAuth in the same browser. The normal LAN HTTP URL cannot be used as an OAuth
   callback by this integration.
   For production HTTPS, set `ROVO_PUBLIC_BASE_URL` to the exact browser origin
   and configure a browser-trusted certificate directly in Uvicorn or at the
   reverse proxy. Session cookies become Secure automatically for that origin.
   See `SOP.md` for direct-IP and proxy setups.
3. Select **Rovo agent > Connect Atlassian account**, then complete Atlassian sign-in
   and consent in the same browser. Do not create a client manually in the developer
   console or paste API tokens: A2A uses dynamic client registration and OAuth.
4. Enter the current Jira key and optionally up to five historical keys. Leaving the
   historical list blank uses Rovo MCP search to find candidate Jira keys. The app
   fetches usable resolution comments and asks Rovo to produce its own cited procedure.

The consent scopes are `read:me`, `offline_access`, and `full_access:chat:rovo`.
Atlassian's Rovo scope includes wider agent capabilities; the app sends a narrow
read-only review request. Anyone who can reach the dashboard can initiate a preview
after connecting an Atlassian account, and these requests are currently manual
previews, outside the scheduled suggestion/publishing workflow.
An A2A answer remains a proposal requiring evidence review, not a verified fix.
No A2A preview is posted to Jira or converted into a source-resolution record.
Partial/interrupted streams and responses lacking a completed task are rejected.
Disconnect removes the connection from this app; revoke upstream access in the
Atlassian account's connected-app settings if required.

Official references: [authentication](https://developer.atlassian.com/cloud/rovo-a2a/authentication/),
[organization enablement](https://developer.atlassian.com/cloud/rovo-a2a/admin-controls/),
and [streaming verification](https://developer.atlassian.com/cloud/rovo-a2a/end-to-end-testing/).

Use Python 3.11 or 3.12. From this directory:

1. Install `requirements.txt` in a fresh virtual environment.
2. Copy `.env.example` to `.env` and set `DATABASE_URL` for PostgreSQL and a
   long random `SESSION_SECRET`. The
   database must already exist, and the application user needs schema create and
   read/write privileges. Set `DATABASE_BACKEND=sqlite` only for a local fallback.
   The backend never switches automatically during a database outage; this keeps
   audit and publication state in one authoritative database. `SESSION_SECRET`
   protects CSRF/OAuth browser state and encrypted integration settings; it is not
   a FalloutAssist login password.
3. The dashboard has no interactive FalloutAssist login. Anyone who can reach its
   network listener can use dashboard controls. Keep local runs on loopback and
   restrict server access through a trusted network gateway or firewall. Jira,
   Atlassian and downstream-agent credentials remain separate integrations.
4. Optionally import existing feedback and rules once:
   `python migrate_legacy.py --from <path-to-old-project>`. Imported rules are
   drafts because failure parsing has changed; review and approve them in the
   new dashboard. Historical Jira resolutions are rebuilt by ingestion.
   If you already used this new application with SQLite, run
   `python migrate_sqlite_to_postgres.py` after setting `DATABASE_URL` to copy
   that application state into an empty PostgreSQL database. Then set
   `DATABASE_BACKEND=postgres` in `.env` before starting the API or worker.
5. From this application directory, start the local app with
   `.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8012 --workers 1 --access-log`.
   Open the dashboard and configure Jira, Azure OpenAI, feedback signing, and optionally Rovo and
   Langfuse in **Settings → Connections**. Set both JQLs for resolved history
   and active inflow. The embedding model downloads on first use unless already
   cached. If Jira has feedback or suggested-workaround custom fields, enter
   their IDs here. For MCP search, enter a cloud ID, email, and **Atlassian MCP V2**
   API token (stored as `ROVO_API_TOKEN`) with `search:rovo:agent-interface`.
   The A2A history path also attempts `searchJiraIssuesUsingJql`, which requires
   the applicable `search:jira:agent-interface` permission. Your Atlassian admin
   must allow token auth. This token is separate from Rovo Dev and A2A OAuth.
   Select **Local only** if Rovo is unavailable. Secret values are encrypted
   in the application database using a key derived from `SESSION_SECRET`, never
   returned to the browser, and omitted from audit events. Preserve that secret
   when migrating the database, or saved credentials cannot be decrypted.
   Existing operational `.env` values continue as fallback until overridden in
   the UI.
6. Start a separate worker: `python worker.py --loop`. Use `--once` for a single
   ingestion and scan pass. You can also trigger both from the dashboard.

The dashboard at `/` shows source quality, Jira operations, suggestion
investigations, feedback, rules, settings, connections (including Rovo and
Salesforce), workaround versions, specialist findings, and audit events. The
**Workaround Library** groups suggestions by order reason, failed step, and
error. Expanding a pattern shows current and previous procedure versions, release
labels, resolution dates, and evidence tickets. A newly observed procedure
becomes current even when an older version had strong feedback; repeated identical
procedures add evidence to the existing version. The **Jira operations** tab
keeps the legacy operator actions:
configure ingestion and scan frequencies, run an incremental sync, run the Jira
commenting scan, or perform a confirmed full re-ingest. A full re-ingest rebuilds
the Chroma vector collection from all issues matching the configured history
JQL. Configure historical retrieval top K, Jira deep-review top K, model
temperature, reranking weights, thresholds, connections, and publishing there.
Changing the embedding model requires a full vector rebuild. A signed
Jira feedback link displays a confirmation page and requires a logged-in user;
opening the link alone never submits a vote.

Jira and Rovo HTTPS connections use the operating system's certificate store by
default. For a company certificate authority, set **Settings → Connections →
Network and TLS → Custom CA bundle file path** to an existing PEM CA bundle.
This controls certificate validation; it does not configure an HTTP proxy.

## Rovo retrieval and composite reranking

The main graph starts with local vector plus PostgreSQL/SQLite full-text retrieval,
then uses the real reranker and applicability/evidence checks. Sufficient local
evidence skips both Rovo integrations. Otherwise MCP runs; reciprocal rank fusion
(RRF) merges its verified Jira evidence with useful local candidates before another
assessment. If combined evidence remains insufficient, a bounded Rovo-agent
investigation supplies structured claims for independent source verification.

The fused pool is scored for failed step, error/code, context and resolution before
the final top K is selected using retrieval, LLM and feedback weights. Errors,
timeouts and unavailable services remain distinct from successful empty/weak
searches in the persisted route. MCP rank never supplies a fabricated vector score;
the gate requires source traceability, applicable actionable evidence and real
reranker support. Exhausted fallbacks, unsupported actions and conflicting
procedures return review-only results. See [the routing decision rule](docs/retrieval-routing.md).

Human votes are aggregated per historical source, workaround version, and
normalized failure using a Beta(2,2) prior, so a small number of votes stays
close to neutral. Votes on superseded versions do not affect the current
procedure. Verified automated success/failure outcomes are recorded against a
specific version through the admin outcome API and displayed in the library.
As those verified outcomes accumulate, their share of the configured outcome
evidence signal grows relative to human votes. The default composite is
`25% retrieval rank + 50% LLM relevance + 25% human and automated outcome evidence`.
Weights are configurable in the dashboard and must sum to 100%; the Jira review
top K prioritizes evidence eligibility, then composite rank within each eligibility
group. Candidate component scores, eligibility reasons and
LLM rationale are saved with the investigation and shown in its detail view.
Suggestion reviews show the selected top tickets as clickable Jira IDs with
their composite match percentages. New comment drafts include the same ticket
IDs and match percentages beside a separate High/Medium/Low overall confidence.
The percentage ranks a historical match; it is not the probability a fix will
work. Missing scores are shown as not recorded, and approved rules do not receive
an invented ticket match percentage. Existing comments are changed only when a
subsequent publishing scan processes the issue and produces a changed comment body.

LLM reranking only orders candidates; the evidence quality gate still controls
whether a suggestion can be high confidence. If the reranker call fails, the
workflow records that failure and falls back to neutral LLM scores, leaving the
retrieval and human ranking available. Reranker scores are relative ranking
signals, not probabilities that a workaround is correct.

## Semantic cache for AI match scores

The reranker uses **Redis + RedisVL `SemanticCache`** by default for shared AI
relevance scores. An optional **Application database** backend uses PostgreSQL or
the explicit SQLite fallback for small/local deployments. An exact repeat reuses
scores without computing another query embedding. RedisVL initializes its vector
dimensions once per worker/model using the existing embedding model. A similar query
can reuse them only when cosine similarity meets the configured threshold and
failed step, exact error text, order type, order reason and release match. Step,
error, order type and order reason must be present for semantic reuse; incomplete
queries allow exact reuse only. Numeric tokens, identifiers, negation and key
operation words must also match. These conservative guards intentionally reduce
cache hits when context is uncertain.

The cache requires identical candidate keys, complete resolution evidence,
procedure versions, source quality, prompt version, chat/embedding model settings
and connection revision. Evidence changes invalidate reuse even outside the text
excerpt sent to the model. Retrieval, feedback/outcome weights, drafting,
Salesforce checks and evidence/publishing gates still run for every investigation.
This caches ranking scores, not suggested comments or live system state. The
existing version-specific procedure synthesis cache remains separate.

In **Settings**, control **Reuse AI match scores**, the cosine similarity threshold
(default 0.97, allowed 0.90–1.00), lifetime (default 60 minutes) and maximum saved
reviews (default 1000). Caching is enabled by default. **Clear saved AI match scores**
invalidates scores across all workers and removes the configured namespace's Redis
entries; the CSRF-protected endpoint is
`POST /api/semantic-cache/clear`. Review details and `rerank_cache_lookup` audit
events distinguish exact hits, semantic hits, misses and cache failures. A hit
keeps its original model trace ID in cache metadata; it does not claim a new trace.

To configure Redis:

1. Install the updated `requirements.txt` in the application environment.
2. Provide a Redis 8+ instance with vector search (or a compatible Redis Cloud /
   Azure Managed Redis instance with RediSearch enabled). A plain Redis deployment
   without vector search cannot serve this cache.
3. In **Settings → Connections → Semantic cache**, save the Redis connection URL,
   such as `rediss://default:YOUR_PASSWORD@YOUR_HOST:6379/0`, and an application
   namespace. Use a unique namespace for each independent application/environment;
   workers sharing one application should use the same namespace and application
   database. The URL is encrypted and masked because it may contain credentials.
   `SEMANTIC_CACHE_REDIS_URL` and `SEMANTIC_CACHE_NAMESPACE` are environment
   fallbacks; neither needs to be checked into source control.
4. Select **Redis (shared cache)** as the match score cache backend. RedisVL creates
   and validates its own vector index on first use; allow the corresponding search,
   hash, expiry, sorted-set, script and unlink operations for this namespace.
   Existing indexes are never overwritten automatically. Model changes use a
   separate index to prevent incompatible vector dimensions from mixing.

The cache uses the existing Hugging Face embedding instance. Only query/scope hashes, embeddings,
validated scores, rationale and trace IDs are stored; query text is not stored in
the cache. Redis applies an absolute per-entry TTL; lookups do not extend it.
Numeric timestamp filters make reduced lifetimes apply immediately. A namespace
sorted set and atomic eviction script limit tracked entries across workers and
embedding models. The database backend prunes expired/oldest records on writes.
Cache or embedding failures fall back to normal AI scoring; Redis failures do not
silently switch to database caching. With Redis unconfigured, scoring still works
and the review records the cache as unavailable. Clearing during an outage still
invalidates old scores through a database generation token and returns a warning;
unremoved Redis keys expire by TTL. Concurrent misses may each calculate fresh
scores; cache upserts are deterministic within each query/scope/mode.

`python -m unittest discover -s tests -v` exercises real RedisVL query/schema/store
code with mocked Redis transport and models, alongside temporary-database tests.
Validate connectivity, Redis permissions and hit quality against the configured
server before relying on production caching; the isolated tests do not require or
modify a live Redis service.

## Jira publishing

The current local 8012 instance was last verified in preview mode. This is a saved
setting, not a permanent restriction or a statement about a remote server. Legacy
8010 deployments may exist separately; verify their process and settings before
operating them. Use the 8012 dashboard controls to enable or disable publishing. A ticket scan posts only
when publishing is enabled and preview-only is off; both settings start in preview mode.
Manual previews never post comments. LangChain comments use the `FA-LANGCHAIN-8012` marker and are kept separate from legacy
`FA-SUGGESTION` comments. If 8012 publishing is enabled later, configure a
reachable `public_base_url` and feedback secret first. The local audit records a
publish intent before the Jira API call and the resulting comment ID after
success. Use the dashboard's audit verification action to check the hash chain.

The supervisor currently routes Jira suggestions and an optional Salesforce
read-only specialist. Register new specialists through
`falloutassist_next/specialist_registry.py`; each must implement the shared
manifest/invoke contract and return findings, source evidence, status, and
applicability in the shared specialist result shape. Configured specialists run
alongside Jira history retrieval; the supervisor combines their evidence before
comment composition. Jira comment publishing remains a separately controlled
final graph node.

## Source map for a coding assistant

All paths below are relative to this application directory.

| File | Responsibility |
| --- | --- |
| `app.py` | ASGI entry point; root `../app.py` also imports this application |
| `falloutassist_next/web.py` | FastAPI routes, open-access dashboard, CSRF/session and dashboard APIs |
| `static/dashboard.html` | Dashboard markup, styles and browser JavaScript; no separate frontend build |
| `falloutassist_next/config.py` | Bootstrap paths, defaults and configuration precedence |
| `falloutassist_next/storage.py` | PostgreSQL/SQLite persistence, encrypted secrets, audit, leases, versions and feedback |
| `worker.py`, `falloutassist_next/service.py` | Separate scheduler process, ingestion/scan leases and scan deduplication |
| `falloutassist_next/plugins.py` | Jira REST client, resolution-comment selection, ingestion, local/MCP fusion and publisher |
| `falloutassist_next/history.py` | Failure parsing, Hugging Face embeddings, Chroma and lexical recall |
| `falloutassist_next/rovo.py` | MCP search adapter used by the main FalloutAssist retrieval path |
| `falloutassist_next/workflow.py` | Investigation LangGraph, supervisor, quality gate, comment composition and optional publishing |
| `falloutassist_next/agent.py`, `jira_specialist.py`, `quality.py` | Grounded Jira generation, applicability and action-evidence checks |
| `falloutassist_next/reranker.py`, `semantic_cache.py`, `redis_cache.py` | Composite relevance scoring, conservative cache reuse and RedisVL backend |
| `falloutassist_next/rovo_history.py` | A2A-specific SAC history discovery, scope validation, evidence fetch and five-minute cache |
| `falloutassist_next/rovo_a2a.py` | OAuth/PKCE, streaming A2A synthesis, manual preview lease and 24-hour synthesis cache |
| `falloutassist_next/knowledge_router.py` | Separate knowledge graph, persistent job/refresh queues and downstream package assembly |
| `falloutassist_next/documents.py`, `synthesis.py` | Uploaded SOP/document indexing and version-specific procedure rewrites |
| `falloutassist_next/specialist_registry.py`, `specialists.py`, `salesforce.py` | Specialist manifests and optional read-only Salesforce investigation |
| `tests/` | Existing isolated unit tests; select relevant files for a change |
| `migrate_legacy.py`, `migrate_sqlite_to_postgres.py` | Explicit migration utilities; inspect current table coverage before use |

### Embedding and configuration details

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`, loaded
through `HuggingFaceEmbeddings`. Azure supplies chat/reranking, not the default
local embeddings. The actual deployment/model values can be overridden in the UI.

`HistoryIndex.upsert_many()` embeds a failure signature plus overlapping passages
from the selected resolution or document reference. `EmbeddingWindow` uses the
loaded model's tokenizer and `max_seq_length` to fit the entire vector input,
including its context prefix and special tokens (256 tokens for default MiniLM).
Every vector retains its original source key; multiple passages cannot vote more
than once for a source in RRF. The persistent Chroma collection is
`jira_history_v1` under `data/vectors/`. Full source/resolution text and quality
remain in the relational database for evidence checks. Token passages do not
replace the complete source or approve a partial SOP procedure.

Local keyword recall prioritizes the explicit error code and error text before
the failed-step words. It deduplicates and bounds queries to 32 terms, with higher
field weights for errors and steps. SQLite uses FTS5 BM25; PostgreSQL uses weighted
full-text `ts_rank`, not BM25. RRF combines the lexical and vector ranks before
the existing AI relevance and evidence checks.

### Updating vectors and measuring retrieval

Existing vectors need to be re-embedded for resolution passages and token-safe
inputs to take effect. After backing up the data and choosing a maintenance window,
run these commands from the application directory:

```powershell
& .\.venv\Scripts\python.exe reindex_history.py
& .\.venv\Scripts\python.exe reindex_history.py --apply
```

The first command only reports the usable source count. `--apply` takes the history
write lease and re-embeds stored usable sources without querying Jira or posting
comments. It preserves source identities, evidence, uploads, feedback and outcomes.
It changes `data/vectors/` and invalidates corpus-dependent score reuse. Normal
ingestion also refreshes unchanged tickets when the embedding-index version changes;
it performs that pass once and refreshes uploaded vectors too. A new model with a
different dimension still requires the existing full vector rebuild procedure.
More passages increase vector count and storage; measure memory after re-indexing.

For a real SAC benchmark, put independently verified cases in an ignored private
file such as `data/evaluation/sac-cases.jsonl`. Each line follows this structure;
the identifiers below are fictional and must be replaced with confirmed cases:

```json
{"id":"case-1","issue_key":"SAC-10001","query_text":"Summary: Add-on cancellation failed\nFailed Step: Cancel Addon\nError: 14047 | Invalid client plan instance identifier","expected_source_keys":["SAC-10000"],"verified":true}
```

Label only historical tickets whose procedures have been independently confirmed
to apply. Do not use the assistant's existing suggestions as assumed correct labels.
The current ticket is excluded automatically to prevent testing against itself.

```powershell
& .\.venv\Scripts\python.exe evaluate_retrieval.py --cases data/evaluation/sac-cases.jsonl --top-k 5
```

This runs local retrieval only and reports hit rate, recall and MRR at top-k, plus
p50/p95 elapsed time. It neither drafts nor posts. Model initialization is included
in elapsed time when cold. Reports omit query text but contain ticket keys: keep
both cases and reports private. Expected source keys must exist in usable local
history. Cases containing only `issue_key` require explicit
`--fetch-current-issues`, which reads current descriptions through Jira REST.
These metrics evaluate historical-ticket recall, not whether a generated fix works.
No production accuracy percentage is established without such verified labels.

Connection values saved in the UI override `.env` fallbacks, including explicitly
saved empty values. `load_dotenv(..., override=False)` means inherited process
environment values take precedence over `.env`. Database and session settings
are bootstrap environment configuration. Changing `.env` alone may therefore
have no effect until the right saved connection and process environment are checked.

### Cache responsibilities

| Cache | Storage / lifetime | Reuses |
| --- | --- | --- |
| AI relevance scores | RedisVL or selected application DB; default 60 minutes, 1000 entries | Reranker scores for guarded query/evidence/model scope; not final comments |
| Version procedure rewrite | Application DB; validity checked against evidence, model and prompt | AI-written rendering of one workaround version |
| Rovo historical evidence | Encrypted `integration_secrets`; valid for 300 seconds | Up to five selected historical sources for matching failure context |
| Rovo A2A synthesis | Encrypted `integration_secrets`; valid for 24 hours | Prior Rovo text and evidence for matching context and connection generation |
| Loaded embeddings / Chroma client | Process-local LRU, up to two model entries each | Model/client objects, not remote Rovo answers |

Rovo cache identity includes site, Jira email, Rovo cloud ID, failed step, error,
order type, order reason and release. A synthesis cache hit does not regenerate
Rovo's text or re-fetch historical comments; its current-issue metadata is rebound.
Treat it as historical knowledge, check its age and evidence, and do not interpret
reuse as a fresh Rovo review of every detail in a different ticket.

`storage.save_integration_secret()` performs write-triggered retention cleanup:
pending OAuth state after 15 minutes, history records after one hour, synthesis
records after 24 hours, refresh state after seven days, and jobs after 30 days.
These retention thresholds differ from lookup-validity TTLs. They are not a
maximum database-size guarantee.

## Implemented limits and open constraints

- A2A history discovery requests up to 30 candidates per exact-JQL and semantic
  search, merges/deduplicates those lists, and returns at most five usable SAC
  Resolved/Closed sources. Its history fetch uses batches of four.
  Main FalloutAssist MCP retrieval is a different path and can fetch with eight
  threads; the A2A limits do not constrain every app operation.
- A shared preview lease lasts 360 seconds. Stream processing checks a
  four-minute elapsed limit, caps received line text at 512,000 characters and
  keeps at most 40,000 output characters. These application checks are not a
  global process-memory cap or a complete HTTP-body-size limit.
- Knowledge jobs admit at most ten queued/running items. Rovo refreshes admit
  at most 50. Each executor has one worker **per process**. Queue locks and
  dispatch state are process-local; do not assume a distributed task broker or
  safe multi-process queue dispatch. Start with one API process and verify any
  scaling change against restart/recovery behavior.
- Cache-first knowledge routing removes cold A2A work from the automatic Rovo
  branch's response path. Manual A2A previews and jobs with explicit historical
  keys can still wait for Rovo. In the normal FalloutAssist branch, MCP discovery
  and remote ticket evidence share `rovo_retrieval_budget_seconds` (default eight;
  configurable from one to 60 in Settings). If the combined evidence is still
  insufficient, the main graph escalates once to its bounded A2A fallback. A shared per-process executor admits at most
  eight remote operations with no queued backlog. Timed-out calls can continue
  occupying a slot until their HTTP operations finish. The budget does not bound
  the separate manual A2A preview. Main investigations have separate stage and
  total deadlines; worker threads may outlive the caller until their transport
  returns. No production latency or memory SLA is established.
- Embeddings, Chroma, ingestion, uploads and concurrent API work can still exhaust
  memory. Lower observed latency is not evidence of OOM safety. See the SOP's
  memory procedure before increasing worker counts or workload.
- Source changes can occur inside the Rovo synthesis reuse window. Cache identity
  does not hash every historical comment on each hit. Prompt changes do not
  automatically appear in that cache key either; plan invalidation explicitly.
- The separate manual A2A preview requests 8010-style headings but remains free-form text.
  A compact, validated Rovo-to-Jira comment formatter is not implemented; the
  preview does not post comments. The main graph's new A2A fallback instead requires
  structured, source-validated actions and passes through the normal final gate.
  Full evidence and proposals remain
  available for review and downstream knowledge use.
- `migrate_sqlite_to_postgres.py` has an explicit table allowlist. It currently
  omits workaround pattern/version/evidence/outcome and semantic-cache tables.
  Do not use it as an assumed complete migration of a populated modern instance;
  reconcile schema/table coverage and verify a restore before switching databases.

For changes, validation choices and operational procedures, continue with the
[SOP](SOP.md#10-coding-assistant-handoff).
