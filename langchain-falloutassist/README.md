# FalloutAssist: LangGraph supervisor with Jira and Salesforce specialists

This application runs in its own port and database configuration. The LangGraph
supervisor coordinates the Jira historical-resolution specialist and an optional
read-only Salesforce specialist through a shared specialist registry. Aria,
Matrixx and Nokia can be added by implementing that contract and registering
their specialist.

## Flow

`Jira inflow -> LangGraph router -> load Jira context -> parallel local/Rovo retrieval and configured specialists
-> reciprocal rank fusion -> hydrate current resolution evidence -> composite reranker
-> Jira suggestion specialist -> supervisor joins specialist findings and makes final decision
-> evidence quality gate -> readable comment composition -> local audit -> optional Jira publish`

The Jira specialist is a LangChain `create_agent` with one bounded tool for reading
selected historical resolutions. The Salesforce specialist is a LangChain agent
whose tool can only query allowlisted objects, fields, and lookup columns. The
graph owns routing, parallel investigation, evidence synthesis, and the publish
gate. Specialists cannot post Jira comments or write to Salesforce. Approved Jira
rules are matched deterministically and require admin review before they can
produce a high quality suggestion.

## What is stored

- PostgreSQL (primary) or SQLite (explicit local fallback): settings, encrypted
  connection secrets, users,
  ingested resolution text, source quality tiers, versioned workarounds and release labels,
  approved rules, investigations, publication IDs, feedback, scan state, and a
  hash-linked application audit log.
- Chroma via LangChain: an independent persistent vector index of the ingested
  resolutions. PostgreSQL full-text search or SQLite FTS5 supplies lexical recall.
- Langfuse, if enabled: model and tool traces. SQLite remains the application
  action audit; Langfuse is optional.

Source quality `high` / `medium` / `low` means how complete an ingested
resolution is. An investigation's tier is a separate decision about whether a
new Jira ticket has enough grounded evidence for a suggested workaround. A
similarity score is not a probability that a fix is correct.

## Readable workaround procedures

Expanding a version in **Find a workaround** calls the authenticated, CSRF-protected
`POST /api/workaround-versions/{version_id}/synthesis` endpoint. It uses the saved
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

Run the isolated synthesis checks from this directory with
`python -m unittest discover -s tests -v`. Model calls are mocked and temporary
databases are used; application data and Jira publishing are not touched.

## Setup

Use Python 3.11 or 3.12. From this directory:

1. Install `requirements.txt` in a fresh virtual environment.
2. Copy `.env.example` to `.env` and set `DATABASE_URL` for PostgreSQL and a
   long random `SESSION_SECRET`. The
   database must already exist, and the application user needs schema create and
   read/write privileges. Set `DATABASE_BACKEND=sqlite` only for a local fallback.
   The backend never switches automatically during a database outage; this keeps
   audit and publication state in one authoritative database. The database and
   session key are bootstrap values because login needs them before the UI loads.
3. Create an admin: `python manage.py create-user you@example.com --role admin`.
   Use the same command with `--role user` for people giving Jira feedback.
4. Optionally import existing accounts, feedback and rules once:
   `python migrate_legacy.py --from <path-to-old-project>`. Imported rules are
   drafts because failure parsing has changed; review and approve them in the
   new dashboard. Historical Jira resolutions are rebuilt by ingestion.
   If you already used this new application with SQLite, run
   `python migrate_sqlite_to_postgres.py` after setting `DATABASE_URL` to copy
   that application state into an empty PostgreSQL database. Then set
   `DATABASE_BACKEND=postgres` in `.env` before starting the API or worker.
5. From the repository root, start the single app with
   `langchain-falloutassist\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010`.
   Sign in and
   configure Jira, Azure OpenAI, feedback signing, and optionally Rovo and
   Langfuse in **Settings → Connections**. Set both JQLs for resolved history
   and active inflow. The embedding model downloads on first use unless already
   cached. If Jira has feedback or suggested-workaround custom fields, enter
   their IDs here. For Rovo, enter a cloud ID, email, and scoped API token with
   `search:rovo:agent-interface`; your Atlassian admin must allow token auth.
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

Rovo MCP semantic search and FalloutAssist's local vector plus PostgreSQL/SQLite
full-text retrieval run concurrently. Reciprocal rank fusion (RRF) merges their
ranked issue lists without assuming their scores use the same scale. FalloutAssist
hydrates candidates from its ingested resolution corpus, so the agent receives
reviewed resolution text and the local evidence quality gate remains
authoritative. The fused retrieval pool is scored by the Azure LLM for relevance
to the failed step, error/code, context, and resolution before the final top K is
selected using the configured retrieval, LLM, and human feedback weights.
If either retriever fails, the workflow continues with the one that succeeded
and records the failure in the audit log. If local similarity is unavailable,
Rovo results can only produce manual-review leads; the system does not infer a
high-confidence score from Rovo's rank. If both sources fail, the investigation
stops before drafting or commenting.

Human votes are aggregated per historical source, workaround version, and
normalized failure using a Beta(2,2) prior, so a small number of votes stays
close to neutral. Votes on superseded versions do not affect the current
procedure. Verified automated success/failure outcomes are recorded against a
specific version through the admin outcome API and displayed in the library.
As those verified outcomes accumulate, their share of the configured outcome
evidence signal grows relative to human votes. The default composite is
`25% retrieval rank + 50% LLM relevance + 25% human and automated outcome evidence`.
Weights are configurable in the dashboard and must sum to 100%; the Jira review
top K is selected from the composite ranking. Candidate component scores and
LLM rationale are saved with the investigation and shown in its detail view.

LLM reranking only orders candidates; the evidence quality gate still controls
whether a suggestion can be high confidence. If the reranker call fails, the
workflow records that failure and falls back to neutral LLM scores, leaving the
retrieval and human ranking available. Reranker scores are relative ranking
signals, not probabilities that a workaround is correct.

## Jira publishing

The 8012 LangChain app is currently configured for preview only while the
independent legacy app on 8010 continues to publish Jira comments. Use the 8012 dashboard controls to enable or disable publishing. A ticket scan posts only
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
