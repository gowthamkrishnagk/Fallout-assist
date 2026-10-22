# FalloutAssist operations and coding-assistant SOP

Reviewed against this checkout on **2026-10-20**. Read the
[application README](README.md) for architecture and the source map. This SOP
describes how to operate and change the app; it does not attest to the revision,
credentials or settings running on a remote server.

## 1. Establish the target before doing work

1. Confirm the requested environment. The current development target is local
   `http://127.0.0.1:8012`. Historical "old 8010" references are not proof of
   which code owns that port now.
2. Run `git status --short` and inspect the relevant diff. This checkout contains
   modified and untracked implementation files. Preserve unrelated work and
   include required new files when preparing a deployment.
3. Read `falloutassist_next/config.py`, then inspect the running command, working
   directory, interpreter and live UI settings. Do not print `.env`, access
   tokens, session keys or OAuth callback query strings into logs or chat.
4. Record the requested outcome and the current publication mode. A manual
   preview is the default validation path for a suggestion change.

Root `app.py` forwards to `langchain-falloutassist/falloutassist_next/web.py`.
`legacy_app.py` and the root legacy modules remain separate reference code.
Running root `app.py` does not select the legacy implementation.

## 2. Fresh local setup

Use Python 3.11 or 3.12. The following PowerShell commands start at the repository
root and apply to a fresh setup. Reuse an existing project `.venv` when appropriate.

```powershell
Set-Location .\langchain-falloutassist
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

Edit `.env` locally. Set a long random `SESSION_SECRET`, select
`DATABASE_BACKEND=postgres` and supply `DATABASE_URL` for an existing database
whose application account has the required schema/read/write privileges. SQLite
requires explicit `DATABASE_BACKEND=sqlite`; it is not automatic failover.
Keep `SESSION_COOKIE_SECURE=false` for local HTTP and use secure cookies with
the intended HTTPS deployment.

The dashboard has no FalloutAssist account login. Anyone who can reach its listener
can use dashboard and administrative controls. Keep local runs bound to loopback;
restrict server runs to a trusted network gateway or firewall. `SESSION_SECRET` is
still required for CSRF/OAuth browser state and encrypted integration settings. It
is not a FalloutAssist user password.

## 3. Start, verify and restart local 8012

From `langchain-falloutassist/`, start the API in the foreground:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8012 --workers 1 --access-log
```

In another terminal, check reachability:

```powershell
(Invoke-WebRequest -Uri 'http://127.0.0.1:8012/' -UseBasicParsing -TimeoutSec 15).StatusCode
```

HTTP 200 proves the page is served, not that Jira, Azure, MCP or A2A work. Open the
dashboard and verify the model display, connections and publishing banner. Use a manual
preview of an authorized issue to validate the required integration. A raw API
request without a session can correctly return 401.

To restart, stop only the identified API process, preferably with Ctrl+C in its
foreground terminal, then run the same command. Do not kill every Python process,
change the target to a remote server, or start another listener as a workaround.
For a hidden/background launcher, use an absolute executable path, the application
working directory and captured stdout/stderr. A successful process-launch command
alone is not a successful startup; check the log and HTTP response.

If privileged process inspection is available, identify the listener with:

```powershell
Get-NetTCPConnection -LocalPort 8012 -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess
```

Do not infer the app is stopped from an access-denied result. The root `.venv`
previously caused a `ModuleNotFoundError: langchain_chroma` during launch; use
`langchain-falloutassist/.venv` consistently.

When launched with this session's existing file redirection, logs are in repository
`tmp/falloutassist-8012.out.log` and `tmp/falloutassist-8012.err.log`. Logging paths
are launcher configuration, not an automatic application feature.

## 4. Configure connections and Jira scope

Use **Settings & publishing > Connections** after signing in.

| Connection | Used for | Verification |
| --- | --- | --- |
| Jira site, email, API token | Current ticket, descriptions, historical resolution comments, ingestion and gated publishing | Preview the saved JQL and open a known issue |
| Azure endpoint, key, API version, chat deployment | FalloutAssist reranking, drafting and procedure rewriting | Generate a manual suggestion with usable evidence |
| Embedding model | Local Chroma retrieval and semantic match-cache embeddings | Ingest/search local history; confirm displayed model |
| Rovo cloud ID, email, scoped MCP token | MCP discovery of matching issue keys | Rovo preview without explicit source keys or main retrieval in `rovo` mode |
| Rovo A2A OAuth connection | Rovo's own generated proposal | Connect in **Rovo agent**, then generate a preview |
| Redis URL and namespace | Optional shared AI relevance-score cache | Inspect exact/semantic hit metadata after repeated eligible requests |
| Salesforce connection | Optional allowlisted read-only specialist | Inspect specialist result/status on a relevant manual preview |
| Langfuse | Optional tracing | Verify model/tool traces without using it as the publication audit |

The configured MCP adapter uses its own `ROVO_API_TOKEN`; it does not use the
A2A OAuth bearer token. A Rovo Dev coding token is not an A2A replacement. Token
permissions must fit the adapter and organization policy; do not infer validity
from a similarly named token product.

For Rovo A2A, use the app's **Connect Atlassian account** flow. Complete sign-in
and consent on Atlassian's website. The app performs dynamic client registration
and OAuth PKCE, stores the connection encrypted, and refreshes access when possible.
It does not require entering an Atlassian password in FalloutAssist. Organization
enablement and account/site permissions are still required. Saved authorization
status alone does not prove a ticket can be read.

Keep the browser origin consistent through sign-in and callback. Local 8012 uses
`http://127.0.0.1:8012`; a tunnel has its own local port/origin. Changing between
origins can lose the app session or mismatch the callback. A remote callback needs
the HTTPS/loopback origin accepted by this integration. Only establish a tunnel
when the intended app is actually remote.

In **Jira setup**, preview and save separate history and inflow JQLs. Previewing
a JQL is read-only and does not save or ingest it. Example history scope:

```jql
project = SAC AND status in (Resolved, Closed) ORDER BY resolutiondate DESC
```

Choose the inflow query for the team's actual active-ticket states. The separate
Rovo A2A history selector currently hardcodes SAC and Resolved/Closed; changing the
saved ingestion JQL does not reconfigure that selector. Up to five valid sources
are used; fewer are acceptable. Unrelated projects, different failures and comments
without usable resolution evidence should be excluded, not used as padding.

## 5. Daily operations and publishing boundaries

1. Check history counts and the latest completed ingestion in the UI. Jira counts
   exclude uploaded document passages. **Refresh past tickets** performs an
   incremental sync; inspect errors before performing a full rebuild.
2. Upload PDF, DOCX, TXT or Markdown material in **Documents & SOPs**. Files are
   limited to 10 MB; PDFs need selectable text and are limited to 100 pages.
   Failed indexing retains the original and offers a retry. Applicability gates
   still apply to uploaded procedures.
3. Use **Show suggestion** for a manual FalloutAssist preview. Inspect confidence,
   matched sources, exact failure/context, release and evidence before judging it.
4. Use **Rovo agent > Generate Rovo preview** for Rovo's independent proposal.
   Leave historical keys blank for automatic discovery; enter at most five keys
   when intentionally selecting evidence. Check timing and cache metadata.
5. Review feedback and audit records. A match percentage ranks a historical
   candidate; it is not a probability of successful remediation.

Publication behavior is implemented in `workflow._publish()`:

| Operation | Posts a Jira comment? |
| --- | --- |
| Manual `POST /api/preview` | No |
| Manual `POST /api/rovo-a2a/preview` | No |
| `POST /api/knowledge/jobs` | No; its FalloutAssist branch uses `publish=False` |
| Manual or scheduled scan | Can post when `publish_enabled=true` and `dry_run=false`, with required feedback/public URL configuration |

The routed graph separates evidence confidence from manual-review status. Medium
evidence stays Medium during a Rovo timeout or failure, and stronger validated
evidence can raise it to High. Medium review-only proposals and all Low results
are not posted automatically. The default search retrieves local top K and Rovo
MCP top K, combines and reranks them once; disable **Always search local + Rovo MCP**
in Settings to restore local-first fallback. Before
running a scan, inspect the live publishing banner. Do not enable publishing as
a troubleshooting shortcut. `public_base_url` must be reachable by feedback users
and `FEEDBACK_SECRET` must be configured. Preserve the `FA-LANGCHAIN-8012` marker,
publication IDs and update/deduplication behavior; legacy comments use another marker.

If a Jira comment attributed to Rovo is too long, first identify its actual
publisher and deployed revision. Current A2A code does not post that text. A
compact Rovo comment adapter remains future work: preserve required resolution
headings, prerequisites, negations and attribution; keep full evidence in the app.
Do not implement a raw substring cut that can remove a condition or verification.

### Scheduler

The API does not run the ordinary ingestion/scan schedule. In a separate terminal,
from this application directory:

```powershell
& .\.venv\Scripts\python.exe worker.py --loop
```

Configure **Jira setup > Automatic schedule** in the UI. Intervals are saved in
the database and checked by the worker roughly every five seconds. Zero disables
a task; the UI exposes this through an enable checkbox. Defaults are 60 minutes
for ingestion and five minutes for scanning. Enabled tasks are due when a worker
starts. `worker.py --once` makes one pass and still honors disabled intervals;
it can include a publishing scan. Inspect settings before starting either form.

Knowledge jobs and Rovo refreshes use executors in the API process and resume
from persisted state at API startup. They are not run by `worker.py --loop`.

## 6. Downstream knowledge-agent procedure

Save a unique `KNOWLEDGE_ROUTER_API_KEY` of at least 32 characters in Connections
or the environment fallback. That bearer key grants the two knowledge-job routes,
not general dashboard access. The dashboard is open to anyone who can reach it;
CSRF protects browser-originated state changes, while the bearer key remains
mandatory for downstream knowledge-job requests.

Example PowerShell request using a key already supplied through the local process
environment; replace the example issue with an authorized real ticket:

```powershell
$faHeaders = @{ Authorization = "Bearer $env:FA_KNOWLEDGE_API_KEY" }
$faBody = @{ issue_key = 'SAC-12345' } | ConvertTo-Json
$faJob = Invoke-RestMethod -Uri 'http://127.0.0.1:8012/api/knowledge/jobs' -Method Post -Headers $faHeaders -ContentType 'application/json' -Body $faBody
Invoke-RestMethod -Uri ("http://127.0.0.1:8012" + $faJob.status_url) -Headers $faHeaders
```

The POST returns 202. Poll the returned URL at a modest interval, such as 2-5
seconds; respect HTTP 429 and `Retry-After`. Pending/running duplicates share a
job. The persisted job record and assembled result include branch progress.

- `ready`: both proposals completed.
- `partial`: one proposal completed; inspect Rovo status to distinguish pending
  refresh from an unavailable branch. Continue polling when it is pending.
- `failed`: neither branch supplies a completed proposal.

Without explicit `source_keys`, the Rovo branch uses a 24-hour synthesis cache or
schedules background refresh on a miss. The separate FalloutAssist branch still
follows `JIRA_HISTORY_BACKEND`, so `rovo` mode can still wait for MCP. Explicit
historical keys take the direct A2A route and can wait for synthesis.

The package preserves separate `proposals.falloutassist` and `proposals.rovo_a2a`,
citations, evidence, timing and `execution_guidance.may_execute=false`. It is not
a merged approved action plan. A future action agent must verify live conditions,
resolve incompatible proposals and use its own authorized tools.

## 7. Troubleshooting

Work from the first backend failure, not just the last browser message.

| Symptom | Check | Corrective action |
| --- | --- | --- |
| 8012 refuses connections | Listener, startup log, executable and working directory | Start the project `.venv` from the application directory; verify HTTP response |
| `ModuleNotFoundError: langchain_chroma` | Interpreter path and installed requirements | Use/install into the application `.venv`, not the root environment |
| Startup requires `DATABASE_URL` | Selected backend and inherited environment | Configure PostgreSQL, or intentionally select local SQLite; do not silently switch on outage |
| Dashboard is reachable without a login | Open-access UI by design | Restrict the listener to loopback or a trusted network gateway/firewall |
| Saved connection changed in `.env` but behavior unchanged | UI override and inherited environment | Change the intended saved connection and restart for bootstrap/process settings |
| `Unexpected token 'I'`, `Internal Server Error`, or non-JSON response | Backend traceback and HTTP status | Diagnose the upstream failure; UI JSON parsing is a secondary symptom |
| Could not reach Jira/Atlassian | DNS, egress, proxy reachability and CA configuration | Repair the actual network or certificate chain; token rotation cannot fix connection refusal |
| Issue not found/inaccessible | Exact key, configured Jira site, connected identity and Browse Project access | Test the ticket with the same identity/site used by that integration |
| MCP unauthorized/unavailable | MCP token, email, cloud/site and organization allowance | Fix MCP configuration; A2A OAuth consent does not supply this adapter's MCP credentials |
| A2A authorization expired/revoked | Saved OAuth connection and refresh error | Reconnect through Atlassian in the same browser/origin |
| Another Rovo preview is running | Active lease, owning process and pending result | Allow it to complete; inspect an apparently orphaned holder before clearing state |
| No usable Rovo resolution | Retrieved keys, SAC/status scope, exact failure and exclusion counts | Check candidate/evidence selection; do not weaken source checks to manufacture five matches |
| Redis unavailable or no hits | Vector-search support, URL, permissions, namespace and cache scope | Scoring falls back to fresh calculation; select DB cache explicitly if desired |
| Long response | `current_issue`, `mcp_search`, `history_total`, `evidence_fetch`, `a2a_synthesis` timing | Address the slow stage; distinguish cold generation from cache reuse |

### Proxy versus TLS

Requests inherits standard proxy environment settings. Inspect whether
`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` and `NO_PROXY` are set without exposing
credential-bearing values. A refusal at a configured proxy is an egress problem,
not proof of invalid Jira credentials. A prior restricted local execution used
an unreachable loopback proxy at `127.0.0.1:9`; an authorized network-enabled launch
resolved that incident. Do not hardcode proxy removal or bypass network policy
as an application fix.

The application initializes `truststore` for operating-system certificate trust.
`TLS_CA_BUNDLE`, when configured, points to a PEM CA bundle for outbound validation.
It does not configure a proxy. Verify the file exists for the service identity.
Keep certificate verification enabled; fix trust or connectivity at its source.

### Production HTTPS and Rovo OAuth

Rovo OAuth requires the browser-visible callback origin to use HTTPS (localhost
is allowed for development). For the direct `https://host:8012` setup, place the
certificate and private key under the app's ignored `tls/` directory, then start
with `./serve_https.sh`. The launcher uses `tls/falloutassist-server.crt` and
`tls/falloutassist-server.key`; override them with `TLS_CERT_FILE` and
`TLS_KEY_FILE` if needed. Keep the private key readable only by the service
account. Never commit either file.

```sh
./.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8012 --workers 1 \
  --ssl-certfile tls/falloutassist-server.crt --ssl-keyfile tls/falloutassist-server.key
```

The certificate must be trusted by browsers and include the exact DNS name or IP
in its Subject Alternative Name. For an IP URL, the certificate needs an IP SAN
for that address. The current direct-IP deployment uses a local root CA and an
IP-SAN server certificate. The public root certificate was added to the current
Windows user's Trusted Root store; another workstation must trust that same root
certificate before its browser accepts the server certificate. This is device-
specific trust, not a publicly trusted certificate. Protect the root private key
and never copy it to the server. For broader access, use an IT-issued certificate
or an approved reverse proxy/load balancer. Alternatively, terminate TLS at the organization's approved reverse proxy/load balancer and
forward requests to the private app listener with the public `Host` header and
`X-Forwarded-Proto: https`.

On a Windows workstation that is authorized to trust this private CA, install
the public certificate for the current user (no administrator rights required)
and restart the browser:

```powershell
certutil.exe -user -addstore Root .\tls\falloutassist-root-ca.crt
```

Set `ROVO_PUBLIC_BASE_URL` in the service environment to the exact browser origin,
for example `https://falloutassist.example.com` or
`https://<your-app-host>:8012` (no path or trailing slash).
The app then registers
`https://falloutassist.example.com/api/rovo-a2a/callback` with Atlassian. Session
cookies become Secure automatically for this HTTPS origin; `SESSION_COOKIE_SECURE=true`
can force Secure cookies, while `false` is for local HTTP. The certificate,
DNS record, proxy host, and Atlassian organization allow-list must all agree.
Leave `ROVO_PUBLIC_BASE_URL` blank for local `127.0.0.1` and SSH-tunnel use.

This private-IP HTTPS origin is suitable for browser access from a workstation
that trusts the local root CA. Whether Atlassian accepts the callback depends on
the OAuth integration's redirect rules; the application registers the exact
origin-derived callback dynamically. Open the app at that exact HTTPS origin
and reconnect Rovo there so OAuth state and callback share the same browser
session. The previous HTTP URL on port 8012 no longer serves the app.

### Cache diagnosis

Keep these caches distinct: Redis/DB AI match scores, DB procedure rewrites,
five-minute encrypted Rovo evidence, and 24-hour encrypted Rovo synthesis.
**Clear saved AI match scores** affects the first cache, not all of them. Inspect
`retrieval.cache`, cache age and timing before declaring a request a fresh Rovo
generation. Changes to a Rovo prompt or source evidence need an explicit
invalidation strategy for existing 24-hour results.

## 8. Memory, OOM and latency procedure

Lower latency and a successful single preview do not establish a safe concurrent
workload. Measure the actual target server. Record process RSS/peak RSS before and
after idle, one cold request, a warm repeat, ingestion and a representative
concurrent workload. Include PostgreSQL, Redis, the API and the scheduler in the
total host budget; do not count only Python.

On a Linux server, useful read-only starting commands are:

```sh
free -h
ps -eo pid,ppid,rss,%mem,args --sort=-rss
journalctl -k --since '1 hour ago'
```

For containers or systemd, inspect the actual memory limits, restart policy and
OOM events through that service manager. Set the memory budget from measured
peaks and host capacity; do not invent a universal safe RAM value. A hard cap
contains host impact but can still terminate the app. Investigate the first
OOM event and whether work resumed after restart.

Current protections include a preview lease, five selected evidence sources,
bounded stream text, and queues capped at ten knowledge jobs and 50 refreshes.
Limits cover different paths, and executor limits are per process. They are not
a global concurrency or RSS cap. The API can retain embedding models/Chroma, and
the separate scheduler can load its own copies. Upload parsing, ingestion and
main-path MCP hydration can have different peaks from an A2A preview.

Start with one API worker and one scheduler for this deployment pattern. Do not
increase Uvicorn workers as an unmeasured latency fix: models can be duplicated
and persisted background-queue dispatch is not a distributed broker. If the
measured workload requires more parallelism, first design coordinated workers,
bounded admission and memory monitoring. Cache growth also consumes database
storage; write-triggered retention is not a total storage-size cap.

## 9. Deployment, backup and recovery

No service-manager/container definition for the active app is supplied in the
current tracked checkout. Discover the actual remote working directory, service
identity, launcher, revision, dependencies and data locations before writing a
deployment command. Do not reuse a legacy deployment script by name alone.

1. Record the deployed revision, active local diff/artifact and current process
   configuration. Confirm the target database and which process owns scheduling.
2. Arrange an application-consistent backup. Preserve the application database,
   `data/vectors/`, bootstrap configuration, and the matching `SESSION_SECRET` in
   protected storage. Uploaded originals are stored in the database. For SQLite,
   use a consistent SQLite backup or stop all writers before copying its files;
   copying only a live main file can omit WAL state.
3. Stop/drain the appropriate scheduler and API for the maintenance window.
   Pending knowledge jobs/refreshes can resume when the API starts. Do not erase
   database state or leases as routine upgrade preparation.
4. Deploy the intended application files, including required untracked new modules,
   and install `langchain-falloutassist/requirements.txt` into its service environment.
   Preserve operational `.env` values and data. Configuration examples are not
   production secrets or a replacement for existing configuration.
5. Start one API process, inspect startup and verify the page, settings, database,
   source counts and one authorized read-only preview.
   Application startup initializes the schema and resumes persisted knowledge work.
6. Verify the publishing mode before restarting the scheduler; enabled schedules
   may run immediately. Check timing/cache metadata, audit and service memory.

For rollback, restore compatible code/dependencies and the corresponding
configuration/data snapshot where schema compatibility requires it. Avoid
restoring old publication state and then immediately scanning: it may no longer
match Jira comments written since the backup. Reconcile publication IDs/audit and
keep publishing off during recovery verification.

`migrate_legacy.py` imports selected old-app state; imported rules are drafts.
`migrate_sqlite_to_postgres.py` requires an empty destination and an explicit
table allowlist. Its current list omits versioned-workaround and semantic-cache
tables. Review and reconcile the schema before using it for a populated instance;
do not call it a complete backup/restore or run it during a routine restart.
Changing the embedding model requires a planned full vector rebuild. A Jira
rebuild preserves uploads and restores their vectors, but is not a login, proxy
or OAuth repair procedure.

### Retrieval upgrade: token passages and Rovo fallback

Keyword search now preserves error terms before long step text. New vectors include
token-bounded, overlapping resolution/SOP passages mapped to their original source.
Keep the configured embedding model unchanged during this upgrade. Back up data
and vectors consistently and pause the relevant writers for the maintenance window,
then run `python reindex_history.py` to inspect the count and
`python reindex_history.py --apply` to update the configured local Chroma index.
This uses stored evidence and does not query Jira, publish comments or reset
feedback. It holds the same history write lease used by ingestion and document
changes. An interrupted pass can be rerun; the version is marked current only after
all sources succeed. Monitor the increased vector count and memory afterwards.

In **Settings & publishing**, **Rovo search time budget (seconds)** controls how
long normal FalloutAssist reviews wait for MCP discovery and ticket evidence.
Default: eight seconds. Main investigations run local retrieval and real relevance
assessment first; sufficient evidence skips Rovo. Insufficient local evidence
escalates once to MCP, then to a 30-second source-validated Rovo-agent fallback if
needed. A total investigation deadline defaults to 90 seconds. Missing support,
conflicts and unfinished stages are review-only and cannot publish. Explicit
local-only previews opt out of both integrations. The separate manual Rovo preview
and downstream Rovo-proposal branch remain independent.

The settings UI/API exposes source-count/relevance thresholds and stage budgets;
see [Retrieval routing](docs/retrieval-routing.md) for defaults and the graph map.
Saved and immediate suggestions expose sanitized route reasons and a review flag.
No schema migration is needed. Retain operational settings and OAuth tokens during
deployment, and verify with a read-only preview rather than a publishing scan.

See the README's **Updating vectors and measuring retrieval** section for the
private JSONL benchmark. Use confirmed ticket/fix labels, exclude the current
ticket, and report hit rate/recall/MRR separately from fix success. Keep real ticket
text, labels and reports under ignored `data/evaluation/`. Never commit them or
substitute unreviewed assistant suggestions for verified outcomes.

## 10. Coding-assistant handoff

### Required first reads

Read root `README.md`, this app's `README.md`, this SOP, applicable `AGENTS.md`
instructions if present, and the current Git diff. Then inspect only the modules
for the requested path using the README source map. Runtime behavior and code
take precedence over old conversation descriptions.

### Working boundaries

- Keep a Rovo-only task within its actual retrieval/synthesis/UI path. Do not alter
  FalloutAssist local retrieval merely to make an A2A preview return a result.
- Preserve open-access behavior, CSRF, encrypted-secret handling, grounded-action checks,
  publication markers, deduplication, feedback signing and audit-before-publish.
- Do not infer an approved fix from similar error codes. Preserve procedure
  prerequisites, context, release, negation and exact source attribution.
- Keep FA and Rovo proposals identifiable in the knowledge package. A future deep
  agent consumes knowledge; this app currently does not authorize remediation.
- Distinguish an implemented feature from a proposal. Compact Rovo Jira comments,
  globally coordinated job workers and proven production OOM protection are not
  established capabilities of this checkout.
- Do not add secrets, real passwords, OAuth codes, customer identifiers or private
  ticket bodies to docs or fixtures. Keep `.env`, data and operational backups out
  of source-control changes.
- A successful local manual preview does not prove production deployment, posting,
  distributed queue safety, cache freshness or memory stability.

### Validation and review

For documentation-only work, verify links, referenced files, commands and factual
claims against source; application tests and a restart are unnecessary. For code
changes, honor the user's current validation constraints, select the relevant
existing tests, and use isolated temporary databases/mocked services. Do not run a
live ingestion, publishing scan, migration or load test as an incidental unit check.

Examples from this application directory, when tests are appropriate:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_rovo_a2a.py' -v
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_rovo_history.py' -v
```

Other relevant suites cover MCP, Jira scope, quality, documents, semantic cache,
Redis and auth bootstrap. Inspect test names before selecting them. For the
knowledge router, inspect its separate graph, queue/recovery and response assembly;
an A2A test alone does not validate downstream job behavior.

Inspect the final diff for unrelated edits and credential leakage. Report what
changed, what was actually verified, whether a process was restarted/deployed and
what remains unverified. Do not claim tests passed when only syntax or links were
checked. Update these docs when ports, schema, cache scope, queue behavior or
publication contracts change.
