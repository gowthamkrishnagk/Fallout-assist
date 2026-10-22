# Installation and required tools

This guide starts from a fresh clone of the standalone **LangChain FalloutAssist**
repository. Commands run in the folder containing `app.py`, `requirements.txt`,
`README.md` and `SOP.md`. In the original combined workspace that folder is
`langchain-falloutassist/`; after cloning this repository, it is the repository root.

Read [README.md](README.md) for architecture and [SOP.md](SOP.md) for ongoing
operations, publishing, troubleshooting, deployment and recovery.

## Required and optional tools

| Tool or service | Needed when | Purpose |
| --- | --- | --- |
| Git and Bitbucket repository access | Installing/updating source | Clone and pull the application |
| Python 3.11 or 3.12 with `venv` and `pip` | Always | Run the FastAPI app and separate scheduler |
| PostgreSQL database and application credentials | Default database mode | Durable configuration, evidence, audit, jobs and publication state |
| SQLite support from Python | Explicit local alternative to PostgreSQL | Local development persistence; not automatic failover |
| Web browser | UI administration | Open dashboard, configure connections, upload documents and review proposals |
| Jira Cloud site, email and Jira REST API token | Reading actual Jira data | Ingestion, ticket lookup and optional gated comment publishing |
| Azure OpenAI endpoint, API key, API version and chat deployment | Generating/reranking FalloutAssist suggestions | Chat model calls; the deployment name is configured in Connections |
| Hugging Face embedding model files | Local semantic retrieval | Default: `sentence-transformers/all-MiniLM-L6-v2`; downloaded on first use unless cached |
| Atlassian MCP V2 token, token owner's email and cloud ID | Optional MCP retrieval / automatic Rovo history discovery | Find matching historical issue keys |
| Atlassian Rovo A2A organization enablement and OAuth account | Optional Rovo-generated proposals | Rovo's independent resolution synthesis |
| Redis with vector-search support | Optional shared AI match-score cache | RedisVL relevance-score reuse; plain Redis without vector search is insufficient |
| Salesforce connection | Optional read-only specialist | Current-system evidence from allowlisted queries |
| Langfuse connection | Optional tracing | Model/tool trace visibility |

Chroma, LangChain, LangGraph, RedisVL and the MCP client are Python dependencies
installed by `requirements.txt`. This app uses an embedded persistent Chroma index;
a separate Chroma server is not required. Node.js, npm, a local LLM server, a local
MCP server and Docker are not required for the documented setup. Redis is not
required for basic operation: choose the database cache explicitly, or allow fresh
scoring if the configured Redis backend is unavailable.

Use a machine with capacity measured for the intended model, index, ingestion and
concurrent workload. No fixed safe production RAM minimum has been established.
See [SOP memory guidance](SOP.md#8-memory-oom-and-latency-procedure).

## 1. Clone

```sh
git clone https://gowthamkrishna1@bitbucket.org/superagent_lcpr/langchain-falloutassist.git
cd langchain-falloutassist
```

Authenticate through your approved Bitbucket credential manager or token flow.
Do not embed a password or token in the Git URL or commit it to configuration.

## 2. Create the Python environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS shell, with Python 3.12 installed:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Python 3.11 is also supported; use its executable consistently. On a Linux
distribution that separates Python's `venv` package, install that distribution's
matching package first. Always install and run through this project's environment.
The dependency file uses compatible version ranges, not a complete lockfile;
retain a protected deployment dependency snapshot for repeatable upgrades.

## 3. Create local bootstrap configuration

Copy the template only for a new configuration. Never overwrite an existing `.env`
as an upgrade step.

Windows:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

Linux/macOS:

```sh
test -f .env || cp .env.example .env
```

Edit `.env` using your local editor. Choose one database mode:

```dotenv
# Default: supply your own existing PostgreSQL database and credentials.
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://APP_USER:URL_ENCODED_PASSWORD@DB_HOST:5432/APP_DATABASE
SESSION_SECRET=REPLACE_WITH_A_LONG_RANDOM_SECRET
SESSION_COOKIE_SECURE=false
```

For a local setup without PostgreSQL, intentionally use:

```dotenv
DATABASE_BACKEND=sqlite
SESSION_SECRET=REPLACE_WITH_A_LONG_RANDOM_SECRET
SESSION_COOKIE_SECURE=false
```

The placeholders above are not working credentials. Supply a cryptographically
random session secret, for example from Python's `secrets.token_urlsafe(48)`.
Preserve it across restarts, deployment and database moves because it also derives
the encryption key for saved connections and integration records. SQLite data is
created at `data/falloutassist.sqlite3`; vectors are stored at `data/vectors/` in
either database mode. Do not commit these paths.

For PostgreSQL, provision an empty application database and account using your
organization's database tools. The app initializes its schema on startup; it
does not provision the server or database. If credentials contain special
characters, URL-encode them correctly in `DATABASE_URL`.

## 4. Open the dashboard and start the app

Windows:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8012 --workers 1 --access-log
```

Linux/macOS:

```sh
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8012 --workers 1 --access-log
```

The interactive FalloutAssist login and account table have been removed. Anyone
who can reach the app listener can use dashboard controls. Keep local use bound to
loopback; restrict server access through a trusted network gateway or firewall.
`SESSION_SECRET` remains required for CSRF/OAuth browser state and encrypted
integration settings, not for a FalloutAssist password.

Open **http://127.0.0.1:8012**. An HTTP 200 means the page is reachable; it does
not verify integration credentials. Keep publishing off during setup.
The documented local launch uses one API worker. Embedded knowledge executors and
models have per-process state; do not increase worker counts without reviewing the
queue/scaling constraints in the README and SOP.

## 5. Configure features in the UI

1. In **Settings & publishing > Connections**, save Jira and Azure settings. UI
   values override `.env` fallbacks. Bootstrap database/session values remain
   environment configuration. Never paste an Atlassian password into this app.
2. Start with **Local only** history retrieval if MCP is not configured. Select
   the application-database AI score cache if you want caching without Redis.
   Default configured backends may otherwise attempt Rovo and Redis.
3. In **Jira setup**, preview and save separate resolved-history and active-inflow
   JQL filters, then select **Refresh past tickets**. A sample history query is
   `project = SAC AND status in (Resolved, Closed) ORDER BY resolutiondate DESC`.
   Ingestion reads Jira and builds local evidence and vectors.
4. Run **Show suggestion** for one authorized issue and inspect source evidence,
   confidence and timing. It does not publish a comment.
5. Optionally upload SOP/documents. Supported inputs are selectable-text PDF,
   DOCX, TXT and Markdown up to 10 MB; the app does not perform OCR.
6. For automatic Rovo discovery, configure the **Atlassian MCP V2** connection
   (`ROVO_API_TOKEN` is the internal setting name). Use the appropriate MCP
   search permissions, not a Rovo Dev coding token.
7. For Rovo's own generated proposal, enable A2A for the organization and connect
   through **Rovo agent > Connect Atlassian account**. Complete OAuth on the
   Atlassian website in the same browser/origin. MCP credentials and A2A OAuth
   serve separate paths; configure both for automatic Rovo A2A history discovery.
8. Verify the separate Rovo preview. It uses resolved/closed SAC evidence, accepts
   up to five historical matches and does not post its output to Jira.

For feedback links and any later publishing, set `public_base_url` to the actual
reachable app URL; the code default still points to port 8010. A local development
URL is not suitable for remote feedback users. Configure `FEEDBACK_SECRET` before
publishing. Enabling a scan's publishing is a separate operational decision.

## 6. Start scheduling only when configured

In a separate terminal in this repository:

```powershell
# Windows
& .\.venv\Scripts\python.exe worker.py --loop
```

```sh
# Linux/macOS
.venv/bin/python worker.py --loop
```

The worker checks saved UI intervals. Enabled tasks can run immediately on startup;
review the publishing banner before starting it. `--once` also honors saved task
enablement and can include a publishing scan. The API itself runs knowledge-job
and Rovo-refresh executors; the scheduler is a separate process for ingestion and
ticket scans.

## 7. Optional downstream agent API

Configure a unique `KNOWLEDGE_ROUTER_API_KEY` of at least 32 characters. Call
`POST /api/knowledge/jobs` with `Authorization: Bearer <key>` and a JSON issue key,
then poll the returned `status_url`. The response contains separate FalloutAssist
and Rovo proposals; cold Rovo work can continue in the background. It does not
post or execute remediation. Authentication, states and a runnable request example
are in [SOP section 6](SOP.md#6-downstream-knowledge-agent-procedure).

## Network and certificate prerequisites

The service identity needs approved outbound connectivity to its database, Jira
site, configured Azure endpoint, and optional Redis/Langfuse endpoints. Rovo uses
`mcp.atlassian.com`, `a2a.atlassian.com` and `auth.atlassian.com`. Initial embedding
downloads also require the configured model host unless model files are preloaded.

Use the operating-system certificate trust store. If your organization requires
a PEM CA bundle, configure `TLS_CA_BUNDLE` as a readable service-side path. Proxy
reachability and TLS trust are separate checks. Do not disable verification or
remove required enterprise network controls as a workaround.

For a server deployment, use the organization's service manager/reverse proxy,
protected bootstrap configuration, backups and measured memory limits. This
repository does not provision those external services or ship live credentials.
See [SOP deployment](SOP.md#9-deployment-backup-and-recovery).

## Verification and next steps

Check the page, selected database, model display, JQL preview, one
manual FalloutAssist suggestion, and optionally a Rovo preview/cache repeat.
Verify publishing remains off during setup and check that the scheduler uses the
same database and environment as the API.

For code changes, existing isolated tests are under `tests/`; choose the relevant
suite and follow user constraints. See [CONTRIBUTING.md](CONTRIBUTING.md). Never
substitute a live publishing scan or production migration for an isolated check.
