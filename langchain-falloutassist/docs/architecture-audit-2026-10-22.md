# FalloutAssist architecture audit — 22 October 2026

Scope: current LangGraph application source, live 8012 configuration, aggregate saved reviews, ingestion, retrieval, agent boundaries, caching, publishing, integration and deployment behavior. This is a code and operational review, not a penetration test, load certification or independently labeled answer-quality benchmark. No production settings, keys, comments, ingestion or investigations were changed for this audit. Read-only diagnostic scripts were uploaded to `/tmp`.

## Executive assessment

The application has a useful architecture for assisted troubleshooting: hybrid retrieval, explicit graph routing, source citations, separate review decisions, bounded external work and an audited publishing boundary. The live deployment should be demonstrated as a supervised knowledge assistant. Its current records do not establish reliable autonomous remediation or an accuracy percentage.

The most consequential findings are open administrative access, no verified resolution benchmark, frequent Rovo-agent timeouts, and a very high review/abstention rate in the sampled records. Better orchestration is implemented; better business outcomes than the old 8010 application have not been demonstrated on a common labeled dataset.

## Live snapshot

Observed at **2026-10-22 09:11 UTC / 14:41 IST**. These are observations, not defaults or promises; counts change with the rolling JQL scope.

| Item | Observed value |
| --- | --- |
| Application | HTTPS on 8012; one web process and one scheduler worker |
| Relational backend | **SQLite**, not PostgreSQL |
| Vector retrieval | Persistent Chroma via LangChain |
| Embedding configuration | `sentence-transformers/all-MiniLM-L6-v2` |
| Azure chat deployment alias | `gpt-4o-mini`; the alias alone does not establish its exact provider model revision |
| Reranking cache | Enabled; **application database** backend, not Redis |
| Cache policy | 60 minutes, 1,000 entries, semantic threshold 0.97 plus strict guards |
| Default retrieval | Local + Rovo MCP; 10 requested candidates per source |
| Merged rerank pool / drafting shortlist | 10 / 5 |
| Evidence routing | At least 1 eligible source, real reranker relevance at least 0.75, plus applicability, source quality and conflict checks |
| Budgets | Local 10s; MCP 8s; rerank 10s; agent 30s; overall investigation 90s |
| Publishing | `publish_enabled=false`, `dry_run=true` |
| Scheduler | History 10 minutes; scan 1 minute; sequential worker, intervals measured after task completion |
| Stored Jira tickets | 11,091 |
| Searchable / excluded | 5,403 / 5,688 |
| Source tiers | High 3,219; Medium 2,184; Low 5,688; these are automatic evidence-quality assessments |
| Uploaded documents | 0 at snapshot time |
| Last completed ingest | 09:06:17 UTC; 11,091 checked, 0 new, 0 updated, 0 removed |
| Rovo OAuth | Connection record saved; this is not proof that every subsequent call succeeds |
| Tracing | Langfuse disabled |
| Named API-key manager | In local source; **not deployed** on 8012 |
| Feedback | 93 positive, 59 negative; not a controlled correctness sample |
| Verified workaround outcomes | No rows in the outcome table |
| Publication records | 36 historical records; current publishing is disabled |
| Process RSS | Web about 172 MiB; worker about 1.25 GiB; one point-in-time reading, not a memory ceiling |
| Public URL setting | A private deployment address was configured as HTTP, despite the HTTPS listener; host omitted from this shared copy |

Local Jira history JQL:

```jql
project = SAC AND status in (Resolved, Cancelled, Closed)
AND "Reporting Area[Dropdown]" = "Order Fallout" AND updated >= -90d
```

Inflow JQL:

```jql
project = SAC AND "Reporting Area[Dropdown]" = "Order Fallout"
AND (status in ("In Progress","Waiting for support") OR created >= -120m)
```

The inflow OR clause also selects recently created tickets outside those two statuses. Rovo evidence scope uses SAC Resolved/Closed and does not automatically inherit all local-history JQL constraints.

## Existing review sample

The latest 100 saved investigations at inspection time contained 90 Low and 10 Medium, with all 100 flagged for review and saved as previews. This sample is neither randomized nor independently labeled. Do not call it 10% accuracy, 90% failure, or proof of safe rejection: a human must determine whether abstention was appropriate.

| Recorded stage/outcome | Observations |
| --- | --- |
| Current ticket load | 82 success; 18 unavailable |
| MCP | 79 success; 3 timeout, among 82 attempts |
| Combined reranker | 81 success; 1 timeout |
| Combined evidence | 10 sufficient; 49 weak; 23 conflicting |
| Rovo agent | 72 attempted: 16 success, 42 timeout, 14 unavailable |
| Frequent final reasons | 84 no concise actionable draft; 66 no approved evidence-grounded steps; 37 applicability mismatch; 36 changed/omitted source details, negation or conditions |

Reason counts overlap. A successful search or successful agent transport does not imply an approved workaround. Even the 10 sufficient retrieval assessments ended as review results after downstream checks.

Observed stage timings from existing route events:

| Stage | Count | p50 | p95 |
| --- | ---: | ---: | ---: |
| Current ticket | 100 | 0.411s | 0.542s |
| Local search | 100 | 1.190s | 1.689s |
| Rovo MCP | 82 | 2.609s | 3.580s |
| Combined rerank | 82 | 5.704s | 7.269s |
| Rovo agent | 72 | 20.001s | 20.002s |

These mixed-status timings describe a small retrospective sample. They are not end-to-end response percentiles, cold-start measurements or an SLA; do not add stage percentiles to calculate a response percentile. The agent stage reserves part of its 30-second allowance for source verification, explaining the approximately 20-second RPC cutoff. Timed-out threads can continue occupying bounded executor slots until the underlying transport returns.

Among the latest 200 cache-lookup audit events, there were 90 exact hits and 110 misses. This is a sample of cache lookups, not an overall application speedup or a semantic-hit benchmark. No semantic hits appeared in that sampled set.

## Actual processing boundaries

1. Jira REST loads current context. A failure produces a review result, not invented ticket details.
2. Local retrieval embeds a bounded failure signature, queries Chroma, runs lexical search, and fuses their ranks with RRF.
3. Default graph routing then calls Rovo MCP. It discovers issue keys; Jira REST hydrates and validates the actual resolution evidence.
4. Local and MCP candidates are deduplicated by source key, fused, capped to the rerank pool and assessed by the structured LLM reranker. Feedback contributes to composite ranking.
5. Insufficient combined evidence invokes Rovo A2A with ticket context and available evidence. New source keys are hydrated through Jira and proposed claims are checked.
6. The Jira specialist can create local-source and MCP-source drafts with the application's own model. Those are different from Rovo's independent A2A synthesis.
7. An optional configured Salesforce specialist supplies read-only context. The supervisor reconciles findings; the final deterministic evidence gate can reject its rewrite.
8. Compose and persist the result and audit. Only an explicitly publishing flow with allowed settings and no review requirement reaches Jira posting.

The knowledge-job graph is separate. It runs FalloutAssist alongside a Rovo synthesis-cache lookup; a cold Rovo miss queues refresh. The FalloutAssist branch itself may already call MCP/A2A. Every knowledge package says `execution_guidance.may_execute=false`.

## Prioritized findings

| Priority | Finding and evidence | Implication | Recommended action |
| --- | --- | --- | --- |
| Critical before broader access | `web.py:75` and `web.py:98` return an anonymous admin when no authorization header is supplied. Browser CSRF is available from the open session endpoint. | API keys do not establish an administrative access boundary. Users reaching the app can operate its open dashboard. This was an earlier requested deployment choice. | Put deployment behind SSO/access control; enforce RBAC and caller authorization on all sensitive routes. Separate reader, operator and admin permissions. |
| High | No labeled end-to-end benchmark or verified outcome records available; recent results overwhelmingly require review. | Output accuracy, recall of valid workarounds and resolution success are unknown. | Obtain an independent operator-labeled test set, including no-answer cases and conflicting procedures. Compare against 8010 on identical cases. |
| High | Rovo agent had 42 timeouts in 72 sampled attempts. | A2A is not a dependable fast path in the sampled workload. MCP is a separate, faster stage. | Keep A2A optional/asynchronous, inspect its actual task traces, improve eligible evidence before escalation, and set user-facing degraded behavior. Do not simply raise all timeouts. |
| High | `quality.py:238`/`:250` treats distinct normalized procedure identities among relevant sources as a conflict. | Complementary or differently worded valid procedures can cause conservative false abstention. 23 combined assessments were conflicting in the sample. | Label the disagreements, distinguish compatible steps from mutually exclusive actions, and retain deterministic checks for negation, values and order. Do not weaken the gate without examples. |
| High | `reranker.py:42`, `agent.py:70` and drafting context limits truncate resolution context; exact action checks then reject omissions. | Long prerequisites can be unavailable to the model or valid paraphrases can fail. 36 sampled results mentioned changed/omitted details. | Supply complete procedure units with bounded retrieval of adjacent context; classify gate failures as true unsupported actions versus faithful equivalents. |
| High for scaling | Knowledge queues use process-local locks plus persisted JSON (`knowledge_router.py:17`, `:93`). Chroma is a local persistent store; the live relational backend is SQLite. | Restart recovery exists, but distributed coordination, horizontal scaling and high availability are not established. | Keep one coordinator until a transactional queue or broker, distributed claims, suitable storage and load validation are implemented. PostgreSQL alone does not solve queue coordination. |
| High | SQL and Chroma updates are separate; vector upsert first deletes old source passages; full rebuild drops the collection (`history.py`, `plugins.py:582`). | Interrupted ingestion can leave index/corpus divergence or temporary retrieval gaps. | Version index builds, switch after validation, reconcile source/vector IDs, and use recovery checkpoints/outbox semantics. |
| High for downstream exposure | Existing bearer keys authorize administration as well as knowledge requests (`web.py:98`). New named keys have no scopes or expiry and are not deployed. | A future action agent could receive more authority than its knowledge role needs. | Add service identities and read-only knowledge scopes, expiry/rotation and revocation policies before broad agent integration. |
| Medium | Dense-search failure raises before lexical retrieval (`history.py:166`–`:167`). | A usable keyword index cannot independently rescue a broken vector stage. | Execute dense and lexical retrieval independently, preserve partial results and report each backend status. |
| Medium | Local candidate hydration makes repeated DB calls for evidence, quality, versions and feedback (`history.py:177` onward). | Candidate growth increases latency and DB connection overhead. | Batch source/context/feedback reads and cache immutable per-investigation settings. |
| Medium | Main default graph invokes local then MCP (`workflow.py:185`) rather than parallel discovery. | Always-search-both has an avoidable serial component. | Evaluate concurrent bounded discovery with one merge/rerank; retain the explicit local-first option for latency-sensitive cases. |
| Medium | Each incremental ingest iterates the complete saved rolling JQL (`plugins.py:610`); unchanged issues skip re-embedding. The worker runs ingestion and scanning sequentially. | Incremental does not mean only changed Jira rows are fetched; ingestion delays scans. | Use a changed-since watermark plus periodic authoritative reconciliation, and independently schedule ingestion/scans with safe leases. |
| Medium | Local scope includes Cancelled and a 90-day/Reporting Area restriction; Rovo scope is SAC Resolved/Closed. | The two retrievers can legitimately produce different candidate populations. | Make and display one policy for status, age and project/business scope, or document the intended difference. |
| Medium | The 24-hour reusable Rovo-synthesis key depends on failure/site context rather than every source's current revision (`rovo_history.py:29`, `rovo_a2a.py:30`). | A matching failure can reuse a procedure after source changes or permission revocation. | Revalidate permissions and source versions, invalidate on corpus/source changes, and expose age prominently. |
| Medium | The knowledge graph's FalloutAssist branch permits Rovo while its other branch can queue independent Rovo work (`knowledge_router.py:165`, `:172`). | Duplicate remote investigation or shared-lease contention is possible; it was not proven as the cause of sampled failures. | Share hydrated evidence/task results, reuse a single investigation coordinator, and distinguish A2A investigation from independent comparison explicitly. |
| Medium | Identifier masking is a small regex policy (`redact.py`); raw source content and uploads live in application storage. | Masking is not a complete PII/DLP policy. Source-system ACLs are not re-created as per-user index ACLs. | Approve data handling, verify what each provider receives, add tenant/document authorization and retention/encryption controls. |
| Medium | Langfuse is off; stage events are not a full cost/SLO telemetry system. | No verified per-ticket spend, e2e p95, queue SLO or provider-error dashboard. | Track end-to-end and queue times, tokens, model version, stage status, cache type and cost; alert on abstention and fallback rates. |
| Medium | `public_base_url` remains HTTP while the listener is HTTPS. | Generated document/feedback links may use an incorrect origin. | Correct and validate the canonical public origin before demonstrating generated links or enabling publishing. |
| Medium | Graphs compile without a checkpointer; persisted jobs restart at job-level boundaries. | A persistent queue is not step-level durable LangGraph execution. | Define replay/idempotency and persist checkpoints if long investigations need mid-graph resume. |
| Medium | Shared open-access actors, hash-linked audit, broad dependency version ranges and manually managed processes. | Caller attribution, independent tamper evidence, reproducible releases and recovery guarantees are limited. | Use identifiable actors, externally retained audit/backup evidence, a locked tested environment, service supervision and restore drills. |

## Accuracy validation plan

Suggested pilot design, requiring owner agreement:

- Start with 50–100 representative SAC tickets spanning common errors, rare failures, missing context, releases and no-answer cases. Ask operators to label valid sources and the full acceptable procedure, not to approve generated answers by default.
- Hold out test tickets and later resolution information to avoid retrieval leakage and self-matches. Split recurring failure families/time periods carefully so nearly identical tickets do not inflate results.
- Measure retrieval hit/recall at K and MRR, then separately label action support, correct prerequisites/order, contradictions, useful-answer coverage and justified abstention. Report accuracy both among answered cases and across all cases.
- Track attempted and confirmed fix outcomes independently of thumbs-up feedback. False confident suggestions require special review; success rate among a small selected subset is insufficient.
- Compare local-only, local+MCP, full fallback, and the old 8010 version using the same cases, source snapshot and manual rubric. Measure warm/cold e2e latency, queue wait and spend.
- Existing `evaluate_retrieval.py` reports local retrieval metrics only. Its loader requires a nonempty expected-source list, so it cannot currently evaluate no-answer behavior; a separate end-to-end evaluation is needed.
- Acceptance thresholds must be agreed with the operational owner; none were inferred or silently declared passed in this audit.

## Validation performed

- Inspected core source modules, graph edges, model/tool contracts, source gate, storage/cache, API authentication, scheduler and knowledge-job behavior.
- Read live HTTPS settings/OpenAPI/ingestion status and aggregated existing DB records. No new Jira request, preview, scan, ingestion, comment, key generation or key revocation was initiated by the audit. Natural scheduled activity continued.
- Ran the existing 95 isolated routing, quality, publication, retrieval, MCP, API, open-access and Jira-scope regressions: all passed. They mock external providers and do not establish production answer accuracy. The first sandbox run could not access Windows temporary SQLite directories; the permitted rerun passed.
- Verified live named-key routes are absent; local named-key code is a pending release and is not fully covered by the existing key tests.
- Did not perform a production load test, provider billing analysis, fresh A2A authorization attempt, security penetration test, restoration drill or 8010 comparative benchmark.

## First decisions after the demo

1. Agree that the near-term product is a supervised knowledge assistant and decide what success metrics justify a wider pilot.
2. Restore an authenticated deployment boundary and decide read-only integration scopes.
3. Label representative failed/abstained examples before changing confidence thresholds or model choice.
4. Separate fast MCP retrieval from slow A2A investigation in UX, metrics and integration contracts.
5. Choose the downstream-agent interface: an MCP tool adapter is a useful next option; the existing async REST job API is available now. Neither inbound MCP nor inbound A2A is currently implemented.

For demo wording, a walkthrough and likely questions with suggested answers, see [Demo guide](demo-guide-2026-10-22.md).
