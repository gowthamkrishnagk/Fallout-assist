# FalloutAssist — Workaround Finder

A retrieval assistant for Salesforce order-fallout support. It indexes resolved
Jira tickets and uploaded workaround documents, then — given a failed **step** and
**error** — finds the past resolutions that match and (optionally) re-ranks them
with an LLM.

## How it works

- **Embeddings** (local `all-MiniLM-L6-v2`) + **Chroma** vector DB.
- Tickets and documents are both matched on **step + error** separately, so a
  result only ranks high when the actual failure matches — not on shared prose.
- Cosine similarity scoring; matches at/above `score_threshold` are "strong".
- Optional **LLM relevance re-rank** filters out near-misses (Groq / Gemini /
  Claude / local Ollama — swappable in the UI).

## Setup

Requires **Python 3.12** (or 3.11) — not 3.14: `chromadb`'s `chroma-hnswlib` dependency
has no Windows wheel for 3.14, and no wheel at all past `chroma-hnswlib==0.7.5` (hence
`chromadb==0.5.4`, not a newer 0.5.x, in requirements.txt — every later 0.5.x pins
`chroma-hnswlib==0.7.6`, which would need a C++ compiler to build from source).

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Config & secrets (kept out of git)
copy config.example.json config.json      # then edit Jira url/email
copy .env.example .env                     # then fill in tokens/keys
```

`.env` keys: `JIRA_EMAIL`, `JIRA_API_TOKEN`; `AZURE_OPENAI_*` for both chat (Synapt) and
embeddings — see below. `GROQ_API_KEY` / `CEREBRAS_API_KEY` / `ANTHROPIC_API_KEY` are
implemented in generate.py but not selectable in the UI (Synapt-only, so org data stays
in-tenant).

## Embeddings

Embeddings run on Azure OpenAI (`text-embedding-ada-002` by default, same deployment
pattern as chat — see `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` in `.env.example`), not a
local model. Changing the embedding model/deployment changes the vector dimension, so
anything already in `trackers/workaround_index` needs a fresh ingest to match — it
can't be reused across a model change.

## Accounts & roles

The app requires sign-in — everyone registers an account (email + password) from the
login screen, and every account is one of two roles:

- **admin** — reviews/approves or rejects user-submitted workarounds (the pending
  suggestions queue), and can manage Jira ingest, LLM settings/keys, documents, and
  app config.
- **user** — can search/ask, and submit a suggested workaround (via 👎 + a note), which
  sits **pending** until an admin approves it. Only an approved suggestion gets
  embedded into the searchable index.

Role is decided at registration time by `config.json` → `auth.admin_emails`: any email
on that list becomes admin; everyone else is a regular user. There's no in-app "make
admin" action — promoting someone means adding their email to that list before they
register (or editing their existing record in `trackers/users.json`, admin accounts
only). Add your own team's emails there before rolling this out.

Optionally set `SESSION_SECRET` in `.env` to a long random string (signs the login
cookie); if left blank a stable key is derived from `JIRA_API_TOKEN`, same fallback
pattern as `FEEDBACK_SECRET`. Accounts are stored in `trackers/users.json` (bcrypt
password hashes only — gitignored, same as the rest of `trackers/`).

## Deterministic workaround rules

The **Rules** tab is a second, separate mechanism from the suggestions queue above —
an approved suggestion gets embedded and found by similarity *later* (a score, never
guaranteed); a rule's Error Description regex either matches the incoming failure
exactly or it doesn't. `suggest.suggest_for_query` (the one function both `/api/ask`
and the Jira auto-suggest bot call) checks approved rules **before** touching the
vector index at all — a match returns the same reviewed fix every time, with no
similarity search run.

Any signed-in user can propose a rule (draft); only an admin can approve one, and
approving refuses (409, with the conflicting rule shown) if it would be genuinely
ambiguous against an already-approved rule for the same error signature — see
`rules.check_overlap`. Rules are stored in `trackers/workaround_rules.json`.

## Run

```bash
.venv\Scripts\uvicorn app:app --reload --port 8010
```

Open http://localhost:8010 — ingest Jira tickets, upload docs, and ask for a
workaround. The LLM provider, API key, re-rank toggle, connection status, and
Groq quota are all managed in the **LLM Settings** tab.

## Notes

- `config.json`, `.env`, and `trackers/` (the vector index + uploaded docs) are
  gitignored — they hold environment-specific config and customer data.
- Embeddings always run locally; only the optional re-rank/synthesis step uses
  the configured LLM provider.
