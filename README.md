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

```bash
pip install -r requirements.txt

# Config & secrets (kept out of git)
copy config.example.json config.json      # then edit Jira url/email
copy .env.example .env                     # then fill in tokens/keys
```

`.env` keys: `JIRA_EMAIL`, `JIRA_API_TOKEN`, and optionally `GROQ_API_KEY` /
`GEMINI_API_KEY` / `ANTHROPIC_API_KEY` for cloud LLMs.

## Run

```bash
uvicorn app:app --reload --port 8010
```

Open http://localhost:8010 — ingest Jira tickets, upload docs, and ask for a
workaround. The LLM provider, API key, re-rank toggle, connection status, and
Groq quota are all managed in the **LLM Settings** tab.

## Notes

- `config.json`, `.env`, and `trackers/` (the vector index + uploaded docs) are
  gitignored — they hold environment-specific config and customer data.
- Embeddings always run locally; only the optional re-rank/synthesis step uses
  the configured LLM provider.
