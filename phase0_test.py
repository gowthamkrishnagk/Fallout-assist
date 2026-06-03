"""
Phase 0 — prove ingest → embed → search → answer loop.
Fetches ~20 SAC resolved tickets, indexes them, then asks a question.

Usage:
    py phase0_test.py                        # uses config.json JQL
    py phase0_test.py "SAC-1234"             # ask about a specific ticket
    py phase0_test.py --query "port out stuck nokia"   # free text
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_jira(cfg):
    from jira import JIRA
    email     = os.getenv("JIRA_EMAIL") or cfg["jira"]["email"]
    api_token = os.getenv("JIRA_API_TOKEN") or cfg["jira"]["api_token"]
    url       = cfg["jira"]["url"]
    return JIRA(server=url, basic_auth=(email, api_token))


def fetch_tickets(jira, jql: str, max_results: int = 20) -> list[dict]:
    print(f"\n[INGEST] Fetching tickets: {jql}")
    issues = jira.search_issues(
        jql,
        maxResults=max_results,
        fields="summary,description,status,comment,resolution",
    )
    print(f"[INGEST] Got {len(issues)} tickets")
    tickets = []
    for issue in issues:
        comments = issue.fields.comment.comments if issue.fields.comment else []
        comment_text = "\n".join(
            f"[{c.author.displayName}]: {c.body}"
            for c in comments
            if c.body and len(c.body.strip()) > 30
        )
        description = issue.fields.description or ""
        text = f"Summary: {issue.fields.summary}\n"
        if description:
            text += f"Description: {description[:500]}\n"
        if comment_text:
            text += f"Comments:\n{comment_text[:2000]}"
        tickets.append({
            "id":      issue.key,
            "url":     f"{jira.server_url}/browse/{issue.key}",
            "summary": issue.fields.summary,
            "status":  issue.fields.status.name,
            "text":    text,
        })
    return tickets


def build_index(tickets: list[dict], index_path: str, embed_model: str):
    from sentence_transformers import SentenceTransformer
    import chromadb

    print(f"\n[EMBED] Loading model: {embed_model}")
    model  = SentenceTransformer(embed_model)
    client = chromadb.PersistentClient(path=index_path)
    col    = client.get_or_create_collection("workarounds")

    print(f"[EMBED] Embedding {len(tickets)} tickets ...")
    texts      = [t["text"] for t in tickets]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    col.upsert(
        ids        = [t["id"] for t in tickets],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [{"source": "ticket", "key": t["id"],
                       "summary": t["summary"], "url": t["url"],
                       "status": t["status"]} for t in tickets],
    )
    print(f"[EMBED] Indexed {len(tickets)} chunks. Total in DB: {col.count()}")
    return model, col


def search(query: str, model, col, top_k: int = 5) -> list[dict]:
    emb     = model.encode([query]).tolist()
    results = col.query(query_embeddings=emb, n_results=top_k)
    hits    = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        hits.append({"doc": doc, "meta": meta, "score": round(1 - dist, 3)})
    return hits


def generate_local(prompt: str, model_name: str, ollama_url: str) -> str:
    import httpx
    resp = httpx.post(
        f"{ollama_url}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=120,
    )
    return resp.json().get("response", "").strip()


def build_prompt(query: str, hits: list[dict]) -> str:
    sources = ""
    for i, h in enumerate(hits, 1):
        sources += f"\n--- Source {i} ({h['meta']['key']} | score {h['score']}) ---\n"
        sources += h["doc"][:800] + "\n"
    return (
        f"You are a Jira order-fallout support assistant. "
        f"A new issue has been raised:\n\n{query}\n\n"
        f"Based ONLY on the following past resolved tickets, suggest a workaround. "
        f"Cite which ticket(s) helped.\n"
        f"{sources}\n"
        f"Workaround:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticket_or_query", nargs="?", help="Ticket ID (SAC-123) or query text")
    parser.add_argument("--query", help="Free-text query")
    parser.add_argument("--max", type=int, default=20, help="Max tickets to ingest")
    parser.add_argument("--skip-ingest", action="store_true", help="Use existing index")
    args = parser.parse_args()

    cfg         = load_config()
    wf_cfg      = cfg["workaround_finder"]
    llm_cfg     = cfg["llm"]
    embed_model = cfg["embed"]["model"]
    index_path  = str(Path(__file__).parent / wf_cfg["index_path"])
    jql         = wf_cfg["ingest_jql"] + f" ORDER BY updated DESC"

    # Connect Jira
    jira = get_jira(cfg)
    print(f"[JIRA] Connected to {jira.server_url}")

    if not args.skip_ingest:
        tickets = fetch_tickets(jira, jql, max_results=args.max)
        if not tickets:
            print("No tickets found. Check your JQL.")
            sys.exit(1)
        embed_model_obj, col = build_index(tickets, index_path, embed_model)
    else:
        from sentence_transformers import SentenceTransformer
        import chromadb
        print("[EMBED] Loading existing index ...")
        embed_model_obj = SentenceTransformer(embed_model)
        client          = chromadb.PersistentClient(path=index_path)
        col             = client.get_or_create_collection("workarounds")
        print(f"[EMBED] {col.count()} chunks in index.")

    # Determine query
    query = args.query or args.ticket_or_query
    if not query:
        query = input("\nEnter ticket ID or error text to search: ").strip()

    # If it looks like a ticket ID, fetch its summary
    if query.upper().startswith("SAC-") and "-" in query:
        print(f"[JIRA] Fetching ticket {query.upper()} ...")
        issue = jira.issue(query.upper(), fields="summary,description")
        query = f"{issue.fields.summary}\n{issue.fields.description or ''}"
        print(f"[JIRA] Query text: {query[:200]}")

    print(f"\n[SEARCH] Searching for: {query[:100]} ...")
    hits = search(query, embed_model_obj, col, top_k=wf_cfg["top_k"])
    print(f"[SEARCH] Top {len(hits)} results:")
    for h in hits:
        print(f"  {h['meta']['key']} (score={h['score']}) — {h['meta']['summary'][:80]}")

    print(f"\n[LLM] Generating workaround via {llm_cfg['provider']} / {llm_cfg['model']} ...")
    prompt   = build_prompt(query, hits)
    answer   = generate_local(prompt, llm_cfg["model"], llm_cfg["ollama_url"])

    print("\n" + "="*60)
    print("SUGGESTED WORKAROUND:")
    print("="*60)
    print(answer)
    print("\nSOURCES:")
    for h in hits:
        print(f"  [{h['meta']['key']}] {h['meta']['summary'][:70]} — {h['meta']['url']}")


if __name__ == "__main__":
    main()
