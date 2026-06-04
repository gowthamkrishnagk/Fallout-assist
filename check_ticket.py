"""Inspect what a Jira ticket actually stored in the index, and what wins a
search for a given error. Run on the SERVER (same machine as the live index):

    py check_ticket.py SAC-234948
    py check_ticket.py SAC-234948 "ReliesOn item should be disconnected"

Shows the ticket's stored step/error metadata, then (if an error string is
given) the top search matches so you can see whether the right ticket ranks.
"""
import json
import sys
from pathlib import Path

import chromadb

cfg        = json.loads(Path("config.json").read_text(encoding="utf-8"))
index_path = str(Path(cfg["workaround_finder"]["index_path"]))
embed_model = cfg["embed"]["model"]

key       = sys.argv[1] if len(sys.argv) > 1 else "SAC-234948"
query_err = sys.argv[2] if len(sys.argv) > 2 else ""

client = chromadb.PersistentClient(path=index_path)

print(f"=== Stored chunks for {key} ===")
found = False
for col_name in ("workarounds_step", "workarounds_error"):
    col = client.get_or_create_collection(col_name)
    got = col.get(where={"key": key})
    print(f"\n[{col_name}] {len(got['ids'])} chunk(s)")
    for cid, meta in zip(got["ids"], got["metadatas"]):
        found = True
        print(f"  id={cid}")
        print(f"    step  = {meta.get('step','')!r}")
        print(f"    error = {meta.get('error','')!r}")
        print(f"    summary = {meta.get('summary','')[:90]!r}")

if not found:
    print(f"\n!! {key} is NOT in the index — it was never ingested "
          f"(JQL mismatch, no usable comment, or pruned).")

if query_err:
    import embedder, vectordb
    print(f"\n=== Top matches searching error = {query_err!r} (error-only) ===")
    error_emb = embedder.embed_one(f"Error: {query_err}", embed_model)
    for h in vectordb.search_dual(None, error_emb, 10, index_path):
        m = h["meta"]
        star = "  <-- TARGET" if m.get("key") == key else ""
        print(f"  {h['score']:.3f}  {m.get('key','?'):12} "
              f"err={m.get('error','')[:45]!r}{star}")
