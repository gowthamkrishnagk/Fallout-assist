"""Quick ingest test — shows assignee comment filtering in action."""
import json, os, sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, '.')

with open('config.json') as f:
    cfg = json.load(f)

import ingest as ing

print("Running ingest (10 tickets)...")
result = ing.ingest_jira(cfg, progress_cb=print)
print("\nResult:", result)

# Show what was stored
import chromadb
client = chromadb.PersistentClient(path='trackers/workaround_index')
col    = client.get_or_create_collection('workarounds')
print(f"\nChunks in DB: {col.count()}")
sample = col.get(limit=3)
for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas'])):
    print(f"\n--- {meta['key']} (assignee: {meta.get('summary','')[:50]}) ---")
    print(doc[:300])
