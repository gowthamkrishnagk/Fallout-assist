import chromadb
client = chromadb.PersistentClient(path='trackers/workaround_index')
col = client.get_or_create_collection('workarounds')
print(f'Chunks in DB: {col.count()}')
sample = col.get(limit=3)
for m in sample['metadatas']:
    print(f"  {m['key']}: {m['summary'][:60]}")
