"""Re-embed stored usable evidence, without a Jira scan or any publishing."""

import argparse
import uuid

from falloutassist_next import storage
from falloutassist_next.history import EMBEDDING_INDEX_VERSION, HistoryIndex


def reindex():
    holder = uuid.uuid4().hex
    if not storage.claim_lease("jira_ingest", holder, 7200):
        raise RuntimeError("A history update or document mutation is running; retry later")
    try:
        index, batch, count = HistoryIndex(), [], 0
        storage.set_state("corpus_version", storage.utc_now())
        for key in storage.all_document_keys():
            row = storage.source_document(key)
            if not row or storage.source_tier(row["source_type"], key) == "low":
                continue
            batch.append({**row, "order_reason": row["metadata"].get("order_reason", "")})
            if len(batch) == 32:
                if not storage.claim_lease("jira_ingest", holder, 7200):
                    raise RuntimeError("Re-index lease was lost")
                index.upsert_many(batch)
                count += len(batch)
                batch.clear()
                print(f"Re-embedded {count} sources", flush=True)
        index.upsert_many(batch)
        count += len(batch)
        storage.set_state("embedding_index_version", EMBEDDING_INDEX_VERSION)
        storage.set_state("corpus_version", storage.utc_now())
        storage.audit("operator", "local_vectors_reindexed", "history",
                      {"sources": count, "embedding_version": EMBEDDING_INDEX_VERSION})
        return count
    finally:
        storage.release_lease("jira_ingest", holder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the new vectors to the configured local index")
    args = parser.parse_args()
    if not args.apply:
        print(f"Usable sources: {storage.usable_document_count()}. Run with --apply to re-embed stored evidence.")
        return
    print(f"Finished: {reindex()} sources re-embedded. Jira was not contacted.")


if __name__ == "__main__":
    main()
