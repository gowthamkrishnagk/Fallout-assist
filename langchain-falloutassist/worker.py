"""Run ingestion and Jira scans once or on independent intervals."""

from __future__ import annotations

import argparse
import time

from falloutassist_next import storage
from falloutassist_next.service import ingest_once, scan_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if not args.once and not args.loop:
        parser.error("Choose --once or --loop")
    storage.initialize()
    next_ingest = next_scan = 0.0
    while True:
        settings = storage.settings()
        now = time.monotonic()
        if settings["ingest_interval_minutes"] and now >= next_ingest:
            try:
                print(ingest_once(), flush=True)
            except Exception as exc:
                print(f"Ingestion failed: {type(exc).__name__}: {exc}", flush=True)
            next_ingest = time.monotonic() + settings["ingest_interval_minutes"] * 60
        if settings["scan_interval_minutes"] and now >= next_scan:
            try:
                print(scan_once(), flush=True)
            except Exception as exc:
                print(f"Scan failed: {type(exc).__name__}: {exc}", flush=True)
            next_scan = time.monotonic() + settings["scan_interval_minutes"] * 60
        if args.once:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
