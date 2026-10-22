"""Evaluate local retrieval without drafting or publishing Jira comments."""

import argparse
import json
from pathlib import Path

from falloutassist_next import storage
from falloutassist_next.history import HistoryIndex
from falloutassist_next.retrieval_evaluation import evaluate, load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="Private, human-verified JSONL cases")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/report.json"))
    parser.add_argument("--fetch-current-issues", action="store_true",
                        help="Explicitly allow Jira REST reads for cases containing only issue_key")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    for case in cases:
        for key in case["expected_source_keys"]:
            row = storage.source_document(key)
            if not row or storage.source_tier(row["source_type"], key) == "low":
                parser.error(f"Expected source {key} is not a usable indexed source")
        if not case.get("query_text"):
            if not args.fetch_current_issues:
                parser.error("Cases without query_text require --fetch-current-issues")
            from falloutassist_next.plugins import JiraPlugin
            _, issue = JiraPlugin().issue(case["issue_key"])
            case["query_text"] = issue["query_text"]
    report = evaluate(cases, HistoryIndex().search, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    print(f"Private report: {args.output}")


if __name__ == "__main__":
    main()
