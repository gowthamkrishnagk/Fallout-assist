"""One-time import of accounts, rules and feedback; no legacy runtime imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from falloutassist_next import storage


def _records(root: Path, config: dict, section: str, name: str, fallback: str) -> list:
    relative = config.get(section, {}).get(name, fallback)
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Legacy data path leaves the selected project")
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def migrate(root: Path) -> dict:
    root = root.resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    storage.initialize()
    counts = {"users": 0, "rules_to_review": 0, "feedback": 0}
    for user in _records(root, config, "auth", "users_path", "trackers/users.json"):
        email = str(user.get("email") or "").strip().lower()
        password_hash = user.get("password_hash") or ""
        if not email or not password_hash or storage.user_by_email(email):
            continue
        storage.create_user(user.get("id") or uuid.uuid4().hex, email, password_hash,
                            "admin" if user.get("role") == "admin" else "user")
        counts["users"] += 1
    for rule in _records(root, config, "workaround_finder", "rules_path",
                         "trackers/workaround_rules.json"):
        match = rule.get("match") or {}
        pattern = (match.get("error_description") or {}).get("pattern") or ""
        workaround = rule.get("workaround_text") or ""
        if not pattern or not workaround or any(r["id"] == rule.get("id") for r in storage.rules()):
            continue
        failed_step = (match.get("failed_step") or {}).get("value") or ""
        order_type = (match.get("order_type") or {}).get("value") or ""
        storage.save_rule({
            "id": rule.get("id") or uuid.uuid4().hex,
            # Parsing/normalization changed; an admin reapproves migrated rules.
            "status": "draft", "error_pattern": pattern,
            "failed_step": failed_step, "order_type": order_type,
            "workaround_text": workaround, "match": match,
            "created_by": rule.get("created_by") or "legacy_import",
        })
        storage.audit("migration", "legacy_rule_imported_for_review", rule.get("id") or "",
                      {"legacy_status": rule.get("status", "")})
        counts["rules_to_review"] += 1
    for index, vote in enumerate(_records(root, config, "workaround_finder", "feedback_path",
                                          "trackers/feedback.json")):
        candidate = str(vote.get("key") or "").strip()
        direction = "up" if vote.get("vote") == 1 else "down" if vote.get("vote") == -1 else ""
        if not candidate or not direction:
            continue
        signature = str(vote.get("error") or vote.get("step") or "")
        signature_hash = hashlib.sha256(signature.encode()).hexdigest()[:12]
        issue = f"legacy:{candidate}:{signature_hash}"
        voter = str(vote.get("voter") or "").strip()
        actor = f"{voter} (legacy #{index})" if voter else f"legacy_import_{index}"
        _, created = storage.add_feedback(issue, candidate, direction, "", actor, None,
                                          str(vote.get("step") or ""),
                                          str(vote.get("error") or ""))
        counts["feedback"] += int(created)
    storage.audit("migration", "legacy_import_completed", str(root), counts)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", required=True, type=Path)
    args = parser.parse_args()
    print(migrate(args.source))
