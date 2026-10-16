"""Match approved rule conditions without invoking a model."""

from __future__ import annotations

import re

from . import storage
from .history import field, parse_failure


def _matches(condition: dict | None, actual: str, error: bool = False) -> bool:
    if not condition:
        return True
    op = condition.get("op")
    if error and op == "matches_pattern":
        try:
            return bool(re.search(condition.get("pattern", ""), actual, re.IGNORECASE | re.DOTALL))
        except re.error:
            return False
    if op == "equals":
        return actual.casefold() == str(condition.get("value", "")).casefold()
    if op == "in":
        return actual.casefold() in {str(v).casefold() for v in condition.get("values") or []}
    return False


def match_approved(query_text: str) -> dict | None:
    step, error = parse_failure(query_text)
    if not error:
        return None
    order_type = field(query_text, "Order Type")
    order_reason = field(query_text, "Order Reason")
    choices = []
    for rule in storage.rules("approved"):
        match = rule.get("match") or {
            "error_description": {"op": "matches_pattern", "pattern": rule["error_pattern"]},
            "failed_step": {"op": "equals", "value": rule["failed_step"]}
            if rule["failed_step"] else None,
        }
        if not _matches(match.get("error_description"), error, True):
            continue
        if not all(_matches(match.get(name), value) for name, value in (
                ("failed_step", step), ("order_type", order_type),
                ("order_reason", order_reason))):
            continue
        specificity = sum(bool(match.get(name)) for name in
                          ("failed_step", "order_type", "order_reason"))
        choices.append((specificity, rule["created_at"], rule))
    return max(choices, key=lambda item: (item[0], item[1], item[2]["id"]))[2] if choices else None
