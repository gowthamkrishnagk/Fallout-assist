"""Mask common customer identifiers before model calls and published comments."""

from __future__ import annotations

import re


def scrub(text: str) -> str:
    value = text or ""
    value = re.sub(r"\b[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-"
                   r"[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}\b", "[ID]", value)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", value)
    value = re.sub(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)", "[NUMBER]", value)
    value = re.sub(r"\b\d{7,}\b", "[NUMBER]", value)
    return value
