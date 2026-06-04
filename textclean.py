"""Canonical text cleaner — the SINGLE source of truth for normalizing step and
error text.

It is applied to BOTH sides of the match:
  - ingest:  the step/error pulled from a ticket before embedding
  - search:  the step/error parsed from a query before embedding

Keeping one function for both is the whole point: when the two sides clean
differently, the *same* failure embeds to two different places and matching
becomes inconsistent (e.g. one ticket keeps a Salesforce ID / emoji / wiki-bold
'*' that its siblings stripped, so it ranks below them for its own error).
"""

import re


def clean_text(text: str) -> str:
    """Normalize step/error text so identical failures collapse to identical
    strings. Removes per-order noise (record IDs, MSISDNs, order numbers),
    Jira/markup junk ('*', empty brackets), and non-ASCII (emoji / mojibake),
    then collapses whitespace. Idempotent."""
    if not text:
        return ""
    # Emoji, mojibake (UTF-8 decoded as latin-1, e.g. 'ðŸ¦'), and any other
    # non-ASCII — pure noise for an English step/error, and a frequent source of
    # split clusters when only some tickets carry it.
    text = re.sub(r'[^\x00-\x7f]+', ' ', text)
    # Salesforce record IDs (15-18 alphanumeric starting with a digit).
    text = re.sub(r'\b[0-9][A-Za-z0-9]{14,17}\b', '', text)
    # MSISDNs (10+ digits) / order numbers (6+ digits). 5-digit error codes kept.
    text = re.sub(r'\b\d{6,}\b', '', text)
    # Hanging Order / Order Id label left after its value was removed.
    text = re.sub(r'(?:Order\s+Id|Order)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*', '', text)            # Jira wiki bold markers
    text = re.sub(r'\(\s*\)', '', text)       # empty ()
    text = re.sub(r'\[\s*\]', '', text)       # empty []
    text = re.sub(r'\s*-\s*', ' ', text)      # hyphen separators -> space
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' -|:•‣●')   # trim stray bullets/seps
