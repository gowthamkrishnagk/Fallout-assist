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


# Comments that merely point at another ticket ("duplicate, please refer to
# SAC-231619") are not workarounds — they must be followed to the real fix or
# dropped, never shown as a resolution.
_POINTER_RE = re.compile(
    r'duplicat\w*'
    r'|please\s+refer\s+to'
    r'|refer\s+to\s+(?:the\s+)?(?:ticket|jira|sac)'
    r'|merged\s+(?:with|into)'
    r'|tracked\s+(?:in|under)'
    r'|raised\s+as\s+(?:a\s+)?duplicate',
    re.IGNORECASE,
)
_TICKET_REF_RE = re.compile(r'browse/([A-Za-z]+-\d+)|\b([A-Z]{2,}-\d{3,})\b')


def is_pointer_comment(body: str) -> bool:
    """True when a comment just points to another ticket instead of describing a
    fix (e.g. 'duplicated ticket please refer to SAC-231619'). Requires both a
    pointer phrase AND a ticket reference, to avoid flagging a real fix that merely
    mentions a ticket in passing."""
    if not body:
        return False
    has_ref = bool(re.search(r'browse/[A-Za-z]+-\d+|smart-?link|\b[A-Z]{2,}-\d{3,}\b', body))
    return bool(_POINTER_RE.search(body)) and has_ref


_FIX_BLOCK_RE = re.compile(r'===\s*FIX\s*===\s*(.*?)\s*===\s*END\s*===',
                           re.IGNORECASE | re.DOTALL)
_FIX_OPEN_RE  = re.compile(r'===\s*FIX\s*===\s*(.+)$', re.IGNORECASE | re.DOTALL)


def extract_fix_block(body: str) -> str:
    """If a comment contains a `=== FIX === ... === END ===` block, return just
    that block (markers normalized), dropping any surrounding chatter — so the
    stored resolution is the clean fix the engineer wrote. Tolerates a missing
    `=== END ===` (takes everything after `=== FIX ===`). Returns the body
    unchanged when there is no block."""
    if not body:
        return body
    m = _FIX_BLOCK_RE.search(body) or _FIX_OPEN_RE.search(body)
    if not m:
        return body
    inner = m.group(1).strip()
    return f"=== FIX ===\n{inner}\n=== END ==="


def referenced_ticket(body: str) -> str:
    """Extract the referenced ticket key from a pointer comment ('...SAC-231619'),
    uppercased, or '' if none found."""
    if not body:
        return ""
    m = _TICKET_REF_RE.search(body)
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").upper()
