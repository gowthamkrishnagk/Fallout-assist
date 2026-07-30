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

# Timestamps, as they appear in the FATAL orchestration dumps
# ("* 6/3/2026 8:59 AM FATAL Exception happened during items processing: ...").
# Pure per-occurrence noise: two tickets hitting the SAME failure differ only by WHEN it
# happened, so embedding the timestamp splits one failure into N unrelated ones. Measured
# on the live corpus: 405 distinct failure signatures collapse to 321, and 218 apparent
# singletons to 148, once these are removed.
_DATETIME = re.compile(
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'                # 6/3/2026, 13-04-26
    r'|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'                 # 2026-06-04 (ISO)
    r'|\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b',      # 8:59 AM, 12:09:33
    re.IGNORECASE)


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
    # Timestamps — BEFORE the hyphen rule below, which would otherwise turn
    # '2026-06-04' into '2026 06 04' and leave the digits behind.
    text = _DATETIME.sub(' ', text)
    # Salesforce record IDs (15-18 alphanumeric starting with a digit).
    text = re.sub(r'\b[0-9][A-Za-z0-9]{14,17}\b', '', text)
    # MSISDNs (10+ digits) / order numbers / ICCIDs / IMSIs. 5-digit error codes kept.
    #
    # Bounded by NON-DIGITS, not by \b. A word boundary needs a word/non-word transition,
    # but letters and digits are both word characters — so '\b\d{6,}\b' never matched an
    # MSISDN glued to a word, and 'MobileVoice19392021851' kept its number. That split one
    # failure into 42 apparent singletons ('Subscribed Service MobileVoice<msisdn> cannot
    # be unblocked'), because every ticket carried a different subscriber. This form
    # matches any maximal run of 6+ digits wherever it sits, and is a strict superset of
    # the old rule. A leading 'code |' error code is protected by clean_error_text, which
    # splits it off before calling this.
    text = re.sub(r'(?<![0-9])\d{6,}(?![0-9])', '', text)
    # Hanging Order / Order Id label left after its value was removed.
    text = re.sub(r'(?:Order\s+Id|Order)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*', '', text)            # Jira wiki bold markers
    text = re.sub(r'\(\s*\)', '', text)       # empty ()
    text = re.sub(r'\[\s*\]', '', text)       # empty []
    text = re.sub(r'\s*-\s*', ' ', text)      # hyphen separators -> space
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' -|:•‣●')   # trim stray bullets/seps


# Leading 'code |' prefix of a Salesforce / SAC BOT error ("14081 | Plan instance
# already cancelled"). Mirrors errormatch._CODE_RE — same tolerance for wiki bold and
# bullet junk ahead of the digits. Duplicated rather than imported because errormatch
# imports THIS module; keep the two in sync.
_ERR_CODE_PREFIX = re.compile(r'^[\s*•·▪◦\-|>]*(\d{1,6})\s*\|(.*)$', re.DOTALL)


def clean_error_text(text: str) -> str:
    """clean_text for an ERROR value, preserving a leading 'code |' prefix.

    clean_text strips runs of 6+ digits (MSISDNs, order numbers), which would also
    erase a 6-digit error CODE — errormatch._CODE_RE accepts 1-6 digits, and that code
    is the sharpest discriminator in the corpus (a different code is a different
    failure). So split the code off, clean only the message, then re-join in the
    canonical 'code | message' form that error_signature parses back out. This is the
    same reason error_signature strips the prefix before cleaning.

    Use this — not clean_text — for anything that is an error value, on BOTH sides
    (ingest's stored error, search's parsed query error), so the identical error never
    normalizes two different ways. Idempotent."""
    m = _ERR_CODE_PREFIX.match(text or "")
    if not m:
        return clean_text(text)
    code, msg = m.group(1), clean_text(m.group(2))
    return f"{code} | {msg}" if msg else code


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
