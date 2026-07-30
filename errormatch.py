"""Vectorless error matching — the precise, embedding-free decider for the error leg.

Salesforce / SAC BOT errors arrive as a structured `code | message` signature
(e.g. "14081 | Plan instance already cancelled") and are ~90% byte-identical across
repeats of the SAME failure. MiniLM cosine is the wrong tool for that: it blurs the
discriminating part (the code — 14081 ≈ 11) and is nudged off by injected junk
(stray emoji, a differing order ID), so it sometimes matches the wrong failure.

This module matches the error LEXICALLY instead:
  - the leading error CODE is a hard gate — a different code is a different failure
  - the message is compared by token overlap after canonical cleaning (textclean)

LLM-free and embedding-free, so it costs no quota and needs no re-ingest (the raw
error is already stored in chunk metadata as meta["error"]).
"""

import re
from textclean import clean_text

# Leading code like "14047 | ...". Tolerate Jira/markup junk in front of it
# ("* 14047 | ...", bullets, stray whitespace) — the stored error keeps the wiki
# bold marker, so the digits are NOT at column 0.
_CODE_RE = re.compile(r'^[\s*•·▪◦\-|>]*(\d{1,6})\s*\|')
_TOKEN   = re.compile(r'[a-z0-9]+')

# Volatile per-occurrence junk that differs between two copies of the SAME error, so it
# must not enter the message comparison:
#   - timestamps ("6/3/2026 9:07 AM") in the FATAL orchestration dumps. textclean.clean_text
#     now strips these too (they were splitting one failure into N singletons), but the
#     pattern is KEPT here as well: the live index still holds values embedded before that
#     change, and the lexical matcher reads those stored values directly.
#   - Salesforce record IDs that start with a LETTER ("a5oPl00000H3QAqIAN") — clean_text
#     only strips digit-leading ones. 15- or 18-char alphanumeric containing a digit.
_DATETIME = re.compile(
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b',
    re.IGNORECASE)
_SF_ID    = re.compile(r'\b(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{15}(?:[A-Za-z0-9]{3})?\b')


def _strip_volatile(text: str) -> str:
    return _SF_ID.sub(' ', _DATETIME.sub(' ', text))

# A matching code is strong evidence on its own — even when the surrounding wording
# differs (a copy with extra chatter, or a human's shortened message). Floor a
# code-equal pair here so it is never treated as a weak match on token overlap alone.
_CODE_MATCH_FLOOR = 0.6


def error_signature(text: str) -> tuple[str, set]:
    """(code, message_tokens) for an error string.
      code           — the leading 'N |' code, or '' if there isn't one
      message_tokens — set of word tokens of the cleaned message (code prefix removed)

    clean_text (textclean.clean_text) strips emoji / mojibake, Salesforce record IDs,
    MSISDNs / order numbers (>=6 digits) and wiki markup — but KEEPS short error codes —
    so two copies of the same Salesforce error collapse to the same signature."""
    raw = text or ""
    m    = _CODE_RE.match(raw)
    code = m.group(1) if m else ""
    # Drop the code prefix before cleaning so its digits don't leak into the message
    # tokens (and so a code-only error yields an empty token set, not the code again).
    msg     = raw[m.end():] if m else raw
    # clean_text removes emoji / wiki markup / digit-leading IDs / 6+ digit numbers;
    # _strip_volatile additionally drops timestamps and letter-leading Salesforce IDs.
    tokens  = set(_TOKEN.findall(_strip_volatile(clean_text(msg)).lower()))
    return code, tokens


def error_similarity(query_err: str, cand_err: str) -> tuple[float, bool]:
    """Lexical error match → (score in [0,1], gated).

      gated=True  → DROP the candidate: both sides carry a code and the codes differ
                    (a different code is a different failure, never a weak match).
      score       → Jaccard token overlap of the cleaned messages. A matching code
                    floors the score (_CODE_MATCH_FLOOR) so a code-equal pair is never
                    buried by mere wording differences.

    Neutral pass: when the QUERY has no usable error (no code and no tokens) this
    returns (1.0, False) so the vector recall stands alone — the error simply doesn't
    constrain the match (e.g. a free-form query that mentions no error)."""
    q_code, q_tokens = error_signature(query_err)
    c_code, c_tokens = error_signature(cand_err)

    # No error on the query side → don't constrain; the vector net decides.
    if not q_code and not q_tokens:
        return 1.0, False

    # Hard code gate: both have a code and they differ → different failure.
    if q_code and c_code and q_code != c_code:
        return 0.0, True

    same_code = bool(q_code) and q_code == c_code

    # Messages carry no comparable tokens → the code alone is the verdict.
    if not q_tokens and not c_tokens:
        return (1.0 if same_code else 0.0), False

    union = len(q_tokens | c_tokens)
    score = len(q_tokens & c_tokens) / union if union else 0.0
    if same_code:
        score = max(score, _CODE_MATCH_FLOOR)
    return round(score, 3), False


def text_similarity(a: str, b: str) -> float:
    """Tolerant lexical match for clean, copy-pasted fields (e.g. the failed step name).
    Cleaned-token Jaccard with a subset bonus, so 'Modify' vs 'Modify Service' still
    scores high. Returns 1.0 on exact/subset overlap, 0.0 on no overlap. Used for the
    step leg now that input is always copied from the system (never paraphrased), so
    exact comparison is sharper than a blurred embedding."""
    ta = set(_TOKEN.findall(_strip_volatile(clean_text(a)).lower()))
    tb = set(_TOKEN.findall(_strip_volatile(clean_text(b)).lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    if ta <= tb or tb <= ta:
        return 1.0
    return round(len(ta & tb) / len(ta | tb), 3)
