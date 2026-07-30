"""The team's 8-field workaround table — the resolution format adopted 2026-07-28.

Two distinct jobs, and it matters which is which:

  * The WORKAROUND an engineer acts on is shown as `=== FIX ===` numbered steps —
    that is the readable, actionable artifact (see `to_fix_block` / `from_fix_answer`).
  * This table is the RESOLUTION COMMENT FORMAT the assignee posts when they close the
    ticket. FalloutAssist pre-fills it and hands it over as a draft, because a
    consistently-formatted resolution today is what makes retrieval work tomorrow.

Engineers close SAC tickets with a two-column table instead of free prose:

    | Order Type Failed | Change order |
    | BAN CAN           | Account checked in SF |
    | MSISDN            | 17873266937 |
    | Cause             | Device protection is not properly mapped under device |
    | Category          | BYOD Change |
    | Solution applied  | Completed the step |
    | System modified   | SF |
    | Customer action   | order item and frls looks good properly mapped |

This module does three jobs:

  parse_table()   — read that table back out of a Jira comment, so ingest can store the
                    fields as dedicated metadata. These tables are far richer than the
                    ~56-char median free-text resolution, so they are the best retrieval
                    material in the corpus.
  ticket_fields() — fill the four fields that belong to THE TICKET BEING ASKED ABOUT,
                    deterministically from its description. Never from a retrieved
                    source and never from the LLM: BAN CAN and MSISDN identify a
                    specific customer, and copying them from a matched ticket would put
                    someone else's account in a workaround. Order Type Failed and
                    Category are likewise verbatim description values, so generating
                    them would only add a chance to be wrong.
  build_prompt() / parse_llm_fields() — the LLM fills only the remaining four
                    (Cause, Solution applied, System modified, Customer action),
                    grounded in retrieved resolutions exactly like the === FIX === path.

Observed variation the parser has to tolerate (from real posted comments): rows omitted
entirely, blank values, "NA" / "N/A" / "" used interchangeably, a markdown separator row
(`| --- | --- |`) when the table came through ADF, and greetings / @mentions / images
wrapped around the table.
"""

import re

# Canonical field order — the table is always rendered in this order.
FIELDS = [
    "Order Type Failed",
    "BAN CAN",
    "MSISDN",
    "Cause",
    "Category",
    "Solution applied",
    "System modified",
    "Customer action",
]

# Filled from the current ticket's description, deterministically. See module docstring
# for why these must never be generated or copied from a matched ticket.
TICKET_FIELDS = ["Order Type Failed", "BAN CAN", "MSISDN", "Category"]

# Filled by the LLM from the retrieved resolutions.
LLM_FIELDS = ["Cause", "Solution applied", "System modified", "Customer action"]

NOT_APPLICABLE = "NA"

# Returned instead of a table when the sources contain no real workaround, so the caller
# can fall back to showing the raw matched comment rather than an invented fix. Same
# contract as search.NO_FIX_SENTINEL.
NO_FIX_SENTINEL = "NO_RELIABLE_WORKAROUND"

# Label spellings seen in real comments -> canonical field.
_ALIASES = {
    "order type failed": "Order Type Failed",
    "order type": "Order Type Failed",
    "failed order type": "Order Type Failed",
    "ban can": "BAN CAN",
    "ban-can": "BAN CAN",
    "bancan": "BAN CAN",
    "ban": "BAN CAN",
    "msisdn": "MSISDN",
    "cause": "Cause",
    "root cause": "Cause",
    "reason": "Cause",
    "category": "Category",
    "solution applied": "Solution applied",
    "solution": "Solution applied",
    "workaround": "Solution applied",
    "workaround applied": "Solution applied",
    "system modified": "System modified",
    "system": "System modified",
    "systems modified": "System modified",
    "customer action": "Customer action",
    "customer": "Customer action",
    "next step": "Customer action",
}

# A markdown table separator row ('| --- | --- |'), produced when an ADF table is
# rendered to text. Carries no data.
_SEP_CELL = re.compile(r'^:?-{2,}:?$')

# Values that mean "nothing here".
_EMPTYISH = {"", "na", "n/a", "n.a.", "none", "-", "--", "nil"}


def _norm_label(s: str) -> str:
    """Lowercase, strip markup/punctuation, collapse whitespace — so '*BAN-CAN:*' and
    'ban can' both resolve to the same alias key."""
    s = re.sub(r'[*_`]', '', s or '')
    s = s.strip().strip(':').strip()
    return re.sub(r'\s+', ' ', s).lower()


def _cells(line: str) -> list[str]:
    """Cells of a pipe-delimited row, or [] if the line isn't one. Handles both the
    Jira wiki form (|a|b|) and the ADF-rendered markdown form (| a | b |)."""
    t = line.strip()
    if not t.startswith("|"):
        return []
    # Jira header rows use '||'; treat them like normal separators.
    t = t.replace("||", "|")
    parts = [p.strip() for p in t.strip("|").split("|")]
    return parts if parts else []


def parse_table(body: str, min_fields: int = 3) -> dict:
    """{canonical field: value} for a workaround table found in `body`, else {}.

    Only the FIRST occurrence of each label wins, so a later quote of the table (a reply
    that includes the original) cannot overwrite the real values. Returns {} unless at
    least `min_fields` canonical fields are present — that threshold is what stops an
    unrelated two-column table in a comment from being mistaken for this format — AND at
    least one of them has a value. An all-empty table is the blank template handed to an
    assignee to fill in; treating it as this format would let a table with no content
    score as a rich resolution."""
    if not body or "|" not in body:
        return {}
    out: dict = {}
    for line in body.splitlines():
        cells = _cells(line)
        if len(cells) < 2:
            continue
        if all(_SEP_CELL.match(c) for c in cells if c):
            continue                                  # markdown separator row
        field = _ALIASES.get(_norm_label(cells[0]))
        if not field or field in out:
            continue
        # A value containing '|' splits into extra cells — rejoin them.
        value = " | ".join(c for c in cells[1:] if c is not None).strip()
        out[field] = re.sub(r'\s+', ' ', value).strip()
    if len(out) < min_fields or not any(clean_value(v) for v in out.values()):
        return {}
    return out


def is_table(body: str) -> bool:
    return bool(parse_table(body))


def rows(rendered: str) -> list[tuple[str, str]]:
    """[(label, value)] for a table produced by `render`, in the order written.

    Deliberately NOT parse_table: that one canonicalises labels, keeps only the first
    occurrence, and returns {} for an all-empty table — correct when reading a resolution
    an engineer wrote, wrong here. This reads back OUR OWN rendering to re-display it, so
    a blank template must survive with its empty rows intact.

    It does not use `_cells` either, for the same reason: `_cells` strips ALL outer pipes,
    so a blank row (`|Solution applied||`) collapses to a single cell and disappears.
    `render` always writes exactly `|label|value|`, so one leading and one trailing pipe
    come off and the rest is label + value."""
    out = []
    for line in (rendered or "").splitlines():
        t = line.strip()
        if not (t.startswith("|") and t.endswith("|") and len(t) > 2):
            continue
        parts = t[1:-1].split("|")
        if len(parts) < 2:
            continue
        # A value containing '|' splits into extra parts — rejoin them.
        label, value = parts[0].strip(), "|".join(parts[1:]).strip()
        if _SEP_CELL.match(label) or (value and _SEP_CELL.match(value)):
            continue                                   # markdown separator row
        out.append((label, value))
    return out


# ── Reading a table back from stored chunk metadata ───────────────────────────────
#
# ingest parses the table ONCE, when the comment is indexed, and stores the rows as
# dedicated `wa_*` metadata. Search-time consumers read them from there instead of
# re-parsing prose on every query — which also side-steps the `comment_body` 1000-char
# cap, since each row is stored under its own limit.
#
# Only the four generated rows plus Category are persisted. The ticket-owned rows (Order
# Type Failed / BAN CAN / MSISDN) belong to the MATCHED ticket and are always re-derived
# from the ticket being asked about, so storing them would serve nothing but a mistake.
META_FIELDS = {
    "wa_cause":    "Cause",
    "wa_solution": "Solution applied",
    "wa_system":   "System modified",
    "wa_customer": "Customer action",
    "wa_category": "Category",
}


def meta_for(body: str) -> dict:
    """The `wa_*` metadata to store for a resolution comment. Written by ingest; the
    single definition of what gets persisted, so the writer and `from_meta` can't drift."""
    tbl = parse_table(body)
    out = {"wa_table": str(bool(tbl))}
    # Each row keeps its own cap: the value ingest stores must not depend on where the
    # 1000-char comment_body truncation happened to land.
    caps = {"wa_cause": 300, "wa_solution": 500, "wa_system": 80,
            "wa_customer": 300, "wa_category": 120}
    for meta_key, field in META_FIELDS.items():
        out[meta_key] = clean_value(tbl.get(field, ""))[:caps[meta_key]]
    return out


def from_meta(meta: dict) -> dict:
    """{field: value} rebuilt from a chunk's stored `wa_*` metadata, or {} when this
    chunk's resolution was not a table (or carried no row worth reusing)."""
    if str((meta or {}).get("wa_table", "")).strip().lower() != "true":
        return {}
    out = {}
    for meta_key, field in META_FIELDS.items():
        v = clean_value((meta or {}).get(meta_key, ""))
        if v:
            out[field] = v
    return out


def table_of(candidate: dict) -> dict:
    """The table rows for a retrieved candidate — the single entry point for every
    search-time consumer.

    Prefers what ingest already parsed. Falls back to parsing the body for anything the
    `wa_*` fields can't cover: chunks indexed before those fields existed, uploaded docs
    and approved user fixes (neither writes them), and a pointer comment resolved live
    from Jira, where the metadata describes the pointer rather than the fetched fix."""
    return (candidate or {}).get("table") or parse_table((candidate or {}).get("comment") or "")


def clean_value(v: str) -> str:
    """A field value normalized for storage/display: '' for the many spellings of
    'nothing here', so downstream code tests one thing instead of six."""
    v = re.sub(r'\s+', ' ', (v or '')).strip()
    return "" if v.lower() in _EMPTYISH else v


# ── Reading the CURRENT ticket's own fields ───────────────────────────────────────

def _raw_field(desc: str, label: str) -> str:
    """A labelled value from a ticket description, VERBATIM — no canonical cleaning.

    Deliberately raw: BAN-CAN and MSISDN are exactly the long digit runs that
    textclean.clean_text strips (correctly, for embedding), but here they are the
    content. This value is only ever rendered, never embedded."""
    m = re.search(
        rf'(?:^|\n|\|)\s*\*{{0,2}}{label}\*{{0,2}}\s*[:|]\s*\*{{0,2}}([^*\n|]{{1,120}}?)\*{{0,2}}\s*(?:\||$|\n)',
        desc or "", re.IGNORECASE | re.MULTILINE)
    return re.sub(r'\s+', ' ', m.group(1)).strip() if m else ""


def ticket_fields(description: str) -> dict:
    """The four ticket-owned fields, from this ticket's description only.

    Order Type Failed: 'Sales' -> 'Sales order', matching how the team writes it
    ('Sales order', 'Change order'). Category: the description's Order Reason verbatim —
    confirmed on both ORDER_FALLOUT examples (Existing, BYOD Change). Non-order tickets
    have neither, and correctly come out as NA."""
    desc = description or ""
    order_type = _raw_field(desc, "Order Type")
    reason     = _raw_field(desc, "Order Reason")
    ban        = _raw_field(desc, "BAN[-\\s]?CAN")
    msisdn     = _raw_field(desc, "MSISDN")

    if order_type and not re.search(r'\border\b', order_type, re.IGNORECASE):
        order_type = f"{order_type} order"

    return {
        "Order Type Failed": order_type or NOT_APPLICABLE,
        "BAN CAN":           ban or NOT_APPLICABLE,
        "MSISDN":            msisdn or NOT_APPLICABLE,
        "Category":          reason or NOT_APPLICABLE,
    }


# ── Rendering ─────────────────────────────────────────────────────────────────────

def render(fields: dict, fill_missing: str = NOT_APPLICABLE) -> str:
    """The 8-row table in Jira wiki markup, always in canonical order. Single pipes =
    all body rows (no header), which is how the team's label/value tables read.

    `fill_missing` is what an absent value becomes: NA for a finished table, and '' for
    a blank template whose empty rows are there for a human to fill in — 'NA' would
    claim the row doesn't apply, which is a different statement."""
    rows = []
    for f in FIELDS:
        v = clean_value(fields.get(f, "")) or fill_missing
        rows.append(f"|{f}|{v}|")
    return "\n".join(rows)


# ── LLM synthesis of the four generated fields ────────────────────────────────────

# One real posted table, as a format exemplar. Worth including because the corpus has no
# other in-band examples the model could imitate (=== FIX === adoption is 0%), and a
# concrete example beats describing the shape in prose.
_EXEMPLAR = (
    "Cause: Device protection is not properly mapped under device\n"
    "Solution applied: Completed the step\n"
    "System modified: SF\n"
    "Customer action: order item and frls looks good properly mapped hence completed the step"
)

_SYSTEM_CHOICES = "Salesforce / SF / Matrixx / Aria / Nokia / Network / EDA"


def build_prompt(query: str, sources: str, has_feedback: bool = False) -> str:
    """Ask for ONLY the four generated fields, one per line.

    Four labelled lines rather than a table: the model cannot mangle the row structure,
    and parse_llm_fields stays trivial. The table itself is assembled in code, so the
    ticket-owned fields cannot be touched by the model.

    `sources` is the same rendered source block the === FIX === prompt uses, so both
    paths are grounded identically."""
    feedback_rule = (
        "- Sources are tagged with human feedback. PREFER steps from VERIFIED / "
        "\U0001F44D-confirmed sources, and avoid a \U0001F44E source unless it is the only "
        "one that clearly fits.\n" if has_feedback else ""
    )
    return (
        "You are a Salesforce order-fallout support assistant. A new issue needs a "
        "workaround:\n\n"
        f"{query}\n\n"
        "Below are the most similar PAST RESOLVED tickets.\n\n"
        "Output EXACTLY these four lines, nothing else — no table, no extra commentary:\n"
        "Cause: <the root cause of THIS failure, one line>\n"
        "Solution applied: <the concrete action(s) that fix it, imperative, one line>\n"
        f"System modified: <which system was changed: {_SYSTEM_CHOICES}, or NA if none>\n"
        "Customer action: <what the agent/customer does next to verify, or NA>\n\n"
        "Rules:\n"
        "- Use ONLY causes, actions, field names and system names that appear in the "
        "sources. Do NOT invent steps and do NOT add generic advice.\n"
        "- Never copy an MSISDN, BAN-CAN, order number or record ID out of a source — "
        "those belong to a different customer. Describe the action without them.\n"
        "- Keep each value to one line. If several actions are needed, join them with "
        "'; ' on the Solution applied line.\n"
        "- 'System modified' names the system that was CHANGED, not one that was merely "
        "checked. NA if nothing was modified.\n"
        f"{feedback_rule}"
        "- Each source shows its own Failed Step / Error. IGNORE any source whose own "
        "failure clearly differs from the issue above.\n"
        f"- If NONE of the sources contain a real workaround (only acknowledgements or "
        f"status notes like 'already activated', 'fixed by L3', 'done', or a bare "
        f"'closed'), reply with exactly: {NO_FIX_SENTINEL}\n\n"
        "Example of the expected output shape:\n"
        f"{_EXEMPLAR}\n"
        f"{sources}\n"
    )


def parse_llm_fields(answer: str) -> dict:
    """{field: value} for the four generated fields from the model's reply, or {} when
    it declined or returned nothing usable.

    Tolerant of a model that wraps the answer in a table or adds stray markup anyway —
    it accepts both 'Cause: x' lines and '|Cause|x|' rows."""
    if not answer or NO_FIX_SENTINEL in answer:
        return {}
    out: dict = {}
    # Table rows, if the model produced them despite the instruction.
    out.update({k: v for k, v in parse_table(answer, min_fields=1).items()
                if k in LLM_FIELDS})
    for line in answer.splitlines():
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        field = _ALIASES.get(_norm_label(label))
        if field in LLM_FIELDS and field not in out:
            out[field] = re.sub(r'\s+', ' ', value).strip().strip('|').strip()
    return {k: v for k, v in out.items() if clean_value(v)}


# ── LLM-free fallback: build the table from a raw resolution comment ──────────────

# Systems named in the 'System modified' row, longest-first so 'Salesforce' wins over a
# bare 'SF' inside it. Values are what the team writes.
_SYSTEMS = [
    (r'salesforce', "Salesforce"), (r'matrixx', "Matrixx"), (r'\baria\b', "Aria"),
    (r'nokia', "Nokia"), (r'\beda\b', "EDA"), (r'network', "Network"),
    (r'asurion', "Asurion"), (r'\bsf\b', "SF"),
]

# Customer identifiers that must never travel from a past ticket into a table row:
# long digit runs (MSISDN / BAN-CAN / order / ICCID) and Salesforce record IDs in both
# digit-leading and letter-leading forms.
_IDENTIFIERS = re.compile(
    r'(?<![0-9])\d{6,}(?![0-9])'
    r'|\b(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{15}(?:[A-Za-z0-9]{3})?\b')

_FIX_MARKERS = re.compile(r'===\s*(?:FIX|END)\s*===', re.IGNORECASE)


def has_identifiers(text: str) -> bool:
    """True when `text` contains something shaped like a customer identifier.

    For text that must NOT be silently rewritten — an engineer's own submitted fix, where
    stripping digit runs would also eat a 6-digit error code (errormatch keys on codes of
    1-6 digits). Flag it for a human instead of mangling it."""
    return bool(_IDENTIFIERS.search(text or ""))


def scrub_identifiers(text: str) -> str:
    """Remove customer identifiers from text lifted out of a PAST ticket's comment.

    The raw comment belongs to a different order, so its MSISDN / BAN-CAN / record IDs
    are wrong for the ticket being answered — and inside a 'Solution applied' row they
    would read as instructions. The action itself survives; only the identifiers go."""
    return re.sub(r'\s{2,}', ' ', _IDENTIFIERS.sub('', text or '')).strip(' -|,;')


def infer_system(text: str) -> str:
    """The system named in a resolution, for the 'System modified' row. '' if none is
    mentioned — deliberately not guessed, since the row means the system that was
    CHANGED and a wrong system sends an engineer to the wrong console."""
    t = (text or "").lower()
    for pattern, name in _SYSTEMS:
        if re.search(pattern, t):
            return name
    return ""


def from_raw(description: str, body: str, fields: dict | None = None) -> str:
    """The table built from a raw resolution comment, with NO LLM.

    `fields` are the matched resolution's already-parsed rows (from `table_of`); pass them
    to skip re-parsing and to use the un-truncated stored values.

    Used whenever synthesis can't run but a workaround still has to be presented in the
    team's format: the LLM is switched off, every provider failed, or the matched
    resolution is an old `=== FIX ===` block. The past comment's own text becomes
    'Solution applied' (identifiers scrubbed, steps joined onto one line) and
    'System modified' is taken from whichever system it names. Cause and Customer action
    are left NA rather than invented — they are genuinely not recoverable without a model.

    If `body` is ALREADY a table, its four generated rows are reused verbatim and only
    the ticket-owned rows are refilled from this ticket."""
    out = ticket_fields(description)

    existing = fields if fields is not None else parse_table(body)
    if existing:
        out.update({f: existing[f] for f in LLM_FIELDS if f in existing})
        return render(out)

    text = _FIX_MARKERS.sub(" ", body or "")
    # Numbered / bulleted steps -> one line, so the row stays a row.
    steps = [re.sub(r'^\s*(?:\d+[.)]|[-*•])\s*', '', ln).strip()
             for ln in text.splitlines() if ln.strip()]
    one_line = "; ".join(s for s in steps if s)
    solution = scrub_identifiers(one_line)[:500]

    out["Solution applied"] = solution or NOT_APPLICABLE
    out["System modified"]  = infer_system(solution) or NOT_APPLICABLE
    return render(out)


def compose(description: str, llm_answer: str) -> str:
    """The finished table: ticket-owned fields from `description`, generated fields from
    the model's reply. Returns '' when the model declined, so the caller falls back to
    the raw matched comment instead of posting a table full of NA."""
    gen = parse_llm_fields(llm_answer)
    if not gen:
        return ""
    return render({**ticket_fields(description), **gen})


# ── The workaround as `=== FIX ===` steps (the artifact an engineer acts on) ───────
#
# The table above is the resolution FORMAT; numbered steps are what someone actually
# reads and follows. Both are produced for every suggestion, from the same material, so
# the fix that is shown and the draft that gets posted can never disagree.

# A step list may arrive ';'-joined on one line (how 'Solution applied' stores it) or as
# separate lines (how a prose comment or a synthesized block writes it).
_STEP_SPLIT = re.compile(r'\s*;\s*|\n')

# 'Cause:' / 'Root Cause:' — a labelled cause line, which is never a step.
_CAUSE_LINE = re.compile(r'(?:^|\n)[ \t*_]*(?:root[ \t]+)?cause[ \t]*:[ \t]*([^\n]+)',
                         re.IGNORECASE)
_CAUSE_ONLY = re.compile(r'^[ \t*_]*(?:root[ \t]+)?cause[ \t]*:', re.IGNORECASE)


def _steps(text: str) -> list[str]:
    """`text` split into individual actions — one per line or ';'-joined clause, with
    list markers and any labelled cause line stripped out."""
    out = []
    for part in _STEP_SPLIT.split(_FIX_MARKERS.sub("\n", text or "")):
        s = re.sub(r'^\s*(?:\d+[.)]|[-*•])\s*', '', part).strip().strip('.;')
        if s and not _CAUSE_ONLY.match(s):
            out.append(s)
    return out


def _cause_of(text: str) -> str:
    m = _CAUSE_LINE.search(text or "")
    return m.group(1).strip() if m else ""


# A 'Customer action' value that states there is nothing to do. It earns its place in
# the table row — the assignee did verify — but not in a numbered procedure, where
# "no action needed" reads as a step somebody still has to carry out.
_NO_ACTION = re.compile(r'(?i)^(?:no|none|nil|nothing|not)\b[^.]{0,40}$')


def render_fix(cause: str, steps: list[str]) -> str:
    """A `=== FIX ===` block from a cause and a step list. Identifiers are scrubbed
    here rather than at the call sites, because every route into this function carries
    text lifted from a DIFFERENT customer's ticket."""
    steps = [s for s in (scrub_identifiers(x) for x in steps) if s]
    if not steps:
        return ""
    lines = []
    if clean_value(cause):
        lines.append(f"Root Cause: {scrub_identifiers(cause)}")
    # Steps often arrive mid-sentence, having been split out of a ';'-joined row.
    lines += [f"{i}. {s[0].upper()}{s[1:]}" for i, s in enumerate(steps, 1)]
    body = "\n".join(lines)
    return f"=== FIX ===\n{body}\n=== END ==="


def to_fix_block(body: str, fields: dict | None = None) -> str:
    """A past resolution comment rendered as the `=== FIX ===` workaround block, with
    NO LLM. '' when it contains no actions to turn into steps.

    Handles either input retrieval can hand us: an 8-field table (the current resolution
    format, so the common case) or a free-prose comment. From a table, 'Solution applied'
    becomes the steps, 'Cause' the Root Cause line, and 'Customer action' the closing
    step — it is the follow-up the engineer still has to do, so dropping it would lose
    part of the fix.

    `fields` are the already-parsed rows (from `table_of`); pass them to skip re-parsing
    and to use the values ingest stored before `comment_body` was truncated."""
    table = fields if fields is not None else parse_table(body)
    if table:
        steps  = _steps(clean_value(table.get("Solution applied", "")))
        follow = clean_value(table.get("Customer action", ""))
        if follow and not _NO_ACTION.match(follow):
            steps += _steps(follow)
        return render_fix(clean_value(table.get("Cause", "")), steps)
    return render_fix(_cause_of(body), _steps(body))


def from_fix_answer(description: str, fix_block: str, extra: dict | None = None) -> str:
    """The prefilled resolution table for a workaround being suggested as `=== FIX ===`
    steps.

    Both artifacts come out of ONE synthesis so they cannot disagree: the block's Root
    Cause becomes 'Cause', its numbered steps join into 'Solution applied', and `extra`
    carries the 'System modified' / 'Customer action' values the model returned alongside
    the block. Anything absent is left NA rather than invented, and identifiers are
    scrubbed even though the prompt forbids them — this text is about to be posted on
    someone else's ticket."""
    gen  = {k: v for k, v in (extra or {}).items() if k in LLM_FIELDS}
    text = _FIX_MARKERS.sub("\n", fix_block or "")

    if not clean_value(gen.get("Cause", "")):
        gen["Cause"] = _cause_of(text)
    if not clean_value(gen.get("Solution applied", "")):
        gen["Solution applied"] = "; ".join(_steps(text))
    for f in ("Cause", "Solution applied", "Customer action"):
        gen[f] = scrub_identifiers(clean_value(gen.get(f, "")))[:500]
    if not clean_value(gen.get("System modified", "")):
        gen["System modified"] = infer_system(gen["Solution applied"])

    return render({**ticket_fields(description), **gen})


def blank_template(description: str) -> str:
    """The empty resolution table with only the identifier rows filled in.

    Posted when nothing in the knowledge base matches the failure: there is no workaround
    to pre-fill, but the assignee still has to close the ticket in this format, and BAN
    CAN / MSISDN are rows we can supply from the ticket itself instead of making someone
    copy them by hand. Every other row is left blank for them to fill."""
    t = ticket_fields(description)
    return render({"BAN CAN": t["BAN CAN"], "MSISDN": t["MSISDN"]}, fill_missing="")
