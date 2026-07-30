"""Proactive Jira auto-suggest bot.

Polls inflow (open) Order Fallout tickets and leaves ONE comment on each: the WORKAROUND —
`=== FIX ===` numbered steps for the engineer to act on — carrying every link it needs:

  👍 "good fix"   → confirm (boost this workaround for the failure), refresh the counts.
  👎 "improve it" → demote/train the shown fix, find the next-best PROPER match, and
                    UPDATE the same comment with it (until accepted or exhausted).
  "Matched tickets" → the past tickets the steps were built from, so the fix can be read
                    at source instead of trusted as a paraphrase.
  "Details"       → the app page listing WHO gave feedback on this workaround. Counts live
                    in the comment; names live in the app, because a comment sits on a
                    customer's ticket.
  "Copy the resolution format" → the app page serving the team's 8-field table, pre-filled
                    from this suggestion and copyable in one click.

That last link REPLACED a second comment. The bot used to post the format reminder as its
own @-mentioning comment; the table now waits behind a click instead of being pushed onto
every ticket. The ask itself still rides in the comment, because the format is what keeps
the corpus parseable — today's well-formatted resolution is what makes tomorrow's match
possible. TEMPLATE_MARKER and _compose_resolution_reminder are retained for tickets that
already carry that older comment: no new one is ever created, but an existing one is kept
in step with the suggestion rather than abandoned mid-thread.

Guarantees (per the product requirements):
  - Accurate / no spam: exactly one comment per ticket, idempotent via per-ticket state
    plus a hidden marker. Never re-posts on a poll.
  - No confident match → say so plainly, and still offer the format link. The ticket has
    to be closed in the team's shape either way, and that ticket's own resolution is the
    only thing that can answer this failure next time.
  - A ticket commented on before the KB knew the answer is not abandoned: later passes
    re-match it and UPGRADE the comment in place once a confident fix exists.
  - Dry-run first: when jira_suggest_dry_run is on, it only logs what it WOULD post.

Pure runtime layer — never touches the Chroma index, so no re-ingest is needed. Reuses
ingest._get_jira / _search_jql / _ticket_to_text, suggest.suggest_for_query, and
feedback.record (the same training store the in-app 👍/👎 feeds)."""

import hashlib
import hmac
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

_DIR        = Path(__file__).parent
_STATE_FILE = _DIR / "trackers" / "jira_suggestions.json"
_META_FILE  = _DIR / "trackers" / "jira_suggest_meta.json"
_lock       = threading.Lock()

MARKER          = "FA-SUGGESTION"         # workaround comment  → idempotency backup
# No longer stamped on anything new — the format reminder became a link. Kept because
# comments carrying it are already sitting on tickets, and once such a ticket closes it
# re-enters ingest_jql: without this marker its blank 8-field table would be ingested as a
# resolution, and _resolution_quality rewards the table shape (+5.0), so an empty form
# would rank as a rich one. Never remove it.
TEMPLATE_MARKER = "FA-RESOLUTION-FORMAT"   # legacy format-reminder comment

# Every marker the bot stamps on its own comments. ingest reads this to make sure a
# FalloutAssist comment is never ingested as though an engineer had written it.
COMMENT_MARKERS = (MARKER, TEMPLATE_MARKER)

# Caution baked into every auto-suggestion. This is a reference fix from a PAST similar
# ticket — the steps (esp. destructive ones like NW cleanup / cancel / delete) must be
# scoped to THIS ticket's own MSISDN/order before acting, or they hit the wrong targets.
WARNING = ("⚠️ *Verify before acting* — this is an auto-suggested fix from a past similar "
           "ticket. Confirm the exact scope (the specific MSISDN/order on THIS ticket) before "
           "applying any step, especially destructive ones like NW cleanup. Do not apply to "
           "other MSISDNs.")


# ── State ──────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict):
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _save_meta(meta: dict):
    _META_FILE.parent.mkdir(parents=True, exist_ok=True)
    _META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# Floor on the poll interval. Each pass is a cursor-paginated JQL search returning every
# queued issue WITH its comments, then per-ticket feedback harvesting — a 1-second cadence
# is ~86k heavy Jira requests a day and risks throttling, while buying nothing: a ticket
# the bot passes over is recorded `no_match` and retried on the very next pass.
MIN_INTERVAL_SECONDS = 15


def interval_seconds(wf: dict) -> int:
    """The EFFECTIVE poll interval in seconds — the single source of truth for both the
    scheduler and the UI, so the dashboard can never show a cadence the poller isn't
    using. `jira_suggest_seconds` wins when present; otherwise the legacy
    `jira_suggest_minutes`. 0 disables. Anything below MIN_INTERVAL_SECONDS is clamped."""
    raw = wf.get("jira_suggest_seconds")
    if raw is None:
        return int(wf.get("jira_suggest_minutes", 1) or 0) * 60
    secs = int(raw or 0)
    if secs <= 0:
        return 0
    return max(secs, MIN_INTERVAL_SECONDS)


def get_status(cfg: dict) -> dict:
    wf = cfg["workaround_finder"]
    try:
        meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    state = _load_state()
    secs = interval_seconds(wf)
    return {
        "enabled":         bool(wf.get("jira_suggest_enabled", False)),
        "dry_run":         bool(wf.get("jira_suggest_dry_run", True)),
        "seconds":         secs,                      # what the poller actually uses
        "min_seconds":     MIN_INTERVAL_SECONDS,
        # Kept for older clients. Reported from the EFFECTIVE interval, not the raw
        # config key, so it can't disagree with `seconds`.
        "minutes":         secs // 60,
        "jql":             wf.get("jira_suggest_jql", ""),
        "public_base_url": wf.get("public_base_url", ""),
        "tracked":         len(state),
        "last_run":        meta,
    }


# Statuses that represent a real suggestion to show in the UI list.
# llm_deferred   = matched, but synthesis is waiting on the LLM rate limit to clear.
# awaiting_match = no fix matched yet; the no-match + blank-template pair is posted and
#                  the ticket is re-checked each pass so both can be upgraded later.
_PREVIEW_STATUSES = {"dry_run", "dry_run_no_match", "posted", "improved", "exhausted",
                     "llm_deferred", "awaiting_match"}


def list_previews(cfg: dict, today_only: bool = True) -> list:
    """The current suggestions across tracked tickets — drives the UI "Pending
    suggestions" list so you don't have to grep the log.

    today_only (default): show just TODAY's inflow suggestions, not the whole tracked
    history. The state file still keeps every ticket (for idempotency / dedup), but
    the panel only lists ones acted on today, matching how inflow is reviewed daily."""
    state    = _load_state()
    jira_url = (cfg.get("jira", {}).get("url", "") or "").rstrip("/")
    today    = datetime.utcnow().date().isoformat()
    def browse(k):
        return f"{jira_url}/browse/{k}" if jira_url and k else ""
    out = []
    for key, st in state.items():
        status = st.get("status")
        if status not in _PREVIEW_STATUSES:
            continue
        if today_only and (st.get("updated", "")[:10] != today):
            continue
        matched = st.get("suggested_key") or ""
        out.append({
            "key":         key,
            "key_url":     browse(key),
            "summary":     st.get("summary", ""),
            "matched":     matched,
            "matched_url": browse(matched),
            "pct":         st.get("pct"),
            "snippet":     st.get("snippet", ""),
            "status":      status,
            "updated":     st.get("updated", ""),
        })
    out.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return out


# ── Link signing ───────────────────────────────────────────────────────────────

def _secret() -> bytes:
    """Stable HMAC key for feedback links. Prefer FEEDBACK_SECRET; otherwise derive a
    deterministic key from the Jira token so links survive restarts without extra config."""
    s = os.getenv("FEEDBACK_SECRET", "").strip()
    if s:
        return s.encode()
    tok = os.getenv("JIRA_API_TOKEN", "") or "fa-default-secret"
    return hashlib.sha256(("fa-fallback:" + tok).encode()).digest()


def sign(key: str, cand: str, action: str) -> str:
    msg = f"{key}|{cand}|{action}".encode()
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:16]


def verify(key: str, cand: str, action: str, sig: str) -> bool:
    return hmac.compare_digest(sign(key, cand, action), sig or "")


def _public_base(cfg: dict) -> str:
    return (cfg["workaround_finder"].get("public_base_url", "") or "").rstrip("/")


def _links(cfg: dict, key: str, cand: str) -> tuple[str, str]:
    base = _public_base(cfg)
    k, c = quote(key, safe=""), quote(cand, safe="")
    up   = f"{base}/api/jira-feedback?key={k}&cand={c}&action=up&sig={sign(key, cand, 'up')}"
    down = f"{base}/api/jira-feedback?key={k}&cand={c}&action=down&sig={sign(key, cand, 'down')}"
    return up, down


# Signed like the vote links, each with its OWN action, so one URL can never be replayed
# as another (a details or format link must not cast a vote). Both are read-only, but they
# disclose ticket content, so they stay behind the same HMAC rather than being guessable
# public endpoints.
DETAILS_ACTION = "details"
FORMAT_ACTION  = "format"


def _app_link(cfg: dict, path: str, key: str, cand: str, action: str) -> str:
    """A signed link into the app. '' when `public_base_url` isn't configured — the
    comment then omits the link rather than emitting a URL that goes nowhere."""
    base = _public_base(cfg)
    if not base:
        return ""
    k, c = quote(key, safe=""), quote(cand, safe="")
    return f"{base}/{path}?key={k}&cand={c}&sig={sign(key, cand, action)}"


def _details_link(cfg: dict, key: str, cand: str) -> str:
    """URL of the app page listing WHO gave feedback on this workaround."""
    return _app_link(cfg, "api/feedback-details", key, cand, DETAILS_ACTION)


def _format_link(cfg: dict, key: str, cand: str) -> str:
    """URL of the app page showing this ticket's resolution-comment format, ready to copy.

    Replaces the second Jira comment the bot used to post: the 8-field table is handed
    over on demand instead of being pushed onto every ticket."""
    return _app_link(cfg, "api/resolution-format", key, cand, FORMAT_ACTION)


# ── Comment composition (Jira wiki markup; /2/ API is plain text) ──────────────

def _sources_block(sources: list, cand: str) -> str:
    """The matched tickets as clickable Jira links, best match first — the PROVENANCE of
    the steps above, so an engineer can open the ticket the fix came from and read the
    whole thread instead of trusting a paraphrase.

    Fed from `search.select_resolutions`, i.e. the same candidates synthesis was grounded
    in, so these links can't disagree with the answer. That pick also drops pointer
    comments, status chatter and screenshot-only bodies — linking one of those would send
    someone to a ticket with no fix on it."""
    if not sources:
        return ""
    lines = []
    for c in sources:
        pct = round((c.get("score") or 0) * 100)
        if c.get("type") == "doc":
            # Uploaded docs / approved user fixes have no Jira issue to link to (search
            # builds them with a filename and no url) — named, so provenance is still
            # visible even though it isn't clickable.
            entry = f"{c.get('filename') or 'uploaded document'} _(doc)_"
            ident = c.get("filename", "")
        else:
            k, url = c.get("key", ""), c.get("url", "")
            entry  = f"[{k}|{url}]" if (k and url) else k
            ident  = k
        if not entry:
            continue
        mine = "  ← *the fix above*" if ident and ident == cand else ""
        lines.append(f"* {entry} · {pct}%{mine}")
    if not lines:
        return ""
    return "*Matched tickets*\n" + "\n".join(lines) + "\n\n"


def _votes_line(cfg: dict, key: str, cand: str, tally: dict) -> str:
    """The 👍/👎 this workaround has collected FOR THIS FAILURE, as bare counts plus a
    *Details* link into the app.

    Scoped like every other feedback read (candidate + failure signature), so it means
    "N people confirmed this fix for this error" — not how popular the ticket is.

    Counts only, deliberately: WHO voted lives in the app (`/api/feedback-details`), not
    in the comment. The comment sits on a customer's ticket, and "<name> said this fix is
    wrong" reads very differently there than on an internal page — so the names are one
    click away for whoever wants them, rather than published by default."""
    up, down = int(tally.get("up", 0)), int(tally.get("down", 0))
    if not up and not down:
        return ""        # a fresh suggestion — "0 👍 · 0 👎" is noise, not information
    parts = []
    if up:
        parts.append(f"👍 {up}")
    if down:
        parts.append(f"👎 {down}")
    line = f"_Feedback so far on this failure: {' · '.join(parts)}_"
    url  = _details_link(cfg, key, cand)
    return f"{line}  [Details|{url}]\n" if url else f"{line}\n"


def _source_refs(sources: list) -> list:
    """Just the link data from `sources` — small enough to keep in per-ticket state.

    Persisted so the comment can be re-rendered later (an up-vote refreshes the vote
    counts) without re-running retrieval: a 👍 must never be able to change which fix
    the comment shows."""
    return [{"type":     c.get("type", "ticket"),
             "key":      c.get("key", ""),
             "url":      c.get("url", ""),
             "filename": c.get("filename", ""),
             "score":    round(float(c.get("score") or 0), 3)}
            for c in (sources or [])]


def _comment_extras(cfg: dict, res: dict, cand: str, kind: str) -> tuple[list, dict]:
    """(source refs, vote tally) for a workaround comment.

    Sources are `select_resolutions` — the candidates synthesis actually used — capped by
    `comment_links`, so the links are provenance for the steps rather than a dump of
    every ticket that scored above the bar (on a common error the strong set runs well
    past 20)."""
    import search as s
    import feedback as fb
    wf    = cfg["workaround_finder"]
    limit = int(wf.get("comment_links", 5) or 0)
    srcs  = s.select_resolutions(res.get("strong") or [], limit) if limit > 0 else []
    tally = fb.vote_counts(cfg, kind, cand,
                           res.get("query_step", ""), res.get("query_error", ""))
    return _source_refs(srcs), tally


def _format_line(cfg: dict, key: str, cand: str) -> str:
    """The link to this ticket's resolution-comment format, pre-filled and copyable.

    This is what replaced the bot's second comment. The format is still the thing that
    keeps the corpus parseable — today's well-formed resolution is what makes tomorrow's
    match possible — so the ask stays in the comment; only the 8-row table moved behind a
    click, where it can be copied to the clipboard instead of read and retyped."""
    url = _format_link(cfg, key, cand)
    if not url:
        return ""
    return (f"*Closing this ticket?*  [📋 Copy the resolution format|{url}]"
            "  — paste it as your resolution comment.\n")


def _compose_comment(cfg: dict, key: str, cand: str, pct: int, answer: str,
                     sources: list | None = None, tally: dict | None = None) -> str:
    up, down = _links(cfg, key, cand)
    wf = cfg["workaround_finder"]
    # Universal feedback path for users without app access: native Jira fields on this
    # ticket. Only advertised when those fields are configured.
    fields_line = ""
    if wf.get("jira_feedback_field") or wf.get("jira_suggestion_field"):
        fields_line = ("\n_No app access?_ Set *Workaround helpful?* on this ticket — "
                       "*Yes* to confirm, or *No* and put the correct steps in "
                       "*Suggested workaround*. It's picked up automatically.\n")
    return (
        f"*💡 Suggested workaround* — matched {cand} · {pct}% confidence\n\n"
        f"{answer}\n\n"
        f"{_sources_block(sources or [], cand)}"
        f"{WARNING}\n\n"
        "----\n"
        f"*Was this helpful?*  [👍 Yes, good fix|{up}]    |    [👎 No, improve it|{down}]\n"
        f"{_votes_line(cfg, key, cand, tally or {})}"
        f"{_format_line(cfg, key, cand)}"
        f"{fields_line}\n"
        f"_🤖 Auto-suggested by FalloutAssist · {MARKER}_"
    )


def _compose_manual_review(cfg: dict, key: str) -> str:
    return (
        "*💡 Suggested workaround*\n\n"
        "No further confident workaround was found for this failure after feedback — "
        "this one needs manual review.\n\n"
        # Still offered: this is now the ONLY place the format is handed over, and a
        # manual-review ticket is precisely the one whose resolution the corpus lacks.
        f"{_format_line(cfg, key, '')}"
        f"\n_🤖 Auto-suggested by FalloutAssist · {MARKER}_"
    )


# Present in the no-match workaround comment and nowhere else, so a ticket's own comments
# are enough to tell "we found nothing (yet)" from "we posted a real fix" when the state
# file is gone.
_NO_MATCH_TAG = "no match found"


def _compose_no_match(cfg: dict, key: str) -> str:
    """The workaround comment when nothing in the KB matches this failure.

    Posted rather than staying silent so the assignee knows the tool looked and found
    nothing — an absent comment is indistinguishable from a bot that isn't running. It
    carries no feedback links: there is no candidate to vote on.

    It DOES carry the resolution-format link, and this is the case where that matters
    most: nothing in the corpus answers this failure, so the assignee's own resolution is
    the only thing that can — and only if it's written in a shape retrieval can read."""
    return (
        f"*💡 Suggested workaround* — {_NO_MATCH_TAG}\n\n"
        "No past resolved ticket shares this failure, so there's no workaround to "
        "suggest yet. This one needs manual investigation.\n\n"
        "_If a matching fix is indexed later, this comment updates itself._\n\n"
        f"{_format_line(cfg, key, '')}"
        f"\n_🤖 Auto-suggested by FalloutAssist · {MARKER}_"
    )


def _mention(issue) -> str:
    """The assignee as a Jira wiki @-mention, or '' when the ticket is unassigned.

    accountId form is the one Jira Cloud renders; the display name is only a fallback
    for a Server/DC instance that doesn't expose one."""
    try:
        a = getattr(issue.fields, "assignee", None)
    except Exception:
        return ""
    if not a:
        return ""
    acct = getattr(a, "accountId", "")
    if acct:
        return f"[~accountid:{acct}]"
    name = getattr(a, "name", "") or getattr(a, "displayName", "")
    return f"[~{name}]" if name else ""


def _compose_resolution_reminder(cfg: dict, issue, template: str,
                                 prefilled: bool) -> str:
    """LEGACY: the format-reminder comment, pre-filled.

    No longer posted on new tickets — the format is served by the "Copy the resolution
    format" link (`_format_line` → app `/api/resolution-format`) instead of a second
    comment. This is retained for tickets that already carry one: `apply_vote` keeps such
    a comment in step with the suggestion, because a draft pre-filled from a fix the user
    just rejected would otherwise have them closing the ticket with it.

    Addressed to the assignee, because they are the one who will close the ticket and
    the whole point is that the closing comment comes back in this shape. `prefilled`
    distinguishes a draft built from a real matched fix (verify and edit) from the blank
    template on a no-match ticket (fill it in as you work)."""
    who  = _mention(issue)
    hail = f"{who} " if who else ""
    if prefilled:
        intro = (
            f"{hail}when you resolve this ticket, *please post your fix as the 8-field "
            "workaround table* — that format is what lets the next person hit this "
            "failure and find your fix.\n\n"
            "Below is a *pre-filled draft* from the suggested workaround. *Verify every "
            "row against what you actually did* and correct it before posting — the "
            "bottom four rows are what get reused on other tickets."
        )
    else:
        intro = (
            f"{hail}no past fix matched this failure, so there's nothing to pre-fill — "
            "but *please still close this ticket with the 8-field workaround table*. "
            "Yours would be the first indexed resolution for this failure, which is "
            "exactly how the next occurrence gets an answer.\n\n"
            "*BAN CAN* and *MSISDN* are filled in from this ticket; complete the rest as "
            "you work it."
        )
    return (
        "*📋 Resolution comment format* — don't forget to close in this format\n\n"
        f"{intro}\n\n"
        f"{template}\n\n"
        "_Keep every row on one line, use *NA* for a row that genuinely doesn't apply, "
        "and keep account numbers / MSISDNs out of the Cause, Solution applied and "
        "Customer action rows — those rows get reused on other customers' orders._\n\n"
        f"_🤖 Format reminder by FalloutAssist · {TEMPLATE_MARKER}_"
    )


def _has_marker(issue, marker: str = MARKER) -> bool:
    try:
        comments = issue.fields.comment.comments if issue.fields.comment else []
    except Exception:
        return False
    return any(marker in (getattr(c, "body", "") or "") for c in comments)


def _recover_state(issue) -> dict:
    """Our own comment ids read back off the ticket, for when the state file is gone
    (redeploy, wiped trackers).

    Recovers the STATUS too, not just the ids: a ticket we told "no match found" is still
    upgradable, and the old marker check froze it as `posted` forever. The ticket's own
    comments are the durable record — trust them over missing state."""
    out: dict = {}
    try:
        comments = issue.fields.comment.comments if issue.fields.comment else []
    except Exception:
        return out
    for c in comments:
        body = getattr(c, "body", "") or ""
        if MARKER in body:
            out["comment_id"] = str(c.id)
            out["status"] = "awaiting_match" if _NO_MATCH_TAG in body else "posted"
        elif TEMPLATE_MARKER in body:
            out["template_comment_id"] = str(c.id)
    return out


def _put_comment(jira, key: str, comment_id: str, body: str) -> str:
    """Create the comment, or edit the one we already own in place. Returns its id.

    One comment per kind, forever — the bot amends rather than appends, which is what
    keeps an upgraded suggestion from reading as spam on a long-running ticket."""
    if comment_id:
        jira.comment(key, comment_id).update(body=body)
        return comment_id
    return str(jira.add_comment(key, body).id)


def _cand_of(top: dict) -> tuple[str, str]:
    """(identity, kind) for the candidate that produced the answer."""
    if top.get("type") == "doc":
        return top.get("filename", ""), "doc"
    return top.get("key", ""), "ticket"


def looks_like_nonfix(text: str) -> bool:
    """LLM-free guard: True when the text is an error description / cause / status
    note rather than an actual workaround. Matters most when the LLM is unavailable
    and we'd otherwise post the raw matched comment verbatim. Conservative — only the
    clear description/status templates are caught; the team's 8-field workaround table,
    a synthesized `=== FIX ===` block, or any numbered steps always pass. (The LLM's own
    NO_RELIABLE_WORKAROUND judgment catches the subtler cases.)"""
    import watable
    t = (text or "").strip().lower()
    if not t:
        return True
    if "=== fix" in t:
        return False
    # The 8-field table IS the current resolution format. Whitelisted explicitly: it has
    # neither '=== fix' nor numbered steps, and it contains a 'Cause' row, so without
    # this it could trip the fault-description heuristics below and the bot would refuse
    # to post a perfectly good workaround.
    if watable.is_table(text):
        return False
    # Numbered, ordered steps → it's an actual procedure, keep it.
    if re.search(r'(?m)^\s*\d+[.)]\s+\S', text):
        return False
    head = t[:80]
    # "Order Failed: … Cause: …" — the classic fault-description template (no steps).
    if "order failed" in t and "cause" in t:
        return True
    if "error description" in head or head.startswith("current error"):
        return True
    if "sla breached" in head:
        return True
    if head.startswith(("cause:", "reason:", "error:", "error :")):
        return True
    return False


# ── Poll + post ────────────────────────────────────────────────────────────────

# ── In-Jira feedback harvesting ────────────────────────────────────────────────
# Users without app-server access give feedback through native Jira fields on the
# ticket: a "Workaround helpful?" single-select (Yes/No) and a "Suggested workaround"
# text field. The bot reads them on its existing poll — pull-only, nothing inbound to
# the restricted subnet. A 👎 + a fix becomes a PENDING suggestion (suggestions.py),
# approved in the in-app review panel.

def _fb_fields(cfg: dict) -> tuple[str, str]:
    wf = cfg["workaround_finder"]
    return (wf.get("jira_feedback_field", "").strip(),
            wf.get("jira_suggestion_field", "").strip())


def _adf_text(node) -> str:
    """Best-effort plain text out of a Jira Cloud ADF (rich-text) value."""
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        return " ".join(_adf_text(c) for c in node.get("content", []))
    if isinstance(node, list):
        return " ".join(_adf_text(c) for c in node)
    return ""


def _field_value(issue, field_id: str) -> str:
    """A single-select custom field's chosen option as plain text ('' if unset)."""
    try:
        v = getattr(issue.fields, field_id, None)
    except Exception:
        return ""
    if v is None:
        return ""
    val = getattr(v, "value", None) or getattr(v, "name", None)
    return str(val if val is not None else v).strip()


def _field_text(issue, field_id: str) -> str:
    """A text custom field as a plain string (handles Cloud ADF dict values)."""
    try:
        v = getattr(issue.fields, field_id, None)
    except Exception:
        return ""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        return _adf_text(v).strip()
    return str(v).strip()


def _field_setter(issue, field_id: str, cfg: dict) -> tuple[str, str]:
    """(voter, voter_source) for whoever last set `field_id` on this issue.

    The truthful answer to "who gave this feedback" is the changelog author who set the
    field — NOT the assignee, and they diverge exactly when it matters (a lead or L3
    reviewer marking a suggestion wrong on someone else's ticket).

    The changelog is fetched PER TICKET, here, rather than by adding expand="changelog"
    to the inflow poll: the poll runs every `jira_suggest_seconds` (30s) over every open
    ticket, while this runs only on the rare pass where someone actually left new
    feedback. Same accuracy, none of the per-poll payload.

    Falls back to ("", "") — an honestly anonymous vote — when the changelog is
    unavailable, rather than attributing the vote to the assignee. Guessing puts a wrong
    name in the training data, which is worse than a blank one."""
    if not field_id:
        return "", ""
    import ingest as ing
    changelog = getattr(issue, "changelog", None)
    if not getattr(changelog, "histories", None):
        # Not expanded on the poll — fetch just this one issue's history.
        try:
            issue = ing._get_jira(cfg).issue(issue.key, expand="changelog")
            changelog = getattr(issue, "changelog", None)
        except Exception as e:
            print(f"[JIRA-SUGGEST] changelog fetch failed for {issue.key}: {str(e)[:100]}")
            return "", ""
    histories = getattr(changelog, "histories", None) or []

    best_epoch, best_author = -1.0, ""
    for h in histories:
        for it in getattr(h, "items", []) or []:
            # fieldId is the custom-field id on Cloud; `field` is the display NAME, which
            # is all a Server/DC changelog carries — match either.
            fid   = getattr(it, "fieldId", "") or ""
            fname = getattr(it, "field", "") or ""
            if fid != field_id and fname != field_id:
                continue
            # Parsed to epoch rather than string-compared: the raw stamps carry a
            # '+0530'-style offset, so lexical order is only chronological while the
            # offset never changes.
            ep = ing._to_epoch(str(getattr(h, "created", "") or ""))
            if ep >= best_epoch:
                author      = getattr(h, "author", None)
                best_epoch  = ep
                best_author = (getattr(author, "displayName", "") or "") if author else ""
    return (best_author, "jira_field") if best_author else ("", "")


def harvest_feedback(issue, st: dict, cfg: dict) -> tuple[dict, bool]:
    """Read the in-Jira feedback fields on `issue` → a vote (+ a pending suggestion
    when a fix is supplied). Idempotent via a snapshot stored in state. Returns
    (state_record, changed) — the caller persists it when changed."""
    wf = cfg["workaround_finder"]
    if not wf.get("suggestions_enabled", True):
        return st, False
    fb_field, sug_field = _fb_fields(cfg)
    if not fb_field and not sug_field:
        return st, False

    helpful = _field_value(issue, fb_field) if fb_field else ""
    fix     = _field_text(issue, sug_field) if sug_field else ""
    if not helpful and not fix:
        return st, False

    snap = {"helpful": helpful,
            "fix_md5": hashlib.md5(fix.encode("utf-8")).hexdigest() if fix else ""}
    if (st or {}).get("fb") == snap:
        return st, False   # this exact feedback already processed

    import feedback as fb
    import suggestions as sg
    import ingest as ing

    # Failure context + the candidate we suggested. Derive step/error from the ticket
    # (LLM-free) when we never matched/commented on it.
    step = (st or {}).get("step", "")
    error = (st or {}).get("error", "")
    if not step and not error:
        try:
            from search import _extract_fields
            step, error, _ = _extract_fields(ing._ticket_to_text(issue))
        except Exception:
            pass
    cand = (st or {}).get("suggested_key", "")
    kind = (st or {}).get("cand_kind", "ticket")

    yes = (wf.get("jira_feedback_yes", "Yes") or "Yes").strip().lower()
    no  = (wf.get("jira_feedback_no",  "No")  or "No").strip().lower()
    h   = helpful.strip().lower()

    # WHO left this feedback. Resolved once, here — this is the only feedback channel
    # that can attribute a vote at all (the app has no auth, and a signed comment link
    # is clicked by whoever happens to be reading the ticket). Looked up off the field
    # the user actually touched: the Yes/No select normally, the free-text fix field
    # when only that was filled in.
    voter, voter_src = _field_setter(issue, fb_field if helpful else sug_field, cfg)

    if h == yes:
        if cand:
            fb.record("up", kind, cand, step, error, issue.key, cfg,
                      voter=voter, voter_source=voter_src)
            print(f"[JIRA-SUGGEST] {issue.key}: field vote UP -> boosted {cand}"
                  f"{f' (by {voter})' if voter else ''}")
    elif h == no or fix:
        if cand:
            fb.record("down", kind, cand, step, error, issue.key, cfg,
                      voter=voter, voter_source=voter_src)
        if fix:
            sg.add(step, error, source_key=issue.key, disliked_key=cand,
                   suggestion=fix, cfg=cfg)
            print(f"[JIRA-SUGGEST] {issue.key}: field vote DOWN + fix -> pending suggestion"
                  f"{f' (by {voter})' if voter else ''}")

    new_st = dict(st or {})
    new_st["fb"] = snap
    # Kept for the audit trail / UI: state records who was last seen giving feedback on
    # this ticket, alongside the snapshot that makes the harvest idempotent.
    if voter:
        new_st["fb_voter"] = voter
    return new_st, True


def run_once(cfg: dict) -> dict:
    """One pass over the inflow JQL. Returns a small summary dict."""
    import ingest as ing
    import suggest

    wf        = cfg["workaround_finder"]
    jql       = wf.get("jira_suggest_jql", "").strip()
    dry       = bool(wf.get("jira_suggest_dry_run", True))
    threshold = wf.get("score_threshold", 0.7)
    if not jql:
        return {"ok": False, "error": "jira_suggest_jql not configured"}

    with _lock:
        # `assignee` is fetched so the format-reminder comment can @-mention the person
        # who will actually close the ticket.
        fields = "summary,description,comment,status,updated,assignee"
        for fid in _fb_fields(cfg):
            if fid:
                fields += f",{fid}"
        try:
            jira   = ing._get_jira(cfg)
            issues = ing._search_jql(jira, jql, fields)
        except Exception as e:
            print(f"[JIRA-SUGGEST] search failed: {str(e)[:160]}")
            return {"ok": False, "error": str(e)[:200]}

        state = _load_state()
        posted = unmatched = skipped = 0
        would  = deferred = 0

        for issue in issues:
            key = issue.key
            st  = state.get(key)

            # Harvest any in-Jira feedback the user left (runs BEFORE the idempotency
            # skips below — feedback lands precisely on tickets we've already commented
            # on, which those skips would otherwise short-circuit past).
            try:
                st, changed = harvest_feedback(issue, st, cfg)
                if changed:
                    state[key] = st
            except Exception as e:
                print(f"[JIRA-SUGGEST] harvest failed for {key}: {str(e)[:120]}")

            # The harvest snapshot must survive every state rewrite below. It is the only
            # record of which in-Jira feedback has been processed, so dropping it makes
            # the next pass count the same 👍/👎 a second time.
            carry = {"fb": st["fb"]} if (st or {}).get("fb") else {}

            # Backup idempotency: our comments are on the ticket but state was lost
            # (redeploy / wiped trackers). Recover the ids AND the status from them.
            if not (st or {}).get("comment_id") and _has_marker(issue):
                st = {**(st or {}), "key": key, **_recover_state(issue),
                      "updated": _now()}
                state[key] = st

            # Idempotency: we already own comments here. Only `awaiting_match` is worth
            # revisiting — that ticket was told "no match found", and a fix may have been
            # indexed since. Everything else is settled, so skip without a search.
            if (st or {}).get("comment_id") and st.get("status") != "awaiting_match":
                skipped += 1
                continue
            # In dry-run, don't re-preview a ticket we already previewed — but only
            # once we have its preview data stored (auto-heals old state from before
            # previews were captured, so the UI list fills on the next pass).
            if dry and st and st.get("status") in ("dry_run", "dry_run_no_match") \
                    and st.get("snippet"):
                skipped += 1
                continue

            try:
                query = ing._ticket_to_text(issue)
                res   = suggest.suggest_for_query(query, cfg)
            except Exception as e:
                print(f"[JIRA-SUGGEST] suggest failed for {key}: {str(e)[:120]}")
                continue

            # LLM rate-limited / unavailable: a strong match was found but synthesis
            # failed because every provider errored (res["llm_note"] set). Don't post
            # a raw fallback or mark the ticket done — show a "check the app" note and
            # let the NEXT poll retry it automatically once the limit clears. Status
            # stays llm_deferred (not dry_run/posted), so it isn't skipped next pass.
            if res.get("llm_note"):
                state[key] = {
                    **carry,
                    "key":           key,
                    "status":        "llm_deferred",
                    "suggested_key": (res.get("top") or {}).get("key", ""),
                    "pct":           round(res.get("best_score", 0) * 100),
                    "snippet":       ("⏳ LLM is rate-limited — workaround synthesis is pending. "
                                      "Open the app and search this failure to see the workaround "
                                      "now; auto-suggest will retry automatically once the limit "
                                      "clears."),
                    "summary":       (getattr(issue.fields, "summary", "") or "")[:140],
                    "step":          res.get("query_step", ""),
                    "error":         res.get("query_error", ""),
                    "updated":       _now(),
                }
                deferred += 1
                print(f"[JIRA-SUGGEST] {key}: LLM rate-limited — deferred, will retry next pass")
                continue

            # ── Is there a real fix to suggest? ────────────────────────────────────
            # Three ways there isn't, all landing in the same place: nothing scored above
            # the bar; the LLM judged the matched sources hold no real workaround (just
            # chatter or a bare "done"); or the LLM-free guard caught an answer that is an
            # error description rather than a fix.
            matched = bool(
                res.get("mode") == "strong_match" and res.get("top")
                and res.get("best_score", 0) >= threshold
                and not res.get("declined")
                and not looks_like_nonfix(res.get("answer", ""))
            )
            if not matched:
                why = ("no reliable workaround in sources" if res.get("declined")
                       else "matched comment is a description/non-fix"
                       if res.get("top") else "no confident match")
                # Already told them so — nothing has changed, so don't touch the ticket.
                if (st or {}).get("comment_id"):
                    skipped += 1
                    continue
                unmatched += 1
                template = res.get("resolution_template") or ""
                base_rec = {**carry, "key": key, "step": res.get("query_step", ""),
                            "error": res.get("query_error", ""),
                            # Served by the "Copy the resolution format" link instead of
                            # being posted as a second comment. Blank template here —
                            # there is no matched fix to pre-fill from.
                            "template": template, "template_prefilled": False,
                            "snippet": "❔ No matching workaround yet — the assignee was "
                                       "asked to close in the team's format so this "
                                       "failure is answerable next time.",
                            "summary": (getattr(issue.fields, "summary", "") or "")[:140],
                            "rejected": [], "updated": _now()}
                if dry:
                    print(f"[JIRA-SUGGEST] (dry-run) would post no-match on {key} ({why})")
                    state[key] = {**base_rec, "status": "dry_run_no_match"}
                    would += 1
                    continue
                try:
                    # ONE comment per ticket. The format reminder is no longer posted —
                    # it's the "Copy the resolution format" link inside this comment, so
                    # the ticket carries the ask without a second notification.
                    fix_id = _put_comment(jira, key, "", _compose_no_match(cfg, key))
                    # awaiting_match, not posted: a later pass re-checks this ticket and
                    # upgrades the comment once the KB can answer it.
                    state[key] = {**base_rec, "comment_id": fix_id,
                                  "status": "awaiting_match"}
                    posted += 1
                    print(f"[JIRA-SUGGEST] {key}: {why} — posted no-match")
                except Exception as e:
                    print(f"[JIRA-SUGGEST] FAILED to comment on {key}: {str(e)[:140]}")
                continue

            # ── A confident fix: one comment, carrying the format link ─────────────
            top        = res["top"]
            cand, kind = _cand_of(top)
            pct        = round(res.get("best_score", 0) * 100)
            template   = res.get("resolution_template") or ""
            # The matched-ticket links and the running vote count shown in the comment.
            srcs, tally = _comment_extras(cfg, res, cand, kind)
            # Set when this ticket already carries our comment — it is then edited in
            # place rather than added, so an upgrade never reads as spam.
            upgrading  = bool((st or {}).get("comment_id"))
            base_rec   = {**carry, "key": key, "suggested_key": cand, "cand_kind": kind,
                          "step": res.get("query_step", ""), "error": res.get("query_error", ""),
                          # Kept so an up-vote can refresh the vote counts in place
                          # without re-running retrieval (see _source_refs).
                          "sources": srcs,
                          # The draft the "Copy the resolution format" link serves.
                          # Pre-filled from this matched fix, so what the assignee copies
                          # is the same fix the comment suggested.
                          "template": template, "template_prefilled": True,
                          # Preview data shown in the UI "Pending suggestions" list —
                          # the full suggested comment text, so it's the actual fix.
                          "pct": pct, "snippet": (res.get("answer") or "")[:4000],
                          "summary": (getattr(issue.fields, "summary", "") or "")[:140],
                          "rejected": [], "updated": _now()}

            if dry:
                verb = "would upgrade" if upgrading else "would comment"
                print(f"[JIRA-SUGGEST] (dry-run) {verb} on {key}: "
                      f"matched {cand} {pct}% — {res['answer'][:80]!r}")
                state[key] = {**base_rec, "status": "dry_run"}
                would += 1
                continue

            try:
                fix_id  = _put_comment(jira, key, (st or {}).get("comment_id", ""),
                                       _compose_comment(cfg, key, cand, pct, res["answer"],
                                                        srcs, tally))
                state[key] = {**base_rec, "comment_id": fix_id, "status": "posted",
                              # Carried forward, never dropped: a ticket commented on
                              # BEFORE this change still owns a format-reminder comment,
                              # and its id is the only way to keep that comment in step
                              # with the suggestion (see the 👎 path in apply_vote).
                              **({"template_comment_id": st["template_comment_id"]}
                                 if (st or {}).get("template_comment_id") else {})}
                posted += 1
                print(f"[JIRA-SUGGEST] {'upgraded' if upgrading else 'commented on'} "
                      f"{key} (matched {cand} {pct}%)")
            except Exception as e:
                print(f"[JIRA-SUGGEST] FAILED to comment on {key}: {str(e)[:140]}")

        _save_state(state)

    # `no_match` replaces the old `silent` count: those tickets are no longer passed over
    # in silence, they get the no-match + blank-template pair.
    summary = {"ok": True, "scanned": len(issues), "posted": posted,
               "would_post": would, "no_match": unmatched, "skipped": skipped,
               "deferred": deferred, "dry_run": dry, "last_run": _now()}
    _save_meta(summary)
    print(f"[JIRA-SUGGEST] run done — {summary}")
    return summary


# ── Feedback (called by the GET link handler) ──────────────────────────────────

def resolution_format(key: str, cfg: dict) -> dict:
    """The resolution-comment format for `key`, for the app's copy-to-clipboard page.

    Prefers the draft stored when the comment was posted, so what the assignee copies is
    the same table the suggestion was built from. Falls back to a blank template derived
    live from the ticket (state lost, or a ticket the bot never commented on) — the four
    generated rows are left empty rather than invented, exactly as watable.blank_template
    does, and BAN CAN / MSISDN still come from the ticket itself."""
    st       = _load_state().get(key) or {}
    template = (st.get("template") or "").strip()
    prefilled = bool(st.get("template_prefilled")) and bool(template)

    if not template:
        try:
            import ingest as ing
            import watable as wt
            template  = wt.blank_template(ing.fetch_ticket_text(key, cfg))
            prefilled = False
        except Exception as e:
            print(f"[JIRA-SUGGEST] format: could not build template for {key}: "
                  f"{str(e)[:100]}")
            return {"key": key, "template": "", "prefilled": False,
                    "matched": st.get("suggested_key", "")}

    return {"key":      key,
            "template": template,
            "prefilled": prefilled,
            "matched":  st.get("suggested_key", ""),
            "status":   st.get("status", "")}


def feedback_details(key: str, cand: str, cfg: dict) -> dict:
    """Who gave feedback on the workaround suggested on `key`, for the app's details page.

    Lives here rather than in app.py because the (kind, step, error) a vote is scoped to
    comes from this module's per-ticket state — the same lookup `apply_vote` does, so the
    page can't disagree with what a click would record.

    Falls back to deriving the failure from the live ticket when state was lost (redeploy /
    wiped trackers); a ticket whose signature can't be recovered returns empty rather than
    guessing, since the wrong signature would show someone else's votes."""
    import feedback as fb
    st = _load_state().get(key) or {}
    kind        = st.get("cand_kind", "ticket")
    step, error = st.get("step", ""), st.get("error", "")

    if not step and not error:
        try:
            import ingest as ing
            from search import _extract_fields
            step, error, _ = _extract_fields(ing.fetch_ticket_text(key, cfg))
        except Exception as e:
            print(f"[JIRA-SUGGEST] details: could not recover failure for {key}: "
                  f"{str(e)[:100]}")

    return {
        "key":    key,
        "cand":   cand,
        "kind":   kind,
        "step":   step,
        "error":  error,
        "status": st.get("status", ""),
        "pct":    st.get("pct"),
        "counts": fb.vote_counts(cfg, kind, cand, step, error),
        "votes":  fb.history(cfg, kind, cand, step, error),
    }

def apply_vote(key: str, cand: str, action: str, cfg: dict) -> dict:
    """Handle a 👍/👎 click from a posted comment.
      up   → record + boost, mark accepted, leave the comment.
      down → record + demote, exclude this cand, re-match, and UPDATE the comment
             with the next-best proper workaround (or a manual-review note).

    A 👎 re-drafts the format-reminder comment too: its table was pre-filled from the fix
    just rejected, so leaving it would have the assignee closing the ticket with the very
    resolution they said was wrong."""
    import ingest as ing
    import suggest
    import feedback as fb

    wf  = cfg["workaround_finder"]
    dry = bool(wf.get("jira_suggest_dry_run", True))

    with _lock:
        state = _load_state()
        st    = state.get(key)
        if not st:
            # No tracked suggestion (state lost). Still record the training signal.
            st = {"key": key, "rejected": [], "step": "", "error": ""}

        kind = st.get("cand_kind", "ticket")
        step, error = st.get("step", ""), st.get("error", "")

        if action == "up":
            fb.record("up", kind, cand, step, error, key, cfg)
            st["status"], st["updated"] = "accepted", _now()
            # Re-render the comment so the vote count reflects the click that just
            # happened. Without this the 👍 would never appear in its own tally: the
            # count is baked in at render time, and a `posted`/`accepted` ticket is
            # skipped by the idempotency check on every later poll.
            #
            # Rebuilt from STORED state (the same fix, links and pct that were posted),
            # never from a fresh search — a confirmation must not be able to change
            # which workaround the comment shows.
            # `snippet` is stored capped at 4000 chars. If it hit that cap the stored copy
            # is not the whole fix, so refreshing would silently re-post a TRUNCATED
            # workaround — a stale vote count is the lesser harm, so skip the refresh.
            snippet = st.get("snippet") or ""
            if not dry and st.get("comment_id") and 0 < len(snippet) < 4000:
                try:
                    jira = ing._get_jira(cfg)
                    jira.comment(key, st["comment_id"]).update(
                        body=_compose_comment(
                            cfg, key, cand, int(st.get("pct") or 0), snippet,
                            st.get("sources") or [],
                            fb.vote_counts(cfg, kind, cand, step, error)))
                except Exception as e:
                    # The vote is already recorded, so a failed refresh costs only the
                    # displayed count — never the training signal.
                    print(f"[JIRA-SUGGEST] vote recorded but comment refresh failed "
                          f"on {key}: {str(e)[:120]}")
            state[key] = st
            _save_state(state)
            return {"ok": True, "action": "up", "key": key, "cand": cand}

        # ── down: train + improve ──────────────────────────────────────────────
        fb.record("down", kind, cand, step, error, key, cfg)
        rejected = list(dict.fromkeys((st.get("rejected") or []) + [cand]))

        # Re-match the live ticket, skipping everything rejected so far.
        query = ""
        try:
            query = ing.fetch_ticket_text(key, cfg)
            res   = suggest.suggest_for_query(query, cfg, exclude_keys=rejected)
        except Exception as e:
            res = {"mode": "error", "best_score": 0, "top": None,
                   "answer": f"(could not re-match: {str(e)[:80]})"}

        threshold = wf.get("score_threshold", 0.7)
        template  = res.get("resolution_template") or ""
        if res.get("mode") == "strong_match" and res.get("top") \
                and res.get("best_score", 0) >= threshold:
            new_cand, new_kind = _cand_of(res["top"])
            pct  = round(res.get("best_score", 0) * 100)
            srcs, tally = _comment_extras(cfg, res, new_cand, new_kind)
            body = _compose_comment(cfg, key, new_cand, pct, res["answer"], srcs, tally)
            st.update(status="improved", suggested_key=new_cand, cand_kind=new_kind,
                      sources=srcs, pct=pct, snippet=(res.get("answer") or "")[:4000])
            improved, answer = True, res["answer"]
            prefilled = True
        else:
            new_cand = None
            body     = _compose_manual_review(cfg, key)
            st.update(status="exhausted", suggested_key=None)
            improved, answer = False, "No further confident workaround — needs manual review."
            # Nothing left to suggest, so the format falls back to the blank template: the
            # ticket still has to be closed in the team's shape, and now the engineer's own
            # fix is the only thing that can fill it.
            prefilled = False
            template  = ""
        if not template:
            import watable as wt
            template = wt.blank_template(query)

        # The draft the format link serves must track the re-match: it was pre-filled from
        # the fix just rejected, so leaving it would have the assignee closing the ticket
        # with the very resolution they said was wrong.
        st["template"], st["template_prefilled"] = template, prefilled
        st["rejected"], st["updated"] = rejected, _now()

        # Edit the comment in place (still one — never spam).
        if not dry and (st.get("comment_id") or st.get("template_comment_id")):
            try:
                jira = ing._get_jira(cfg)
                if st.get("comment_id"):
                    jira.comment(key, st["comment_id"]).update(body=body)
                # Legacy only: tickets commented on before the format reminder became a
                # link still carry that second comment. It is kept in step rather than
                # abandoned mid-thread — but no NEW one is ever created.
                if st.get("template_comment_id"):
                    jira.comment(key, st["template_comment_id"]).update(
                        body=_compose_resolution_reminder(
                            cfg, jira.issue(key), template, prefilled))
            except Exception as e:
                print(f"[JIRA-SUGGEST] failed to update comment on {key}: {str(e)[:120]}")

        state[key] = st
        _save_state(state)
        return {"ok": True, "action": "down", "key": key,
                "improved": improved, "new_cand": new_cand, "answer": answer}
