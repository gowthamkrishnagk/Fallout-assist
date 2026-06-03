"""
Ingestion pipeline — two sources:
  1. Jira resolved/closed/cancelled SAC tickets via JQL
  2. Uploaded workaround documents (PDF / Word / txt / xlsx)

Both write to the same Chroma collection via vectordb.py.
"""

import hashlib
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import embedder
import vectordb

_STATE_FILE  = Path(__file__).parent / "trackers" / "ingest_state.json"
_ingest_lock = threading.Lock()

# Structured fields to extract from ticket descriptions (ORDER_FALLOUT format)
_STRUCTURED_FIELDS = [
    "Order Reason", "Order Type", "Step", "Failed Step",
    "Error Description", "Error Code", "Sub Type", "Order Sub Type",
    "Service Type", "Action",
]


# ── Jira ingestion ────────────────────────────────────────────────────────────

def _get_jira(cfg: dict):
    from jira import JIRA
    email     = os.getenv("JIRA_EMAIL") or cfg["jira"]["email"]
    api_token = os.getenv("JIRA_API_TOKEN") or cfg["jira"]["api_token"]
    url       = cfg["jira"]["url"]
    return JIRA(server=url, basic_auth=(email, api_token))


def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _clean_for_embed(text: str) -> str:
    """Strip MSISDNs, order numbers, Salesforce IDs — keep step/error context.
    Removes noise that pollutes embeddings without adding semantic meaning."""
    # Salesforce record IDs (15-18 alphanumeric starting with a digit)
    text = re.sub(r'\b[0-9][A-Za-z0-9]{14,17}\b', '', text)
    # Long numeric sequences: MSISDNs (10+ digits), order numbers (6+ digits)
    text = re.sub(r'\b\d{6,}\b', '', text)
    # Clean up hanging labels after value removal
    text = re.sub(r'(?:Order\s+Id|Order)\s*:\s*', '', text, flags=re.IGNORECASE)
    # Normalize separators and whitespace
    text = re.sub(r'\s*-\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_description_fields(issue) -> str:
    """Extract structured fields (Order Reason, Order Type, Step, Error Description…)
    from the ticket description. Handles Jira wiki markup bold (*Field:*) and plain text."""
    desc = getattr(issue.fields, 'description', '') or ''
    extracted = []
    for field in _STRUCTURED_FIELDS:
        # Matches: *Order Reason:* value  |  Order Reason: value  |  | Order Reason | value |
        m = re.search(
            rf'(?:^|\n|\|)\s*\*?{re.escape(field)}\*?\s*[:\|]\s*\*?([^*\n\|]{{1,200}}?)\*?\s*(?:\||$|\n)',
            desc, re.IGNORECASE | re.MULTILINE
        )
        if m:
            val = m.group(1).strip().strip('*').strip()
            # Skip pure numbers (they're order/account IDs, not meaningful text)
            if val and not re.match(r'^\d+$', val):
                extracted.append(f"{field}: {val}")
    return '\n'.join(extracted)


# Authors whose comments are automated noise — never real workarounds
_BOT_AUTHORS = {"sac bot", "automation for jira"}


def _is_bot(display_name: str) -> bool:
    return display_name.lower().strip() in _BOT_AUTHORS


def _extract_error_context(issue) -> str:
    """Pull error description from two sources:
    1. PR_Error_Description in SAC BOT comments (e.g. '14081 | Plan instance already cancelled')
    2. Error Description field in the ticket description (for errors not logged by SAC BOT)
    Both are embedded into every chunk so searching by error text finds the right resolution."""
    # Source 1: SAC BOT PR_Error_Description field
    comments = issue.fields.comment.comments if issue.fields.comment else []
    for c in reversed(comments):
        if c.author.displayName.lower().strip() != "sac bot":
            continue
        if not c.body:
            continue
        for line in c.body.splitlines():
            if "PR_Error_Description" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    err = parts[1].strip()
                    err = re.sub(r'^\*+\s*', '', err)          # strip leading wiki bold *
                    err = re.sub(r'\*', '', err)               # strip remaining * markers
                    err = re.sub(r'\b[0-9][A-Za-z0-9]{14,17}\b', '', err)  # strip SF IDs
                    err = re.sub(r'\(\s*\)', '', err)          # strip empty parens
                    err = re.sub(r'\s+', ' ', err).strip()
                    if err:
                        return err

    # Source 2: Description field — handles errors SAC BOT doesn't log (e.g. Asurion errors)
    desc = getattr(issue.fields, 'description', '') or ''
    if isinstance(desc, str) and desc.strip() and not desc.strip().startswith('{'):
        for line in desc.splitlines():
            line = line.strip().strip('*').strip()
            # Look for lines that contain "Error" label or look like an error message
            m = re.search(r'Error(?:\s+Description)?\s*[:\|]\s*(.{20,300})', line, re.IGNORECASE)
            if m:
                return m.group(1).strip()

    return ""


def _get_assignee_comments(issue) -> list[dict]:
    """Return assignee's substantive comments as individual dicts.
    Bots (SAC BOT, Automation for Jira) are excluded from both primary and fallback.
    Falls back to last 3 substantive human comments if assignee left none."""
    comments      = issue.fields.comment.comments if issue.fields.comment else []
    assignee      = getattr(issue.fields, "assignee", None)
    assignee_id   = getattr(assignee, "accountId", None)
    assignee_name = getattr(assignee, "displayName", None)

    def _is_assignee(c) -> bool:
        if assignee_id and getattr(c.author, "accountId", None) == assignee_id:
            return True
        if assignee_name and getattr(c.author, "displayName", None) == assignee_name:
            return True
        return False

    # All substantive human comments (bots excluded)
    human = [
        {"author": c.author.displayName, "body": _clean(c.body), "is_assignee": _is_assignee(c)}
        for c in comments
        if c.body and len(c.body.strip()) > 40 and not _is_bot(c.author.displayName)
    ]

    assignee_only = [c for c in human if c["is_assignee"]]

    # Fallback: no assignee comments — use last 3 substantive human comments
    return assignee_only if assignee_only else human[-3:]


def ingest_jira(cfg: dict, progress_cb=None) -> dict:
    """Pull tickets, create one chunk per assignee comment. Returns {ok, indexed, skipped, error}."""
    with _ingest_lock:
        wf          = cfg["workaround_finder"]
        jql         = wf["ingest_jql"] + " ORDER BY updated DESC"
        index_path  = str(Path(__file__).parent / wf["index_path"])
        embed_model = cfg["embed"]["model"]

        try:
            jira   = _get_jira(cfg)
            issues = jira.search_issues(jql, maxResults=False,
                                        fields="summary,description,status,comment,resolution,assignee")
            total  = len(issues)
            if progress_cb:
                progress_cb(f"Fetched {total} tickets from Jira")

            ids = []; step_texts = []; error_texts = []; display_texts = []; metas = []
            skipped = 0

            for i, issue in enumerate(issues):
                assignee      = getattr(issue.fields, "assignee", None)
                assignee_name = getattr(assignee, "displayName", "") or ""
                summary       = issue.fields.summary
                url           = f"{jira.server_url}/browse/{issue.key}"
                comments      = _get_assignee_comments(issue)

                if not comments:
                    skipped += 1
                    continue

                error_ctx   = _extract_error_context(issue)
                clean_sum   = _clean_for_embed(summary)
                desc_fields = _extract_description_fields(issue)

                # Extract step name from cleaned summary for focused embedding
                step_match = re.search(r'Failed\s+Step:\s*([^\n]{5,80}?)(?:\s*$)', clean_sum, re.IGNORECASE)
                step_name  = step_match.group(1).strip() if step_match else ''

                for j, c in enumerate(comments):
                    chunk_id = f"{issue.key}_c{j}"

                    # ── Step embed: step name + desc fields (50% of score) ──
                    step_parts = [f"Failed Step: {step_name}"] if step_name else [clean_sum]
                    if desc_fields:
                        step_parts.append(desc_fields)
                    step_text = '\n'.join(step_parts)[:500]

                    # ── Error embed: error description only (50% of score) ──
                    # None if no error context — chunk won't be in error collection
                    error_text = f"Error: {error_ctx}"[:500] if error_ctx else None

                    # ── Display text: full context stored for showing to user ──
                    error_line = f"Error: {error_ctx}\n" if error_ctx else ""
                    desc_line  = f"{desc_fields}\n" if desc_fields else ""
                    display_text = (
                        f"Ticket: {clean_sum}\n"
                        f"{desc_line}"
                        f"{error_line}"
                        f"Assignee: {assignee_name}\n"
                        f"Resolution by {c['author']}:\n{c['body']}"
                    )[:2000]

                    ids.append(chunk_id)
                    step_texts.append(step_text)
                    error_texts.append(error_text)
                    display_texts.append(display_text)
                    metas.append({
                        "source":         "ticket",
                        "source_id":      issue.key,
                        "key":            issue.key,
                        "summary":        summary[:200],
                        "step":           step_name,
                        "error":          error_ctx[:200] if error_ctx else "",
                        "description":    desc_fields[:500] if desc_fields else "",
                        "status":         issue.fields.status.name,
                        "url":            url,
                        "assignee":       assignee_name,
                        "comment_author": c["author"],
                        "comment_body":   c["body"][:1000],
                        "is_assignee":    str(c["is_assignee"]),
                        "comment_index":  j,
                    })

                if progress_cb and i % 20 == 0:
                    progress_cb(f"Processing ticket {i+1}/{total}")

            if not ids:
                return {"ok": False, "error": "No comments found to index"}

            if progress_cb:
                progress_cb(f"Embedding {len(ids)} chunks — step + error separately...")

            step_embs  = embedder.embed(step_texts, embed_model)
            # Embed only non-None error texts; keep None positions as None
            error_texts_clean = [t if t else "" for t in error_texts]
            error_embs_all    = embedder.embed(error_texts_clean, embed_model)
            error_embs = [emb if error_texts[i] else None
                          for i, emb in enumerate(error_embs_all)]

            if progress_cb:
                progress_cb("Clearing old ticket chunks and storing fresh ones...")
            vectordb.delete_tickets(index_path)
            vectordb.add_tickets(ids, step_embs, error_embs, display_texts, metas, index_path)
            _save_state({"last_jira_sync": datetime.utcnow().isoformat(),
                         "jira_count": len(ids), "ticket_count": total})
            return {"ok": True, "indexed": len(ids), "skipped": skipped}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ── Document ingestion ────────────────────────────────────────────────────────

def _parse_doc(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt" or ext == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if ext in (".docx", ".doc"):
        from docx import Document
        doc   = Document(str(path))
        lines = []
        for p in doc.paragraphs:
            t = p.text.strip()
            if not t:
                continue
            # Word renders list numbers/bullets outside p.text, so they're lost on
            # extract. Re-add a marker for list-style paragraphs (unless the text
            # already carries one) so step-by-step instructions stay readable.
            style = (p.style.name or "").lower() if p.style is not None else ""
            if "list" in style and not re.match(r'^\s*([-*•‣●]|\d+[.)])', t):
                t = f"• {t}"
            lines.append(t)
        return "\n".join(lines)
    if ext in (".xlsx", ".xls"):
        from openpyxl import load_workbook
        wb   = load_workbook(str(path), data_only=True)
        rows = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                line = " | ".join(str(c) for c in row if c is not None)
                if line.strip():
                    rows.append(line)
        return "\n".join(rows)
    raise ValueError(f"Unsupported file type: {ext}")


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Chunk into ~chunk_size-word windows while PRESERVING line breaks, so a
    document's step-by-step / list structure survives into the stored text and
    renders properly (the UI uses white-space: pre-wrap)."""
    lines  = text.splitlines()
    chunks = []
    cur, cur_words = [], 0
    for line in lines:
        w = len(line.split())
        if cur_words + w > chunk_size and cur:
            chunks.append("\n".join(cur))
            # Carry the last ~overlap words (whole lines) into the next chunk.
            keep, kept = [], 0
            for l in reversed(cur):
                lw = len(l.split())
                if kept + lw > overlap:
                    break
                keep.insert(0, l)
                kept += lw
            cur, cur_words = keep, kept
        cur.append(line)
        cur_words += w
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if len(c.strip()) > 50]


def _extract_doc_step_error(text: str) -> tuple[str, str]:
    """Pull the failed-step and error lines out of a workaround document so it is
    matched on the same (step, error) basis as Jira tickets. Tolerates both
    'failed step- ...' / 'Step: ...' and 'error response- ...' / 'N | ...' forms."""
    step = error = ""
    m = re.search(r'(?:failed\s+)?step\s*[-:]\s*([^\n]{3,120})', text, re.IGNORECASE)
    if m:
        step = m.group(1).strip()
    m = re.search(r'error(?:\s+response|\s+description)?\s*[-:]\s*([^\n]{3,200})',
                  text, re.IGNORECASE)
    if m:
        error = m.group(1).strip()
    if not error:
        m = re.search(r'(\d{1,6}\s*\|[^\n]{3,150})', text)
        if m:
            error = m.group(1).strip()
    # Drop volatile identifiers so they don't dilute the semantic match.
    error = re.sub(r'\(?\bExternalId\s*=\s*[^)\n]*\)?', '', error, flags=re.IGNORECASE)
    error = re.sub(r'\b\d{6,}\b', '', error).strip(' -|')
    step  = re.sub(r'\b\d{6,}\b', '', step).strip(' -|')
    return step, error


def ingest_document(file_bytes: bytes, filename: str, cfg: dict) -> dict:
    """Parse a document, extract its failed step + error, and index it on that
    (step, error) basis — the same as tickets — so it only matches a query whose
    step/error are semantically close, not one that merely shares boilerplate.
    Re-uploading the same filename replaces the previous version."""
    wf          = cfg["workaround_finder"]
    index_path  = str(Path(__file__).parent / wf["index_path"])
    docs_path   = Path(__file__).parent / wf["docs_path"]
    embed_model = cfg["embed"]["model"]

    docs_path.mkdir(parents=True, exist_ok=True)
    dest   = docs_path / filename
    doc_id = hashlib.md5(filename.encode()).hexdigest()[:12]

    # Remove old version's vectors before indexing new ones
    existing = _load_doc_meta()
    if doc_id in existing:
        vectordb.delete_docs_by_source(doc_id, index_path)

    dest.write_bytes(file_bytes)

    try:
        text = _parse_doc(dest)
        if len(text.strip()) < 50:
            return {"ok": False, "error": "No text extracted from document"}

        step, error = _extract_doc_step_error(text)
        # A workaround doc MUST declare its failed step and error — that's what it
        # gets matched on. Reject anything missing either so it can never surface
        # as an unrelated match later.
        if not step or not error:
            dest.unlink(missing_ok=True)
            missing = " and ".join(
                lbl for lbl, val in (("a 'failed step' line", step),
                                     ("an 'error' line", error)) if not val)
            return {"ok": False,
                    "error": f"Document must contain {missing}. "
                             "Add lines like 'failed step- <name>' and "
                             "'error response- <code | message>', then re-upload."}

        # Frame step/error exactly like tickets/queries so vectors share one space.
        step_text  = f"Failed Step: {step}"
        error_text = f"Error: {error}"

        # Full text (newlines preserved) is the display body shown to the user.
        display    = text[:4000]
        meta       = {"source": "doc", "source_id": doc_id, "filename": filename,
                      "chunk": 0, "step": step, "error": error}
        step_embs  = [embedder.embed_one(step_text,  embed_model)]
        error_embs = [embedder.embed_one(error_text, embed_model)]
        vectordb.add_docs_dual([doc_id], step_embs, error_embs, [display], [meta], index_path)

        _save_doc_meta(doc_id, filename, 1)
        return {"ok": True, "doc_id": doc_id, "chunks": 1, "filename": filename,
                "step": step, "error": error}
    except Exception as e:
        dest.unlink(missing_ok=True)
        return {"ok": False, "error": str(e)}


def delete_document(doc_id: str, cfg: dict) -> dict:
    index_path = str(Path(__file__).parent / cfg["workaround_finder"]["index_path"])
    deleted    = vectordb.delete_docs_by_source(doc_id, index_path)
    docs_path  = Path(__file__).parent / cfg["workaround_finder"]["docs_path"]
    meta       = _load_doc_meta()
    entry      = meta.pop(doc_id, {})
    filename   = entry.get("filename", "")
    if filename:
        (docs_path / filename).unlink(missing_ok=True)
    _save_doc_meta_raw(meta)
    return {"ok": True, "deleted_chunks": deleted, "filename": filename}


def list_documents(cfg: dict) -> list[dict]:
    return list(_load_doc_meta().values())


# ── Jira ticket preview ───────────────────────────────────────────────────────

def _ticket_to_text(issue) -> str:
    """Build a clean search query from a fetched ticket.
    Extracts structured fields (category, step, description fields, error) —
    strips MSISDNs, order numbers, and Salesforce IDs so they don't dilute the match."""
    summary = issue.fields.summary or ''

    # Category prefix: ORDER_FALLOUT, PORT_OUT, SUSPEND, etc.
    cat_match = re.match(r'^([A-Z_]+)\s*:', summary)
    category  = cat_match.group(1).strip() if cat_match else ''

    # Step from summary: "… - Failed Step: Create Commercial Order Aria - …"
    step_match = re.search(r'Failed\s+Step:\s*([^-\n]+)', summary, re.IGNORECASE)
    step       = step_match.group(1).strip() if step_match else ''

    # Structured fields from description body
    desc_fields = _extract_description_fields(issue)

    # Error code from SAC BOT comment
    error_ctx = _extract_error_context(issue)

    parts = []
    if category:
        parts.append(category)
    if step:
        parts.append(f"Failed Step: {step}")
    if desc_fields:
        parts.append(desc_fields)
    if error_ctx:
        parts.append(f"Error: {error_ctx}")

    # Fallback: cleaned summary if we could extract nothing structured
    return '\n'.join(parts) if parts else _clean_for_embed(summary)


def preview_jql(jql: str, cfg: dict) -> dict:
    try:
        jira   = _get_jira(cfg)
        issues = jira.search_issues(jql, maxResults=False, fields="summary,status")
        return {
            "ok":    True,
            "count": len(issues),
            "sample": [{"key": i.key, "summary": i.fields.summary,
                        "status": i.fields.status.name}
                       for i in issues[:5]],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_ticket_text(ticket_id: str, cfg: dict) -> str:
    jira  = _get_jira(cfg)
    issue = jira.issue(ticket_id, fields="summary,description,comment,assignee")
    return _ticket_to_text(issue)


# ── State helpers ─────────────────────────────────────────────────────────────

def get_status(cfg: dict) -> dict:
    state     = _load_state()
    index_path = str(Path(__file__).parent / cfg["workaround_finder"]["index_path"])
    return {
        "total_chunks":    vectordb.count(index_path),
        "last_jira_sync":  state.get("last_jira_sync"),
        "jira_count":      state.get("jira_count", 0),
        "doc_count":       len(_load_doc_meta()),
    }


def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(update: dict):
    state = _load_state()
    state.update(update)
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


_DOC_META_FILE = Path(__file__).parent / "trackers" / "doc_meta.json"


def _load_doc_meta() -> dict:
    try:
        return json.loads(_DOC_META_FILE.read_text())
    except Exception:
        return {}


def _save_doc_meta(doc_id: str, filename: str, chunks: int):
    meta = _load_doc_meta()
    meta[doc_id] = {"doc_id": doc_id, "filename": filename, "chunks": chunks,
                    "indexed_at": datetime.utcnow().isoformat()}
    _save_doc_meta_raw(meta)


def _save_doc_meta_raw(meta: dict):
    _DOC_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DOC_META_FILE.write_text(json.dumps(meta, indent=2))
