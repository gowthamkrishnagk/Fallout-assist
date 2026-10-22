"""Uploaded document/SOP evidence, retained in the application database."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import uuid
import zipfile
from xml.etree import ElementTree

from . import storage
from .history import HistoryIndex, field, parse_failure
from .quality import source_quality

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
CHUNK_SIZE = 6000
FORMATS = {".txt": "text/plain", ".md": "text/markdown", ".pdf": "application/pdf",
           ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


class IndexBusy(ValueError):
    pass


def extract_text(filename: str, content: bytes) -> tuple[str, str]:
    extension = "." + filename.rsplit(".", 1)[-1].lower()
    if extension not in FORMATS:
        raise ValueError("Choose a PDF, DOCX, TXT or Markdown file.")
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("Choose a non-empty file no larger than 10 MB.")
    try:
        if extension in {".txt", ".md"}:
            text = content.decode("utf-8-sig")
        elif extension == ".docx":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                info = archive.getinfo("word/document.xml")
                if info.file_size > 4 * MAX_FILE_BYTES:
                    raise ValueError("The Word document contains too much expanded text.")
                raw = archive.read(info)
            if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
                raise ValueError("This Word XML format is not supported.")
            root = ElementTree.fromstring(raw)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            parts = []
            body = root.find("w:body", ns)
            if body is None:
                raise ValueError("The Word document has no readable body.")
            for node in body:
                if node.tag.endswith("}p"):
                    parts.append("".join(node.itertext()))
                elif node.tag.endswith("}tbl"):
                    for row in node.findall("w:tr", ns):
                        cells = [" ".join("".join(p.itertext()) for p in cell.findall(".//w:p", ns))
                                 for cell in row.findall("w:tc", ns)]
                        parts.append("|" + "|".join(cells) + "|")
            text = "\n".join(parts)
        else:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("Upload an unencrypted PDF.")
            if len(reader.pages) > 100:
                raise ValueError("PDFs are limited to 100 pages.")
            parts = []
            for page in reader.pages:
                stream = page.get_contents()
                if stream is not None and len(stream.get_data()) > 20 * MAX_FILE_BYTES:
                    raise ValueError("This PDF page is too complex to extract reliably.")
                parts.append(page.extract_text() or "")
                if sum(map(len, parts)) > MAX_TEXT_CHARS:
                    raise ValueError("Extracted text must be no longer than 200,000 characters.")
            text = "\n\n".join(parts)
    except (UnicodeDecodeError, zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError("The file could not be read. Text files must use UTF-8.") from exc
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("The file could not be parsed. Upload a valid, unencrypted document.") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\x00" in text or len(text) < 20:
        raise ValueError("No usable text found. Scanned PDFs need a text layer before upload.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("Extracted text must be no longer than 200,000 characters.")
    return text, FORMATS[extension]


def document(document_id: str, include_file: bool = False) -> dict | None:
    with storage.connect() as db:
        columns = "*" if include_file else "id,title,kind,filename,media_type,text_content,context_json,byte_size,chunk_count,searchable_chunks,status,uploaded_by,created_at"
        row = db.execute(f"SELECT {columns} FROM uploaded_documents WHERE id=?", (document_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["context"] = json.loads(result.pop("context_json"))
    return result


def list_documents() -> list[dict]:
    with storage.connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT id,title,kind,filename,byte_size,chunk_count,searchable_chunks,status,uploaded_by,created_at "
            "FROM uploaded_documents ORDER BY created_at DESC")]


def _chunks(text: str) -> list[str]:
    chunks = []
    while text:
        boundary = min(len(text), CHUNK_SIZE)
        if boundary < len(text):
            split = text.rfind("\n", CHUNK_SIZE // 2, boundary)
            if split < 0:
                split = text.rfind(" ", CHUNK_SIZE // 2, boundary)
            if split >= 0:
                boundary = split
        chunks.append(text[:boundary].strip())
        text = text[boundary:].lstrip()
    return chunks


def _source_items(item: dict) -> list[dict]:
    chunks = _chunks(item["text_content"])
    result = []
    for number, body in enumerate(chunks, 1):
        metadata = {**item["context"], "document_id": item["id"], "filename": item["filename"],
                    "kind": item["kind"], "chunk_number": number, "chunk_count": len(chunks)}
        result.append({"source_key": f"DOC:{item['id']}:{number}", "source_type": "doc",
                       "issue_updated": item["created_at"], "summary": item["title"],
                       "step": metadata.get("step", ""), "error": metadata.get("error", ""),
                       "body": body, "url": f"/api/documents/{item['id']}/file",
                       "order_reason": metadata.get("order_reason", ""), "metadata": metadata,
                       "content_hash": hashlib.sha256(body.encode()).hexdigest()})
    return result


def _index(item: dict, actor: str, index: HistoryIndex) -> dict:
    items = _source_items(item)
    keys = [source["source_key"] for source in items]
    usable = []
    assessed = []
    for source in items:
        meta = {**source["metadata"], "step": source["step"], "error": source["error"]}
        tier, reasons = source_quality("doc", meta, source["body"])
        assessed.append((source, tier, reasons))
        if tier != "low":
            usable.append(source)
    try:
        if usable:
            index.upsert_many(usable)
        for source, tier, reasons in assessed:
            storage.upsert_document(source, tier, reasons)
        with storage.connect() as db:
            db.execute("UPDATE uploaded_documents SET status='ready',chunk_count=?,searchable_chunks=? WHERE id=?",
                       (len(items), len(usable), item["id"]))
        storage.set_state("corpus_version", storage.utc_now())
        storage.audit(actor, "document_indexed", item["id"], {"chunks": len(items), "searchable": len(usable)})
    except Exception as exc:
        for key in keys:
            storage.remove_document(key)
        try:
            index.delete_many(keys)
        except Exception:
            pass  # Orphan vectors cannot be retrieved without their source rows.
        with storage.connect() as db:
            db.execute("UPDATE uploaded_documents SET status='failed',chunk_count=0,searchable_chunks=0 WHERE id=?", (item["id"],))
        storage.audit(actor, "document_index_failed", item["id"], {"error_type": type(exc).__name__})
    return document(item["id"])


def add_document(filename: str, content: bytes, title: str, kind: str,
                 context: dict, actor: str) -> dict:
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if len(filename) > 200 or not filename or any(ord(char) < 32 for char in filename):
        raise ValueError("Use a filename with 1–200 printable characters.")
    if kind not in {"document", "sop"}:
        raise ValueError("Choose Document or SOP.")
    title = (title or filename).strip()
    if not title or len(title) > 200 or any(not isinstance(value, str) or len(value) > 500 for value in context.values()):
        raise ValueError("Title is limited to 200 characters; context fields to 500.")
    text, media_type = extract_text(filename, content)
    step, error = parse_failure(text)
    context = {"step": context.get("step") or step, "error": context.get("error") or error,
               "order_type": context.get("order_type") or field(text, "Order Type", "Order Type Failed"),
               "order_reason": context.get("order_reason") or field(text, "Order Reason"),
               "release": context.get("release") or field(text, "Release")}
    holder = uuid.uuid4().hex
    if not storage.claim_lease("jira_ingest", holder, 7200):
        raise IndexBusy("A history update is running. Try the upload again when it finishes.")
    try:
        identifier = uuid.uuid4().hex
        with storage.connect() as db:
            db.execute("INSERT INTO uploaded_documents "
                       "(id,title,kind,filename,media_type,file_base64,text_content,context_json,byte_size,status,uploaded_by,created_at) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (identifier, title, kind, filename, media_type, base64.b64encode(content).decode(),
                        text, storage._json(context), len(content), "indexing", actor, storage.utc_now()))
        storage.audit(actor, "document_uploaded", identifier, {"kind": kind, "filename": filename})
        return _index(document(identifier), actor, HistoryIndex())
    finally:
        storage.release_lease("jira_ingest", holder)


def retry_index(document_id: str, actor: str) -> dict | None:
    item = document(document_id)
    if item is None:
        return None
    holder = uuid.uuid4().hex
    if not storage.claim_lease("jira_ingest", holder, 7200):
        raise IndexBusy("A history update is running. Try again when it finishes.")
    try:
        return _index(item, actor, HistoryIndex())
    finally:
        storage.release_lease("jira_ingest", holder)


def delete_document(document_id: str, actor: str) -> bool:
    item = document(document_id)
    if item is None:
        return False
    holder = uuid.uuid4().hex
    if not storage.claim_lease("jira_ingest", holder, 7200):
        raise IndexBusy("A history update is running. Try again when it finishes.")
    try:
        keys = [source["source_key"] for source in _source_items(item)]
        for key in keys:
            storage.remove_document(key)
        with storage.connect() as db:
            db.execute("DELETE FROM uploaded_documents WHERE id=?", (document_id,))
        try:
            HistoryIndex().delete_many(keys)
        except Exception as exc:
            storage.audit(actor, "document_vector_cleanup_failed", document_id, {"error_type": type(exc).__name__})
        storage.set_state("corpus_version", storage.utc_now())
        storage.audit(actor, "document_deleted", document_id, {})
        return True
    finally:
        storage.release_lease("jira_ingest", holder)


def restore_document_index(index: HistoryIndex) -> None:
    """A Jira rebuild must retain uploaded references in the shared vector index."""
    for item in list_documents():
        if item["status"] == "ready":
            rows = [source for source in _source_items(document(item["id"]))
                    if storage.source_tier("doc", source["source_key"]) != "low"]
            if rows:
                index.upsert_many(rows)
