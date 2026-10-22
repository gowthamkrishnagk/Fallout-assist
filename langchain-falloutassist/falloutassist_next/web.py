"""Dashboard, administration, preview, and signed feedback endpoints."""

from __future__ import annotations

import html
import logging
import base64
import json
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, quote

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, URLSafeSerializer
from requests import HTTPError, RequestException
from starlette.middleware.sessions import SessionMiddleware

from . import documents, storage, rovo_a2a, knowledge_router, api_docs
from .config import PROJECT_DIR, config_value, database_backend, feedback_secret, session_secret
from .plugins import ISSUE_KEY, JiraPlugin, registry
from .specialist_registry import specialist_registry
from .service import ingest_once, scan_once
from .workflow import investigate


@asynccontextmanager
async def lifespan(_: FastAPI):
    storage.initialize()
    knowledge_router.resume_pending()
    yield


app = FastAPI(title="FalloutAssist API", lifespan=lifespan)


def _rovo_public_base_url(request: Request) -> str:
    """Return the canonical browser origin used for the Rovo OAuth redirect.

    A reverse proxy may terminate HTTPS before forwarding to Uvicorn. In that
    deployment the explicit origin avoids constructing an HTTP redirect URI
    from the internal app listener or an untrusted forwarded header.
    """
    configured = os.getenv("ROVO_PUBLIC_BASE_URL", "").strip()
    return configured or str(request.base_url)


def _secure_session_cookie() -> bool:
    override = os.getenv("SESSION_COOKIE_SECURE", "").strip().lower()
    https_origin = os.getenv("ROVO_PUBLIC_BASE_URL", "").strip().lower().startswith("https://")
    return override == "true" or (override not in {"false", "0", "no"} and https_origin)


class _OAuthLogFilter(logging.Filter):
    def filter(self, record):
        # OAuth codes and state must not appear in Uvicorn's access logs.
        if isinstance(record.args, tuple) and len(record.args) == 5:
            args = list(record.args)
            if str(args[2]).split("?", 1)[0] == rovo_a2a.CALLBACK_PATH:
                args[2] = rovo_a2a.CALLBACK_PATH
                record.args = tuple(args)
        return True


logging.getLogger("uvicorn.access").addFilter(_OAuthLogFilter())
app.add_middleware(SessionMiddleware, secret_key=session_secret(),
                   session_cookie="fa_next_session", same_site="lax",
                   https_only=_secure_session_cookie())


_OPEN_ACCESS_USER = {"id": "open-access", "email": "open-access", "role": "admin"}


def _current_user(request: Request) -> dict:
    """Return the anonymous operator identity; this deployment has no UI login."""
    return dict(_OPEN_ACCESS_USER)


def _csrf_token(request: Request) -> str:
    if "csrf" not in request.session:
        request.session["csrf"] = secrets.token_urlsafe(32)
    return request.session["csrf"]


def _check_csrf(request: Request, submitted: str = "") -> None:
    if _api_key_authenticated(request):
        return
    expected = request.session.get("csrf", "")
    received = submitted or request.headers.get("x-csrf-token", "")
    if not expected or not secrets.compare_digest(expected, received):
        raise HTTPException(403, "Invalid CSRF token")


def _admin(request: Request) -> dict:
    """Provide the dashboard operator or a validated service identity."""
    if _api_key_authenticated(request):
        return {"id": "falloutassist-api", "email": "falloutassist_api", "role": "admin"}
    return dict(_OPEN_ACCESS_USER)


def _api_key_authenticated(request: Request) -> bool:
    """Service calls use the FalloutAssist key; browser writes retain CSRF."""
    supplied = request.headers.get("authorization", "")
    if not supplied:
        return False
    token = supplied[7:].strip() if supplied.lower().startswith("bearer ") else ""
    if token and storage.validate_falloutassist_api_key(token):
        return True
    expected = config_value("KNOWLEDGE_ROUTER_API_KEY") or ""
    if len(expected) >= 32 and token and secrets.compare_digest(token.encode(), expected.encode()):
        return True
    raise HTTPException(401, "A valid FalloutAssist API key is required",
                        headers={"WWW-Authenticate": "Bearer"})


def _knowledge_actor(request: Request) -> str:
    """Require the separately configured key for downstream-agent requests."""
    if _api_key_authenticated(request):
        return "knowledge_agent"
    raise HTTPException(401, "A valid FalloutAssist API key is required",
                        headers={"WWW-Authenticate": "Bearer"})


def _signed(token: str) -> dict:
    secret = feedback_secret()
    if not secret:
        raise HTTPException(503, "Feedback links are not configured")
    try:
        data = URLSafeSerializer(secret, salt="falloutassist-feedback-v1").loads(token)
    except BadSignature as exc:
        raise HTTPException(400, "Invalid link") from exc
    if (not isinstance(data, dict) or data.get("vote") not in ("up", "down", "format")
        or not isinstance(data.get("issue"), str)
        or not isinstance(data.get("candidate"), str)
        or ("version" in data and not isinstance(data.get("version"), str))):
        raise HTTPException(400, "Invalid link")
    return data


@app.get("/")
def home():
    return FileResponse(PROJECT_DIR / "static" / "dashboard.html")


@app.get("/api/session")
def me(request: Request):
    return {"user": _current_user(request), "csrf": _csrf_token(request)}


@app.get("/api/dashboard")
def dashboard(_: dict = Depends(_admin)):
    investigations = storage.latest_investigations(100)
    tiers = {tier: sum(1 for item in investigations if item["tier"] == tier)
             for tier in ("high", "medium", "low")}
    last_ingest = storage.latest_ingestion()
    jira_counts = storage.jira_ingestion_counts()
    return {"source_counts": storage.source_counts(),
            "ingest_running": storage.lease_active("jira_ingest"),
            "ingested_documents": len(storage.all_document_keys()),
            "usable_documents": storage.usable_document_count(),
            "last_ingest": last_ingest,
            "database_backend": database_backend(),
            "models": {"chat_deployment": config_value("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                       "embedding": config_value("EMBEDDING_MODEL")},
            "uploaded_documents": len(documents.list_documents()),
            "jira_documents": jira_counts["stored"],
            "jira_ingestion": jira_counts,
            "recent_investigation_counts": tiers,
            "recent_feedback": len(storage.recent_feedback(100)),
            "settings": storage.settings(),
            "corpus_version": storage.get_state("corpus_version"),
            "plugins": registry.catalog(),
            "specialists": specialist_registry.catalog()}


@app.get("/api/documents")
def document_list(_: dict = Depends(_admin)):
    return {"documents": documents.list_documents()}


@app.post("/api/documents")
def document_upload(request: Request, file: UploadFile = File(...), title: str = Form(""),
                    kind: str = Form("document"), step: str = Form(""), error: str = Form(""),
                    order_type: str = Form(""), order_reason: str = Form(""), release: str = Form(""),
                    admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        if file.size is not None and file.size > documents.MAX_FILE_BYTES:
            raise HTTPException(413, "Files are limited to 10 MB.")
        content = file.file.read(documents.MAX_FILE_BYTES + 1)
        return documents.add_document(file.filename or "", content, title, kind,
                                      {"step": step, "error": error, "order_type": order_type,
                                       "order_reason": order_reason, "release": release}, admin["email"])
    except documents.IndexBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        file.file.close()


@app.get("/api/documents/{document_id}")
def document_detail(document_id: str, _: dict = Depends(_admin)):
    item = documents.document(document_id)
    if item is None:
        raise HTTPException(404, "Document not found")
    return item


@app.get("/api/documents/{document_id}/file")
def document_file(document_id: str, _: dict = Depends(_admin)):
    item = documents.document(document_id, include_file=True)
    if item is None:
        raise HTTPException(404, "Document not found")
    return Response(base64.b64decode(item["file_base64"]), media_type=item["media_type"],
                    headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(item["filename"], safe=""),
                             "X-Content-Type-Options": "nosniff"})


@app.post("/api/documents/{document_id}/index")
def document_reindex(document_id: str, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        item = documents.retry_index(document_id, admin["email"])
    except documents.IndexBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    if item is None:
        raise HTTPException(404, "Document not found")
    return item


@app.delete("/api/documents/{document_id}")
def document_delete(document_id: str, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        deleted = documents.delete_document(document_id, admin["email"])
    except documents.IndexBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, "Document not found")
    return {"ok": True}


@app.get("/api/sources")
def sources(tier: str = "", limit: int = 100, offset: int = 0,
            _: dict = Depends(_admin)):
    if tier not in ("", "high", "medium", "low") or not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(400, "Invalid source filter")
    return {"rows": storage.source_rows(tier, limit, offset), "counts": storage.source_counts()}


@app.get("/api/sources/{source_key}")
def source_detail(source_key: str, _: dict = Depends(_admin)):
    item = storage.source_detail(source_key)
    if not item:
        raise HTTPException(404, "Source not found")
    return item


@app.get("/api/workarounds")
def workarounds(order_reason: str = "", step: str = "", error: str = "",
                issue_key: str = "", limit: int = 100, _: dict = Depends(_admin)):
    issue_key = issue_key.strip().upper()
    if (not 1 <= limit <= 250 or any(len(value) > 300 for value in (order_reason, step, error))
            or (issue_key and not ISSUE_KEY.fullmatch(issue_key))):
        raise HTTPException(400, "Invalid workaround filter")
    return {"patterns": storage.workaround_rows(order_reason, step, error, limit,
                                                issue_key=issue_key)}


@app.post("/api/workaround-versions/{version_id}/outcomes")
def record_workaround_outcome(version_id: str, body: dict, request: Request,
                              admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        return storage.record_workaround_outcome(
            version_id, str(body.get("source_kind", "human")),
            str(body.get("outcome", "unknown")), body.get("verified", False),
            str(body.get("release", "")), str(body.get("evidence", "")),
            str(body.get("external_reference", "")), admin["email"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/workaround-versions/{version_id}")
def workaround_version_detail(version_id: str, _: dict = Depends(_admin)):
    """Serve source comments and valid saved synthesis without starting model work."""
    version = storage.workaround_version(version_id)
    if not version:
        raise HTTPException(404, "Workaround version not found")
    from .synthesis import cached_synthesis, source_evidence
    return {"version_id": version_id, "original_comments": source_evidence(version),
            "saved_synthesis": cached_synthesis(version)}


@app.post("/api/workaround-versions/{version_id}/synthesis")
def workaround_synthesis(version_id: str, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    version = storage.workaround_version(version_id)
    if not version:
        raise HTTPException(404, "Workaround version not found")
    from .synthesis import prepare_synthesis, source_evidence
    return {**prepare_synthesis(version, admin["email"]),
            "original_comments": source_evidence(version)}


@app.get("/api/investigations")
def investigations(limit: int = 100, _: dict = Depends(_admin)):
    if not 1 <= limit <= 500:
        raise HTTPException(400, "Invalid limit")
    return storage.latest_investigations(limit)


@app.get("/api/investigations/{investigation_id}")
def investigation(investigation_id: str, _: dict = Depends(_admin)):
    item = storage.investigation(investigation_id)
    if not item:
        raise HTTPException(404, "Investigation not found")
    for source, target in (("reason_json", "reasons"), ("source_keys_json", "source_keys"),
                           ("settings_json", "settings"), ("rerank_json", "rerank")):
        item[target] = json.loads(item.pop(source))
    item["specialist_findings"] = item["settings"].get("specialist_findings", {})
    item["specialists"] = item["settings"].get("specialists", ["jira"])
    item["supervisor_summary"] = item["settings"].get("supervisor_summary", "")
    item["branch_drafts"] = item["settings"].get("branch_drafts", {})
    item["branch_agreement"] = item["settings"].get("branch_agreement", "")
    item["supervisor_result"] = item["settings"].get("supervisor_result", {})
    item["retrieval_route"] = item["settings"].get("retrieval_route", [])
    item["review_required"] = item["settings"].get("review_required", item["tier"] == "low")
    item["source_details"] = item["settings"].get("source_details", [])
    return item


@app.get("/api/feedback")
def feedback_list(limit: int = 100, _: dict = Depends(_admin)):
    if not 1 <= limit <= 500:
        raise HTTPException(400, "Invalid limit")
    return storage.recent_feedback(limit)


@app.get("/api/audit")
def audit_list(limit: int = 100, _: dict = Depends(_admin)):
    if not 1 <= limit <= 500:
        raise HTTPException(400, "Invalid limit")
    return storage.recent_audit(limit)


@app.get("/api/audit/verify")
def audit_verify(_: dict = Depends(_admin)):
    return storage.verify_audit()


@app.get("/api/settings")
def get_settings(_: dict = Depends(_admin)):
    return storage.settings()


@app.put("/api/settings")
def put_settings(body: dict, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        return storage.update_settings(body, admin["email"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/connections")
def get_connections(_: dict = Depends(_admin)):
    return storage.connection_snapshot()


@app.get("/api/rovo-a2a")
def rovo_a2a_status(_: dict = Depends(_admin)):
    return rovo_a2a.status()


@app.post("/api/rovo-a2a/connect")
def rovo_a2a_connect(request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    browser = request.session.setdefault("rovo_a2a_browser", secrets.token_urlsafe(32))
    try:
        return rovo_a2a.start(_rovo_public_base_url(request), admin["id"], browser)
    except rovo_a2a.A2AError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/rovo-a2a/callback")
def rovo_a2a_callback(request: Request, code: str = "", state: str = "", error: str = "",
                      admin: dict = Depends(_admin)):
    headers = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
    try:
        if error:
            raise rovo_a2a.A2AError("Atlassian sign-in was cancelled or denied. Start the connection again.")
        rovo_a2a.finish(state, code, admin, request.session.get("rovo_a2a_browser", ""))
    except rovo_a2a.A2AError as exc:
        return HTMLResponse("<!doctype html><title>Rovo connection</title><main><h1>Rovo connection needs attention</h1>"
                            f"<p>{html.escape(str(exc))}</p><a href='/?rovo=retry'>Return to FalloutAssist</a></main>",
                            status_code=400, headers=headers)
    return RedirectResponse("/?rovo=connected", status_code=303, headers=headers)


@app.post("/api/rovo-a2a/disconnect")
def rovo_a2a_disconnect(request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        rovo_a2a.disconnect(admin["email"])
    except rovo_a2a.A2AError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True}


@app.post("/api/rovo-a2a/preview")
def rovo_a2a_preview(body: dict, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        return rovo_a2a.resolution(body.get("issue_key", ""), body.get("source_keys", []), admin["email"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except rovo_a2a.A2ABusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except rovo_a2a.A2AError as exc:
        raise HTTPException(502, str(exc)) from exc
    except RequestException as exc:
        raise HTTPException(
            502,
            "Could not reach Jira or Atlassian while preparing the Rovo preview. "
            "Check the local network, proxy, and TLS trust settings.",
        ) from exc


@app.post("/api/knowledge/jobs", status_code=202)
def start_knowledge_job(body: dict, request: Request, actor: str = Depends(_knowledge_actor)):
    """Queue FA and Rovo suggestions for a downstream agent; this route is read-only."""
    if actor != "knowledge_agent":
        _check_csrf(request)
    issue_key = str(body.get("issue_key", "")).strip().upper()
    source_keys = body.get("source_keys", [])
    if not ISSUE_KEY.fullmatch(issue_key):
        raise HTTPException(400, "Enter a valid Jira ticket key.")
    if (not isinstance(source_keys, list) or len(source_keys) > 5
            or any(not isinstance(key, str) or not ISSUE_KEY.fullmatch(key)
                   for key in source_keys)):
        raise HTTPException(400, "Use up to five valid historical Jira ticket keys.")
    try:
        result = knowledge_router.submit(issue_key, source_keys, actor)
    except RuntimeError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc
    storage.audit(actor, "knowledge_package_requested", issue_key,
                  {"job_id": result["job_id"], "deduplicated": result["deduplicated"]})
    result["status_url"] = f"/api/knowledge/jobs/{result['job_id']}"
    return result


@app.get("/api/knowledge/jobs/{job_id}")
def get_knowledge_job(job_id: str, _: str = Depends(_knowledge_actor)):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "Knowledge job not found.")
    result = knowledge_router.get_job(job_id)
    if not result:
        raise HTTPException(404, "Knowledge job not found.")
    return result


@app.post("/api/semantic-cache/clear")
def clear_semantic_cache(request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    storage.clear_semantic_cache(admin["email"])
    if storage.settings()["semantic_cache_backend"] == "redis":
        try:
            from .redis_cache import clear_remote
            clear_remote()
        except Exception as exc:
            storage.audit(admin["email"], "semantic_cache_remote_clear_failed", "reranker",
                          {"error_type": type(exc).__name__})
            return {"ok": True, "warning": "Scores invalidated for all workers. Redis entries could not be removed and will expire by TTL."}
    return {"ok": True}


@app.post("/api/connections/test-jql")
def test_jql(body: dict, request: Request, _: dict = Depends(_admin)):
    _check_csrf(request)
    kind = str(body.get("kind", ""))
    jql = str(body.get("jql", "")).strip()
    if kind not in {"history", "inflow"}:
        raise HTTPException(400, "Choose a history or inflow query")
    if not jql or len(jql) > 4000:
        raise HTTPException(400, "Enter a JQL query up to 4,000 characters")
    try:
        return {**JiraPlugin().test_jql(jql), "jql": jql, "kind": kind}
    except HTTPError as exc:
        response = exc.response
        message = "Jira rejected the request"
        if response is not None and response.status_code == 400:
            try:
                payload = response.json()
                errors = payload.get("errorMessages") or []
                message = "; ".join(str(error) for error in errors)[:300] or "Check the JQL syntax and field names"
            except ValueError:
                message = "Check the JQL syntax and field names"
            raise HTTPException(400, f"JQL is invalid: {message}") from exc
        if response is not None and response.status_code in {401, 403}:
            raise HTTPException(400, "Jira rejected the saved credentials or search permissions") from exc
        raise HTTPException(502, "Jira could not complete the query check") from exc
    except RequestException as exc:
        raise HTTPException(502, "Could not connect to Jira. Check the Jira URL and network settings") from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/connections")
def put_connections(body: dict, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        return storage.update_connection_config(body, admin["email"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/falloutassist-api-keys")
def get_falloutassist_api_keys(_: dict = Depends(_admin)):
    """List named key metadata only; plaintext keys are never returned here."""
    return storage.falloutassist_api_keys()


@app.post("/api/falloutassist-api-keys", status_code=201)
def create_falloutassist_api_key(body: dict, request: Request,
                                 admin: dict = Depends(_admin)):
    _check_csrf(request)
    try:
        return storage.create_falloutassist_api_key(str(body.get("name", "")), admin["email"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/falloutassist-api-keys/{key_id}")
def delete_falloutassist_api_key(key_id: str, request: Request,
                                 admin: dict = Depends(_admin)):
    _check_csrf(request)
    if not re.fullmatch(r"[0-9a-f]{32}", key_id):
        raise HTTPException(404, "FalloutAssist API key not found")
    if not storage.revoke_falloutassist_api_key(key_id, admin["email"]):
        raise HTTPException(404, "FalloutAssist API key not found")
    return {"ok": True}


@app.post("/api/ingest")
def ingest(body: dict, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    full = body.get("full", False)
    if not isinstance(full, bool):
        raise HTTPException(400, "full must be true or false")
    return ingest_once(admin["email"], full=full)


@app.get("/api/ingestion/status")
def ingestion_status(_: dict = Depends(_admin)):
    """Stored Jira totals, latest completed ingest metrics and current configuration."""
    return {"running": storage.lease_active("jira_ingest"),
            "totals": storage.jira_ingestion_counts(),
            "last_completed_run": storage.latest_ingestion(),
            "configuration": {"history_jql": config_value("JIRA_INGEST_JQL"),
                              "inflow_jql": config_value("JIRA_INFLOW_JQL"),
                              "settings": storage.settings()}}


@app.post("/api/scan")
def scan(request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    return scan_once(admin["email"])


@app.post("/api/preview")
def preview(body: dict, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    return investigate(str(body.get("issue_key", "")), admin["email"], publish=False,
                       local_only=body.get("local_only") is True)


@app.get("/api/rules")
def list_rules(_: dict = Depends(_admin)):
    return storage.rules()


@app.post("/api/rules")
def create_rule(body: dict, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    pattern = str(body.get("error_pattern", "")).strip()
    text = str(body.get("workaround_text", "")).strip()
    if not pattern or not text or len(pattern) > 500 or len(text) > 8000:
        raise HTTPException(400, "Error pattern and workaround text are required")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise HTTPException(400, "Invalid error pattern") from exc
    item = {"id": uuid.uuid4().hex, "status": "draft", "error_pattern": pattern,
            "failed_step": str(body.get("failed_step", ""))[:300],
            "order_type": str(body.get("order_type", ""))[:200],
            "workaround_text": text, "created_by": admin["email"]}
    item["match"] = {"error_description": {"op": "matches_pattern", "pattern": pattern}}
    if item["failed_step"]:
        item["match"]["failed_step"] = {"op": "equals", "value": item["failed_step"]}
    if item["order_type"]:
        item["match"]["order_type"] = {"op": "equals", "value": item["order_type"]}
    storage.save_rule(item)
    return item


@app.post("/api/rules/{rule_id}/approve")
def approve_rule(rule_id: str, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    selected = next((r for r in storage.rules() if r["id"] == rule_id), None)
    if not selected:
        raise HTTPException(404, "Rule not found")
    if any(r["id"] != rule_id and r["status"] == "approved"
           and r["match"] == selected["match"] for r in storage.rules()):
        raise HTTPException(409, "An approved rule has the same match conditions")
    if not storage.set_rule_status(rule_id, "approved", admin["email"]):
        raise HTTPException(404, "Rule not found")
    storage.set_state("corpus_version", storage.utc_now())
    return {"ok": True}


@app.post("/api/rules/{rule_id}/deprecate")
def deprecate_rule(rule_id: str, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    if not storage.set_rule_status(rule_id, "deprecated", admin["email"]):
        raise HTTPException(404, "Rule not found")
    storage.set_state("corpus_version", storage.utc_now())
    return {"ok": True}


@app.get("/feedback/{token}")
def feedback_page(token: str, request: Request):
    data = _signed(token)
    if data["vote"] == "format":
        raise HTTPException(400, "Use the resolution link")
    title = "Helpful" if data["vote"] == "up" else "Needs improvement"
    identity = f"{html.escape(data['issue'])} / {html.escape(data['candidate'])}"
    csrf = html.escape(_csrf_token(request))
    content = ("<p>No FalloutAssist sign-in is required.</p>"
               f"<form method='post' action='/feedback/{html.escape(token)}'>"
               f"<input type='hidden' name='csrf' value='{csrf}'>"
               "<label>Optional correction or note<br><textarea name='note' maxlength='4000'></textarea></label>"
               f"<p><button type='submit'>Submit {title.lower()} feedback</button></p></form>")
    return HTMLResponse(f"<!doctype html><meta name='viewport' content='width=device-width'>"
                        f"<title>FalloutAssist feedback</title><main><h1>{title}</h1>"
                        f"<p>{identity}</p>{content}</main>", headers={"Cache-Control": "no-store"})


@app.post("/feedback/{token}")
async def record_feedback(token: str, request: Request):
    data = _signed(token)
    if data["vote"] not in ("up", "down"):
        raise HTTPException(400, "Invalid feedback link")
    raw = await request.body()
    if len(raw) > 5000:
        raise HTTPException(413, "Form too large")
    form = parse_qs(raw.decode("utf-8", errors="replace"))
    _check_csrf(request, (form.get("csrf") or [""])[0])
    note = (form.get("note") or [""])[0][:4000]
    publication = storage.publication(data["issue"])
    investigation = (storage.investigation(publication["investigation_id"])
                     if publication else None)
    feedback_id, created = storage.add_feedback(
        data["issue"], data["candidate"], data["vote"], note,
        "open-access", investigation["id"] if investigation else None,
        investigation["query_step"] if investigation else "",
        investigation["query_error"] if investigation else "", data.get("version", ""))
    message = "Feedback recorded." if created else "You already submitted feedback for this workaround version."
    return HTMLResponse(f"<!doctype html><title>FalloutAssist feedback</title>"
                        f"<main><h1>{html.escape(message)}</h1><p>Reference {feedback_id}</p>"
                        "<a href='/'>Dashboard</a></main>", headers={"Cache-Control": "no-store"})


@app.get("/resolution/{token}")
def resolution_template(token: str, request: Request):
    data = _signed(token)
    if data["vote"] != "format":
        raise HTTPException(400, "Invalid resolution link")
    _, issue = registry.get("jira").issue(data["issue"])
    template = (f"Ticket: {issue['key']}\nFailed Step: {issue['step']}\n"
                f"Error Description: {issue['error']}\nCause: \n=== FIX ===\n1. \n"
                "System modified: \nCustomer action: \nVerification: ")
    return HTMLResponse("<!doctype html><title>Resolution format</title>"
                        f"<main><h1>Resolution format for {html.escape(issue['key'])}</h1>"
                        f"<pre>{html.escape(template)}</pre></main>",
                        headers={"Cache-Control": "no-store"})


api_docs.configure_api_docs(app)
