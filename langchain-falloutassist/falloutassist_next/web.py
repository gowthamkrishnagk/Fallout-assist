"""Dashboard, administration, preview, and signed feedback endpoints."""

from __future__ import annotations

import html
import json
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from urllib.parse import parse_qs

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from itsdangerous import BadSignature, URLSafeSerializer
from requests import HTTPError, RequestException
from starlette.middleware.sessions import SessionMiddleware

from . import storage
from .auth import authenticate, public
from .config import PROJECT_DIR, database_backend, feedback_secret, session_secret
from .plugins import JiraPlugin, registry
from .specialist_registry import specialist_registry
from .service import ingest_once, scan_once
from .workflow import investigate


@asynccontextmanager
async def lifespan(_: FastAPI):
    storage.initialize()
    yield


app = FastAPI(title="FalloutAssist Next", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=session_secret(),
                   session_cookie="fa_next_session", same_site="lax",
                   https_only=os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true")


def _current_user(request: Request) -> dict | None:
    user_id = request.session.get("uid")
    return public(storage.user_by_id(user_id)) if user_id else None


def _csrf_token(request: Request) -> str:
    if "csrf" not in request.session:
        request.session["csrf"] = secrets.token_urlsafe(32)
    return request.session["csrf"]


def _check_csrf(request: Request, submitted: str = "") -> None:
    expected = request.session.get("csrf", "")
    received = submitted or request.headers.get("x-csrf-token", "")
    if not expected or not secrets.compare_digest(expected, received):
        raise HTTPException(403, "Invalid CSRF token")


def _admin(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(401, "Sign in required")
    if user["role"] != "admin":
        raise HTTPException(403, "Admin role required")
    return user


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


@app.get("/api/auth/me")
def me(request: Request):
    return {"user": _current_user(request), "csrf": _csrf_token(request)}


@app.post("/api/auth/login")
def login(body: dict, request: Request):
    _check_csrf(request)
    user = authenticate(str(body.get("email", "")), str(body.get("password", "")))
    if not user:
        raise HTTPException(401, "Invalid email or password")
    request.session.clear()
    request.session["uid"] = user["id"]
    token = _csrf_token(request)
    storage.audit(user["email"], "login", "dashboard", {})
    return {"user": user, "csrf": token}


@app.post("/api/auth/logout")
def logout(request: Request):
    _check_csrf(request)
    user = _current_user(request)
    if user:
        storage.audit(user["email"], "logout", "dashboard", {})
    request.session.clear()
    return {"ok": True}


@app.get("/api/dashboard")
def dashboard(_: dict = Depends(_admin)):
    investigations = storage.latest_investigations(100)
    tiers = {tier: sum(1 for item in investigations if item["tier"] == tier)
             for tier in ("high", "medium", "low")}
    last_ingest = next((event for event in storage.recent_audit(200)
                        if event["action"] == "ingest_completed"), None)
    return {"source_counts": storage.source_counts(),
            "ingest_running": storage.lease_active("jira_ingest"),
            "ingested_documents": len(storage.all_document_keys()),
            "usable_documents": storage.usable_document_count(),
            "last_ingest": ({"created_at": last_ingest["created_at"],
                             **last_ingest["data"]} if last_ingest else None),
            "database_backend": database_backend(),
            "recent_investigation_counts": tiers,
            "recent_feedback": len(storage.recent_feedback(100)),
            "settings": storage.settings(),
            "corpus_version": storage.get_state("corpus_version"),
            "plugins": registry.catalog(),
            "specialists": specialist_registry.catalog()}


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
                limit: int = 100, _: dict = Depends(_admin)):
    if not 1 <= limit <= 250 or any(len(value) > 300 for value in (order_reason, step, error)):
        raise HTTPException(400, "Invalid workaround filter")
    return {"patterns": storage.workaround_rows(order_reason, step, error, limit)}


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
        return JiraPlugin().test_jql(jql)
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


@app.post("/api/ingest")
def ingest(body: dict, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    full = body.get("full", False)
    if not isinstance(full, bool):
        raise HTTPException(400, "full must be true or false")
    return ingest_once(admin["email"], full=full)


@app.post("/api/scan")
def scan(request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    return scan_once(admin["email"])


@app.post("/api/preview")
def preview(body: dict, request: Request, admin: dict = Depends(_admin)):
    _check_csrf(request)
    return investigate(str(body.get("issue_key", "")), admin["email"], publish=False)


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
    user = _current_user(request)
    title = "Helpful" if data["vote"] == "up" else "Needs improvement"
    identity = f"{html.escape(data['issue'])} / {html.escape(data['candidate'])}"
    if not user:
        content = ("<p>Sign in to FalloutAssist, then open this link again.</p>"
                   "<p><a href='/'>Open sign in</a></p>")
    else:
        csrf = html.escape(_csrf_token(request))
        content = (f"<p>Signed in as {html.escape(user['email'])}</p>"
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
    user = _current_user(request)
    if not user:
        raise HTTPException(401, "Sign in required")
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
        user["email"], investigation["id"] if investigation else None,
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
    if not _current_user(request):
        return HTMLResponse("<p>Sign in to FalloutAssist, then open this link again.</p>"
                            "<a href='/'>Open sign in</a>", status_code=401)
    _, issue = registry.get("jira").issue(data["issue"])
    template = (f"Ticket: {issue['key']}\nFailed Step: {issue['step']}\n"
                f"Error Description: {issue['error']}\nCause: \n=== FIX ===\n1. \n"
                "System modified: \nCustomer action: \nVerification: ")
    return HTMLResponse("<!doctype html><title>Resolution format</title>"
                        f"<main><h1>Resolution format for {html.escape(issue['key'])}</h1>"
                        f"<pre>{html.escape(template)}</pre></main>",
                        headers={"Cache-Control": "no-store"})
