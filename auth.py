"""User accounts + roles for FalloutAssist — registration, login, and the admin/user
distinction that gates the suggestion-approval flow (and other admin-only actions).

Mirrors suggestions.py / feedback.py: a single JSON file (default trackers/users.json),
lock-guarded writes, reads cached on the file's mtime. Passwords are hashed with bcrypt —
the raw password is never stored or logged.

Role assignment: an email is admin if (case-insensitively) it appears in
config.json -> auth.admin_emails; every other registered email becomes a regular user.
There is no in-app "make this user an admin" action on purpose — promoting someone means
editing config.json, the same trust boundary as every other setting in this app.
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path

import bcrypt

_DIR  = Path(__file__).parent
_lock = threading.Lock()

# Read cache: (mtime -> list), same pattern as suggestions.py / feedback.py.
_cache: dict = {"mtime": None, "records": []}

ROLES = ("admin", "user")


def _path(cfg: dict) -> Path:
    rel = cfg.get("auth", {}).get("users_path", "trackers/users.json")
    return _DIR / rel


def _load_raw(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _all(cfg: dict) -> list:
    """Every account record, cached on the file's mtime."""
    path = _path(cfg)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    if _cache["mtime"] == mtime:
        return _cache["records"]
    records = _load_raw(path)
    _cache["mtime"], _cache["records"] = mtime, records
    return records


def _save(records: list, cfg: dict):
    path = _path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    _cache["mtime"] = None   # force re-read next time


def _role_for(email: str, cfg: dict) -> str:
    admins = {e.strip().lower() for e in cfg.get("auth", {}).get("admin_emails", []) if e.strip()}
    return "admin" if email.strip().lower() in admins else "user"


def _public(user: dict) -> dict:
    """A user record with the password hash stripped — the only form that should ever
    leave this module (into a session, an API response, or a log line)."""
    return {k: v for k, v in user.items() if k != "password_hash"}


def get_by_email(email: str, cfg: dict) -> dict | None:
    email = (email or "").strip().lower()
    return next((u for u in _all(cfg) if u.get("email", "").lower() == email), None)


def get_by_id(uid: str, cfg: dict) -> dict | None:
    """Public form (no password hash) — the shape a session/dependency should hold."""
    rec = next((u for u in _all(cfg) if u.get("id") == uid), None)
    return _public(rec) if rec else None


def register(email: str, password: str, cfg: dict) -> dict:
    """Create a new account. Raises ValueError on bad input or a duplicate email.
    Returns the public (no password hash) record."""
    email = (email or "").strip().lower()
    if not email or "@" not in email or " " in email:
        raise ValueError("enter a valid email address")
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    with _lock:
        records = _load_raw(_path(cfg))
        if any(u.get("email", "").lower() == email for u in records):
            raise ValueError("an account with this email already exists")
        rec = {
            "id":            uuid.uuid4().hex,
            "email":         email,
            "password_hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
            "role":          _role_for(email, cfg),
            "created_ts":    datetime.utcnow().isoformat(),
        }
        records.append(rec)
        _save(records, cfg)
    return _public(rec)


def authenticate(email: str, password: str, cfg: dict) -> dict | None:
    """The public user record on success, else None. Never raises — a malformed hash
    (or no account at all) is just a failed login, not a server error."""
    user = get_by_email(email, cfg)
    if not user:
        return None
    try:
        ok = bcrypt.checkpw((password or "").encode(), user["password_hash"].encode())
    except Exception:
        ok = False
    return _public(user) if ok else None
