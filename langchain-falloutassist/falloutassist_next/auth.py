"""Local account authentication, separate from the old application."""

from __future__ import annotations

import uuid

import bcrypt

from . import storage


def public(user: dict | None) -> dict | None:
    if not user:
        return None
    return {"id": user["id"], "email": user["email"], "role": user["role"]}


def authenticate(email: str, password: str) -> dict | None:
    user = storage.user_by_email(email.strip().lower())
    if not user:
        return None
    try:
        if bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            return public(user)
    except (ValueError, TypeError):
        pass
    return None


def create(email: str, password: str, role: str = "admin") -> dict:
    if "@" not in email or len(password) < 12 or role not in ("admin", "user"):
        raise ValueError("Valid email, password of at least 12 characters, and role required")
    user_id = uuid.uuid4().hex
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    storage.create_user(user_id, email.strip().lower(), password_hash, role)
    return public(storage.user_by_id(user_id)) or {}
