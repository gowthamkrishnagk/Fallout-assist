"""Standalone Jira-only application configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DATABASE_PATH = DATA_DIR / "falloutassist.sqlite3"
VECTOR_DIR = DATA_DIR / "vectors"
load_dotenv(PROJECT_DIR / ".env", override=False)


def database_backend() -> str:
    backend = os.getenv("DATABASE_BACKEND", "postgres").strip().lower()
    if backend not in {"postgres", "sqlite"}:
        raise ValueError("DATABASE_BACKEND must be postgres or sqlite")
    return backend

DEFAULTS = {
    "retrieval_top_k": 20,
    "rerank_pool_k": 20,
    "jira_review_top_k": 5,
    "temperature": 0.1,
    "rerank_temperature": 0.0,
    "llm_rerank_enabled": True,
    "retrieval_weight": 0.25,
    "llm_rerank_weight": 0.50,
    "human_feedback_weight": 0.25,
    "min_match_score": 0.70,
    "publish_enabled": False,
    "dry_run": True,
    "ingest_interval_minutes": 60,
    "scan_interval_minutes": 5,
    "langfuse_enabled": False,
    "public_base_url": "http://localhost:8010",
    "allow_mass_prune": False,
}

# Operational connections are editable by an administrator. The database and
# session key remain bootstrap settings because both are needed before login.
CONNECTION_DEFAULTS = {
    "JIRA_BASE_URL": "", "JIRA_EMAIL": "", "JIRA_API_TOKEN": "",
    "JIRA_HISTORY_BACKEND": "rovo", "JIRA_INGEST_JQL": "",
    "JIRA_INFLOW_JQL": "", "JIRA_FEEDBACK_FIELD_ID": "",
    "JIRA_SUGGESTED_WORKAROUND_FIELD_ID": "",
    "ROVO_CLOUD_ID": "", "ROVO_EMAIL": "", "ROVO_API_TOKEN": "",
    "SALESFORCE_INSTANCE_URL": "", "SALESFORCE_ACCESS_TOKEN": "",
    "SALESFORCE_API_VERSION": "v64.0",
    "AZURE_OPENAI_ENDPOINT": "", "AZURE_OPENAI_API_KEY": "",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "FEEDBACK_SECRET": "", "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "", "LANGFUSE_BASE_URL": "",
    "TLS_CA_BUNDLE": "",
}
SECRET_KEYS = {"JIRA_API_TOKEN", "ROVO_API_TOKEN", "AZURE_OPENAI_API_KEY",
               "FEEDBACK_SECRET", "LANGFUSE_SECRET_KEY", "SALESFORCE_ACCESS_TOKEN"}


def config_value(name: str) -> str:
    if name not in CONNECTION_DEFAULTS:
        raise KeyError(name)
    from . import storage
    stored = storage.connection_value(name)
    return stored if stored is not None else os.getenv(name, CONNECTION_DEFAULTS[name]).strip()


def required_env(name: str) -> str:
    value = config_value(name) if name in CONNECTION_DEFAULTS else os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def feedback_secret() -> str:
    return config_value("FEEDBACK_SECRET")


def session_secret() -> str:
    return required_env("SESSION_SECRET")


def validated_settings(raw: dict) -> dict:
    result = {**DEFAULTS, **raw}
    for key, minimum, maximum in (
        ("retrieval_top_k", 1, 100), ("rerank_pool_k", 1, 50),
        ("jira_review_top_k", 1, 20),
        ("ingest_interval_minutes", 0, 1440), ("scan_interval_minutes", 0, 1440),
    ):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be an integer from {minimum} to {maximum}")
    if result["jira_review_top_k"] > result["retrieval_top_k"]:
        raise ValueError("jira_review_top_k cannot exceed retrieval_top_k")
    if result["rerank_pool_k"] > result["retrieval_top_k"]:
        raise ValueError("rerank_pool_k cannot exceed retrieval_top_k")
    if result["jira_review_top_k"] > result["rerank_pool_k"]:
        raise ValueError("jira_review_top_k cannot exceed rerank_pool_k")
    for key in ("temperature", "rerank_temperature", "min_match_score",
                "retrieval_weight", "llm_rerank_weight", "human_feedback_weight"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if abs(sum(result[key] for key in ("retrieval_weight", "llm_rerank_weight",
                                       "human_feedback_weight")) - 1.0) > 0.001:
        raise ValueError("Rerank weights must sum to 1")
    for key in ("publish_enabled", "dry_run", "langfuse_enabled", "llm_rerank_enabled", "allow_mass_prune"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be true or false")
    url = result["public_base_url"]
    if not isinstance(url, str) or not url.startswith(("http://", "https://")) or len(url) > 300:
        raise ValueError("public_base_url must be an http(s) URL")
    return {key: result[key] for key in DEFAULTS}
