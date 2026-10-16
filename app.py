"""Canonical FalloutAssist entry point: the LangChain Jira application."""

from pathlib import Path
import sys

_LANGCHAIN_PROJECT = Path(__file__).resolve().parent / "langchain-falloutassist"
sys.path.insert(0, str(_LANGCHAIN_PROJECT))

from falloutassist_next.web import app  # noqa: E402
