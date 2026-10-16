"""Create Langfuse callbacks from the administrator's saved configuration."""

from __future__ import annotations

from .config import config_value


def callback_handler():
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    public = config_value("LANGFUSE_PUBLIC_KEY")
    secret = config_value("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        raise RuntimeError("Langfuse keys are incomplete")
    Langfuse(public_key=public, secret_key=secret,
             base_url=config_value("LANGFUSE_BASE_URL") or None)
    return CallbackHandler(public_key=public)
