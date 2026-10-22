"""LangChain Jira specialist: draft only from selected historical evidence."""

from __future__ import annotations

from typing import Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from .redact import scrub
from .config import config_value
from .execution import deadline_context, remaining
from .tracing import callback_handler
PROMPT_VERSION = "jira-draft-v3"


class DraftSupport(BaseModel):
    source_key: str = Field(description="Exact additional supporting source key")
    source_excerpt: str = Field(description="Exact excerpt supporting this same action")


class DraftStep(BaseModel):
    text: str = Field(description="A specific proposed action grounded in the source")
    source_key: str = Field(description="Exact source key from the supplied candidates")
    source_excerpt: str = Field(description="An exact excerpt from that source supporting the action")
    supporting_sources: list[DraftSupport] = Field(default_factory=list)


class DraftResponse(BaseModel):
    decline: bool = Field(description="True if the evidence does not support a usable fix")
    reason: str = Field(description="Short explanation of the evidence decision")
    steps: list[DraftStep] = Field(default_factory=list)


def _scrub(text: str) -> str:
    return scrub(text)


def chat_model(temperature: float) -> AzureChatOpenAI:
    endpoint = config_value("AZURE_OPENAI_ENDPOINT")
    key = config_value("AZURE_OPENAI_API_KEY")
    deployment = config_value("AZURE_OPENAI_CHAT_DEPLOYMENT")
    version = config_value("AZURE_OPENAI_API_VERSION")
    if not all((endpoint, key, deployment, version)):
        raise RuntimeError("Azure OpenAI chat configuration is incomplete")
    if "," in key:
        raise RuntimeError("Configure one Azure OpenAI API key for this project")
    return AzureChatOpenAI(
        azure_endpoint=endpoint, api_key=key, azure_deployment=deployment,
        api_version=version, temperature=temperature,
        max_retries=0 if deadline_context.get() is not None else 1, timeout=remaining(45),
    )


def draft_fix(query_text: str, candidates: list[dict], temperature: float,
              use_langfuse: bool) -> tuple[dict, Optional[str]]:
    """Run the Jira specialist. Tool access is limited to the selected sources."""
    if not candidates:
        return {"decline": True, "reason": "No candidates", "steps": []}, None
    allowed = {(c.get("key") or c.get("filename") or ""): c for c in candidates}

    @tool
    def read_historical_resolution(source_key: str) -> str:
        """Read a selected historical Jira resolution or approved document by exact source key."""
        candidate = allowed.get(source_key)
        if candidate is None:
            return "Source key is not in the selected candidate list."
        return _scrub(candidate.get("comment", ""))[:2000]

    descriptions = []
    for c in candidates:
        key = c.get("key") or c.get("filename") or ""
        descriptions.append(
            f"- {key}: failed step={_scrub(c.get('step', ''))[:120]}; "
            f"error={_scrub(c.get('error', ''))[:180]}; "
            f"order reason={_scrub(c.get('order_reason', ''))[:100]}; "
            f"order type={_scrub(c.get('order_type', ''))[:80]}; "
            f"release={_scrub(c.get('release', ''))[:50]}; "
            f"workaround version={(c.get('workaround_version') or {}).get('version_number', 'unversioned')}; "
            f"raw match score={float(c.get('score') or 0):.3f}; "
            f"resolution={_scrub(c.get('comment', ''))[:1000]}"
        )
    prompt = (
        "Investigate this Jira order fallout using only the selected historical sources. "
        "Read a source with the tool if you need more context. Return a structured draft. "
        "Every action must name one source key and quote an exact supporting excerpt "
        "from that source. Do not invent record IDs, field values, API operations, or "
        "steps. If no source gives a clear applicable action, decline. "
        "Preserve the source's action clauses, objects, values, negation and conditions; "
        "change only tense or articles. Do not omit preconditions or reorder dependent actions. "
        "A similar failure is a lead, not proof that the fix applies to this customer.\n\n"
        f"Current failure:\n{_bounded_query(_scrub(query_text))}\n\n"
        "Selected sources:\n" + "\n".join(descriptions)
    )
    agent = create_agent(
        model=chat_model(temperature),
        tools=[read_historical_resolution],
        response_format=DraftResponse,
        system_prompt=(
            "You are the Jira historical-resolution specialist for FalloutAssist. "
            "Use only supplied evidence. Return concise, actionable steps with exact "
            "source excerpts. Never claim that a historical match is a verified fix "
            "for the current ticket."
        ),
    )
    callbacks = []
    handler = None
    if use_langfuse:
        try:
            handler = callback_handler()
            callbacks.append(handler)
        except Exception:
            handler = None  # Tracing is optional; the application audit is not.
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                          config={"callbacks": callbacks, "recursion_limit": 8,
                                  "metadata": {"prompt_version": PROMPT_VERSION}})
    structured = result.get("structured_response")
    if isinstance(structured, DraftResponse):
        draft = structured.model_dump()
    elif isinstance(structured, dict):
        draft = DraftResponse.model_validate(structured).model_dump()
    else:
        raise RuntimeError("Jira specialist returned no structured draft")
    trace_id = getattr(handler, "last_trace_id", None) if handler else None
    return draft, trace_id


def _bounded_query(value: str, limit: int = 1800) -> str:
    if len(value) <= limit:
        return value
    head = limit // 2
    return value[:head] + "\n...[middle omitted]...\n" + value[-(limit - head):]

