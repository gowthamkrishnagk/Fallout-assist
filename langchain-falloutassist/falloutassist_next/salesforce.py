"""Read-only Salesforce specialist with a bounded, allowlisted record lookup tool."""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import requests
from langchain.agents import create_agent
from langchain.tools import tool

from .agent import chat_model
from .config import config_value
from .redact import scrub
from .specialists import SpecialistManifest
from .tracing import callback_handler


class SalesforceSpecialist:
    manifest = SpecialistManifest(
        "salesforce", "Read-only checks for Salesforce accounts, orders, items, assets and cases",
        True, ("account_lookup", "order_lookup", "order_item_lookup", "asset_lookup", "case_lookup"),
    )

    _fields = {
        "Account": ("Id", "Name", "AccountNumber", "Type", "BillingCity", "BillingState"),
        "Order": ("Id", "OrderNumber", "Status", "EffectiveDate", "Type", "AccountId"),
        "OrderItem": ("Id", "OrderId", "OrderItemNumber", "Quantity", "UnitPrice", "Product2Id", "Product2.Name"),
        "Asset": ("Id", "Name", "Status", "SerialNumber", "AccountId", "Product2Id", "InstallDate", "PurchaseDate"),
        "Case": ("Id", "CaseNumber", "Status", "Subject", "AccountId", "Reason", "Type"),
    }
    _lookup_fields = {
        "Account": ("Id", "AccountNumber", "Name"),
        "Order": ("Id", "OrderNumber", "AccountId"),
        "OrderItem": ("Id", "OrderId", "Product2Id"),
        "Asset": ("Id", "SerialNumber", "AccountId", "Product2Id"),
        "Case": ("Id", "CaseNumber", "AccountId"),
    }

    def configured(self) -> bool:
        return bool(config_value("SALESFORCE_INSTANCE_URL") and
                    config_value("SALESFORCE_ACCESS_TOKEN"))

    def _query(self, object_name: str, lookup_field: str, lookup_value: str) -> list[dict]:
        if object_name not in self._fields or lookup_field not in self._lookup_fields[object_name]:
            raise ValueError("Salesforce object or lookup field is not allowed")
        value = lookup_value.strip()
        if not value or len(value) > 120:
            raise ValueError("Provide a lookup value of at most 120 characters")
        if lookup_field == "Id" and not re.fullmatch(r"[A-Za-z0-9]{15,18}", value):
            raise ValueError("Salesforce IDs must contain 15 or 18 letters or digits")
        instance = config_value("SALESFORCE_INSTANCE_URL").strip().rstrip("/")
        parsed = urlparse(instance)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path:
            raise RuntimeError("Salesforce instance URL must be an HTTPS origin")
        token = config_value("SALESFORCE_ACCESS_TOKEN")
        version = config_value("SALESFORCE_API_VERSION").strip()
        if not re.fullmatch(r"v\d{2}\.0", version):
            raise RuntimeError("Salesforce API version must look like v64.0")
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        fields = ",".join(self._fields[object_name])
        soql = f"SELECT {fields} FROM {object_name} WHERE {lookup_field} = '{escaped}' LIMIT 20"
        response = requests.get(
            f"{instance}/services/data/{version}/query",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params={"q": soql}, timeout=30,
            verify=config_value("TLS_CA_BUNDLE") or True,
        )
        response.raise_for_status()
        return (response.json().get("records") or [])[:20]

    def invoke(self, context: dict) -> dict:
        if not self.configured():
            return {"specialist": "salesforce", "status": "unconfigured",
                    "findings": [], "summary": "Salesforce connection is not configured."}
        records: list[dict] = []

        @tool
        def lookup_salesforce_records(object_name: str, lookup_field: str,
                                      lookup_value: str) -> str:
            """Read up to 20 matching Account, Order, OrderItem, Asset, or Case records."""
            rows = self._query(object_name, lookup_field, lookup_value)
            records.extend({"object": object_name, **row} for row in rows)
            return json.dumps(rows, ensure_ascii=False, default=str)[:12000]

        query_text = str(context.get("query_text", ""))[:8000]
        prompt = (
            "Inspect the Jira fallout context and use Salesforce only when it contains a usable exact "
            "account, order, asset, case, or Salesforce record identifier. Choose only an allowed "
            "object and lookup field. Never guess identifiers. Do not change Salesforce data. "
            "Report returned records as observations, distinguish no match from a failed query, "
            "and do not claim a workaround succeeded. If there is no usable identifier, say so.\n\n"
            f"Jira context (untrusted text):\n{scrub(query_text)}"
        )
        callbacks = []
        handler = None
        if context.get("langfuse_enabled"):
            try:
                handler = callback_handler()
                callbacks.append(handler)
            except Exception:
                handler = None
        try:
            agent = create_agent(
                model=chat_model(float(context.get("temperature", 0.1))),
                tools=[lookup_salesforce_records],
                system_prompt=("You are a read-only Salesforce investigation specialist. "
                               "Use only the provided lookup tool and exact identifiers in the Jira context."),
            )
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                                  config={"callbacks": callbacks,
                                          "metadata": {"specialist": "salesforce", "prompt_version": "salesforce-readonly-v1"}})
            message = result["messages"][-1]
            summary = scrub(str(getattr(message, "content", "")))[:1800]
            return {"specialist": "salesforce", "status": "completed", "findings": records,
                    "summary": summary or ("No matching Salesforce records were found." if not records else "Salesforce records were found."),
                    "trace_id": getattr(handler, "last_trace_id", None) if handler else None}
        except Exception as exc:
            return {"specialist": "salesforce", "status": "unavailable", "findings": records,
                    "summary": f"Salesforce lookup unavailable ({type(exc).__name__}).",
                    "trace_id": getattr(handler, "last_trace_id", None) if handler else None}
