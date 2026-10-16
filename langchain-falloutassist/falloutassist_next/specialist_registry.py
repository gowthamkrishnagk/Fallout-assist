"""Configured specialist agents. New specialists implement the shared contract and register here."""

from .jira_specialist import JiraSuggestionSpecialist
from .salesforce import SalesforceSpecialist
from .specialists import SpecialistRegistry

specialist_registry = SpecialistRegistry()
specialist_registry.register(JiraSuggestionSpecialist())
specialist_registry.register(SalesforceSpecialist())
