"""Contract and registry for independently deployable system specialists."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpecialistManifest:
    name: str
    description: str
    read_only: bool
    capabilities: tuple[str, ...]


class Specialist(Protocol):
    manifest: SpecialistManifest

    def configured(self) -> bool: ...

    def invoke(self, context: dict) -> dict: ...


class SpecialistRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Specialist] = {}

    def register(self, specialist: Specialist) -> None:
        name = specialist.manifest.name
        if name in self._items:
            raise ValueError(f"Duplicate specialist: {name}")
        self._items[name] = specialist

    def get(self, name: str) -> Specialist:
        return self._items[name]

    def catalog(self) -> list[dict]:
        return [{**asdict(item.manifest), "configured": item.configured()}
                for item in self._items.values()]
