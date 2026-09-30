from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TransitionWindow:
    min_seconds: float = 0.0
    max_seconds: float = float("inf")

    def allows(self, gap_seconds: float) -> bool:
        return self.min_seconds <= gap_seconds <= self.max_seconds


class CameraTopology:
    def __init__(
        self,
        transitions: dict[str, dict[str, TransitionWindow]] | None = None,
        strict: bool = False,
    ) -> None:
        self.transitions = transitions or {}
        self.strict = strict

    @classmethod
    def from_yaml(cls, path: str | Path, strict: bool = False) -> "CameraTopology":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        transitions: dict[str, dict[str, TransitionWindow]] = {}
        for source, destinations in (raw.get("transitions") or {}).items():
            transitions[source] = {}
            for destination, window in (destinations or {}).items():
                window = window or {}
                transitions[source][destination] = TransitionWindow(
                    min_seconds=float(window.get("min_seconds", 0.0)),
                    max_seconds=float(window.get("max_seconds", float("inf"))),
                )
        return cls(transitions=transitions, strict=strict)

    def allows(self, source: str, destination: str, gap_seconds: float) -> bool:
        if source == destination:
            return gap_seconds >= 0
        window = self.transitions.get(source, {}).get(destination)
        if window is None:
            return not self.strict
        return window.allows(gap_seconds)
