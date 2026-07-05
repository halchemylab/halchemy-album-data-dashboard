from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AgentTraceStep:
    phase: str
    detail: str


@dataclass(frozen=True)
class AgentAnswer:
    question: str
    summary: str
    detail: pd.DataFrame
    skill: str
    mode: str = "deterministic"
    trace: tuple[AgentTraceStep, ...] = ()
    dashboard_action: dict[str, object] | None = None


@dataclass(frozen=True)
class ProactivePrompt:
    key: str
    message: str
    actions: tuple[str, ...]
    reason: str = ""
    category: str = "general"


AgentContext = dict[str, Any]
AgentMemory = dict[str, Any]
