from __future__ import annotations

import os

import streamlit as st


AGENT_QUESTION_KEY = "agent_question"
AGENT_PENDING_QUESTION_KEY = "agent_pending_question"
AGENT_HISTORY_KEY = "agent_history"
AGENT_CONTEXT_KEY = "agent_context"
AGENT_PIN_CONTEXT_KEY = "agent_pin_context"
AGENT_ACTIVE_NUDGE_KEY = "agent_active_nudge"
AGENT_LAST_INTERACTION_KEY = "agent_last_interaction_at"
AGENT_IDLE_SIGNATURE_KEY = "agent_idle_signature"
AGENT_PROACTIVE_SEEN_KEY = "agent_proactive_seen"
AGENT_PROACTIVE_MUTED_KEY = "agent_proactive_muted"
AGENT_ACTION_NOTICE_KEY = "agent_action_notice"
AGENT_IDLE_SECONDS = 60


def optional_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def truthy_setting(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def assistant_debug_enabled() -> bool:
    return truthy_setting(os.getenv("SHOW_ASSISTANT_DEBUG") or optional_secret("SHOW_ASSISTANT_DEBUG"))
