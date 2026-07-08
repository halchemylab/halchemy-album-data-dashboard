from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from agent.models import AgentAnswer, AgentContext, AgentMemory, AgentTraceStep
from agent.runtime import answer_question, run_skill
from agent.skills import _clean_records, _context_rows, _format_context_album, _rows_trace, _scope_trace
from agent.tools import AGENT_TOOLS


def _skill_payload(answer: AgentAnswer) -> dict[str, object]:
    return {
        "skill": answer.skill,
        "summary": answer.summary,
        "rows": _clean_records(answer.detail),
    }


def context_summary(context: AgentContext | None) -> str:
    if not context:
        return "No previous agent context is active."
    lines = [
        f"Previous question: {context.get('last_question', '-')}",
        f"Previous skill: {context.get('last_skill', '-')}",
        f"Previous summary: {context.get('last_summary', '-')}",
    ]
    selected = context.get("selected_album")
    if isinstance(selected, dict):
        lines.append(f"Selected album from previous result: {_format_context_album(selected)}")
    rows = _context_rows(context)[:5]
    if rows:
        row_text = "; ".join(f"{index + 1}. {_format_context_album(row)}" for index, row in enumerate(rows))
        lines.append(f"Previous result rows: {row_text}")
    return "\n".join(lines)


def memory_summary(memory: AgentMemory | None) -> str:
    if not memory:
        return "No durable taste memory is active."
    catalog = memory.get("catalog", {})
    lines = [
        "Durable taste memory:",
        (
            f"{catalog.get('albums', 0):,} albums, {catalog.get('rated', 0):,} rated, "
            f"{catalog.get('unrated', 0):,} unresolved, {catalog.get('genres', 0):,} genres."
        ),
    ]
    favorite_genres = memory.get("favorite_genres", [])
    if isinstance(favorite_genres, list) and favorite_genres:
        labels = [
            f"{item.get('Genre')} ({float(item.get('AvgRating', 0)):.2f})"
            for item in favorite_genres[:3]
            if isinstance(item, dict)
        ]
        if labels:
            lines.append("Favorite genre signals: " + ", ".join(labels))
    reliable_artists = memory.get("reliable_artists", [])
    if isinstance(reliable_artists, list) and reliable_artists:
        labels = [
            f"{item.get('Artist')} ({float(item.get('AvgRating', 0)):.2f})"
            for item in reliable_artists[:3]
            if isinstance(item, dict)
        ]
        if labels:
            lines.append("Reliable artists: " + ", ".join(labels))
    above_consensus = memory.get("above_consensus", [])
    if isinstance(above_consensus, list) and above_consensus:
        labels = [_format_context_album(item) for item in above_consensus[:2] if isinstance(item, dict)]
        if labels:
            lines.append("Recurring above-consensus examples: " + "; ".join(labels))
    unresolved = memory.get("unresolved_queue", [])
    if isinstance(unresolved, list) and unresolved:
        labels = [_format_context_album(item) for item in unresolved[:3] if isinstance(item, dict)]
        if labels:
            lines.append("Unresolved listening queue: " + "; ".join(labels))
    return "\n".join(lines)


def _get_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def _function_calls(response: object) -> list[Any]:
    return [item for item in getattr(response, "output", []) or [] if getattr(item, "type", None) == "function_call"]


def answer_question_with_openai(
    question: str,
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    *,
    api_key: str | None = None,
    model: str | None = None,
    context: AgentContext | None = None,
    memory: AgentMemory | None = None,
    filter_df: pd.DataFrame | None = None,
    filter_exploded: pd.DataFrame | None = None,
) -> AgentAnswer:
    cleaned = question.strip()
    if not cleaned:
        return answer_question(
            question,
            df,
            exploded,
            context=context,
            memory=memory,
            filter_df=filter_df,
            filter_exploded=filter_exploded,
        )

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        fallback = answer_question(
            question,
            df,
            exploded,
            context=context,
            memory=memory,
            filter_df=filter_df,
            filter_exploded=filter_exploded,
        )
        return AgentAnswer(
            question=fallback.question,
            summary=fallback.summary,
            detail=fallback.detail,
            skill=fallback.skill,
            mode="deterministic fallback",
            trace=(
                AgentTraceStep("Plan", "No OpenAI API key was configured, so the local router handled the request."),
                *fallback.trace,
            ),
            dashboard_action=fallback.dashboard_action,
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to use the OpenAI-backed agent.") from exc

    client = OpenAI(api_key=resolved_api_key)
    selected_summary = (
        f"Current filtered data: {len(df):,} albums, "
        f"{df['RatingNum'].notna().sum():,} personally rated, "
        f"{exploded['Genre'].nunique():,} genres. "
        f"Available columns: {', '.join(df.columns)}."
    )
    input_items: list[dict[str, object]] = [
        {
            "role": "system",
            "content": (
                "You are a skill-based album analytics agent inside a Streamlit dashboard. "
                "Use the provided tools for factual answers. Do not invent albums, ratings, genres, or notes. "
                "When the user asks a follow-up, use the previous context to resolve phrases like this, that, "
                "the second one, more like this, or why. "
                "Keep answers concise, distinguish evidence from interpretation, and cite the skill result "
                "in plain language. Calibrate certainty: say when a recommendation or pattern is high, medium, "
                "or low confidence based on the amount and specificity of evidence."
            ),
        },
        {"role": "user", "content": selected_summary},
        {"role": "user", "content": "Previous agent context:\n" + context_summary(context)},
        {"role": "user", "content": memory_summary(memory)},
        {"role": "user", "content": cleaned},
    ]

    responses_api: Any = client.responses
    response = responses_api.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5.5"),
        input=input_items,
        tools=AGENT_TOOLS,
        tool_choice="auto",
    )

    calls = _function_calls(response)
    if not calls:
        text = _get_response_text(response)
        fallback = answer_question(
            cleaned,
            df,
            exploded,
            context=context,
            memory=memory,
            filter_df=filter_df,
            filter_exploded=filter_exploded,
        )
        return AgentAnswer(
            question=cleaned,
            summary=text or fallback.summary,
            detail=fallback.detail,
            skill=fallback.skill,
            mode="openai",
            trace=(
                _scope_trace(df, exploded),
                AgentTraceStep("Plan", "Asked OpenAI to route the request, but it answered without a tool call."),
                AgentTraceStep("Tool", f"Used the local {fallback.skill} skill for evidence rows."),
                _rows_trace(fallback),
            ),
            dashboard_action=fallback.dashboard_action,
        )

    for item in getattr(response, "output", []) or []:
        if hasattr(item, "model_dump"):
            input_items.append(item.model_dump(exclude_none=True))
        else:
            input_items.append(item)
    last_answer: AgentAnswer | None = None
    for call in calls:
        raw_arguments = getattr(call, "arguments", "{}") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
        skill_name = str(getattr(call, "name", "catalog_overview"))
        uses_full_catalog = skill_name in {"dashboard_walkthrough", "set_dashboard_filters"}
        skill_df = filter_df if uses_full_catalog and filter_df is not None else df
        skill_exploded = filter_exploded if uses_full_catalog and filter_exploded is not None else exploded
        last_answer = run_skill(skill_name, cleaned, skill_df, skill_exploded, arguments)
        argument_text = ", ".join(f"{key}={value}" for key, value in arguments.items()) or "no arguments"
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(_skill_payload(last_answer), default=str),
            }
        )

    final_response = responses_api.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5.5"),
        input=input_items,
        tools=AGENT_TOOLS,
    )
    final_text = _get_response_text(final_response)
    if last_answer is None:
        last_answer = answer_question(
            cleaned,
            df,
            exploded,
            context=context,
            memory=memory,
            filter_df=filter_df,
            filter_exploded=filter_exploded,
        )
    return AgentAnswer(
        question=cleaned,
        summary=final_text or last_answer.summary,
        detail=last_answer.detail,
        skill=last_answer.skill,
        mode="openai",
        trace=(
            _scope_trace(df, exploded),
            AgentTraceStep("Plan", "Asked OpenAI to choose the best album-analysis tool."),
            AgentTraceStep("Tool", f"OpenAI called {last_answer.skill} with {argument_text}."),
            _rows_trace(last_answer),
            AgentTraceStep("Explain", "Sent the tool result back to OpenAI for the final wording."),
        ),
        dashboard_action=last_answer.dashboard_action,
    )
