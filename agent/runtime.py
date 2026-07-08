from __future__ import annotations

import pandas as pd

from agent.models import AgentAnswer, AgentContext, AgentMemory, AgentTraceStep
from agent.router import choose_skill
from agent.skills import (
    SKILLS,
    _empty_detail,
    _filter_skill_data,
    _rows_trace,
    _scope_trace,
    _with_trace,
    answer_context_followup,
    catalog_overview,
    dashboard_walkthrough,
    genre_analysis,
    notes_search,
    playlist_builder,
    recommendations,
    set_dashboard_filters,
    taste_gaps,
)


def run_skill(
    skill_name: str,
    question: str,
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    arguments: dict[str, object] | None = None,
) -> AgentAnswer:
    arguments = arguments or {}
    if skill_name == "dashboard_walkthrough":
        return dashboard_walkthrough(question, df, exploded, arguments)
    if skill_name == "set_dashboard_filters":
        return set_dashboard_filters(question, df, exploded, arguments)
    if skill_name == "recommendations":
        selected, selected_genres = _filter_skill_data(
            df,
            exploded,
            genre=str(arguments["genre"]) if arguments.get("genre") else None,
            decade=str(arguments["decade"]) if arguments.get("decade") else None,
            artist=str(arguments["artist"]) if arguments.get("artist") else None,
        )
        return recommendations(question, selected, selected_genres)
    if skill_name == "playlist_builder":
        return playlist_builder(question, df, exploded, arguments)
    if skill_name == "taste_gaps":
        direction = str(arguments.get("direction", "above"))
        skill_question = question
        if direction == "below":
            skill_question = question + " below lower overrated"
        return taste_gaps(skill_question, df, exploded)
    if skill_name == "genre_analysis" and arguments.get("genre"):
        return genre_analysis(f"{question} {arguments['genre']}", df, exploded)
    if skill_name == "notes_search":
        terms = arguments.get("terms")
        if isinstance(terms, list) and terms:
            return notes_search("Find notes that mention " + " ".join(str(term) for term in terms), df, exploded)
    return SKILLS.get(skill_name, catalog_overview)(question, df, exploded)


def answer_question(
    question: str,
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    context: AgentContext | None = None,
    memory: AgentMemory | None = None,
    filter_df: pd.DataFrame | None = None,
    filter_exploded: pd.DataFrame | None = None,
) -> AgentAnswer:
    cleaned = question.strip()
    if not cleaned:
        return AgentAnswer(
            question=question,
            summary="Ask me about recommendations, genre patterns, taste gaps, notes, or the current catalog.",
            detail=_empty_detail(),
            skill="help",
        )
    followup = answer_context_followup(cleaned, df, context)
    if followup is not None:
        return _with_trace(
            followup,
            _scope_trace(df, exploded),
            AgentTraceStep(
                "Memory", "Loaded durable taste memory." if memory else "No durable taste memory was loaded."
            ),
            AgentTraceStep("Plan", "Resolved the question as a follow-up using the active agent context."),
            AgentTraceStep("Tool", "Ran the context_followup skill against the filtered catalog."),
            _rows_trace(followup),
        )
    skill_name = choose_skill(cleaned)
    uses_full_catalog = skill_name in {"dashboard_walkthrough", "set_dashboard_filters"}
    skill_df = filter_df if uses_full_catalog and filter_df is not None else df
    skill_exploded = filter_exploded if uses_full_catalog and filter_exploded is not None else exploded
    answer = SKILLS[skill_name](cleaned, skill_df, skill_exploded)
    return _with_trace(
        answer,
        _scope_trace(df, exploded),
        AgentTraceStep("Memory", "Loaded durable taste memory." if memory else "No durable taste memory was loaded."),
        AgentTraceStep("Plan", f"Classified the request as {skill_name}."),
        AgentTraceStep("Tool", f"Ran the {skill_name} skill with deterministic pandas analysis."),
        _rows_trace(answer),
    )
