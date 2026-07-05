from __future__ import annotations

from html import escape
import os
from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from agent import (
    AgentAnswer,
    ProactivePrompt,
    answer_question,
    answer_question_with_openai,
    build_proactive_prompt,
    choose_skill,
)
from album_data import AlbumDataError, RATING_ORDER, load_data, notes_keywords
from album_memory import build_agent_memory, ensure_agent_memory, save_agent_memory
from album_missions import add_mission, load_missions
from ui.assistant import (
    AGENT_ACTION_NOTICE_KEY,
    AGENT_ACTIVE_NUDGE_KEY,
    AGENT_CONTEXT_KEY,
    AGENT_HISTORY_KEY,
    AGENT_IDLE_SECONDS,
    AGENT_IDLE_SIGNATURE_KEY,
    AGENT_LAST_INTERACTION_KEY,
    AGENT_PENDING_QUESTION_KEY,
    AGENT_PIN_CONTEXT_KEY,
    AGENT_PROACTIVE_MUTED_KEY,
    AGENT_PROACTIVE_SEEN_KEY,
    AGENT_QUESTION_KEY,
    assistant_debug_enabled,
    optional_secret,
)
from ui.charts import (
    CHART_HEIGHT,
    RATING_COLOR_MAP,
    WIDE_CHART_HEIGHT,
    compact_table,
    polish_chart,
    render_rating_key,
)
from ui.filters import (
    FILTER_DECADES_KEY,
    FILTER_GENRES_KEY,
    FILTER_ORIGINS_KEY,
    FILTER_SEARCH_KEY,
    FILTER_STATUSES_KEY,
    FILTER_YEAR_RANGE_KEY,
    active_filter_labels,
    ensure_filter_defaults,
    rating_display_label,
    render_active_filters,
)
from ui.explorer import (
    album_key,
    album_selector_label,
    display_number,
    display_text,
    genre_set,
    render_album_detail,
)
from ui.styles import render_global_styles

DATA_PATH = Path(__file__).resolve().parents[1] / "albums.csv"
EXPLORER_ALBUM_KEY = "explorer_album_key"
@st.cache_data(show_spinner=False)
def cached_load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_data(path)



def render_album_assistant(album: pd.Series, selected: pd.DataFrame) -> None:
    st.markdown("**Assistant Context**")
    rating = album.get("RatingNum")
    global_rating = album.get("Global Rating")
    delta = album.get("RatingDelta")
    if pd.notna(rating) and pd.notna(global_rating):
        if pd.notna(delta) and float(delta) >= 0:
            st.caption(f"You rate this {float(delta):+.2f} above the global signal.")
        elif pd.notna(delta):
            st.caption(f"You rate this {float(delta):+.2f} below the global signal.")
        else:
            st.caption("This has both personal and global rating context.")
    elif album.get("RatingStatus") == "unrated":
        st.caption("This is unresolved, so it can become a useful mission target.")
    else:
        st.caption("Use this album as the active context for a focused agent follow-up.")

    source_genres = genre_set(album.get("Genres"))
    similar = selected.copy()
    same_album = (
        similar["Artist"].astype(str).eq(str(album.get("Artist")))
        & similar["Album"].astype(str).eq(str(album.get("Album")))
        & similar["Released"].eq(album.get("Released"))
    )
    similar = similar.loc[~same_album]
    if source_genres:
        similar = similar.loc[similar["Genres"].fillna("").apply(lambda value: bool(genre_set(value) & source_genres))]
    liked = similar.dropna(subset=["RatingNum"]).sort_values(["RatingNum", "Global Rating"], ascending=[False, False]).head(1)
    unresolved = similar.loc[similar["RatingStatus"].eq("unrated")].sort_values(
        ["Global Rating", "Released"],
        ascending=[False, False],
        na_position="last",
    ).head(1)
    cols = st.columns(2)
    cols[0].metric("Closest liked", album_selector_label(liked.iloc[0]) if not liked.empty else "-")
    cols[1].metric("Unresolved match", album_selector_label(unresolved.iloc[0]) if not unresolved.empty else "-")

    context_album = album.where(pd.notna(album), None).to_dict()
    action_cols = st.columns(3)
    if action_cols[0].button("Explain", key="album_assistant_explain", use_container_width=True):
        queue_agent_followup("Why this?", context_album)
    if action_cols[1].button("Similar", key="album_assistant_similar", use_container_width=True):
        queue_agent_followup("Show more like this", context_album)
    if action_cols[2].button("Mission", key="album_assistant_mission", use_container_width=True):
        queue_agent_followup("Create a listening mission from this album", context_album)


def render_agent_scope(
    selected: pd.DataFrame,
    selected_genres: pd.DataFrame,
    active_filters: list[str],
) -> None:
    rated = selected["RatingNum"].dropna()
    scope_cols = st.columns(4)
    scope_cols[0].metric("Scope Albums", f"{len(selected):,}")
    scope_cols[1].metric("Rated", f"{rated.count():,}")
    scope_cols[2].metric("Avg Rating", f"{rated.mean():.2f}" if not rated.empty else "-")
    scope_cols[3].metric("Genres", f"{selected_genres['Genre'].nunique():,}")

    st.markdown("**Current scope**")
    if active_filters:
        chips = " ".join(f"<span class='filter-chip'>{escape(label)}</span>" for label in active_filters)
        st.markdown(f"<div class='filter-strip'>{chips}</div>", unsafe_allow_html=True)
    else:
        st.caption("All albums are included.")


def make_agent_context(answer: AgentAnswer) -> dict[str, object]:
    rows = []
    if not answer.detail.empty:
        clean = answer.detail.head(8).copy()
        clean = clean.where(pd.notna(clean), None)
        rows = clean.to_dict(orient="records")
    selected_album = rows[0] if rows else None
    return {
        "last_question": answer.question,
        "last_skill": answer.skill,
        "last_summary": answer.summary,
        "last_rows": rows,
        "selected_album": selected_album,
    }


def render_agent_context_controls() -> None:
    context = st.session_state.get(AGENT_CONTEXT_KEY)
    left, right = st.columns([0.76, 0.24], vertical_alignment="center")
    with left:
        st.markdown("**Follow-up Context**")
        if context:
            selected = context.get("selected_album") if isinstance(context, dict) else None
            if isinstance(selected, dict) and selected.get("Artist") and selected.get("Album"):
                label = f"{selected.get('Artist')} - {selected.get('Album')} ({selected.get('Released', '-')})"
                st.markdown(f"<span class='filter-chip'>Using: {escape(label)}</span>", unsafe_allow_html=True)
            else:
                st.caption("Using the previous agent result.")
        else:
            st.caption("Ask a question to start a follow-up thread.")
    with right:
        st.toggle("Pin context", key=AGENT_PIN_CONTEXT_KEY, disabled=not bool(context))
        if st.button("Clear context", disabled=not bool(context), use_container_width=True):
            mark_agent_interaction()
            st.session_state[AGENT_CONTEXT_KEY] = None
            st.session_state[AGENT_PIN_CONTEXT_KEY] = False
            st.rerun()


def suggested_followups(answer: AgentAnswer | None) -> list[str]:
    if answer is None:
        return [
            "Show more like this",
            "Why this?",
            "Only unrated albums",
            "Compare against my overall taste",
        ]

    if answer.skill == "listening_mission":
        return [
            "Show unresolved high-signal albums",
            "Build a 3-album starter pack",
            "What hypotheses explain my taste patterns?",
            "Create a taste report",
        ]
    if answer.skill == "taste_hypotheses":
        return [
            "Create a listening mission",
            "Create a taste report",
            "Where do I disagree with consensus?",
            "Build a starter pack",
        ]
    if answer.skill == "taste_gaps":
        return [
            "Show below consensus albums",
            "Show above consensus albums",
            "What hypotheses explain my taste patterns?",
            "Create a taste report",
        ]
    if answer.skill == "genre_analysis":
        return [
            "Build a starter pack",
            "Create a listening mission",
            "Where do I disagree with consensus?",
            "Show unresolved high-signal albums",
        ]
    if answer.skill == "playlist_builder":
        return [
            "Create a listening mission",
            "Only unrated albums",
            "Compare against my overall taste",
            "What hypotheses explain my taste patterns?",
        ]
    if answer.skill == "dashboard_walkthrough":
        return [
            "What should I inspect first?",
            "Create a taste report",
            "Where do I disagree with consensus?",
            "Reset dashboard filters",
        ]
    if answer.skill == "notes_search":
        return [
            "Recommend from these albums",
            "Create a listening mission",
            "Compare against my overall taste",
            "Create a taste report",
        ]

    return [
        "Show more like this",
        "Why this?",
        "Only unrated albums",
        "Compare against my overall taste",
    ]


def render_followup_buttons(answer: AgentAnswer | None = None, *, compact: bool = False) -> None:
    if not st.session_state.get(AGENT_CONTEXT_KEY):
        return
    followups = suggested_followups(answer)
    st.markdown("**Suggested follow-ups**")
    if compact:
        for followup in followups:
            if st.button(followup, use_container_width=True):
                queue_agent_followup(followup)
        return
    cols = st.columns(len(followups))
    for col, followup in zip(cols, followups):
        if col.button(followup, use_container_width=True):
            queue_agent_followup(followup)


def render_agent_trace(answer: AgentAnswer) -> None:
    if not answer.trace:
        return
    with st.expander("Agent plan and tool trace", expanded=False):
        for index, step in enumerate(answer.trace, start=1):
            st.markdown(f"**{index}. {escape(step.phase)}**")
            st.caption(step.detail)


def mark_agent_interaction() -> None:
    st.session_state[AGENT_LAST_INTERACTION_KEY] = time.time()
    st.session_state.pop(AGENT_ACTIVE_NUDGE_KEY, None)


def queue_agent_followup(
    question: str,
    selected_album: dict[str, object] | None = None,
) -> None:
    mark_agent_interaction()
    if selected_album:
        st.session_state[AGENT_CONTEXT_KEY] = {
            "last_question": "Quick action",
            "last_skill": "row_action",
            "last_summary": f"Selected {selected_album.get('Artist')} - {selected_album.get('Album')}",
            "last_rows": [selected_album],
            "selected_album": selected_album,
        }
    st.session_state[AGENT_PENDING_QUESTION_KEY] = question
    st.rerun()


def proactive_signature(selected: pd.DataFrame, selected_genres: pd.DataFrame, active_filters: list[str]) -> str:
    rated = int(selected["RatingNum"].notna().sum()) if "RatingNum" in selected.columns else 0
    unresolved = int(selected["RatingStatus"].eq("unrated").sum()) if "RatingStatus" in selected.columns else 0
    genres = int(selected_genres["Genre"].nunique()) if "Genre" in selected_genres.columns else 0
    return "|".join([str(len(selected)), str(rated), str(unresolved), str(genres), *active_filters])


def render_proactive_prompt(prompt: ProactivePrompt) -> None:
    with st.chat_message("assistant"):
        st.write(prompt.message)


def render_idle_agent_nudge(
    selected: pd.DataFrame,
    selected_genres: pd.DataFrame,
    active_filters: list[str],
    memory: dict[str, object],
) -> None:
    signature = proactive_signature(selected, selected_genres, active_filters)
    if st.session_state.get(AGENT_IDLE_SIGNATURE_KEY) != signature:
        st.session_state[AGENT_IDLE_SIGNATURE_KEY] = signature
        st.session_state[AGENT_LAST_INTERACTION_KEY] = time.time()
        st.session_state.pop(AGENT_ACTIVE_NUDGE_KEY, None)
    st.session_state.setdefault(AGENT_LAST_INTERACTION_KEY, time.time())
    st.session_state.setdefault(AGENT_PROACTIVE_SEEN_KEY, [])
    st.session_state.setdefault(AGENT_PROACTIVE_MUTED_KEY, [])

    active_nudge = st.session_state.get(AGENT_ACTIVE_NUDGE_KEY)
    if isinstance(active_nudge, dict) and active_nudge.get("signature") == signature:
        message = str(active_nudge.get("message", "")).strip()
        if message:
            render_proactive_prompt(
                ProactivePrompt(
                    key=str(active_nudge.get("key", "")),
                    message=message,
                    actions=(),
                    reason="",
                    category=str(active_nudge.get("category", "")),
                )
            )
        return

    prompt = build_proactive_prompt(
        selected,
        selected_genres,
        memory=memory,
        context=st.session_state.get(AGENT_CONTEXT_KEY),
    )
    muted = st.session_state.get(AGENT_PROACTIVE_MUTED_KEY, [])
    if (
        prompt is None
        or prompt.key in st.session_state.get(AGENT_PROACTIVE_SEEN_KEY, [])
        or prompt.category in muted
    ):
        return

    idle_for = time.time() - float(st.session_state[AGENT_LAST_INTERACTION_KEY])
    if idle_for < AGENT_IDLE_SECONDS:
        if st_autorefresh is not None:
            st_autorefresh(interval=1000, limit=AGENT_IDLE_SECONDS + 1, key=f"agent_idle_refresh_{signature}")
        else:
            st.caption("The assistant can surface a nudge after one minute when streamlit-autorefresh is installed.")
        return

    seen = list(st.session_state.get(AGENT_PROACTIVE_SEEN_KEY, []))
    st.session_state[AGENT_PROACTIVE_SEEN_KEY] = sorted(set([*seen, prompt.key]))
    st.session_state[AGENT_ACTIVE_NUDGE_KEY] = {
        "key": prompt.key,
        "message": prompt.message,
        "category": prompt.category,
        "signature": signature,
    }
    render_proactive_prompt(prompt)


def render_saved_missions() -> None:
    data = load_missions()
    missions = data.get("missions", [])
    if not isinstance(missions, list) or not missions:
        return
    with st.expander("Saved listening missions", expanded=False):
        for mission in missions[:4]:
            if not isinstance(mission, dict):
                continue
            st.markdown(f"**{escape(str(mission.get('title', 'Listening mission')))}**")
            st.caption(f"{mission.get('status', 'not started')} | {mission.get('created_at', '-')}")


def render_mission_answer(answer: AgentAnswer, answer_index: int, *, compact: bool = False) -> None:
    if answer.detail.empty:
        return
    st.markdown("**Mission Path**")
    for _, row in answer.detail.iterrows():
        if compact:
            st.markdown(f"**{int(row.get('Step', 0))}. {escape(str(row.get('Role', '-')))}**")
            st.write(f"{row.get('Artist', '-')} - {row.get('Album', '-')}")
            st.caption(f"{row.get('Status', '-')} | {row.get('Why', '')}")
            continue
        cols = st.columns([0.08, 0.2, 0.28, 0.16, 0.28], vertical_alignment="center")
        cols[0].metric("Step", int(row.get("Step", 0)))
        cols[1].markdown(f"**{escape(str(row.get('Role', '-')))}**")
        cols[2].write(f"{row.get('Artist', '-')} - {row.get('Album', '-')}")
        cols[3].caption(str(row.get("Status", "-")))
        cols[4].caption(str(row.get("Why", "")))
    save_key = f"save_mission_{answer_index}_{len(st.session_state.get(AGENT_HISTORY_KEY, []))}"
    if st.button("Save mission", key=save_key):
        mark_agent_interaction()
        add_mission(
            {
                "title": answer.summary.split(".")[0],
                "summary": answer.summary,
                "steps": answer.detail.where(pd.notna(answer.detail), None).to_dict(orient="records"),
            }
        )
        st.toast("Mission saved.")


def render_hypothesis_answer(answer: AgentAnswer, *, compact: bool = False) -> None:
    if answer.detail.empty:
        return
    st.markdown("**Taste Hypotheses**")
    for index, row in answer.detail.iterrows():
        st.markdown(f"**{index + 1}. {escape(str(row.get('Hypothesis', '-')))}**")
        st.caption(f"Confidence: {row.get('Confidence', '-')}")
        if compact:
            st.write(f"Evidence: {row.get('Evidence', '-')}")
            st.write(f"Counterexample: {row.get('Counterexample', '-')}")
            st.write(f"Next: {row.get('Action', '-')}")
            continue
        evidence_col, counter_col, action_col = st.columns(3)
        evidence_col.write(f"Evidence: {row.get('Evidence', '-')}")
        counter_col.write(f"Counterexample: {row.get('Counterexample', '-')}")
        action_col.write(f"Next: {row.get('Action', '-')}")


def render_agent_row_actions(answer: AgentAnswer, answer_index: int, *, compact: bool = False) -> None:
    if answer.detail.empty or not {"Artist", "Album"}.issubset(answer.detail.columns):
        return
    rows = answer.detail.head(3).where(pd.notna(answer.detail.head(3)), None).to_dict(orient="records")
    if not rows:
        return
    st.markdown("**Quick actions**")
    for row_index, row in enumerate(rows):
        label = f"{row.get('Artist', '-')} - {row.get('Album', '-')}"
        st.caption(label)
        if compact:
            if st.button("Find similar", key=f"similar_{answer_index}_{row_index}", use_container_width=True):
                queue_agent_followup("Show more like this", row)
            if st.button("Explain", key=f"explain_{answer_index}_{row_index}", use_container_width=True):
                queue_agent_followup("Why this?", row)
            if st.button("Make mission", key=f"mission_{answer_index}_{row_index}", use_container_width=True):
                queue_agent_followup("Create a listening mission from this album", row)
            continue
        cols = st.columns(3)
        if cols[0].button("Find similar", key=f"similar_{answer_index}_{row_index}", use_container_width=True):
            queue_agent_followup("Show more like this", row)
        if cols[1].button("Explain", key=f"explain_{answer_index}_{row_index}", use_container_width=True):
            queue_agent_followup("Why this?", row)
        if cols[2].button("Make mission", key=f"mission_{answer_index}_{row_index}", use_container_width=True):
            queue_agent_followup("Create a listening mission from this album", row)


def apply_agent_dashboard_action(answer: AgentAnswer, full_catalog: pd.DataFrame, full_genres: pd.DataFrame) -> bool:
    action = answer.dashboard_action
    if not isinstance(action, dict) or action.get("type") != "set_filters":
        return False

    year_min, year_max = int(full_catalog["Released"].min()), int(full_catalog["Released"].max())
    filters = action.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    if action.get("clear_existing"):
        reset_filter_state(year_min, year_max)

    valid_genres = set(full_genres["Genre"].dropna().astype(str))
    valid_origins = set(full_catalog["OriginLabel"].dropna().astype(str))
    valid_decades = set(full_catalog["Decade"].dropna().astype(str))
    valid_statuses = set(RATING_ORDER)

    search = str(filters.get("search", "") or "").strip()
    st.session_state[FILTER_SEARCH_KEY] = search
    st.session_state[FILTER_GENRES_KEY] = [
        str(value) for value in filters.get("genres", []) if str(value) in valid_genres
    ]
    st.session_state[FILTER_ORIGINS_KEY] = [
        str(value) for value in filters.get("origins", []) if str(value) in valid_origins
    ]
    st.session_state[FILTER_DECADES_KEY] = [
        str(value) for value in filters.get("decades", []) if str(value) in valid_decades
    ]
    st.session_state[FILTER_STATUSES_KEY] = [
        str(value) for value in filters.get("statuses", []) if str(value) in valid_statuses
    ]

    year_range = filters.get("year_range")
    if isinstance(year_range, list) and len(year_range) == 2:
        try:
            start, end = sorted((int(year_range[0]), int(year_range[1])))
        except (TypeError, ValueError):
            start, end = year_min, year_max
        st.session_state[FILTER_YEAR_RANGE_KEY] = (
            max(year_min, min(year_max, start)),
            max(year_min, min(year_max, end)),
        )

    return True


def render_agent_memory(memory: dict[str, object]) -> None:
    catalog = memory.get("catalog", {}) if isinstance(memory.get("catalog"), dict) else {}
    st.caption(
        f"{catalog.get('rated', 0):,} rated albums, "
        f"{catalog.get('unrated', 0):,} unresolved albums, "
        f"{catalog.get('genres', 0):,} genre signals."
    )

    favorite_genres = memory.get("favorite_genres", [])
    unresolved = memory.get("unresolved_queue", [])
    if isinstance(favorite_genres, list) and favorite_genres:
        st.markdown("**Favorite genre signals**")
        st.dataframe(pd.DataFrame(favorite_genres), use_container_width=True, hide_index=True, height=190)
    if isinstance(unresolved, list) and unresolved:
        st.markdown("**Unresolved listening queue**")
        st.dataframe(pd.DataFrame(unresolved), use_container_width=True, hide_index=True, height=220)


def render_agent_router_controls() -> tuple[bool, str | None, str]:
    api_key = os.getenv("OPENAI_API_KEY") or optional_secret("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL") or optional_secret("OPENAI_MODEL", "gpt-5.5")
    use_openai = st.toggle(
        "Use OpenAI skill router",
        value=bool(api_key),
        disabled=not bool(api_key),
        help="Set OPENAI_API_KEY as an environment variable or in .streamlit/secrets.toml.",
    )
    if not api_key:
        st.caption("Running in local fallback mode because no OpenAI API key is configured.")
    return use_openai, api_key, model


def run_agent_turn(
    question: str,
    selected: pd.DataFrame,
    selected_genres: pd.DataFrame,
    full_catalog: pd.DataFrame,
    full_genres: pd.DataFrame,
    memory: dict[str, object],
    *,
    use_openai: bool,
    api_key: str | None,
    model: str,
) -> AgentAnswer:
    try:
        skill_name = choose_skill(question)
        if skill_name in {"set_dashboard_filters", "dashboard_walkthrough"}:
            return answer_question(
                question,
                selected,
                selected_genres,
                context=st.session_state.get(AGENT_CONTEXT_KEY),
                memory=memory,
                filter_df=full_catalog,
                filter_exploded=full_genres,
            )
        if use_openai:
            return answer_question_with_openai(
                question,
                selected,
                selected_genres,
                api_key=str(api_key),
                model=str(model),
                context=st.session_state.get(AGENT_CONTEXT_KEY),
                memory=memory,
                filter_df=full_catalog,
                filter_exploded=full_genres,
            )
        return answer_question(
            question,
            selected,
            selected_genres,
            context=st.session_state.get(AGENT_CONTEXT_KEY),
            memory=memory,
            filter_df=full_catalog,
            filter_exploded=full_genres,
        )
    except Exception as exc:
        fallback = answer_question(
            question,
            selected,
            selected_genres,
            context=st.session_state.get(AGENT_CONTEXT_KEY),
            memory=memory,
            filter_df=full_catalog,
            filter_exploded=full_genres,
        )
        return fallback.__class__(
            question=fallback.question,
            summary=(
                f"{fallback.summary}\n\n"
                f"OpenAI routing was unavailable, so this answer used the local skill router. Error: {exc}"
            ),
            detail=fallback.detail,
            skill=fallback.skill,
            mode="deterministic fallback",
            trace=fallback.trace,
            dashboard_action=fallback.dashboard_action,
        )


def save_agent_turn(answer: AgentAnswer) -> None:
    st.session_state[AGENT_HISTORY_KEY].insert(0, answer)
    st.session_state[AGENT_HISTORY_KEY] = st.session_state[AGENT_HISTORY_KEY][:6]
    if not st.session_state.get(AGENT_PIN_CONTEXT_KEY):
        st.session_state[AGENT_CONTEXT_KEY] = make_agent_context(answer)


def render_agent(
    selected: pd.DataFrame,
    selected_genres: pd.DataFrame,
    active_filters: list[str],
    full_catalog: pd.DataFrame,
    full_genres: pd.DataFrame,
) -> None:
    st.subheader("AI Agent")
    st.caption("Ask about the currently filtered albums. The agent chooses a data skill, runs it, and explains the result.")

    memory = ensure_agent_memory(full_catalog, full_genres)
    st.session_state.setdefault(AGENT_HISTORY_KEY, [])
    st.session_state.setdefault(AGENT_CONTEXT_KEY, None)
    st.session_state.setdefault(AGENT_PIN_CONTEXT_KEY, False)

    with st.expander("Current scope", expanded=False):
        render_agent_scope(selected, selected_genres, active_filters)
    with st.expander("Taste memory", expanded=False):
        refresh_memory = st.button("Refresh durable memory", use_container_width=False)
        if refresh_memory:
            mark_agent_interaction()
            memory = build_agent_memory(full_catalog, full_genres)
            save_agent_memory(memory)
            st.rerun()
        render_agent_memory(memory)
    render_saved_missions()
    with st.expander("Current focus", expanded=False):
        render_agent_context_controls()
    render_idle_agent_nudge(selected, selected_genres, active_filters, memory)

    with st.expander("Router mode", expanded=False):
        use_openai, api_key, model = render_agent_router_controls()

    examples = [
        "What should I listen to next?",
        "Create a listening mission",
        "What hypotheses explain my taste patterns?",
        "Build a 3-album starter pack",
    ]
    example_cols = st.columns(len(examples))
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            mark_agent_interaction()
            st.session_state[AGENT_QUESTION_KEY] = example
    latest_answer = st.session_state[AGENT_HISTORY_KEY][0] if st.session_state[AGENT_HISTORY_KEY] else None
    render_followup_buttons(latest_answer)

    with st.form("agent_form", clear_on_submit=False):
        question = st.text_input(
            "Ask a question",
            key=AGENT_QUESTION_KEY,
            placeholder="Try: recommend a rock album from the 1970s",
        )
        submitted = st.form_submit_button("Ask agent", use_container_width=True)

    pending_question = st.session_state.pop(AGENT_PENDING_QUESTION_KEY, None)
    question_to_answer = str(pending_question or question).strip()

    if (submitted or pending_question) and question_to_answer:
        mark_agent_interaction()
        answer = run_agent_turn(
            question_to_answer,
            selected,
            selected_genres,
            full_catalog,
            full_genres,
            memory,
            use_openai=use_openai,
            api_key=api_key,
            model=model,
        )
        save_agent_turn(answer)
        if apply_agent_dashboard_action(answer, full_catalog, full_genres):
            st.rerun()

    if not st.session_state[AGENT_HISTORY_KEY]:
        st.info(
            "The agent can recommend albums, summarize genre patterns, compare you against consensus, "
            "build listening paths, search notes, and produce capstone-ready insights."
        )
        return

    for answer_index, answer in enumerate(st.session_state[AGENT_HISTORY_KEY]):
        with st.container(border=True):
            st.markdown(f"**You:** {escape(answer.question)}")
            st.write(answer.summary)
            st.caption(f"Skill used: {answer.skill} | Mode: {answer.mode}")
            render_agent_trace(answer)
            if answer.skill == "listening_mission":
                render_mission_answer(answer, answer_index)
            elif answer.skill == "taste_hypotheses":
                render_hypothesis_answer(answer)
            elif not answer.detail.empty:
                compact_table(answer.detail, answer.detail.columns.tolist(), height=260)
            render_agent_row_actions(answer, answer_index)


def render_soundprint(selected: pd.DataFrame, selected_genres: pd.DataFrame) -> None:
    rated = selected.dropna(subset=["RatingNum"]).copy()
    rated_genres = selected_genres.dropna(subset=["RatingNum"]).copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if not rated_genres.empty:
            favorite_genre = (
                rated_genres.groupby("Genre", as_index=False)
                .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"))
                .query("Albums >= 2")
                .sort_values(["AvgRating", "Albums"], ascending=[False, False])
                .head(1)
            )
            label = favorite_genre.iloc[0]["Genre"] if not favorite_genre.empty else "-"
            detail = f"{favorite_genre.iloc[0]['AvgRating']:.2f} avg" if not favorite_genre.empty else None
        else:
            label = "-"
            detail = None
        st.metric("Signature Genre", label, detail)
    with c2:
        if not rated.empty:
            favorite_decade = (
                rated.groupby("Decade", as_index=False)
                .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"))
                .sort_values(["AvgRating", "Albums"], ascending=[False, False])
                .head(1)
            )
            st.metric("Best Era", favorite_decade.iloc[0]["Decade"], f"{favorite_decade.iloc[0]['AvgRating']:.2f} avg")
        else:
            st.metric("Best Era", "-")
    with c3:
        if not rated.empty:
            artist = (
                rated.groupby("Artist", as_index=False)
                .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"))
                .query("Albums >= 2")
                .sort_values(["AvgRating", "Albums"], ascending=[False, False])
                .head(1)
            )
            label = artist.iloc[0]["Artist"] if not artist.empty else "-"
            detail = f"{artist.iloc[0]['Albums']:.0f} albums" if not artist.empty else None
        else:
            label = "-"
            detail = None
        st.metric("Reliable Artist", label, detail)
    with c4:
        gap = selected["RatingDelta"].dropna()
        st.metric("Consensus Bias", f"{gap.mean():+.2f}" if not gap.empty else "-", help="Average personal rating minus global rating")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Favorite Genre Signals**")
        genre_summary = (
            rated_genres.groupby("Genre", as_index=False)
            .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
            .query("Albums >= 2")
            .sort_values(["AvgRating", "Albums"], ascending=[False, False])
        )
        if genre_summary.empty:
            st.caption("Rate more albums to build genre signals.")
        else:
            genre_summary["Delta"] = genre_summary["AvgRating"] - genre_summary["AvgGlobal"]
            compact_table(
                genre_summary.head(12),
                ["Genre", "Albums", "AvgRating", "AvgGlobal", "Delta"],
                height=330,
            )

    with right:
        st.markdown("**Biggest Personal Splits**")
        split_df = selected.dropna(subset=["RatingNum", "Global Rating", "RatingDelta"]).copy()
        if split_df.empty:
            st.caption("Add personal and global ratings to see where your taste diverges.")
        else:
            compact_table(
                split_df.reindex(split_df["RatingDelta"].abs().sort_values(ascending=False).index).head(12),
                ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta"],
                height=330,
            )

    st.markdown("**Taste Map**")
    if rated_genres.empty:
        st.caption("Rate more albums to generate a taste map.")
    else:
        taste_map = (
            rated_genres.groupby(["Decade", "Genre"], as_index=False)
            .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"))
            .query("Albums >= 2")
        )
        if taste_map.empty:
            st.caption("More rated albums per genre and decade will unlock the taste map.")
        else:
            decade_order = sorted(taste_map["Decade"].unique(), key=lambda value: int(str(value).rstrip("s")))
            fig_taste_map = px.scatter(
                taste_map,
                x="Decade",
                y="Genre",
                size="Albums",
                color="AvgRating",
                hover_data={"Albums": True, "AvgRating": ":.2f"},
                title="Where your ratings cluster by era and genre",
                category_orders={"Decade": decade_order},
                color_continuous_scale="RdYlGn",
                size_max=42,
            )
            st.plotly_chart(
                polish_chart(fig_taste_map, height=500, x_title=None, y_title=None),
                use_container_width=True,
            )


def current_filtered_data(df: pd.DataFrame, exploded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    year_min, year_max = int(df["Released"].min()), int(df["Released"].max())
    query = str(st.session_state.get(FILTER_SEARCH_KEY, "")).strip().lower()
    genres = list(st.session_state.get(FILTER_GENRES_KEY, []))
    origins = list(st.session_state.get(FILTER_ORIGINS_KEY, []))
    decades = list(st.session_state.get(FILTER_DECADES_KEY, []))
    statuses = list(st.session_state.get(FILTER_STATUSES_KEY, []))
    year_range = tuple(st.session_state.get(FILTER_YEAR_RANGE_KEY, (year_min, year_max)))
    mask = df["Released"].between(year_range[0], year_range[1])
    if query:
        mask &= df["SearchText"].str.contains(query, regex=False, na=False)
    if genres:
        album_keys = exploded.loc[exploded["Genre"].isin(genres), ["Artist", "Album", "Released"]].drop_duplicates()
        key = pd.MultiIndex.from_frame(album_keys)
        own_key = pd.MultiIndex.from_frame(df[["Artist", "Album", "Released"]])
        mask &= own_key.isin(key)
    if origins:
        mask &= df["OriginLabel"].isin(origins)
    if decades:
        mask &= df["Decade"].isin(decades)
    if statuses:
        mask &= df["RatingStatus"].isin(statuses)

    selected = df.loc[mask].copy()
    selected_keys = pd.MultiIndex.from_frame(selected[["Artist", "Album", "Released"]])
    exploded_keys = pd.MultiIndex.from_frame(exploded[["Artist", "Album", "Released"]])
    labels = active_filter_labels(
        query=query,
        genres=genres,
        origins=origins,
        decades=decades,
        statuses=statuses,
        year_range=year_range,
        full_year_range=(year_min, year_max),
    )
    return selected, exploded.loc[exploded_keys.isin(selected_keys)].copy(), labels


def render_sidebar_answer_body(answer: AgentAnswer, answer_index: int) -> None:
    st.write(answer.summary)
    if assistant_debug_enabled():
        st.caption(f"Skill: {answer.skill} | Mode: {answer.mode}")
        render_agent_trace(answer)
    if answer.skill == "listening_mission":
        render_mission_answer(answer, answer_index, compact=True)
    elif answer.skill == "taste_hypotheses":
        render_hypothesis_answer(answer, compact=True)
    elif not answer.detail.empty:
        compact_table(answer.detail, answer.detail.columns.tolist(), height=220)
    render_agent_row_actions(answer, answer_index, compact=True)


def render_sidebar_answer(answer: AgentAnswer) -> None:
    st.sidebar.markdown(f"**You:** {escape(answer.question)}")
    with st.sidebar:
        render_sidebar_answer_body(answer, 0)


def render_sidebar_assistant(
    selected: pd.DataFrame,
    selected_genres: pd.DataFrame,
    active_filters: list[str],
    full_catalog: pd.DataFrame,
    full_genres: pd.DataFrame,
) -> None:
    st.session_state.setdefault(AGENT_HISTORY_KEY, [])
    st.session_state.setdefault(AGENT_CONTEXT_KEY, None)
    st.session_state.setdefault(AGENT_PIN_CONTEXT_KEY, False)
    memory = ensure_agent_memory(full_catalog, full_genres)
    api_key = os.getenv("OPENAI_API_KEY") or optional_secret("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL") or optional_secret("OPENAI_MODEL", "gpt-5.5")
    use_openai = bool(api_key)

    st.sidebar.markdown("### Assistant")
    st.sidebar.caption("Ask a question. The dashboard can answer or update itself.")
    notice = st.session_state.pop(AGENT_ACTION_NOTICE_KEY, None)
    if notice:
        st.sidebar.success(str(notice))

    if assistant_debug_enabled():
        with st.sidebar.expander("Assistant debug", expanded=False):
            render_agent_context_controls()
            st.caption(f"Router: {'OpenAI' if use_openai else 'local fallback'}")
            if not api_key:
                st.caption("Set OPENAI_API_KEY to enable OpenAI routing.")
            refresh_memory = st.button("Refresh durable memory", use_container_width=True)
            if refresh_memory:
                mark_agent_interaction()
                memory = build_agent_memory(full_catalog, full_genres)
                save_agent_memory(memory)
                st.rerun()
            render_agent_scope(selected, selected_genres, active_filters)
            render_agent_memory(memory)

    pending_question = st.session_state.pop(AGENT_PENDING_QUESTION_KEY, None)
    with st.sidebar:
        chat_question = st.chat_input("Ask about this dashboard")
    question_to_answer = str(pending_question or chat_question or "").strip()
    if question_to_answer:
        mark_agent_interaction()
        answer = run_agent_turn(
            question_to_answer,
            selected,
            selected_genres,
            full_catalog,
            full_genres,
            memory,
            use_openai=use_openai,
            api_key=api_key,
            model=model,
        )
        save_agent_turn(answer)
        if apply_agent_dashboard_action(answer, full_catalog, full_genres):
            st.session_state[AGENT_ACTION_NOTICE_KEY] = "Updated the dashboard filters."
            st.rerun()

    with st.sidebar:
        render_idle_agent_nudge(selected, selected_genres, active_filters, memory)
    history = st.session_state[AGENT_HISTORY_KEY]
    latest_answer = history[0] if history else None
    if not history:
        st.sidebar.info("Try: recommend an album, explain a pattern, or change filters.")
    else:
        with st.sidebar:
            for answer_index, answer in reversed(list(enumerate(history))):
                with st.chat_message("user"):
                    st.write(answer.question)
                with st.chat_message("assistant"):
                    render_sidebar_answer_body(answer, answer_index)
            render_followup_buttons(latest_answer, compact=True)
    with st.sidebar:
        render_saved_missions()
    st.sidebar.divider()


def filtered_data(df: pd.DataFrame, exploded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    active_filter_container = st.sidebar.container()

    year_min, year_max = int(df["Released"].min()), int(df["Released"].max())
    all_genres = sorted(exploded["Genre"].dropna().unique())
    query = st.sidebar.text_input(
        "Search",
        placeholder="artist, album, note, genre",
        key=FILTER_SEARCH_KEY,
    ).strip().lower()
    genres = st.sidebar.multiselect("Genres", all_genres, key=FILTER_GENRES_KEY)
    origins = st.sidebar.multiselect("Origin", sorted(df["OriginLabel"].dropna().unique()), key=FILTER_ORIGINS_KEY)
    decades = st.sidebar.multiselect("Decades", sorted(df["Decade"].unique()), key=FILTER_DECADES_KEY)
    statuses = st.sidebar.multiselect(
        "Personal rating label",
        RATING_ORDER,
        key=FILTER_STATUSES_KEY,
        format_func=rating_display_label,
    )
    year_range = st.sidebar.slider("Release years", year_min, year_max, key=FILTER_YEAR_RANGE_KEY)

    mask = df["Released"].between(year_range[0], year_range[1])
    if query:
        mask &= df["SearchText"].str.contains(query, regex=False, na=False)
    if genres:
        album_keys = exploded.loc[exploded["Genre"].isin(genres), ["Artist", "Album", "Released"]].drop_duplicates()
        key = pd.MultiIndex.from_frame(album_keys)
        own_key = pd.MultiIndex.from_frame(df[["Artist", "Album", "Released"]])
        mask &= own_key.isin(key)
    if origins:
        mask &= df["OriginLabel"].isin(origins)
    if decades:
        mask &= df["Decade"].isin(decades)
    if statuses:
        mask &= df["RatingStatus"].isin(statuses)

    selected = df.loc[mask].copy()
    selected_keys = pd.MultiIndex.from_frame(selected[["Artist", "Album", "Released"]])
    exploded_keys = pd.MultiIndex.from_frame(exploded[["Artist", "Album", "Released"]])
    labels = active_filter_labels(
        query=query,
        genres=genres,
        origins=origins,
        decades=decades,
        statuses=statuses,
        year_range=year_range,
        full_year_range=(year_min, year_max),
    )
    render_active_filters(labels, year_min, year_max, active_filter_container)
    return selected, exploded.loc[exploded_keys.isin(selected_keys)].copy(), labels


def main() -> None:
    try:
        df, exploded = cached_load_data(DATA_PATH)
    except AlbumDataError as exc:
        st.error("albums.csv has data issues.")
        st.markdown("Fix these rows, then rerun the dashboard:")
        for message in str(exc).splitlines():
            st.write(f"- {message}")
        st.stop()

    year_min, year_max = int(df["Released"].min()), int(df["Released"].max())
    ensure_filter_defaults(year_min, year_max)
    assistant_selected, assistant_selected_genres, assistant_filters = current_filtered_data(df, exploded)
    render_sidebar_assistant(assistant_selected, assistant_selected_genres, assistant_filters, df, exploded)
    selected, selected_genres, active_filters = filtered_data(df, exploded)

    st.title("🎧 Halchemy Album Dashboard")
    st.caption(
        "Start with the catalog, follow the taste patterns, then drill into the albums worth inspecting."
    )
    render_global_styles()

    if selected.empty:
        st.warning("No albums match the current filters.")
        return

    rated = selected["RatingNum"].dropna()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Albums", f"{len(selected):,}", help="Filtered catalog size")
    with c2:
        st.metric("Rated", f"{rated.count():,}", f"{selected['RatingStatus'].eq('did-not-listen').sum():,} did not listen")
    with c3:
        st.metric("Avg Rating", f"{rated.mean():.2f}" if not rated.empty else "-", help="Your personal mean")
    with c4:
        st.metric("Avg Global", f"{selected['Global Rating'].mean():.2f}", help="External consensus")
    with c5:
        delta = selected["RatingDelta"].mean()
        st.metric("Taste Gap", f"{delta:+.2f}" if pd.notna(delta) else "-", help="You minus global")

    section = st.segmented_control(
        "Section",
        ["Catalog", "Soundprint", "Taste", "Outliers", "Explorer"],
        default="Catalog",
        label_visibility="collapsed",
    )

    if section == "Catalog":
        st.subheader("Catalog")

        left, right = st.columns([1.05, 1])
        with left:
            render_rating_key(selected["RatingStatus"])
        rating_counts = (
            selected["RatingStatus"]
            .value_counts()
            .reindex([r for r in RATING_ORDER if r in selected["RatingStatus"].unique()])
            .reset_index()
        )
        rating_counts.columns = ["RatingStatus", "Albums"]
        rating_counts["Rating"] = rating_counts["RatingStatus"].map(rating_display_label)
        fig_rating = px.bar(
            rating_counts,
            x="Rating",
            y="Albums",
            color="RatingStatus",
            title="Rating mix",
            category_orders={
                "Rating": [rating_display_label(status) for status in RATING_ORDER],
                "RatingStatus": RATING_ORDER,
            },
            color_discrete_map=RATING_COLOR_MAP,
        )
        fig_rating.update_layout(showlegend=False)
        left.plotly_chart(polish_chart(fig_rating, x_title=None, y_title="Albums"), use_container_width=True)

        decade = (
            selected.groupby("Decade", as_index=False)
            .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
            .sort_values("Decade")
        )
        fig_decade = px.bar(
            decade,
            x="Decade",
            y="Albums",
            color="AvgRating",
            title="Era spread",
        )
        right.plotly_chart(polish_chart(fig_decade, x_title=None, y_title="Albums"), use_container_width=True)

        timeline_df = selected.sort_values(["Released", "Artist", "Album"]).copy()
        timeline_df["TimelineSize"] = timeline_df["RatingNum"].fillna(2.5) + 3
        decade_order = sorted(
            timeline_df["Decade"].dropna().unique(),
            key=lambda value: int(str(value).rstrip("s")),
        )
        fig_release_timeline = px.scatter(
            timeline_df,
            x="Released",
            y="Decade",
            color="RatingStatus",
            size="TimelineSize",
            hover_data={
                "Artist": True,
                "Album": True,
                "Released": True,
                "Genres": True,
                "RatingLabel": True,
                "TimelineSize": False,
            },
            title="Album timeline",
            category_orders={"Decade": decade_order, "RatingStatus": RATING_ORDER},
            color_discrete_map=RATING_COLOR_MAP,
        )
        fig_release_timeline.update_traces(marker=dict(line=dict(width=0.7, color="rgba(49, 51, 63, 0.35)")))
        fig_release_timeline.update_xaxes(dtick=5)
        st.plotly_chart(
            polish_chart(fig_release_timeline, height=430, x_title="Release year", y_title=None),
            use_container_width=True,
        )

        by_month = selected.dropna(subset=["MonthAdded"]).groupby("MonthAdded", as_index=False).size()
        fig_month = px.area(by_month, x="MonthAdded", y="size", title="Listening timeline")
        st.plotly_chart(
            polish_chart(fig_month, height=WIDE_CHART_HEIGHT, x_title=None, y_title="Albums"),
            use_container_width=True,
        )

    elif section == "Soundprint":
        st.subheader("Your Soundprint")
        st.caption("Personal patterns from the current catalog slice.")
        render_soundprint(selected, selected_genres)

    elif section == "Taste":
        st.subheader("Taste")

        genre_summary = (
            selected_genres.groupby("Genre", as_index=False)
            .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
            .query("Albums >= 3")
            .sort_values(["AvgRating", "Albums"], ascending=[False, False])
        )
        genre_summary["Delta"] = genre_summary["AvgRating"] - genre_summary["AvgGlobal"]

        left, right = st.columns([1, 1])
        fig_genre = px.bar(
            genre_summary.head(16).sort_values("AvgRating"),
            x="AvgRating",
            y="Genre",
            orientation="h",
            color="Albums",
            title="Genre signals",
        )
        fig_genre.update_xaxes(range=[0, 5])
        left.plotly_chart(polish_chart(fig_genre, x_title="Avg rating", y_title=None), use_container_width=True)

        origin = (
            selected.groupby("OriginLabel", as_index=False)
            .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
            .sort_values("Albums", ascending=False)
        )
        fig_origin = px.scatter(
            origin,
            x="AvgGlobal",
            y="AvgRating",
            size="Albums",
            color="OriginLabel",
            text="OriginLabel",
            title="Origin patterns",
        )
        fig_origin.update_traces(textposition="top center")
        fig_origin.update_xaxes(range=[2.5, 3.8])
        fig_origin.update_yaxes(range=[1, 5])
        right.plotly_chart(polish_chart(fig_origin, x_title="Global", y_title="Rating"), use_container_width=True)

        words = notes_keywords(selected)
        if not words.empty:
            fig_words = px.bar(
                words.sort_values("Count"),
                x="Count",
                y="Word",
                orientation="h",
                title="Language in your notes",
                color="Count",
            )
            st.plotly_chart(
                polish_chart(fig_words, height=WIDE_CHART_HEIGHT, x_title="Mentions", y_title=None),
                use_container_width=True,
            )

        if not genre_summary.empty:
            fig_genre_map = px.scatter(
                genre_summary,
                x="Albums",
                y="AvgRating",
                size="Albums",
                color="Delta",
                text="Genre",
                hover_data={
                    "Genre": True,
                    "Albums": True,
                    "AvgRating": ":.2f",
                    "AvgGlobal": ":.2f",
                    "Delta": ":+.2f",
                },
                title="Genre map",
                size_max=46,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )
            fig_genre_map.update_traces(textposition="top center")
            fig_genre_map.update_yaxes(range=[0.8, 5.2])
            st.plotly_chart(
                polish_chart(fig_genre_map, height=430, x_title="Albums", y_title="Avg rating"),
                use_container_width=True,
            )

    elif section == "Outliers":
        st.subheader("Outliers")

        gap_df = selected.dropna(subset=["RatingNum", "Global Rating", "RatingDelta"]).copy()
        fig_gap = px.scatter(
            gap_df,
            x="Global Rating",
            y="RatingNum",
            color="RatingDelta",
            size="Released",
            hover_data={
                "Artist": True,
                "Album": True,
                "Released": True,
                "Genres": True,
                "RatingDelta": ":.2f",
                "Global Rating": ":.2f",
                "RatingNum": ":.1f",
            },
            title="You vs consensus",
        )
        fig_gap.add_shape(type="line", x0=1, y0=1, x1=5, y1=5, line=dict(dash="dash"))
        fig_gap.update_yaxes(range=[0.7, 5.3], dtick=1)
        fig_gap.update_xaxes(range=[1.7, 4.6])
        st.plotly_chart(
            polish_chart(fig_gap, height=430, x_title="Global", y_title="Rating"),
            use_container_width=True,
        )

        love, reject = st.columns(2)
        with love:
            st.markdown("**Above Consensus**")
            compact_table(
                gap_df.sort_values("RatingDelta", ascending=False).head(10),
                ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"],
                height=300,
            )
        with reject:
            st.markdown("**Below Consensus**")
            compact_table(
                gap_df.sort_values("RatingDelta").head(10),
                ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"],
                height=300,
            )

    elif section == "Explorer":
        st.subheader("Explorer")

        sort_options = {
            "Newest release": ("Released", False),
            "Oldest release": ("Released", True),
            "Highest personal rating": ("RatingNum", False),
            "Highest global rating": ("Global Rating", False),
            "Biggest positive gap": ("RatingDelta", False),
            "Biggest negative gap": ("RatingDelta", True),
            "Recently added": ("GeneratedDate", False),
        }
        sort_label = st.selectbox("Sort albums", list(sort_options.keys()), index=0)
        sort_col, ascending = sort_options[sort_label]
        table_df = selected.sort_values(sort_col, ascending=ascending, na_position="last").copy()
        table_df["AlbumKey"] = table_df.apply(album_key, axis=1)
        album_labels = {
            row["AlbumKey"]: album_selector_label(row)
            for _, row in table_df[["AlbumKey", "Artist", "Album", "Released"]].iterrows()
        }
        album_keys = table_df["AlbumKey"].tolist()
        if st.session_state.get(EXPLORER_ALBUM_KEY) not in album_keys:
            st.session_state[EXPLORER_ALBUM_KEY] = album_keys[0]

        selected_album_key = st.selectbox(
            "Select album",
            album_keys,
            key=EXPLORER_ALBUM_KEY,
            format_func=album_labels.get,
        )
        selected_album = table_df.loc[table_df["AlbumKey"].eq(selected_album_key)].iloc[0]
        explorer_cols = [
            "Cover",
            "Artist",
            "Album",
            "Released",
            "RatingLabel",
            "RatingNum",
            "Global Rating",
            "RatingDelta",
            "Genres",
            "OriginLabel",
            "Notes",
        ]

        detail_col, table_col = st.columns([0.36, 0.64], gap="large")
        with detail_col:
            render_album_detail(selected_album)
            render_album_assistant(selected_album, selected)
        with table_col:
            st.markdown("**Top Matches**")
            compact_table(table_df.head(25), explorer_cols, height=430)

        with st.expander("Full table"):
            compact_table(table_df, explorer_cols, height=640)


if __name__ == "__main__":
    main()
