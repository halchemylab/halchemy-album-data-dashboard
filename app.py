from __future__ import annotations

from html import escape
from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from album_agent import AgentAnswer, answer_question, answer_question_with_openai
from album_data import AlbumDataError, RATING_LABEL_MAP, RATING_ORDER, load_data, notes_keywords
from album_memory import build_agent_memory, ensure_agent_memory, save_agent_memory


DATA_PATH = Path(__file__).with_name("albums.csv")
RATING_COLOR_MAP = {
    "1": "#d73027",
    "2": "#f46d43",
    "3": "#fdae61",
    "4": "#4575b4",
    "5": "#1a9850",
    "did-not-listen": "#8c8c8c",
    "unrated": "#c9c9c9",
}
CHART_HEIGHT = 390
WIDE_CHART_HEIGHT = 340
FILTER_SEARCH_KEY = "filter_search"
FILTER_GENRES_KEY = "filter_genres"
FILTER_ORIGINS_KEY = "filter_origins"
FILTER_DECADES_KEY = "filter_decades"
FILTER_STATUSES_KEY = "filter_statuses"
FILTER_YEAR_RANGE_KEY = "filter_year_range"
EXPLORER_ALBUM_KEY = "explorer_album_key"
AGENT_QUESTION_KEY = "agent_question"
AGENT_HISTORY_KEY = "agent_history"
AGENT_CONTEXT_KEY = "agent_context"
AGENT_PIN_CONTEXT_KEY = "agent_pin_context"


st.set_page_config(
    page_title="Halchemy Album Dashboard",
    page_icon=":notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_data(path)


def optional_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def compact_table(data: pd.DataFrame, cols: list[str], height: int = 330) -> None:
    st.dataframe(
        data[cols],
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config={
            "Album": st.column_config.TextColumn("Album", width="medium"),
            "Cover": st.column_config.ImageColumn("Cover", width="small"),
            "Artist": st.column_config.TextColumn("Artist", width="medium"),
            "RatingNum": st.column_config.NumberColumn("Rating", format="%.1f"),
            "Global Rating": st.column_config.NumberColumn("Global", format="%.2f"),
            "RatingDelta": st.column_config.NumberColumn("Delta", format="%+.2f"),
            "Released": st.column_config.NumberColumn("Year", format="%d"),
            "Genres": st.column_config.TextColumn("Genres", width="medium"),
            "OriginLabel": st.column_config.TextColumn("Origin"),
            "RatingStatus": st.column_config.TextColumn("Status"),
            "RatingLabel": st.column_config.TextColumn("Personal Label"),
            "Albums": st.column_config.NumberColumn("Albums", format="%d"),
            "AvgRating": st.column_config.NumberColumn("Avg Rating", format="%.2f"),
            "AvgGlobal": st.column_config.NumberColumn("Avg Global", format="%.2f"),
            "Delta": st.column_config.NumberColumn("Delta", format="%+.2f"),
        },
    )


def polish_chart(fig, *, height: int = CHART_HEIGHT, x_title: str | None = None, y_title: str | None = None):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        legend_title_text=None,
    )
    if x_title is not None:
        fig.update_xaxes(title=x_title)
    if y_title is not None:
        fig.update_yaxes(title=y_title)
    return fig


def reset_filter_state(year_min: int, year_max: int) -> None:
    st.session_state[FILTER_SEARCH_KEY] = ""
    st.session_state[FILTER_GENRES_KEY] = []
    st.session_state[FILTER_ORIGINS_KEY] = []
    st.session_state[FILTER_DECADES_KEY] = []
    st.session_state[FILTER_STATUSES_KEY] = []
    st.session_state[FILTER_YEAR_RANGE_KEY] = (year_min, year_max)


def ensure_filter_defaults(year_min: int, year_max: int) -> None:
    defaults = {
        FILTER_SEARCH_KEY: "",
        FILTER_GENRES_KEY: [],
        FILTER_ORIGINS_KEY: [],
        FILTER_DECADES_KEY: [],
        FILTER_STATUSES_KEY: [],
        FILTER_YEAR_RANGE_KEY: (year_min, year_max),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    current_min, current_max = st.session_state[FILTER_YEAR_RANGE_KEY]
    clamped_min = max(year_min, min(year_max, current_min))
    clamped_max = max(year_min, min(year_max, current_max))
    if (clamped_min, clamped_max) != (current_min, current_max):
        st.session_state[FILTER_YEAR_RANGE_KEY] = (clamped_min, clamped_max)


def rating_display_label(status: str) -> str:
    label = RATING_LABEL_MAP.get(status, status)
    if status in {"did-not-listen", "unrated"}:
        return label
    return f"{label} ({status})"


def active_filter_labels(
    *,
    query: str,
    genres: list[str],
    origins: list[str],
    decades: list[str],
    statuses: list[str],
    year_range: tuple[int, int],
    full_year_range: tuple[int, int],
) -> list[str]:
    labels: list[str] = []
    if query:
        labels.append(f'Search: "{query}"')
    if genres:
        labels.append("Genres: " + ", ".join(genres))
    if origins:
        labels.append("Origin: " + ", ".join(origins))
    if decades:
        labels.append("Decades: " + ", ".join(decades))
    if statuses:
        labels.append("Status: " + ", ".join(rating_display_label(status) for status in statuses))
    if year_range != full_year_range:
        labels.append(f"Years: {year_range[0]}-{year_range[1]}")
    return labels


def render_active_filters(labels: list[str], year_min: int, year_max: int) -> None:
    left, right = st.columns([0.82, 0.18], vertical_alignment="center")
    with left:
        st.markdown("**Active Filters**")
        if labels:
            chips = " ".join(f"<span class='filter-chip'>{escape(label)}</span>" for label in labels)
            st.markdown(f"<div class='filter-strip'>{chips}</div>", unsafe_allow_html=True)
        else:
            st.caption("All albums are included.")
    with right:
        st.button(
            "Clear filters",
            disabled=not labels,
            use_container_width=True,
            on_click=reset_filter_state,
            args=(year_min, year_max),
        )


def render_rating_key(statuses: pd.Series) -> None:
    present = [status for status in RATING_ORDER if status in set(statuses)]
    swatches = " ".join(
        "<span class='rating-key-item'>"
        f"<span class='rating-key-dot' style='background:{RATING_COLOR_MAP[status]}'></span>"
        f"{escape(rating_display_label(status))}</span>"
        for status in present
    )
    st.markdown(f"<div class='rating-key'>{swatches}</div>", unsafe_allow_html=True)


def album_key(album: pd.Series) -> str:
    return f"{album['Artist']}\0{album['Album']}\0{album['Released']}"


def album_selector_label(album: pd.Series) -> str:
    return f"{album['Artist']} - {album['Album']} ({album['Released']})"


def display_text(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "-"
    return str(value)


def display_number(value: object, precision: int = 2, signed: bool = False) -> str:
    if pd.isna(value):
        return "-"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{precision}f}"


def display_date(value: object) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def detail_row(label: str, value: object) -> None:
    st.markdown(
        "<div class='detail-row'>"
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(display_text(value))}</strong>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_album_detail(album: pd.Series) -> None:
    rating_label = display_text(album["RatingLabel"]).title()
    if not pd.isna(album["RatingNum"]):
        rating_label = f"{rating_label} ({display_number(album['RatingNum'], precision=1)})"

    delta = album["RatingDelta"]
    if pd.isna(delta):
        delta_class = "neutral"
    elif delta > 0:
        delta_class = "positive"
    elif delta < 0:
        delta_class = "negative"
    else:
        delta_class = "neutral"

    st.markdown("**Selected Album**")
    st.image(album["Cover"], width=128)
    st.markdown(f"### {escape(display_text(album['Album']))}", unsafe_allow_html=True)
    st.caption(f"{display_text(album['Artist'])} - {display_text(album['Released'])}")

    personal, global_rating, gap = st.columns(3)
    personal.metric("Personal", rating_label)
    global_rating.metric("Consensus", display_number(album["Global Rating"]))
    with gap:
        st.markdown(
            f"<div class='gap-metric {delta_class}'>"
            "<span>Gap</span>"
            f"<strong>{escape(display_number(delta, signed=True))}</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

    detail_row("Genres", album["Genres"])
    detail_row("Origin", album["OriginLabel"])
    detail_row("Added", display_date(album["GeneratedDate"]))

    st.markdown("**Notes**")
    notes = display_text(album["Notes"])
    if notes == "-":
        st.caption("No notes yet.")
    else:
        st.write(notes)


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
            st.session_state[AGENT_CONTEXT_KEY] = None
            st.session_state[AGENT_PIN_CONTEXT_KEY] = False
            st.rerun()


def render_followup_buttons() -> None:
    if not st.session_state.get(AGENT_CONTEXT_KEY):
        return
    followups = [
        "Show more like this",
        "Why this?",
        "Only unrated albums",
        "Compare against my overall taste",
    ]
    st.markdown("**Suggested follow-ups**")
    cols = st.columns(len(followups))
    for col, followup in zip(cols, followups):
        if col.button(followup, use_container_width=True):
            st.session_state[AGENT_QUESTION_KEY] = followup


def render_agent_trace(answer: AgentAnswer) -> None:
    if not answer.trace:
        return
    with st.expander("Agent plan and tool trace", expanded=False):
        for index, step in enumerate(answer.trace, start=1):
            st.markdown(f"**{index}. {escape(step.phase)}**")
            st.caption(step.detail)


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
    st.markdown("**Durable Taste Memory**")
    st.caption(
        f"{catalog.get('rated', 0):,} rated albums, "
        f"{catalog.get('unrated', 0):,} unresolved albums, "
        f"{catalog.get('genres', 0):,} genre signals."
    )

    with st.expander("Memory snapshot", expanded=False):
        favorite_genres = memory.get("favorite_genres", [])
        unresolved = memory.get("unresolved_queue", [])
        if isinstance(favorite_genres, list) and favorite_genres:
            st.markdown("**Favorite genre signals**")
            st.dataframe(pd.DataFrame(favorite_genres), use_container_width=True, hide_index=True, height=190)
        if isinstance(unresolved, list) and unresolved:
            st.markdown("**Unresolved listening queue**")
            st.dataframe(pd.DataFrame(unresolved), use_container_width=True, hide_index=True, height=220)


def render_agent(
    selected: pd.DataFrame,
    selected_genres: pd.DataFrame,
    active_filters: list[str],
    full_catalog: pd.DataFrame,
    full_genres: pd.DataFrame,
) -> None:
    st.subheader("Album Agent")
    st.caption("Ask about the currently filtered albums. The agent chooses a data skill, runs it, and explains the result.")

    memory = ensure_agent_memory(full_catalog, full_genres)
    st.session_state.setdefault(AGENT_HISTORY_KEY, [])
    st.session_state.setdefault(AGENT_CONTEXT_KEY, None)
    st.session_state.setdefault(AGENT_PIN_CONTEXT_KEY, False)
    render_agent_scope(selected, selected_genres, active_filters)
    refresh_memory = st.button("Refresh durable memory", use_container_width=False)
    if refresh_memory:
        memory = build_agent_memory(full_catalog, full_genres)
        save_agent_memory(memory)
        st.rerun()
    render_agent_memory(memory)
    render_agent_context_controls()

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

    examples = [
        "What should I listen to next?",
        "Build a 3-album starter pack",
        "What genres do I rate highest?",
        "Create a one-page taste report",
    ]
    example_cols = st.columns(len(examples))
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state[AGENT_QUESTION_KEY] = example
    render_followup_buttons()

    with st.form("agent_form", clear_on_submit=False):
        question = st.text_input(
            "Ask a question",
            key=AGENT_QUESTION_KEY,
            placeholder="Try: recommend a rock album from the 1970s",
        )
        submitted = st.form_submit_button("Ask agent", use_container_width=True)

    if submitted and question.strip():
        try:
            if use_openai:
                answer = answer_question_with_openai(
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
            else:
                answer = answer_question(
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
            answer = fallback.__class__(
                question=fallback.question,
                summary=f"{fallback.summary}\n\nOpenAI routing was unavailable, so this answer used the local skill router. Error: {exc}",
                detail=fallback.detail,
                skill=fallback.skill,
                mode="deterministic fallback",
                trace=fallback.trace,
                dashboard_action=fallback.dashboard_action,
            )
        st.session_state[AGENT_HISTORY_KEY].insert(0, answer)
        st.session_state[AGENT_HISTORY_KEY] = st.session_state[AGENT_HISTORY_KEY][:6]
        if not st.session_state.get(AGENT_PIN_CONTEXT_KEY):
            st.session_state[AGENT_CONTEXT_KEY] = make_agent_context(answer)
        if apply_agent_dashboard_action(answer, full_catalog, full_genres):
            st.rerun()

    if not st.session_state[AGENT_HISTORY_KEY]:
        st.info(
            "The agent can recommend albums, summarize genre patterns, compare you against consensus, "
            "build listening paths, search notes, and produce capstone-ready insights."
        )
        return

    for answer in st.session_state[AGENT_HISTORY_KEY]:
        with st.container(border=True):
            st.markdown(f"**You:** {escape(answer.question)}")
            st.write(answer.summary)
            st.caption(f"Skill used: {answer.skill} | Mode: {answer.mode}")
            render_agent_trace(answer)
            if not answer.detail.empty:
                compact_table(answer.detail, answer.detail.columns.tolist(), height=260)


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


def filtered_data(df: pd.DataFrame, exploded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    st.sidebar.markdown("### Filters")

    year_min, year_max = int(df["Released"].min()), int(df["Released"].max())
    ensure_filter_defaults(year_min, year_max)

    query = st.sidebar.text_input(
        "Search",
        placeholder="artist, album, note, genre",
        key=FILTER_SEARCH_KEY,
    ).strip().lower()
    all_genres = sorted(exploded["Genre"].dropna().unique())
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

    selected, selected_genres, active_filters = filtered_data(df, exploded)

    st.title("Halchemy Album Dashboard")
    st.caption(
        "Start with the catalog, follow the taste patterns, then drill into the albums worth inspecting."
    )
    st.markdown(
        """
        <style>
        .filter-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: -0.35rem;
        }
        .filter-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            background: rgba(49, 51, 63, 0.04);
            color: rgb(49, 51, 63);
            font-size: 0.85rem;
            line-height: 1.35;
        }
        .rating-key {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin: -0.25rem 0 0.85rem;
            color: rgba(49, 51, 63, 0.76);
            font-size: 0.84rem;
        }
        .rating-key-item {
            display: inline-flex;
            align-items: center;
            gap: 0.28rem;
            white-space: nowrap;
        }
        .rating-key-dot {
            width: 0.62rem;
            height: 0.62rem;
            border-radius: 999px;
            box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.12);
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid rgba(49, 51, 63, 0.12);
            padding: 0.45rem 0;
            font-size: 0.92rem;
        }
        .detail-row span {
            color: rgba(49, 51, 63, 0.66);
        }
        .detail-row strong {
            max-width: 68%;
            text-align: right;
            overflow-wrap: anywhere;
        }
        .gap-metric {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
            min-height: 4.5rem;
            justify-content: center;
        }
        .gap-metric span {
            color: rgba(49, 51, 63, 0.66);
            font-size: 0.88rem;
        }
        .gap-metric strong {
            font-size: 1.85rem;
            line-height: 1.15;
        }
        .gap-metric.positive strong {
            color: #1a9850;
        }
        .gap-metric.negative strong {
            color: #d73027;
        }
        .gap-metric.neutral strong {
            color: #8c8c8c;
        }
        @media (prefers-color-scheme: dark) {
            .filter-chip {
                border-color: rgba(250, 250, 250, 0.2);
                background: rgba(250, 250, 250, 0.08);
                color: rgb(250, 250, 250);
            }
            .rating-key {
                color: rgba(250, 250, 250, 0.78);
            }
            .detail-row {
                border-bottom-color: rgba(250, 250, 250, 0.16);
            }
            .detail-row span,
            .gap-metric span {
                color: rgba(250, 250, 250, 0.68);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if selected.empty:
        render_active_filters(active_filters, int(df["Released"].min()), int(df["Released"].max()))
        st.warning("No albums match the current filters.")
        return

    render_active_filters(active_filters, int(df["Released"].min()), int(df["Released"].max()))

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

    render_rating_key(selected["RatingStatus"])

    tab_overview, tab_soundprint, tab_taste, tab_gaps, tab_explorer, tab_agent = st.tabs(
        ["Catalog", "Soundprint", "Taste", "Outliers", "Explorer", "Agent"]
    )

    with tab_overview:
        st.subheader("Catalog")

        left, right = st.columns([1.05, 1])
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

    with tab_soundprint:
        st.subheader("Your Soundprint")
        st.caption("Personal patterns from the current catalog slice.")
        render_soundprint(selected, selected_genres)

    with tab_taste:
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

    with tab_gaps:
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

    with tab_explorer:
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
        with table_col:
            st.markdown("**Top Matches**")
            compact_table(table_df.head(25), explorer_cols, height=430)

        with st.expander("Full table"):
            compact_table(table_df, explorer_cols, height=640)

    with tab_agent:
        render_agent(selected, selected_genres, active_filters, df, exploded)


if __name__ == "__main__":
    main()
