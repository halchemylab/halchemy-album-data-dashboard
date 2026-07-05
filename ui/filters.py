from __future__ import annotations

from html import escape

import streamlit as st

from album_data import RATING_LABEL_MAP


FILTER_SEARCH_KEY = "filter_search"
FILTER_GENRES_KEY = "filter_genres"
FILTER_ORIGINS_KEY = "filter_origins"
FILTER_DECADES_KEY = "filter_decades"
FILTER_STATUSES_KEY = "filter_statuses"
FILTER_YEAR_RANGE_KEY = "filter_year_range"


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


def render_active_filters(labels: list[str], year_min: int, year_max: int, container=st.sidebar) -> None:
    container.markdown("### Filters")
    if labels:
        chips = " ".join(f"<span class='filter-chip'>{escape(label)}</span>" for label in labels)
        container.markdown(f"<div class='filter-strip'>{chips}</div>", unsafe_allow_html=True)
    else:
        container.caption("All albums are included.")
    container.button(
        "Clear filters",
        disabled=not labels,
        use_container_width=True,
        on_click=reset_filter_state,
        args=(year_min, year_max),
    )
