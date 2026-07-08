from __future__ import annotations

import pandas as pd
import streamlit as st

from agent import ProactivePrompt, build_proactive_prompt
from album_data import RATING_ORDER, notes_keywords
from album_memory import ensure_agent_memory


@st.cache_data(show_spinner=False)
def filtered_catalog(
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    *,
    query: str,
    genres: tuple[str, ...],
    origins: tuple[str, ...],
    decades: tuple[str, ...],
    statuses: tuple[str, ...],
    year_range: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return selected, exploded.loc[exploded_keys.isin(selected_keys)].copy()


@st.cache_data(show_spinner=False)
def cached_agent_memory(df: pd.DataFrame, exploded: pd.DataFrame) -> dict[str, object]:
    return ensure_agent_memory(df, exploded)


@st.cache_data(show_spinner=False)
def cached_notes_keywords(df: pd.DataFrame) -> pd.DataFrame:
    return notes_keywords(df)


@st.cache_data(show_spinner=False)
def cached_catalog_tab_data(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    rating_counts = (
        selected["RatingStatus"]
        .value_counts()
        .reindex([rating for rating in RATING_ORDER if rating in selected["RatingStatus"].unique()])
        .reset_index()
    )
    rating_counts.columns = ["RatingStatus", "Albums"]
    decade = (
        selected.groupby("Decade", as_index=False)
        .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
        .sort_values("Decade")
    )
    timeline = selected.sort_values(["Released", "Artist", "Album"]).copy()
    timeline["TimelineSize"] = timeline["RatingNum"].fillna(2.5) + 3
    decade_order = sorted(
        timeline["Decade"].dropna().unique(),
        key=lambda value: int(str(value).rstrip("s")),
    )
    by_month = selected.dropna(subset=["MonthAdded"]).groupby("MonthAdded", as_index=False).size()
    return rating_counts, decade, timeline, decade_order, by_month


@st.cache_data(show_spinner=False)
def cached_taste_tab_data(selected: pd.DataFrame, selected_genres: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    genre_summary = (
        selected_genres.groupby("Genre", as_index=False)
        .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
        .query("Albums >= 3")
        .sort_values(["AvgRating", "Albums"], ascending=[False, False])
    )
    genre_summary["Delta"] = genre_summary["AvgRating"] - genre_summary["AvgGlobal"]
    origin = (
        selected.groupby("OriginLabel", as_index=False)
        .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
        .sort_values("Albums", ascending=False)
    )
    return genre_summary, origin


@st.cache_data(show_spinner=False)
def cached_explorer_table(selected: pd.DataFrame, sort_col: str, ascending: bool) -> pd.DataFrame:
    table = selected.sort_values(sort_col, ascending=ascending, na_position="last").copy()
    table["AlbumKey"] = (
        table["Artist"].astype(str) + "\0" + table["Album"].astype(str) + "\0" + table["Released"].astype(str)
    )
    return table


@st.cache_data(show_spinner=False)
def cached_proactive_prompt(
    selected: pd.DataFrame,
    selected_genres: pd.DataFrame,
    *,
    memory: dict[str, object],
    context: dict[str, object] | None,
) -> ProactivePrompt | None:
    return build_proactive_prompt(selected, selected_genres, memory=memory, context=context)
