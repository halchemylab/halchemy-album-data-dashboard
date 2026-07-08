from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st


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
        f"<div class='detail-row'><span>{escape(label)}</span><strong>{escape(display_text(value))}</strong></div>",
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


def genre_set(value: object) -> set[str]:
    return {genre.strip().casefold() for genre in str(value or "").split(",") if genre.strip()}
