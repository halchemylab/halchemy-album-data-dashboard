from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from album_data import RATING_ORDER
from ui.filters import rating_display_label

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


def render_rating_key(statuses: pd.Series) -> None:
    present = [status for status in RATING_ORDER if status in set(statuses)]
    swatches = " ".join(
        "<span class='rating-key-item'>"
        f"<span class='rating-key-dot' style='background:{RATING_COLOR_MAP[status]}'></span>"
        f"{escape(rating_display_label(status))}</span>"
        for status in present
    )
    st.markdown(f"<div class='rating-key'>{swatches}</div>", unsafe_allow_html=True)
