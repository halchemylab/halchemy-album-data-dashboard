from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from album_data import AlbumDataError, RATING_ORDER, load_data, notes_keywords


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


st.set_page_config(
    page_title="Halchemy Album Dashboard",
    page_icon=":notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_data(path)


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
        labels.append("Status: " + ", ".join(statuses))
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
        f"{escape(status)}</span>"
        for status in present
    )
    st.markdown(f"<div class='rating-key'>{swatches}</div>", unsafe_allow_html=True)


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
    statuses = st.sidebar.multiselect("Rating status", RATING_ORDER, key=FILTER_STATUSES_KEY)
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
        @media (prefers-color-scheme: dark) {
            .filter-chip {
                border-color: rgba(250, 250, 250, 0.2);
                background: rgba(250, 250, 250, 0.08);
                color: rgb(250, 250, 250);
            }
            .rating-key {
                color: rgba(250, 250, 250, 0.78);
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

    tab_overview, tab_soundprint, tab_taste, tab_gaps, tab_explorer = st.tabs(
        ["Catalog", "Soundprint", "Taste", "Outliers", "Explorer"]
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
        rating_counts.columns = ["Rating", "Albums"]
        fig_rating = px.bar(
            rating_counts,
            x="Rating",
            y="Albums",
            color="Rating",
            title="Rating mix",
            category_orders={"Rating": RATING_ORDER},
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
                "RatingStatus": True,
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
        table_df = selected.sort_values(sort_col, ascending=ascending, na_position="last")
        explorer_cols = [
            "Cover",
            "Artist",
            "Album",
            "Released",
            "RatingStatus",
            "RatingNum",
            "Global Rating",
            "RatingDelta",
            "Genres",
            "OriginLabel",
            "Notes",
        ]

        st.markdown("**Top Matches**")
        compact_table(table_df.head(25), explorer_cols, height=430)

        with st.expander("Full table"):
            compact_table(table_df, explorer_cols, height=640)


if __name__ == "__main__":
    main()
