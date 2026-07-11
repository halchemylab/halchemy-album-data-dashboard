from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from album_data import RATING_ORDER, AlbumDataError, load_data
from ui.assistant import render_album_assistant, render_sidebar_assistant
from ui.charts import (
    RATING_COLOR_MAP,
    WIDE_CHART_HEIGHT,
    compact_table,
    polish_chart,
    render_rating_key,
)
from ui.derived import (
    cached_catalog_tab_data,
    cached_explorer_table,
    cached_notes_keywords,
    cached_taste_tab_data,
    filtered_catalog,
)
from ui.explorer import (
    album_selector_label,
    render_album_detail,
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
from ui.styles import render_global_styles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA_PATH = PROJECT_ROOT / "albums.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_albums.csv"
DATA_PATH = PRIVATE_DATA_PATH if PRIVATE_DATA_PATH.exists() else SAMPLE_DATA_PATH
EXPLORER_ALBUM_KEY = "explorer_album_key"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


@st.cache_data(show_spinner=False)
def cached_load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_data(path)


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
        st.metric(
            "Consensus Bias",
            f"{gap.mean():+.2f}" if not gap.empty else "-",
            help="Average personal rating minus global rating",
        )

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
    selected, selected_genres = filtered_catalog(
        df,
        exploded,
        query=query,
        genres=tuple(genres),
        origins=tuple(origins),
        decades=tuple(decades),
        statuses=tuple(statuses),
        year_range=(int(year_range[0]), int(year_range[1])),
    )
    labels = active_filter_labels(
        query=query,
        genres=genres,
        origins=origins,
        decades=decades,
        statuses=statuses,
        year_range=year_range,
        full_year_range=(year_min, year_max),
    )
    return selected, selected_genres, labels


def filtered_data(df: pd.DataFrame, exploded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    active_filter_container = st.sidebar.container()

    year_min, year_max = int(df["Released"].min()), int(df["Released"].max())
    all_genres = sorted(exploded["Genre"].dropna().unique())
    query = (
        st.sidebar.text_input(
            "Search",
            placeholder="artist, album, note, genre",
            key=FILTER_SEARCH_KEY,
        )
        .strip()
        .lower()
    )
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

    selected, selected_genres = filtered_catalog(
        df,
        exploded,
        query=query,
        genres=tuple(genres),
        origins=tuple(origins),
        decades=tuple(decades),
        statuses=tuple(statuses),
        year_range=(int(year_range[0]), int(year_range[1])),
    )
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
    return selected, selected_genres, labels


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
    st.caption("Start with the catalog, follow the taste patterns, then drill into the albums worth inspecting.")
    render_global_styles()

    if selected.empty:
        st.warning("No albums match the current filters.")
        return

    rated = selected["RatingNum"].dropna()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Albums", f"{len(selected):,}", help="Filtered catalog size")
    with c2:
        st.metric(
            "Rated", f"{rated.count():,}", f"{selected['RatingStatus'].eq('did-not-listen').sum():,} did not listen"
        )
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
        rating_counts, decade, timeline_df, decade_order, by_month = cached_catalog_tab_data(selected)
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
        left.plotly_chart(
            polish_chart(fig_rating, x_title=None, y_title="Albums"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        fig_decade = px.bar(
            decade,
            x="Decade",
            y="Albums",
            color="AvgRating",
            title="Era spread",
        )
        right.plotly_chart(
            polish_chart(fig_decade, x_title=None, y_title="Albums"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
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
            render_mode="webgl",
        )
        fig_release_timeline.update_traces(marker=dict(line=dict(width=0.7, color="rgba(49, 51, 63, 0.35)")))
        fig_release_timeline.update_xaxes(dtick=5)
        st.plotly_chart(
            polish_chart(fig_release_timeline, height=430, x_title="Release year", y_title=None),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        fig_month = px.area(by_month, x="MonthAdded", y="size", title="Listening timeline")
        st.plotly_chart(
            polish_chart(fig_month, height=WIDE_CHART_HEIGHT, x_title=None, y_title="Albums"),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    elif section == "Soundprint":
        st.subheader("Your Soundprint")
        st.caption("Personal patterns from the current catalog slice.")
        render_soundprint(selected, selected_genres)

    elif section == "Taste":
        st.subheader("Taste")

        genre_summary, origin = cached_taste_tab_data(selected, selected_genres)

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

        words = cached_notes_keywords(selected)
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
        table_df = cached_explorer_table(selected, sort_col, ascending)
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
