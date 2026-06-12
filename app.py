from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).with_name("albums.csv")
RATING_ORDER = ["1", "2", "3", "4", "5", "did-not-listen", "unrated"]


st.set_page_config(
    page_title="Halchemy Album Dashboard",
    page_icon=":notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df["RatingNum"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["RatingStatus"] = df["Rating"].fillna("unrated").astype(str)
    df.loc[df["RatingStatus"].eq("nan"), "RatingStatus"] = "unrated"
    df["GeneratedDate"] = pd.to_datetime(df["Generated Date"], errors="coerce", utc=True).dt.tz_convert(None)
    df["YearAdded"] = df["GeneratedDate"].dt.year
    df["MonthAdded"] = df["GeneratedDate"].dt.to_period("M").dt.to_timestamp()
    df["Decade"] = (df["Released"] // 10 * 10).astype(str) + "s"
    df["OriginLabel"] = df["Origin"].fillna("unknown").str.upper()
    df["RatingDelta"] = df["RatingNum"] - df["Global Rating"]
    df["PrimaryGenre"] = df["Genres"].fillna("unknown").str.split(",").str[0].str.strip()
    df["SearchText"] = (
        df["Artist"].fillna("")
        + " "
        + df["Album"].fillna("")
        + " "
        + df["Notes"].fillna("")
        + " "
        + df["Genres"].fillna("")
    ).str.lower()

    exploded = (
        df.assign(Genre=df["Genres"].fillna("unknown").str.split(","))
        .explode("Genre")
        .assign(Genre=lambda data: data["Genre"].astype(str).str.strip())
    )
    exploded.loc[exploded["Genre"].eq(""), "Genre"] = "unknown"
    return df, exploded


def compact_table(data: pd.DataFrame, cols: list[str], height: int = 330) -> None:
    st.dataframe(
        data[cols],
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config={
            "RatingNum": st.column_config.NumberColumn("Rating", format="%.1f"),
            "Global Rating": st.column_config.NumberColumn("Global", format="%.2f"),
            "RatingDelta": st.column_config.NumberColumn("Delta", format="%+.2f"),
            "Released": st.column_config.NumberColumn("Year", format="%d"),
        },
    )


def filtered_data(df: pd.DataFrame, exploded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.markdown("### Filters")

    query = st.sidebar.text_input("Search", placeholder="artist, album, note, genre").strip().lower()
    all_genres = sorted(exploded["Genre"].dropna().unique())
    genres = st.sidebar.multiselect("Genres", all_genres)
    origins = st.sidebar.multiselect("Origin", sorted(df["OriginLabel"].dropna().unique()))
    decades = st.sidebar.multiselect("Decades", sorted(df["Decade"].unique()))
    statuses = st.sidebar.multiselect("Rating status", RATING_ORDER, default=[])
    year_min, year_max = int(df["Released"].min()), int(df["Released"].max())
    year_range = st.sidebar.slider("Release years", year_min, year_max, (year_min, year_max))

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


def notes_keywords(df: pd.DataFrame) -> pd.DataFrame:
    stopwords = {
        "the",
        "and",
        "was",
        "with",
        "that",
        "this",
        "for",
        "but",
        "are",
        "not",
        "just",
        "like",
        "very",
        "kinda",
        "felt",
        "music",
        "album",
        "song",
        "songs",
        "really",
        "more",
        "good",
    }
    words: Counter[str] = Counter()
    for note in df["Notes"].dropna():
        cleaned = "".join(ch.lower() if ch.isalpha() else " " for ch in str(note))
        words.update(word for word in cleaned.split() if len(word) > 3 and word not in stopwords)
    return pd.DataFrame(words.most_common(25), columns=["Word", "Count"])


def main() -> None:
    df, exploded = load_data(DATA_PATH)
    selected, selected_genres = filtered_data(df, exploded)

    st.title("Halchemy Album Dashboard")
    st.caption(
        "A living map of ratings, genre instincts, era bias, outlier loves, "
        "and albums still waiting for a verdict."
    )

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

    tab_overview, tab_taste, tab_gaps, tab_explorer = st.tabs(
        ["Overview", "Taste Profile", "Global Gap", "Explorer"]
    )

    with tab_overview:
        st.subheader("Catalog Pulse")
        st.caption("How the collection is distributed across ratings, release eras, and entry dates.")

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
            title="Rating distribution",
        )
        fig_rating.update_layout(showlegend=False)
        left.plotly_chart(fig_rating, use_container_width=True)

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
            title="Albums by decade",
        )
        right.plotly_chart(fig_decade, use_container_width=True)

        by_month = selected.dropna(subset=["MonthAdded"]).groupby("MonthAdded", as_index=False).size()
        fig_month = px.area(by_month, x="MonthAdded", y="size", title="Albums added over time")
        fig_month.update_xaxes(title=None)
        fig_month.update_yaxes(title="Albums")
        st.plotly_chart(fig_month, use_container_width=True)

    with tab_taste:
        st.subheader("Taste Profile")
        st.caption("Where your ratings cluster by genre, origin, and era.")

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
            title="Highest-rated genres",
        )
        fig_genre.update_xaxes(range=[0, 5], title="Average rating")
        left.plotly_chart(fig_genre, use_container_width=True)

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
            title="Origin: you vs global",
        )
        fig_origin.update_traces(textposition="top center")
        fig_origin.update_xaxes(range=[2.5, 3.8])
        fig_origin.update_yaxes(range=[1, 5])
        right.plotly_chart(fig_origin, use_container_width=True)

        words = notes_keywords(selected)
        if not words.empty:
            fig_words = px.bar(
                words.sort_values("Count"),
                x="Count",
                y="Word",
                orientation="h",
                title="Most-used note words",
                color="Count",
            )
            st.plotly_chart(fig_words, use_container_width=True)

    with tab_gaps:
        st.subheader("Consensus Breakers")
        st.caption("The albums where your ear most disagrees with the crowd.")

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
            title="Your rating vs global rating",
        )
        fig_gap.add_shape(type="line", x0=1, y0=1, x1=5, y1=5, line=dict(dash="dash"))
        fig_gap.update_yaxes(range=[0.7, 5.3], dtick=1)
        fig_gap.update_xaxes(range=[1.7, 4.6])
        st.plotly_chart(fig_gap, use_container_width=True)

        love, reject = st.columns(2)
        with love:
            st.markdown("**You rate these far above consensus**")
            compact_table(
                gap_df.sort_values("RatingDelta", ascending=False).head(12),
                ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"],
            )
        with reject:
            st.markdown("**Consensus likes these more than you do**")
            compact_table(
                gap_df.sort_values("RatingDelta").head(12),
                ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"],
            )

    with tab_explorer:
        st.subheader("Album Explorer")
        st.caption("Search, sort, and inspect notes without leaving the dashboard.")

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
        st.dataframe(
            table_df[
                [
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
            ],
            use_container_width=True,
            hide_index=True,
            height=640,
            column_config={
                "RatingStatus": st.column_config.TextColumn("Status"),
                "RatingNum": st.column_config.NumberColumn("Rating", format="%.1f"),
                "Global Rating": st.column_config.NumberColumn("Global", format="%.2f"),
                "RatingDelta": st.column_config.NumberColumn("Delta", format="%+.2f"),
                "OriginLabel": st.column_config.TextColumn("Origin"),
                "Released": st.column_config.NumberColumn("Year", format="%d"),
                "Notes": st.column_config.TextColumn("Notes", width="large"),
            },
        )


if __name__ == "__main__":
    main()
