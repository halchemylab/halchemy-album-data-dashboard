from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).with_name("albums.csv")
REQUIRED_COLUMNS = [
    "Artist",
    "Album",
    "Released",
    "Rating",
    "Notes",
    "Global Rating",
    "Genres",
    "Origin",
    "Generated Date",
]
RATING_ORDER = ["1", "2", "3", "4", "5", "did-not-listen", "unrated"]
CHART_HEIGHT = 390
WIDE_CHART_HEIGHT = 340


st.set_page_config(
    page_title="Halchemy Album Dashboard",
    page_icon=":notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)


class AlbumDataError(ValueError):
    """Raised when albums.csv cannot be safely used by the dashboard."""


def clean_text_status(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def row_numbers(mask: pd.Series) -> str:
    rows = (mask[mask].index + 2).astype(str).tolist()
    if len(rows) <= 8:
        return ", ".join(rows)
    return ", ".join(rows[:8]) + f", and {len(rows) - 8} more"


def normalize_rating_status(series: pd.Series) -> pd.Series:
    status = clean_text_status(series).str.lower()
    numeric = pd.to_numeric(status, errors="coerce")
    numeric_rating = numeric.notna() & numeric.between(1, 5) & numeric.mod(1).eq(0)
    status.loc[numeric_rating] = numeric.loc[numeric_rating].astype(int).astype(str)
    status.loc[status.eq("")] = "unrated"
    return status


def validate_albums_csv(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return errors

    artist_blank = clean_text_status(df["Artist"]).eq("")
    if artist_blank.any():
        errors.append(f"Artist is blank on row(s): {row_numbers(artist_blank)}")

    album_blank = clean_text_status(df["Album"]).eq("")
    if album_blank.any():
        errors.append(f"Album is blank on row(s): {row_numbers(album_blank)}")

    released = pd.to_numeric(df["Released"], errors="coerce")
    released_invalid = released.isna() | released.lt(0) | released.mod(1).ne(0)
    if released_invalid.any():
        errors.append(f"Released must be a whole year on row(s): {row_numbers(released_invalid)}")

    global_raw = clean_text_status(df["Global Rating"])
    global_rating = pd.to_numeric(global_raw, errors="coerce")
    global_invalid = global_raw.ne("") & global_rating.isna()
    if global_invalid.any():
        errors.append(f"Global Rating must be numeric when present on row(s): {row_numbers(global_invalid)}")

    generated_date = pd.to_datetime(df["Generated Date"], errors="coerce", utc=True)
    generated_invalid = generated_date.isna()
    if generated_invalid.any():
        errors.append(f"Generated Date must be parseable on row(s): {row_numbers(generated_invalid)}")

    rating_raw = clean_text_status(df["Rating"]).str.lower()
    numeric_rating = pd.to_numeric(rating_raw, errors="coerce")
    rating_valid = (
        rating_raw.eq("")
        | rating_raw.eq("did-not-listen")
        | (numeric_rating.notna() & numeric_rating.between(1, 5) & numeric_rating.mod(1).eq(0))
    )
    rating_invalid = ~rating_valid
    if rating_invalid.any():
        errors.append(
            "Rating must be 1-5, did-not-listen, or blank on row(s): "
            + row_numbers(rating_invalid)
        )

    return errors


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    errors = validate_albums_csv(df)
    if errors:
        raise AlbumDataError("\n".join(errors))

    df["Released"] = pd.to_numeric(df["Released"], errors="raise").astype(int)
    df["Global Rating"] = pd.to_numeric(df["Global Rating"], errors="coerce")
    df["RatingStatus"] = normalize_rating_status(df["Rating"])
    df["RatingNum"] = pd.to_numeric(df["RatingStatus"], errors="coerce")
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
            "Album": st.column_config.TextColumn("Album", width="medium"),
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
    try:
        df, exploded = load_data(DATA_PATH)
    except AlbumDataError as exc:
        st.error("albums.csv has data issues.")
        st.markdown("Fix these rows, then rerun the dashboard:")
        for message in str(exc).splitlines():
            st.write(f"- {message}")
        st.stop()

    selected, selected_genres = filtered_data(df, exploded)

    st.title("Halchemy Album Dashboard")
    st.caption(
        "Start with the catalog, follow the taste patterns, then drill into the albums worth inspecting."
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

    tab_overview, tab_taste, tab_gaps, tab_explorer = st.tabs(["Catalog", "Taste", "Outliers", "Explorer"])

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

        by_month = selected.dropna(subset=["MonthAdded"]).groupby("MonthAdded", as_index=False).size()
        fig_month = px.area(by_month, x="MonthAdded", y="size", title="Listening timeline")
        st.plotly_chart(
            polish_chart(fig_month, height=WIDE_CHART_HEIGHT, x_title=None, y_title="Albums"),
            use_container_width=True,
        )

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
