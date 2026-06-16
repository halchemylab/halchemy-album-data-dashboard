from __future__ import annotations

import base64
import hashlib
from collections import Counter
from html import escape
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
COVER_PALETTES = [
    ("#264653", "#2a9d8f", "#e9c46a"),
    ("#1d3557", "#457b9d", "#f1faee"),
    ("#3a0ca3", "#7209b7", "#f72585"),
    ("#283618", "#606c38", "#fefae0"),
    ("#14213d", "#fca311", "#e5e5e5"),
    ("#2b2d42", "#8d99ae", "#edf2f4"),
    ("#5f0f40", "#9a031e", "#fb8b24"),
    ("#0f4c5c", "#e36414", "#f6f1d1"),
]


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


def short_cover_line(text: object, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def album_cover_data_uri(artist: object, album: object, genre: object) -> str:
    key = f"{artist}|{album}|{genre}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    bg, accent, text = COVER_PALETTES[digest[0] % len(COVER_PALETTES)]
    angle = digest[1] % 360
    ring_x = 22 + digest[2] % 58
    ring_y = 18 + digest[3] % 64
    album_line = escape(short_cover_line(album, 18))
    artist_line = escape(short_cover_line(artist, 17))
    genre_line = escape(short_cover_line(genre, 18).upper())
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
<defs>
<linearGradient id="g" gradientTransform="rotate({angle})">
<stop offset="0" stop-color="{bg}"/>
<stop offset="1" stop-color="{accent}"/>
</linearGradient>
</defs>
<rect width="96" height="96" rx="12" fill="url(#g)"/>
<circle cx="{ring_x}" cy="{ring_y}" r="26" fill="none" stroke="{text}" stroke-opacity=".22" stroke-width="11"/>
<path d="M0 72 C24 58 42 86 96 64 L96 96 L0 96 Z" fill="{text}" fill-opacity=".13"/>
<text x="9" y="45" fill="{text}" font-family="Arial, sans-serif" font-size="11" font-weight="700">{album_line}</text>
<text x="9" y="61" fill="{text}" font-family="Arial, sans-serif" font-size="9" opacity=".86">{artist_line}</text>
<text x="9" y="82" fill="{text}" font-family="Arial, sans-serif" font-size="7" opacity=".74">{genre_line}</text>
</svg>"""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


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
    df["Cover"] = [
        album_cover_data_uri(row.Artist, row.Album, row.PrimaryGenre)
        for row in df[["Artist", "Album", "PrimaryGenre"]].itertuples(index=False)
    ]
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
