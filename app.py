from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).with_name("albums.csv")
RATING_ORDER = ["1", "2", "3", "4", "5", "did-not-listen", "unrated"]
PALETTE = {
    "paper": "#f7f3ea",
    "ink": "#171614",
    "muted": "#6f675a",
    "line": "#ded4c2",
    "gold": "#c88f2d",
    "coral": "#d66a50",
    "teal": "#287c78",
    "plum": "#6b4c7c",
    "green": "#587d46",
    "blue": "#376a94",
}


st.set_page_config(
    page_title="Halchemy Album Dashboard",
    page_icon=":notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --paper: {PALETTE["paper"]};
            --ink: {PALETTE["ink"]};
            --muted: {PALETTE["muted"]};
            --line: {PALETTE["line"]};
            --gold: {PALETTE["gold"]};
            --coral: {PALETTE["coral"]};
            --teal: {PALETTE["teal"]};
            --plum: {PALETTE["plum"]};
            --green: {PALETTE["green"]};
            --blue: {PALETTE["blue"]};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 18% 0%, rgba(200, 143, 45, 0.18), transparent 28rem),
                radial-gradient(circle at 86% 14%, rgba(40, 124, 120, 0.16), transparent 30rem),
                linear-gradient(180deg, #fbf8ef 0%, #f2eadc 100%);
            color: var(--ink);
        }}

        section[data-testid="stSidebar"] {{
            background: rgba(247, 243, 234, 0.86);
            border-right: 1px solid rgba(23, 22, 20, 0.1);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--ink);
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2.8rem;
            max-width: 1540px;
        }}

        .hero {{
            position: relative;
            overflow: hidden;
            min-height: 260px;
            padding: 2.15rem;
            border: 1px solid rgba(23, 22, 20, 0.12);
            background:
                linear-gradient(135deg, rgba(23,22,20,0.94), rgba(52,48,42,0.9)),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.04) 0 1px, transparent 1px 42px);
            box-shadow: 0 24px 70px rgba(63, 52, 36, 0.18);
        }}

        .hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(200,143,45,0.28), transparent 26%),
                repeating-radial-gradient(circle at 78% 46%, transparent 0 11px, rgba(247,243,234,0.16) 12px 13px);
            pointer-events: none;
        }}

        .hero-content {{
            position: relative;
            z-index: 1;
            max-width: 920px;
        }}

        .eyebrow {{
            color: #e0b76b;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }}

        .hero h1 {{
            color: #fff7e7;
            font-size: clamp(2.2rem, 5vw, 5.3rem);
            line-height: 0.94;
            letter-spacing: 0;
            margin: 0 0 1rem 0;
            max-width: 820px;
        }}

        .hero p {{
            color: rgba(255, 247, 231, 0.78);
            max-width: 760px;
            font-size: 1.03rem;
            line-height: 1.65;
            margin: 0;
        }}

        .metric-card {{
            min-height: 132px;
            padding: 1.15rem 1.2rem;
            border: 1px solid rgba(23, 22, 20, 0.12);
            background: rgba(255, 252, 244, 0.72);
            box-shadow: 0 18px 45px rgba(72, 58, 38, 0.1);
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }}

        .metric-value {{
            color: var(--ink);
            font-size: 2.25rem;
            font-weight: 800;
            line-height: 1;
        }}

        .metric-note {{
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.7rem;
        }}

        .section-title {{
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 800;
            margin: 1.8rem 0 0.3rem;
        }}

        .section-subtitle {{
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 1.1rem;
        }}

        div[data-testid="stTabs"] button {{
            font-weight: 700;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(23, 22, 20, 0.12);
        }}

        .stPlotlyChart {{
            border: 1px solid rgba(23, 22, 20, 0.1);
            background: rgba(255, 252, 244, 0.55);
            box-shadow: 0 16px 42px rgba(72, 58, 38, 0.08);
        }}

        .small-table-title {{
            font-size: 0.95rem;
            color: var(--ink);
            font-weight: 800;
            margin: 0.35rem 0 0.55rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
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


def fig_theme(fig):
    fig.update_layout(
        font_family="Inter, sans-serif",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,252,244,0.42)",
        colorway=[
            PALETTE["gold"],
            PALETTE["teal"],
            PALETTE["coral"],
            PALETTE["plum"],
            PALETTE["green"],
            PALETTE["blue"],
        ],
        margin=dict(l=28, r=24, t=48, b=34),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#171614", font_color="#fff7e7", bordercolor="#171614"),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(23,22,20,0.18)", title_font_color=PALETTE["muted"])
    fig.update_yaxes(gridcolor="rgba(23,22,20,0.08)", linecolor="rgba(23,22,20,0.18)", title_font_color=PALETTE["muted"])
    return fig


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    inject_css()
    df, exploded = load_data(DATA_PATH)
    selected, selected_genres = filtered_data(df, exploded)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="eyebrow">CSV-powered listening intelligence</div>
                <h1>Halchemy Album Dashboard</h1>
                <p>
                    A living map of ratings, genre instincts, era bias, outlier loves,
                    and albums still waiting for a verdict.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rated = selected["RatingNum"].dropna()
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Albums", f"{len(selected):,}", "Filtered catalog size")
    with c2:
        metric_card("Rated", f"{rated.count():,}", f"{selected['RatingStatus'].eq('did-not-listen').sum():,} did not listen")
    with c3:
        metric_card("Avg Rating", f"{rated.mean():.2f}" if not rated.empty else "-", "Your personal mean")
    with c4:
        metric_card("Avg Global", f"{selected['Global Rating'].mean():.2f}", "External consensus")
    with c5:
        delta = selected["RatingDelta"].mean()
        metric_card("Taste Gap", f"{delta:+.2f}" if pd.notna(delta) else "-", "You minus global")

    if selected.empty:
        st.warning("No albums match the current filters.")
        return

    tab_overview, tab_taste, tab_gaps, tab_explorer = st.tabs(
        ["Overview", "Taste Profile", "Global Gap", "Explorer"]
    )

    with tab_overview:
        st.markdown('<div class="section-title">Catalog Pulse</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">How the collection is distributed across ratings, release eras, and entry dates.</div>',
            unsafe_allow_html=True,
        )

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
            color_discrete_sequence=[
                PALETTE["coral"],
                PALETTE["gold"],
                PALETTE["green"],
                PALETTE["teal"],
                PALETTE["blue"],
                PALETTE["plum"],
                PALETTE["muted"],
            ],
        )
        fig_theme(fig_rating).update_layout(showlegend=False)
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
            color_continuous_scale=["#d66a50", "#c88f2d", "#287c78"],
        )
        fig_theme(fig_decade)
        right.plotly_chart(fig_decade, use_container_width=True)

        by_month = selected.dropna(subset=["MonthAdded"]).groupby("MonthAdded", as_index=False).size()
        fig_month = px.area(by_month, x="MonthAdded", y="size", title="Albums added over time")
        fig_month.update_traces(line_color=PALETTE["teal"], fillcolor="rgba(40,124,120,0.22)")
        fig_theme(fig_month).update_xaxes(title=None).update_yaxes(title="Albums")
        st.plotly_chart(fig_month, use_container_width=True)

    with tab_taste:
        st.markdown('<div class="section-title">Taste Profile</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Where your ratings cluster by genre, origin, and era.</div>',
            unsafe_allow_html=True,
        )

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
            color_continuous_scale=["#ead8b3", "#c88f2d", "#287c78"],
        )
        fig_theme(fig_genre).update_xaxes(range=[0, 5], title="Average rating")
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
            color_discrete_sequence=[PALETTE["gold"], PALETTE["teal"], PALETTE["coral"], PALETTE["plum"]],
        )
        fig_origin.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="#171614")))
        fig_theme(fig_origin).update_xaxes(range=[2.5, 3.8]).update_yaxes(range=[1, 5])
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
                color_continuous_scale=["#ead8b3", "#d66a50", "#6b4c7c"],
            )
            fig_theme(fig_words)
            st.plotly_chart(fig_words, use_container_width=True)

    with tab_gaps:
        st.markdown('<div class="section-title">Consensus Breakers</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">The albums where your ear most disagrees with the crowd.</div>',
            unsafe_allow_html=True,
        )

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
            color_continuous_scale=["#d66a50", "#f1d390", "#287c78"],
        )
        fig_gap.add_shape(type="line", x0=1, y0=1, x1=5, y1=5, line=dict(color="rgba(23,22,20,0.32)", dash="dash"))
        fig_theme(fig_gap).update_yaxes(range=[0.7, 5.3], dtick=1).update_xaxes(range=[1.7, 4.6])
        st.plotly_chart(fig_gap, use_container_width=True)

        love, reject = st.columns(2)
        with love:
            st.markdown('<div class="small-table-title">You rate these far above consensus</div>', unsafe_allow_html=True)
            compact_table(
                gap_df.sort_values("RatingDelta", ascending=False).head(12),
                ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"],
            )
        with reject:
            st.markdown('<div class="small-table-title">Consensus likes these more than you do</div>', unsafe_allow_html=True)
            compact_table(
                gap_df.sort_values("RatingDelta").head(12),
                ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"],
            )

    with tab_explorer:
        st.markdown('<div class="section-title">Album Explorer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Search, sort, and inspect notes without leaving the dashboard.</div>',
            unsafe_allow_html=True,
        )

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
