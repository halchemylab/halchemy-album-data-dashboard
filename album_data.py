from __future__ import annotations

import base64
import hashlib
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd


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
