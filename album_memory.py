from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from album_data import notes_keywords


MEMORY_VERSION = 1
MEMORY_PATH = Path(__file__).with_name("agent_memory.json")


def catalog_signature(df: pd.DataFrame) -> str:
    columns = [
        column
        for column in ["Artist", "Album", "Released", "RatingStatus", "RatingNum", "Global Rating", "Genres", "Notes"]
        if column in df.columns
    ]
    records = df.loc[:, columns].fillna("").astype(str).sort_values(columns[:3]).to_json(orient="records")
    return hashlib.sha256(records.encode("utf-8")).hexdigest()


def _records(data: pd.DataFrame, columns: list[str], limit: int) -> list[dict[str, object]]:
    if data.empty:
        return []
    available = [column for column in columns if column in data.columns]
    clean = data.loc[:, available].head(limit).copy()
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def build_agent_memory(df: pd.DataFrame, exploded: pd.DataFrame) -> dict[str, Any]:
    rated = df.dropna(subset=["RatingNum"]).copy()
    gaps = df.dropna(subset=["RatingNum", "Global Rating", "RatingDelta"]).copy()
    rated_genres = exploded.dropna(subset=["RatingNum"]).copy()

    favorite_genres = pd.DataFrame()
    if not rated_genres.empty:
        favorite_genres = (
            rated_genres.groupby("Genre", as_index=False)
            .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
            .query("Albums >= 2")
            .sort_values(["AvgRating", "Albums"], ascending=[False, False])
        )
        favorite_genres["Delta"] = favorite_genres["AvgRating"] - favorite_genres["AvgGlobal"]

    reliable_artists = pd.DataFrame()
    if not rated.empty:
        reliable_artists = (
            rated.groupby("Artist", as_index=False)
            .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"))
            .query("Albums >= 2")
            .sort_values(["AvgRating", "Albums"], ascending=[False, False])
        )

    low_rated_genres = pd.DataFrame()
    low_rated = rated_genres.loc[rated_genres["RatingNum"].le(2)].copy() if not rated_genres.empty else pd.DataFrame()
    if not low_rated.empty:
        low_rated_genres = (
            low_rated.groupby("Genre", as_index=False)
            .agg(LowRatedAlbums=("Album", "count"), AvgRating=("RatingNum", "mean"))
            .sort_values(["LowRatedAlbums", "AvgRating"], ascending=[False, True])
        )

    unresolved = df.loc[df["RatingStatus"].eq("unrated")].sort_values(
        ["Global Rating", "Released"],
        ascending=[False, False],
        na_position="last",
    )

    return {
        "version": MEMORY_VERSION,
        "catalog_signature": catalog_signature(df),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "catalog": {
            "albums": int(len(df)),
            "rated": int(df["RatingNum"].notna().sum()),
            "unrated": int(df["RatingStatus"].eq("unrated").sum()),
            "genres": int(exploded["Genre"].nunique()),
        },
        "favorite_genres": _records(favorite_genres, ["Genre", "Albums", "AvgRating", "AvgGlobal", "Delta"], 6),
        "reliable_artists": _records(reliable_artists, ["Artist", "Albums", "AvgRating"], 6),
        "above_consensus": _records(
            gaps.sort_values("RatingDelta", ascending=False),
            ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"],
            5,
        ),
        "below_consensus": _records(
            gaps.sort_values("RatingDelta", ascending=True),
            ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"],
            5,
        ),
        "avoid_signals": _records(low_rated_genres, ["Genre", "LowRatedAlbums", "AvgRating"], 5),
        "note_keywords": _records(notes_keywords(df), ["Word", "Count"], 10),
        "unresolved_queue": _records(
            unresolved,
            ["Artist", "Album", "Released", "Global Rating", "Genres"],
            8,
        ),
    }


def load_agent_memory(path: Path = MEMORY_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("version") != MEMORY_VERSION:
        return None
    return data


def save_agent_memory(memory: dict[str, Any], path: Path = MEMORY_PATH) -> None:
    path.write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_agent_memory(df: pd.DataFrame, exploded: pd.DataFrame, path: Path = MEMORY_PATH) -> dict[str, Any]:
    current = load_agent_memory(path)
    signature = catalog_signature(df)
    if current and current.get("catalog_signature") == signature:
        return current
    memory = build_agent_memory(df, exploded)
    save_agent_memory(memory, path)
    return memory
