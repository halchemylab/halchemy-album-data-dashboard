from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class AgentTraceStep:
    phase: str
    detail: str


@dataclass(frozen=True)
class AgentAnswer:
    question: str
    summary: str
    detail: pd.DataFrame
    skill: str
    mode: str = "deterministic"
    trace: tuple[AgentTraceStep, ...] = ()
    dashboard_action: dict[str, object] | None = None


SkillHandler = Callable[[str, pd.DataFrame, pd.DataFrame], AgentAnswer]
AgentContext = dict[str, Any]
AgentMemory = dict[str, Any]


def _empty_detail() -> pd.DataFrame:
    return pd.DataFrame()


def _with_trace(answer: AgentAnswer, *steps: AgentTraceStep) -> AgentAnswer:
    return AgentAnswer(
        question=answer.question,
        summary=answer.summary,
        detail=answer.detail,
        skill=answer.skill,
        mode=answer.mode,
        trace=(*steps, *answer.trace),
        dashboard_action=answer.dashboard_action,
    )


def _scope_trace(df: pd.DataFrame, exploded: pd.DataFrame) -> AgentTraceStep:
    rated = df["RatingNum"].notna().sum() if "RatingNum" in df.columns else 0
    genres = exploded["Genre"].nunique() if "Genre" in exploded.columns else 0
    return AgentTraceStep(
        "Scope",
        f"Inspected {len(df):,} albums, {rated:,} rated entries, and {genres:,} genres in the current filter.",
    )


def _rows_trace(answer: AgentAnswer) -> AgentTraceStep:
    row_count = len(answer.detail)
    if row_count:
        return AgentTraceStep("Evidence", f"Returned {row_count:,} evidence rows for the UI table.")
    return AgentTraceStep("Evidence", "No table rows were needed for this answer.")


def _top_table(data: pd.DataFrame, columns: list[str], limit: int = 8) -> pd.DataFrame:
    available = [column for column in columns if column in data.columns]
    return data.loc[:, available].head(limit).reset_index(drop=True)


def _clean_records(data: pd.DataFrame, limit: int = 8) -> list[dict[str, object]]:
    if data.empty:
        return []
    clean = data.head(limit).copy()
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def _records_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _format_album(row: pd.Series) -> str:
    return f"{row['Artist']} - {row['Album']} ({int(row['Released'])})"


def _format_context_album(album: dict[str, object]) -> str:
    released = album.get("Released", "-")
    return f"{album.get('Artist', '-')} - {album.get('Album', '-')} ({released})"


def _row_identity(row: pd.Series | dict[str, object]) -> tuple[str, str, int | None]:
    get = row.get if isinstance(row, dict) else row.__getitem__
    try:
        released = int(get("Released"))
    except (TypeError, ValueError):
        released = None
    return str(get("Artist")), str(get("Album")), released


def _context_rows(context: AgentContext | None) -> list[dict[str, object]]:
    if not context:
        return []
    rows = context.get("last_rows", [])
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _context_album(question: str, context: AgentContext | None) -> dict[str, object] | None:
    rows = _context_rows(context)
    lowered = question.lower()
    ordinal_indexes = {
        "first": 0,
        "1st": 0,
        "top": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4,
    }
    for word, index in ordinal_indexes.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered) and index < len(rows):
            return rows[index]
    selected = context.get("selected_album") if context else None
    if isinstance(selected, dict):
        return selected
    if rows and any(phrase in lowered for phrase in ["this", "that", "it", "why", "similar", "more like"]):
        return rows[0]
    return None


def _genre_tokens(value: object) -> set[str]:
    return {genre.strip().casefold() for genre in str(value or "").split(",") if genre.strip()}


def _matching_album(df: pd.DataFrame, album: dict[str, object]) -> pd.Series | None:
    artist, title, released = _row_identity(album)
    mask = df["Artist"].astype(str).eq(artist) & df["Album"].astype(str).eq(title)
    if released is not None:
        mask &= df["Released"].eq(released)
    matches = df.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def _similar_album_answer(question: str, df: pd.DataFrame, album: dict[str, object]) -> AgentAnswer:
    source = _matching_album(df, album)
    if source is None:
        return AgentAnswer(
            question=question,
            summary=f"I could not find {_format_context_album(album)} in the current filtered data.",
            detail=_empty_detail(),
            skill="context_followup",
        )

    source_genres = _genre_tokens(source.get("Genres"))
    source_decade = str(source.get("Decade", ""))
    data = df.copy()
    same_album = (
        data["Artist"].astype(str).eq(str(source["Artist"]))
        & data["Album"].astype(str).eq(str(source["Album"]))
        & data["Released"].eq(source["Released"])
    )
    data = data.loc[~same_album]

    if "unrated" in question.lower() or "not rated" in question.lower():
        data = data.loc[data["RatingStatus"].eq("unrated")]
    else:
        data = data.dropna(subset=["RatingNum"])

    if source_genres:
        genre_match = data["Genres"].fillna("").apply(lambda value: bool(_genre_tokens(value) & source_genres))
        data = data.loc[genre_match]
    if data.empty and source_decade:
        data = df.loc[df["Decade"].eq(source_decade) & ~same_album].copy()

    if data.empty:
        return AgentAnswer(
            question=question,
            summary=f"I could not find another album like {_format_album(source)} in the current filtered data.",
            detail=_empty_detail(),
            skill="context_followup",
        )

    sort_columns = ["RatingNum", "Global Rating", "RatingDelta"]
    data = data.sort_values(sort_columns, ascending=[False, False, False], na_position="last")
    genre_text = ", ".join(sorted(source_genres)) if source_genres else source_decade
    summary = f"Using {_format_album(source)} as context, these are the closest matches by genre signal"
    if genre_text:
        summary += f" ({genre_text})"
    summary += "."
    return AgentAnswer(
        question=question,
        summary=summary,
        detail=_top_table(data, ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"]),
        skill="context_followup",
    )


def _explain_album_answer(question: str, df: pd.DataFrame, album: dict[str, object]) -> AgentAnswer:
    source = _matching_album(df, album)
    if source is None:
        return AgentAnswer(
            question=question,
            summary=f"I could not find {_format_context_album(album)} in the current filtered data.",
            detail=_empty_detail(),
            skill="context_followup",
        )

    parts = [
        f"{_format_album(source)} is in context.",
        f"Your rating is {source['RatingNum']:.1f}." if pd.notna(source.get("RatingNum")) else "It is not personally rated yet.",
        f"The global rating is {source['Global Rating']:.2f}." if pd.notna(source.get("Global Rating")) else "It has no global rating in the data.",
    ]
    if pd.notna(source.get("RatingDelta")):
        parts.append(f"That puts your taste gap at {source['RatingDelta']:+.2f}.")
    if str(source.get("Genres", "")).strip():
        parts.append(f"Genres: {source['Genres']}.")
    if str(source.get("Notes", "")).strip():
        parts.append(f"Your notes say: {source['Notes']}")
    return AgentAnswer(
        question=question,
        summary=" ".join(parts),
        detail=_records_frame([source.where(pd.notna(source), None).to_dict()]),
        skill="context_followup",
    )


def _compare_album_answer(question: str, df: pd.DataFrame, album: dict[str, object]) -> AgentAnswer | None:
    source = _matching_album(df, album)
    rated = df.dropna(subset=["RatingNum"]).copy()
    if source is None or rated.empty:
        return None

    avg_rating = rated["RatingNum"].mean()
    avg_global = df["Global Rating"].mean()
    avg_gap = df["RatingDelta"].mean()
    rows = [
        {
            "Metric": "Selected album",
            "Value": _format_album(source),
            "Catalog Average": "-",
        },
        {
            "Metric": "Personal rating",
            "Value": f"{source['RatingNum']:.2f}" if pd.notna(source.get("RatingNum")) else "-",
            "Catalog Average": f"{avg_rating:.2f}",
        },
        {
            "Metric": "Global rating",
            "Value": f"{source['Global Rating']:.2f}" if pd.notna(source.get("Global Rating")) else "-",
            "Catalog Average": f"{avg_global:.2f}" if pd.notna(avg_global) else "-",
        },
        {
            "Metric": "Taste gap",
            "Value": f"{source['RatingDelta']:+.2f}" if pd.notna(source.get("RatingDelta")) else "-",
            "Catalog Average": f"{avg_gap:+.2f}" if pd.notna(avg_gap) else "-",
        },
    ]
    if pd.notna(source.get("RatingNum")):
        relation = "above" if source["RatingNum"] > avg_rating else "below" if source["RatingNum"] < avg_rating else "right at"
        summary = f"{_format_album(source)} sits {relation} your current average personal rating of {avg_rating:.2f}."
    else:
        summary = f"{_format_album(source)} is not personally rated yet, so only consensus and catalog context are available."
    return AgentAnswer(
        question=question,
        summary=summary,
        detail=_records_frame(rows),
        skill="context_followup",
    )


def answer_context_followup(question: str, df: pd.DataFrame, context: AgentContext | None) -> AgentAnswer | None:
    album = _context_album(question, context)
    if album is None:
        return None
    lowered = question.lower()
    if any(phrase in lowered for phrase in ["compare", "overall taste", "usual taste", "average"]):
        return _compare_album_answer(question, df, album)
    if any(phrase in lowered for phrase in ["more like", "similar", "another", "else", "only unrated", "unrated"]):
        return _similar_album_answer(question, df, album)
    if any(phrase in lowered for phrase in ["why", "explain", "context", "that one", "this one"]):
        return _explain_album_answer(question, df, album)
    return None


def _extract_artist(question: str, df: pd.DataFrame) -> str | None:
    lowered = question.lower()
    artists = sorted(df["Artist"].dropna().unique(), key=lambda value: len(str(value)), reverse=True)
    for artist in artists:
        if str(artist).lower() in lowered:
            return str(artist)
    return None


def _extract_genre(question: str, exploded: pd.DataFrame) -> str | None:
    lowered = question.lower()
    genres = sorted(exploded["Genre"].dropna().unique(), key=lambda value: len(str(value)), reverse=True)
    for genre in genres:
        if str(genre).lower() in lowered:
            return str(genre)
    return None


def _extract_decade(question: str) -> str | None:
    match = re.search(r"\b(19|20)\d0s\b", question.lower())
    if match:
        return match.group(0)
    year_match = re.search(r"\b((19|20)\d{2})\b", question)
    if year_match:
        year = int(year_match.group(1))
        return f"{year // 10 * 10}s"
    return None


def _extract_decades(question: str, df: pd.DataFrame) -> list[str]:
    valid_decades = sorted(str(value) for value in df["Decade"].dropna().unique()) if "Decade" in df.columns else []
    found = {match.group(0) for match in re.finditer(r"\b(?:19|20)\d0s\b", question.lower())}
    for match in re.finditer(r"\b((?:19|20)\d{2})\b", question):
        year = int(match.group(1))
        found.add(f"{year // 10 * 10}s")
    return [decade for decade in valid_decades if decade.lower() in found]


def _extract_filter_values(question: str, values: list[object]) -> list[str]:
    lowered = question.lower()
    matches: list[str] = []
    for value in sorted({str(item) for item in values if str(item).strip()}, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(value.lower())}(?!\w)", lowered):
            matches.append(value)
    return matches


def _extract_statuses(question: str) -> list[str]:
    lowered = question.lower()
    status_aliases = {
        "unrated": ["unrated", "not rated", "haven't rated", "have not rated", "no rating"],
        "did-not-listen": ["did not listen", "didn't listen", "skipped"],
        "1": ["1 star", "one star", "avoid"],
        "2": ["2 star", "two star", "not for me"],
        "3": ["3 star", "three star", "mixed"],
        "4": ["4 star", "four star", "strong"],
        "5": ["5 star", "five star", "essential"],
    }
    matches: list[str] = []
    for status, aliases in status_aliases.items():
        if any(alias in lowered for alias in aliases):
            matches.append(status)
    return matches


def _extract_year_range(question: str, df: pd.DataFrame) -> tuple[int, int] | None:
    if "Released" not in df.columns or df.empty:
        return None
    catalog_min, catalog_max = int(df["Released"].min()), int(df["Released"].max())
    lowered = question.lower()
    between = re.search(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b", lowered)
    if between:
        start, end = sorted((int(between.group(1)), int(between.group(2))))
        return max(catalog_min, start), min(catalog_max, end)
    after = re.search(r"\b(?:after|since|from)\s+((?:19|20)\d{2})\b", lowered)
    before = re.search(r"\b(?:before|through|until|to)\s+((?:19|20)\d{2})\b", lowered)
    if after or before:
        start = int(after.group(1)) if after else catalog_min
        end = int(before.group(1)) if before else catalog_max
        if start <= end:
            return max(catalog_min, start), min(catalog_max, end)
    return None


def _extract_search_text(question: str, df: pd.DataFrame) -> str:
    lowered = question.lower()
    artists = _extract_filter_values(question, df["Artist"].dropna().unique().tolist() if "Artist" in df.columns else [])
    albums = _extract_filter_values(question, df["Album"].dropna().unique().tolist() if "Album" in df.columns else [])
    candidates = albums or artists
    if candidates:
        return candidates[0]
    quoted = re.search(r"['\"]([^'\"]{2,})['\"]", question)
    if quoted:
        return quoted.group(1).strip()
    search_match = re.search(r"\b(?:search|find)\s+(?:for\s+)?([a-z0-9][a-z0-9 '&.-]{1,40})", lowered)
    if search_match:
        value = search_match.group(1).strip()
        value = re.sub(r"\b(?:albums?|records?|music|from|in|with|that|mention|mentions)\b.*$", "", value).strip()
        return value
    return ""


def _filter_action_labels(filters: dict[str, object]) -> list[str]:
    labels: list[str] = []
    search = str(filters.get("search", "")).strip()
    if search:
        labels.append(f'Search "{search}"')
    for key, label in [("genres", "Genres"), ("origins", "Origins"), ("decades", "Decades"), ("statuses", "Statuses")]:
        values = filters.get(key, [])
        if isinstance(values, list) and values:
            labels.append(f"{label}: " + ", ".join(str(value) for value in values))
    year_range = filters.get("year_range")
    if isinstance(year_range, list) and len(year_range) == 2:
        labels.append(f"Years: {year_range[0]}-{year_range[1]}")
    return labels


def set_dashboard_filters(
    question: str,
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    arguments: dict[str, object] | None = None,
) -> AgentAnswer:
    arguments = arguments or {}
    lowered = question.lower()
    reset = any(phrase in lowered for phrase in ["clear filters", "reset filters", "reset dashboard", "show everything", "all albums"])
    if reset:
        action = {"type": "set_filters", "clear_existing": True, "filters": {}}
        return AgentAnswer(
            question=question,
            summary="I cleared the dashboard filters.",
            detail=_records_frame([{"Filter": "All filters", "Value": "cleared"}]),
            skill="set_dashboard_filters",
            dashboard_action=action,
        )

    valid_genres = sorted(exploded["Genre"].dropna().unique()) if "Genre" in exploded.columns else []
    valid_origins = sorted(df["OriginLabel"].dropna().unique()) if "OriginLabel" in df.columns else []

    filters: dict[str, object] = {
        "search": str(arguments.get("search", "") or "").strip() or _extract_search_text(question, df),
        "genres": arguments.get("genres") if isinstance(arguments.get("genres"), list) else _extract_filter_values(question, valid_genres),
        "origins": arguments.get("origins") if isinstance(arguments.get("origins"), list) else _extract_filter_values(question, valid_origins),
        "decades": arguments.get("decades") if isinstance(arguments.get("decades"), list) else _extract_decades(question, df),
        "statuses": arguments.get("statuses") if isinstance(arguments.get("statuses"), list) else _extract_statuses(question),
    }
    year_range = arguments.get("year_range")
    if isinstance(year_range, list) and len(year_range) == 2:
        try:
            filters["year_range"] = [int(year_range[0]), int(year_range[1])]
        except (TypeError, ValueError):
            filters["year_range"] = None
    else:
        extracted_range = _extract_year_range(question, df)
        filters["year_range"] = list(extracted_range) if extracted_range else None

    valid_genre_set = {str(value).casefold(): str(value) for value in valid_genres}
    valid_origin_set = {str(value).casefold(): str(value) for value in valid_origins}
    valid_decade_set = {str(value).casefold(): str(value) for value in df["Decade"].dropna().unique()} if "Decade" in df.columns else {}
    valid_status_set = set(["1", "2", "3", "4", "5", "did-not-listen", "unrated"])
    filters["genres"] = [valid_genre_set[str(value).casefold()] for value in filters["genres"] if str(value).casefold() in valid_genre_set]
    filters["origins"] = [valid_origin_set[str(value).casefold()] for value in filters["origins"] if str(value).casefold() in valid_origin_set]
    filters["decades"] = [valid_decade_set[str(value).casefold()] for value in filters["decades"] if str(value).casefold() in valid_decade_set]
    filters["statuses"] = [str(value) for value in filters["statuses"] if str(value) in valid_status_set]

    labels = _filter_action_labels(filters)
    if not labels:
        return AgentAnswer(
            question=question,
            summary="I could not identify a valid dashboard filter from that request.",
            detail=_empty_detail(),
            skill="set_dashboard_filters",
        )

    action = {"type": "set_filters", "clear_existing": True, "filters": filters}
    rows = [{"Filter": label.split(":", 1)[0], "Value": label.split(":", 1)[-1].strip()} for label in labels]
    return AgentAnswer(
        question=question,
        summary="I applied dashboard filters: " + "; ".join(labels) + ".",
        detail=_records_frame(rows),
        skill="set_dashboard_filters",
        dashboard_action=action,
    )


def _data_for_filter_action(
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    filters: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    search = str(filters.get("search", "") or "").strip().lower()
    if search:
        data = data.loc[data["SearchText"].fillna("").str.contains(re.escape(search), na=False)]

    values = filters.get("origins", [])
    if isinstance(values, list) and values:
        data = data.loc[data["OriginLabel"].astype(str).isin([str(value) for value in values])]
    values = filters.get("decades", [])
    if isinstance(values, list) and values:
        data = data.loc[data["Decade"].astype(str).isin([str(value) for value in values])]
    values = filters.get("statuses", [])
    if isinstance(values, list) and values:
        data = data.loc[data["RatingStatus"].astype(str).isin([str(value) for value in values])]

    year_range = filters.get("year_range")
    if isinstance(year_range, list) and len(year_range) == 2:
        start, end = sorted((int(year_range[0]), int(year_range[1])))
        data = data.loc[data["Released"].between(start, end)]

    values = filters.get("genres", [])
    if isinstance(values, list) and values:
        genre_keys = exploded.loc[
            exploded["Genre"].astype(str).isin([str(value) for value in values]),
            ["Artist", "Album", "Released"],
        ].drop_duplicates()
        data_key = pd.MultiIndex.from_frame(data[["Artist", "Album", "Released"]])
        genre_key = pd.MultiIndex.from_frame(genre_keys)
        data = data.loc[data_key.isin(genre_key)]

    if data.empty:
        return data.copy(), exploded.iloc[0:0].copy()
    selected_keys = pd.MultiIndex.from_frame(data[["Artist", "Album", "Released"]])
    exploded_keys = pd.MultiIndex.from_frame(exploded[["Artist", "Album", "Released"]])
    return data.copy(), exploded.loc[exploded_keys.isin(selected_keys)].copy()


def dashboard_walkthrough(
    question: str,
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    arguments: dict[str, object] | None = None,
) -> AgentAnswer:
    filter_answer = set_dashboard_filters(question, df, exploded, arguments)
    action = filter_answer.dashboard_action if filter_answer.dashboard_action else None
    filters = action.get("filters", {}) if isinstance(action, dict) else {}
    clear_existing = bool(action.get("clear_existing")) if isinstance(action, dict) else False
    if isinstance(filters, dict) and filters:
        selected, selected_genres = _data_for_filter_action(df, exploded, filters)
        filter_labels = _filter_action_labels(filters)
    else:
        selected, selected_genres = df.copy(), exploded.copy()
        filter_labels = []

    rated = selected.dropna(subset=["RatingNum"]).copy()
    gaps = selected.dropna(subset=["RatingNum", "Global Rating", "RatingDelta"]).copy()
    top_album = _album_label_or_dash(rated.sort_values(["RatingNum", "Global Rating"], ascending=[False, False]).head(1))
    top_genre, top_genre_metric = _top_genre_summary(selected_genres)

    rows = [
        {
            "Step": "1. Focus the dashboard",
            "Dashboard move": "; ".join(filter_labels) if filter_labels else "Use the current filters",
            "What to inspect": "Confirm the scope metrics before reading the charts.",
            "Evidence": f"{len(selected):,} albums, {rated['RatingNum'].count():,} rated",
        },
        {
            "Step": "2. Read the soundprint",
            "Dashboard move": "Open Soundprint",
            "What to inspect": "Start with the strongest repeated genre signal.",
            "Evidence": f"{top_genre}: {top_genre_metric}",
        },
        {
            "Step": "3. Check the anchor album",
            "Dashboard move": "Open Explorer",
            "What to inspect": "Use the top album as the narrative anchor for this slice.",
            "Evidence": top_album,
        },
        {
            "Step": "4. Inspect disagreement",
            "Dashboard move": "Open Outliers",
            "What to inspect": "Look for where your rating diverges most from global consensus.",
            "Evidence": (
                _album_label_or_dash(gaps.sort_values("RatingDelta", ascending=False).head(1))
                if not gaps.empty
                else "No consensus gaps available"
            ),
        },
    ]
    summary = (
        f"I built a guided walkthrough for {len(selected):,} albums"
        + (f" and applied filters: {'; '.join(filter_labels)}." if filter_labels else " using the current dashboard filters.")
    )
    dashboard_action = {"type": "set_filters", "clear_existing": clear_existing, "filters": filters} if filter_labels else None
    return AgentAnswer(
        question=question,
        summary=summary,
        detail=_records_frame(rows),
        skill="dashboard_walkthrough",
        dashboard_action=dashboard_action,
    )


def _filter_skill_data(
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    *,
    genre: str | None = None,
    decade: str | None = None,
    artist: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    if genre:
        keys = exploded.loc[
            exploded["Genre"].astype(str).str.casefold().eq(genre.casefold()),
            ["Artist", "Album", "Released"],
        ].drop_duplicates()
        data_key = pd.MultiIndex.from_frame(data[["Artist", "Album", "Released"]])
        genre_key = pd.MultiIndex.from_frame(keys)
        data = data.loc[data_key.isin(genre_key)]
    if decade:
        normalized_decade = decade if decade.endswith("s") else f"{int(decade) // 10 * 10}s"
        data = data.loc[data["Decade"].eq(normalized_decade)]
    if artist:
        data = data.loc[data["Artist"].astype(str).str.casefold().eq(artist.casefold())]

    selected_keys = pd.MultiIndex.from_frame(data[["Artist", "Album", "Released"]])
    exploded_keys = pd.MultiIndex.from_frame(exploded[["Artist", "Album", "Released"]])
    return data.copy(), exploded.loc[exploded_keys.isin(selected_keys)].copy()


def _skill_payload(answer: AgentAnswer) -> dict[str, object]:
    return {
        "skill": answer.skill,
        "summary": answer.summary,
        "rows": _clean_records(answer.detail),
    }


def catalog_overview(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
    rated = df["RatingNum"].dropna()
    favorite = df.dropna(subset=["RatingNum"]).sort_values(
        ["RatingNum", "Global Rating", "Released"],
        ascending=[False, False, False],
    )
    genre_counts = exploded.groupby("Genre", as_index=False).agg(Albums=("Album", "count")).sort_values(
        "Albums",
        ascending=False,
    )
    top_genre = genre_counts.iloc[0]["Genre"] if not genre_counts.empty else "unknown"
    summary = (
        f"The current slice has {len(df):,} albums, {rated.count():,} rated albums, "
        f"and an average personal rating of {rated.mean():.2f}."
        if not rated.empty
        else f"The current slice has {len(df):,} albums, but no personal ratings yet."
    )
    if not favorite.empty:
        summary += f" Your highest-rated album here is {_format_album(favorite.iloc[0])}."
    summary += f" The most common genre is {top_genre}."
    return AgentAnswer(
        question=question,
        summary=summary,
        detail=_top_table(genre_counts, ["Genre", "Albums"]),
        skill="catalog_overview",
    )


def recommendations(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
    data = df.dropna(subset=["RatingNum"]).copy()
    if data.empty:
        return AgentAnswer(
            question=question,
            summary="I need at least one rated album before I can recommend a direction.",
            detail=_empty_detail(),
            skill="recommendations",
        )

    genre = _extract_genre(question, exploded)
    decade = _extract_decade(question)
    artist = _extract_artist(question, df)
    if genre:
        keys = exploded.loc[exploded["Genre"].str.casefold().eq(genre.casefold()), ["Artist", "Album", "Released"]]
        data = data.loc[pd.MultiIndex.from_frame(data[["Artist", "Album", "Released"]]).isin(pd.MultiIndex.from_frame(keys))]
    if decade:
        data = data.loc[data["Decade"].eq(decade)]
    if artist:
        data = data.loc[data["Artist"].str.casefold().eq(artist.casefold())]

    data = data.sort_values(["RatingNum", "Global Rating", "RatingDelta"], ascending=[False, False, False])
    if data.empty:
        context = ", ".join(part for part in [genre, decade, artist] if part)
        return AgentAnswer(
            question=question,
            summary=f"I could not find rated albums for that request ({context}). Try asking with a broader genre, artist, or decade.",
            detail=_empty_detail(),
            skill="recommendations",
        )

    top = data.iloc[0]
    context = ", ".join(part for part in [genre, decade, artist] if part)
    scope = f" matching {context}" if context else ""
    return AgentAnswer(
        question=question,
        summary=f"I would start with {_format_album(top)}. It has your rating {top['RatingNum']:.1f} and a global rating of {top['Global Rating']:.2f}{scope}.",
        detail=_top_table(data, ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"]),
        skill="recommendations",
    )


def taste_gaps(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
    data = df.dropna(subset=["RatingNum", "Global Rating", "RatingDelta"]).copy()
    if data.empty:
        return AgentAnswer(
            question=question,
            summary="I need both personal and global ratings to compare your taste against consensus.",
            detail=_empty_detail(),
            skill="taste_gaps",
        )

    lowered = question.lower()
    if any(word in lowered for word in ["overrated", "below", "hate", "dislike", "lower"]):
        data = data.sort_values("RatingDelta", ascending=True)
        direction = "below consensus"
    else:
        data = data.sort_values("RatingDelta", ascending=False)
        direction = "above consensus"
    top = data.iloc[0]
    return AgentAnswer(
        question=question,
        summary=f"Your strongest {direction} signal is {_format_album(top)} with a {top['RatingDelta']:+.2f} gap.",
        detail=_top_table(data, ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta", "Genres"]),
        skill="taste_gaps",
    )


def genre_analysis(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
    data = exploded.dropna(subset=["RatingNum"]).copy()
    if data.empty:
        return AgentAnswer(
            question=question,
            summary="I need rated albums before I can describe genre preferences.",
            detail=_empty_detail(),
            skill="genre_analysis",
        )

    genre = _extract_genre(question, exploded)
    if genre:
        genre_data = data.loc[data["Genre"].str.casefold().eq(genre.casefold())]
        if genre_data.empty:
            return AgentAnswer(
                question=question,
                summary=f"I found the genre {genre}, but there are no rated albums for it in the current slice.",
                detail=_empty_detail(),
                skill="genre_analysis",
            )
        summary = (
            f"For {genre}, you have {len(genre_data):,} rated albums with an average rating "
            f"of {genre_data['RatingNum'].mean():.2f}."
        )
        detail = genre_data.sort_values(["RatingNum", "Global Rating"], ascending=[False, False])
        return AgentAnswer(
            question=question,
            summary=summary,
            detail=_top_table(detail, ["Artist", "Album", "Released", "RatingNum", "Global Rating", "RatingDelta"]),
            skill="genre_analysis",
        )

    summary_df = (
        data.groupby("Genre", as_index=False)
        .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
        .query("Albums >= 2")
    )
    summary_df["Delta"] = summary_df["AvgRating"] - summary_df["AvgGlobal"]
    summary_df = summary_df.sort_values(["AvgRating", "Albums"], ascending=[False, False])
    if summary_df.empty:
        return AgentAnswer(
            question=question,
            summary="There are not enough repeated rated genres yet. Rate more albums to make genre comparisons stronger.",
            detail=_empty_detail(),
            skill="genre_analysis",
        )
    top = summary_df.iloc[0]
    return AgentAnswer(
        question=question,
        summary=f"Your strongest genre signal is {top['Genre']} with {top['Albums']:.0f} albums averaging {top['AvgRating']:.2f}.",
        detail=_top_table(summary_df, ["Genre", "Albums", "AvgRating", "AvgGlobal", "Delta"]),
        skill="genre_analysis",
    )


def notes_search(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
    terms = [
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", question.lower())
        if word not in {"find", "show", "album", "albums", "notes", "note", "that", "mention", "mentions", "about"}
    ]
    if not terms:
        return AgentAnswer(
            question=question,
            summary="Ask me for a specific word or phrase to search in your notes.",
            detail=_empty_detail(),
            skill="notes_search",
        )
    mask = df["Notes"].fillna("").str.lower().apply(lambda note: all(term in note for term in terms))
    matches = df.loc[mask].sort_values(["RatingNum", "Global Rating"], ascending=[False, False])
    phrase = " ".join(terms)
    if matches.empty:
        return AgentAnswer(
            question=question,
            summary=f"I did not find notes matching '{phrase}' in the current slice.",
            detail=_empty_detail(),
            skill="notes_search",
        )
    return AgentAnswer(
        question=question,
        summary=f"I found {len(matches):,} albums with notes matching '{phrase}'.",
        detail=_top_table(matches, ["Artist", "Album", "Released", "RatingLabel", "RatingNum", "Notes"]),
        skill="notes_search",
    )


def _album_label_or_dash(data: pd.DataFrame) -> str:
    if data.empty:
        return "-"
    return _format_album(data.iloc[0])


def _top_genre_summary(exploded: pd.DataFrame) -> tuple[str, str]:
    rated_genres = exploded.dropna(subset=["RatingNum"]).copy()
    if rated_genres.empty:
        return "-", "No rated genre signal yet"
    genre_summary = (
        rated_genres.groupby("Genre", as_index=False)
        .agg(Albums=("Album", "count"), AvgRating=("RatingNum", "mean"), AvgGlobal=("Global Rating", "mean"))
        .sort_values(["AvgRating", "Albums"], ascending=[False, False])
    )
    top = genre_summary.iloc[0]
    return str(top["Genre"]), f"{top['Albums']:.0f} albums, {top['AvgRating']:.2f} avg rating"


def taste_report(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
    rated = df.dropna(subset=["RatingNum"]).copy()
    gap = df.dropna(subset=["RatingNum", "Global Rating", "RatingDelta"]).copy()
    unresolved = df.loc[df["RatingStatus"].eq("unrated")].sort_values(
        ["Global Rating", "Released"],
        ascending=[False, False],
        na_position="last",
    )

    if rated.empty:
        return AgentAnswer(
            question=question,
            summary="I need rated albums before I can build a useful taste report.",
            detail=_empty_detail(),
            skill="story_insights",
        )

    best = rated.sort_values(["RatingNum", "Global Rating"], ascending=[False, False]).head(1)
    lowest = rated.sort_values(["RatingNum", "Global Rating"], ascending=[True, False]).head(1)
    top_genre, top_genre_metric = _top_genre_summary(exploded)
    above = gap.sort_values("RatingDelta", ascending=False).head(1)
    below = gap.sort_values("RatingDelta", ascending=True).head(1)
    avg_rating = rated["RatingNum"].mean()
    avg_global = rated["Global Rating"].mean()

    rows = [
        {
            "Section": "Taste identity",
            "Narrative": f"You lean toward albums that earn a {avg_rating:.2f} average personal rating in this slice.",
            "Evidence": _album_label_or_dash(best),
            "Metric": f"{len(rated):,} rated albums",
        },
        {
            "Section": "Strongest genre",
            "Narrative": f"{top_genre} is the clearest repeated genre signal.",
            "Evidence": top_genre,
            "Metric": top_genre_metric,
        },
        {
            "Section": "Signature favorite",
            "Narrative": "This is the strongest anchor for describing what works for you.",
            "Evidence": _album_label_or_dash(best),
            "Metric": f"{best.iloc[0]['RatingNum']:.1f} personal rating" if not best.empty else "-",
        },
        {
            "Section": "Resistance point",
            "Narrative": "This is useful contrast for explaining what does not connect.",
            "Evidence": _album_label_or_dash(lowest),
            "Metric": f"{lowest.iloc[0]['RatingNum']:.1f} personal rating" if not lowest.empty else "-",
        },
    ]
    if not above.empty:
        rows.append(
            {
                "Section": "Above consensus",
                "Narrative": "This is where your taste is more enthusiastic than the global signal.",
                "Evidence": _album_label_or_dash(above),
                "Metric": f"{above.iloc[0]['RatingDelta']:+.2f} gap",
            }
        )
    if not below.empty:
        rows.append(
            {
                "Section": "Below consensus",
                "Narrative": "This is where the broader audience is warmer than you are.",
                "Evidence": _album_label_or_dash(below),
                "Metric": f"{below.iloc[0]['RatingDelta']:+.2f} gap",
            }
        )
    if not unresolved.empty:
        rows.append(
            {
                "Section": "Next listening question",
                "Narrative": "This unresolved album can test whether the current pattern holds.",
                "Evidence": _album_label_or_dash(unresolved),
                "Metric": (
                    f"{unresolved.iloc[0]['Global Rating']:.2f} global rating"
                    if pd.notna(unresolved.iloc[0].get("Global Rating"))
                    else "unrated by you"
                ),
            }
        )

    report_type = "slide outline" if any(word in question.lower() for word in ["slide", "deck"]) else "taste report"
    summary = (
        f"I built a {report_type} with {len(rows):,} sections from {len(rated):,} rated albums. "
        f"Your current average personal rating is {avg_rating:.2f}"
    )
    if pd.notna(avg_global):
        summary += f" against a {avg_global:.2f} average global rating."
    else:
        summary += "."
    return AgentAnswer(question=question, summary=summary, detail=_records_frame(rows), skill="story_insights")


def story_insights(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
    lowered = question.lower()
    if any(word in lowered for word in ["report", "profile", "slide", "deck", "one-page", "one page"]):
        return taste_report(question, df, exploded)

    rated = df.dropna(subset=["RatingNum"]).copy()
    gap = df.dropna(subset=["RatingNum", "Global Rating", "RatingDelta"]).copy()
    genre_answer = genre_analysis(question, df, exploded)

    rows: list[dict[str, object]] = []
    if not rated.empty:
        best = rated.sort_values(["RatingNum", "Global Rating"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "Insight": "Highest-rated album",
                "Evidence": _format_album(best),
                "Metric": f"{best['RatingNum']:.1f} personal rating",
            }
        )
    if not gap.empty:
        above = gap.sort_values("RatingDelta", ascending=False).iloc[0]
        below = gap.sort_values("RatingDelta", ascending=True).iloc[0]
        rows.extend(
            [
                {
                    "Insight": "Most above consensus",
                    "Evidence": _format_album(above),
                    "Metric": f"{above['RatingDelta']:+.2f} gap",
                },
                {
                    "Insight": "Most below consensus",
                    "Evidence": _format_album(below),
                    "Metric": f"{below['RatingDelta']:+.2f} gap",
                },
            ]
        )
    if not genre_answer.detail.empty:
        genre_row = genre_answer.detail.iloc[0]
        rows.append(
            {
                "Insight": "Strongest genre pattern",
                "Evidence": genre_row.get("Genre", "-"),
                "Metric": f"{float(genre_row.get('AvgRating', 0)):.2f} avg rating",
            }
        )

    detail = _records_frame(rows)
    if detail.empty:
        summary = "I need more rated albums before I can produce useful story insights."
    else:
        summary = f"I found {len(detail):,} story-ready insights from the current filtered catalog."
    return AgentAnswer(question=question, summary=summary, detail=detail, skill="story_insights")


SKILLS: dict[str, SkillHandler] = {
    "catalog_overview": catalog_overview,
    "recommendations": recommendations,
    "taste_gaps": taste_gaps,
    "genre_analysis": genre_analysis,
    "notes_search": notes_search,
    "story_insights": story_insights,
    "dashboard_walkthrough": lambda question, df, exploded: dashboard_walkthrough(question, df, exploded),
    "set_dashboard_filters": lambda question, df, exploded: set_dashboard_filters(question, df, exploded),
}


AGENT_TOOLS = [
    {
        "type": "function",
        "name": "catalog_overview",
        "description": "Summarize the current filtered album catalog, including counts, average ratings, and common genres.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "recommendations",
        "description": "Recommend high-signal albums from the current filtered catalog. Can narrow by genre, decade, or artist.",
        "parameters": {
            "type": "object",
            "properties": {
                "genre": {"type": "string", "description": "Optional genre to narrow recommendations."},
                "decade": {"type": "string", "description": "Optional decade such as 1970s or 1990s."},
                "artist": {"type": "string", "description": "Optional artist name to narrow recommendations."},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "taste_gaps",
        "description": "Find where personal ratings differ most from global ratings.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["above", "below", "both"],
                    "description": "Use above for underrated-by-consensus albums, below for overrated-by-consensus albums, or both.",
                }
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "genre_analysis",
        "description": "Analyze genre-level taste patterns or summarize one specific genre.",
        "parameters": {
            "type": "object",
            "properties": {
                "genre": {"type": "string", "description": "Optional specific genre to inspect."}
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "notes_search",
        "description": "Search freeform album notes for words or phrases.",
        "parameters": {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Words or short phrases that must appear in the notes.",
                }
            },
            "required": ["terms"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "story_insights",
        "description": "Create capstone-friendly narrative insights backed by rows from the current filtered catalog.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "dashboard_walkthrough",
        "description": "Guide the user through a dashboard slice by setting filters and returning a step-by-step analysis path.",
        "parameters": {
            "type": "object",
            "properties": {
                "search": {"type": "string", "description": "Optional search text for artist, album, notes, or genres."},
                "genres": {"type": "array", "items": {"type": "string"}, "description": "Genres to select."},
                "origins": {"type": "array", "items": {"type": "string"}, "description": "Origin labels to select."},
                "decades": {"type": "array", "items": {"type": "string"}, "description": "Decades such as 1970s."},
                "statuses": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["1", "2", "3", "4", "5", "did-not-listen", "unrated"]},
                    "description": "Personal rating statuses to select.",
                },
                "year_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [start_year, end_year] release-year range.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_dashboard_filters",
        "description": "Change the dashboard filters when the user asks to show, filter, focus, narrow, reset, or clear the dashboard.",
        "parameters": {
            "type": "object",
            "properties": {
                "search": {"type": "string", "description": "Optional search text for artist, album, notes, or genres."},
                "genres": {"type": "array", "items": {"type": "string"}, "description": "Genres to select."},
                "origins": {"type": "array", "items": {"type": "string"}, "description": "Origin labels to select."},
                "decades": {"type": "array", "items": {"type": "string"}, "description": "Decades such as 1970s."},
                "statuses": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["1", "2", "3", "4", "5", "did-not-listen", "unrated"]},
                    "description": "Personal rating statuses to select.",
                },
                "year_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [start_year, end_year] release-year range.",
                },
            },
            "additionalProperties": False,
        },
    },
]


def choose_skill(question: str) -> str:
    lowered = question.lower()
    if any(phrase in lowered for phrase in ["walk me through", "guide me through", "dashboard walkthrough", "guided tour"]):
        return "dashboard_walkthrough"
    if any(phrase in lowered for phrase in ["clear filters", "reset filters", "reset dashboard", "show everything"]):
        return "set_dashboard_filters"
    filter_verbs = ["filter", "set filters", "show me", "only show", "focus on", "narrow to", "switch to"]
    filter_terms = [
        "unrated",
        "not rated",
        "did not listen",
        "skipped",
        "albums from",
        "from the",
        "in the",
        "rock",
        "pop",
        "jazz",
        "hip-hop",
        "folk",
        "soul",
    ]
    if any(verb in lowered for verb in filter_verbs) and (
        any(term in lowered for term in filter_terms) or re.search(r"\b(?:19|20)\d0s\b", lowered)
    ):
        return "set_dashboard_filters"
    if any(
        word in lowered
        for word in [
            "insight",
            "story",
            "presentation",
            "summarize my taste",
            "takeaway",
            "report",
            "profile",
            "slide",
            "deck",
            "one-page",
            "one page",
        ]
    ):
        return "story_insights"
    if any(word in lowered for word in ["recommend", "suggest", "should i listen", "best", "favorite", "top"]):
        return "recommendations"
    if any(word in lowered for word in ["gap", "consensus", "overrated", "underrated", "global"]):
        return "taste_gaps"
    if "genre" in lowered or any(word in lowered for word in ["rock", "pop", "jazz", "hip-hop", "folk", "soul"]):
        return "genre_analysis"
    if any(word in lowered for word in ["note", "notes", "mention", "find"]):
        return "notes_search"
    return "catalog_overview"


def run_skill(
    skill_name: str,
    question: str,
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    arguments: dict[str, object] | None = None,
) -> AgentAnswer:
    arguments = arguments or {}
    if skill_name == "dashboard_walkthrough":
        return dashboard_walkthrough(question, df, exploded, arguments)
    if skill_name == "set_dashboard_filters":
        return set_dashboard_filters(question, df, exploded, arguments)
    if skill_name == "recommendations":
        selected, selected_genres = _filter_skill_data(
            df,
            exploded,
            genre=str(arguments["genre"]) if arguments.get("genre") else None,
            decade=str(arguments["decade"]) if arguments.get("decade") else None,
            artist=str(arguments["artist"]) if arguments.get("artist") else None,
        )
        return recommendations(question, selected, selected_genres)
    if skill_name == "taste_gaps":
        direction = str(arguments.get("direction", "above"))
        skill_question = question
        if direction == "below":
            skill_question = question + " below lower overrated"
        return taste_gaps(skill_question, df, exploded)
    if skill_name == "genre_analysis" and arguments.get("genre"):
        return genre_analysis(f"{question} {arguments['genre']}", df, exploded)
    if skill_name == "notes_search":
        terms = arguments.get("terms")
        if isinstance(terms, list) and terms:
            return notes_search("Find notes that mention " + " ".join(str(term) for term in terms), df, exploded)
    return SKILLS.get(skill_name, catalog_overview)(question, df, exploded)


def answer_question(
    question: str,
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    context: AgentContext | None = None,
    memory: AgentMemory | None = None,
    filter_df: pd.DataFrame | None = None,
    filter_exploded: pd.DataFrame | None = None,
) -> AgentAnswer:
    cleaned = question.strip()
    if not cleaned:
        return AgentAnswer(
            question=question,
            summary="Ask me about recommendations, genre patterns, taste gaps, notes, or the current catalog.",
            detail=_empty_detail(),
            skill="help",
        )
    followup = answer_context_followup(cleaned, df, context)
    if followup is not None:
        return _with_trace(
            followup,
            _scope_trace(df, exploded),
            AgentTraceStep("Memory", "Loaded durable taste memory." if memory else "No durable taste memory was loaded."),
            AgentTraceStep("Plan", "Resolved the question as a follow-up using the active agent context."),
            AgentTraceStep("Tool", "Ran the context_followup skill against the filtered catalog."),
            _rows_trace(followup),
        )
    skill_name = choose_skill(cleaned)
    uses_full_catalog = skill_name in {"dashboard_walkthrough", "set_dashboard_filters"}
    skill_df = filter_df if uses_full_catalog and filter_df is not None else df
    skill_exploded = filter_exploded if uses_full_catalog and filter_exploded is not None else exploded
    answer = SKILLS[skill_name](cleaned, skill_df, skill_exploded)
    return _with_trace(
        answer,
        _scope_trace(df, exploded),
        AgentTraceStep("Memory", "Loaded durable taste memory." if memory else "No durable taste memory was loaded."),
        AgentTraceStep("Plan", f"Classified the request as {skill_name}."),
        AgentTraceStep("Tool", f"Ran the {skill_name} skill with deterministic pandas analysis."),
        _rows_trace(answer),
    )


def context_summary(context: AgentContext | None) -> str:
    if not context:
        return "No previous agent context is active."
    lines = [
        f"Previous question: {context.get('last_question', '-')}",
        f"Previous skill: {context.get('last_skill', '-')}",
        f"Previous summary: {context.get('last_summary', '-')}",
    ]
    selected = context.get("selected_album")
    if isinstance(selected, dict):
        lines.append(f"Selected album from previous result: {_format_context_album(selected)}")
    rows = _context_rows(context)[:5]
    if rows:
        row_text = "; ".join(f"{index + 1}. {_format_context_album(row)}" for index, row in enumerate(rows))
        lines.append(f"Previous result rows: {row_text}")
    return "\n".join(lines)


def memory_summary(memory: AgentMemory | None) -> str:
    if not memory:
        return "No durable taste memory is active."
    catalog = memory.get("catalog", {})
    lines = [
        "Durable taste memory:",
        (
            f"{catalog.get('albums', 0):,} albums, {catalog.get('rated', 0):,} rated, "
            f"{catalog.get('unrated', 0):,} unresolved, {catalog.get('genres', 0):,} genres."
        ),
    ]
    favorite_genres = memory.get("favorite_genres", [])
    if isinstance(favorite_genres, list) and favorite_genres:
        labels = [
            f"{item.get('Genre')} ({float(item.get('AvgRating', 0)):.2f})"
            for item in favorite_genres[:3]
            if isinstance(item, dict)
        ]
        if labels:
            lines.append("Favorite genre signals: " + ", ".join(labels))
    reliable_artists = memory.get("reliable_artists", [])
    if isinstance(reliable_artists, list) and reliable_artists:
        labels = [
            f"{item.get('Artist')} ({float(item.get('AvgRating', 0)):.2f})"
            for item in reliable_artists[:3]
            if isinstance(item, dict)
        ]
        if labels:
            lines.append("Reliable artists: " + ", ".join(labels))
    above_consensus = memory.get("above_consensus", [])
    if isinstance(above_consensus, list) and above_consensus:
        labels = [
            _format_context_album(item)
            for item in above_consensus[:2]
            if isinstance(item, dict)
        ]
        if labels:
            lines.append("Recurring above-consensus examples: " + "; ".join(labels))
    unresolved = memory.get("unresolved_queue", [])
    if isinstance(unresolved, list) and unresolved:
        labels = [
            _format_context_album(item)
            for item in unresolved[:3]
            if isinstance(item, dict)
        ]
        if labels:
            lines.append("Unresolved listening queue: " + "; ".join(labels))
    return "\n".join(lines)


def _get_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def _function_calls(response: object) -> list[object]:
    return [
        item
        for item in getattr(response, "output", []) or []
        if getattr(item, "type", None) == "function_call"
    ]


def answer_question_with_openai(
    question: str,
    df: pd.DataFrame,
    exploded: pd.DataFrame,
    *,
    api_key: str | None = None,
    model: str | None = None,
    context: AgentContext | None = None,
    memory: AgentMemory | None = None,
    filter_df: pd.DataFrame | None = None,
    filter_exploded: pd.DataFrame | None = None,
) -> AgentAnswer:
    cleaned = question.strip()
    if not cleaned:
        return answer_question(
            question,
            df,
            exploded,
            context=context,
            memory=memory,
            filter_df=filter_df,
            filter_exploded=filter_exploded,
        )

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        fallback = answer_question(
            question,
            df,
            exploded,
            context=context,
            memory=memory,
            filter_df=filter_df,
            filter_exploded=filter_exploded,
        )
        return AgentAnswer(
            question=fallback.question,
            summary=fallback.summary,
            detail=fallback.detail,
            skill=fallback.skill,
            mode="deterministic fallback",
            trace=(
                AgentTraceStep("Plan", "No OpenAI API key was configured, so the local router handled the request."),
                *fallback.trace,
            ),
            dashboard_action=fallback.dashboard_action,
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package to use the OpenAI-backed agent.") from exc

    client = OpenAI(api_key=resolved_api_key)
    selected_summary = (
        f"Current filtered data: {len(df):,} albums, "
        f"{df['RatingNum'].notna().sum():,} personally rated, "
        f"{exploded['Genre'].nunique():,} genres. "
        f"Available columns: {', '.join(df.columns)}."
    )
    input_items: list[dict[str, object]] = [
        {
            "role": "system",
            "content": (
                "You are a skill-based album analytics agent inside a Streamlit dashboard. "
                "Use the provided tools for factual answers. Do not invent albums, ratings, genres, or notes. "
                "When the user asks a follow-up, use the previous context to resolve phrases like this, that, "
                "the second one, more like this, or why. "
                "Keep answers concise and cite the skill result in plain language."
            ),
        },
        {"role": "user", "content": selected_summary},
        {"role": "user", "content": "Previous agent context:\n" + context_summary(context)},
        {"role": "user", "content": memory_summary(memory)},
        {"role": "user", "content": cleaned},
    ]

    response = client.responses.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5.5"),
        input=input_items,
        tools=AGENT_TOOLS,
        tool_choice="auto",
    )

    calls = _function_calls(response)
    if not calls:
        text = _get_response_text(response)
        fallback = answer_question(
            cleaned,
            df,
            exploded,
            context=context,
            memory=memory,
            filter_df=filter_df,
            filter_exploded=filter_exploded,
        )
        return AgentAnswer(
            question=cleaned,
            summary=text or fallback.summary,
            detail=fallback.detail,
            skill=fallback.skill,
            mode="openai",
            trace=(
                _scope_trace(df, exploded),
                AgentTraceStep("Plan", "Asked OpenAI to route the request, but it answered without a tool call."),
                AgentTraceStep("Tool", f"Used the local {fallback.skill} skill for evidence rows."),
                _rows_trace(fallback),
            ),
            dashboard_action=fallback.dashboard_action,
        )

    for item in getattr(response, "output", []) or []:
        if hasattr(item, "model_dump"):
            input_items.append(item.model_dump(exclude_none=True))
        else:
            input_items.append(item)
    last_answer: AgentAnswer | None = None
    for call in calls:
        raw_arguments = getattr(call, "arguments", "{}") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
        skill_name = str(getattr(call, "name", "catalog_overview"))
        uses_full_catalog = skill_name in {"dashboard_walkthrough", "set_dashboard_filters"}
        skill_df = filter_df if uses_full_catalog and filter_df is not None else df
        skill_exploded = filter_exploded if uses_full_catalog and filter_exploded is not None else exploded
        last_answer = run_skill(skill_name, cleaned, skill_df, skill_exploded, arguments)
        argument_text = ", ".join(f"{key}={value}" for key, value in arguments.items()) or "no arguments"
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": getattr(call, "call_id"),
                "output": json.dumps(_skill_payload(last_answer), default=str),
            }
        )

    final_response = client.responses.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5.5"),
        input=input_items,
        tools=AGENT_TOOLS,
    )
    final_text = _get_response_text(final_response)
    if last_answer is None:
        last_answer = answer_question(
            cleaned,
            df,
            exploded,
            context=context,
            memory=memory,
            filter_df=filter_df,
            filter_exploded=filter_exploded,
        )
    return AgentAnswer(
        question=cleaned,
        summary=final_text or last_answer.summary,
        detail=last_answer.detail,
        skill=last_answer.skill,
        mode="openai",
        trace=(
            _scope_trace(df, exploded),
            AgentTraceStep("Plan", "Asked OpenAI to choose the best album-analysis tool."),
            AgentTraceStep("Tool", f"OpenAI called {last_answer.skill} with {argument_text}."),
            _rows_trace(last_answer),
            AgentTraceStep("Explain", "Sent the tool result back to OpenAI for the final wording."),
        ),
        dashboard_action=last_answer.dashboard_action,
    )
