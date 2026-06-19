from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class AgentAnswer:
    question: str
    summary: str
    detail: pd.DataFrame
    skill: str
    mode: str = "deterministic"


SkillHandler = Callable[[str, pd.DataFrame, pd.DataFrame], AgentAnswer]


def _empty_detail() -> pd.DataFrame:
    return pd.DataFrame()


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


def story_insights(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
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
]


def choose_skill(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ["insight", "story", "presentation", "summarize my taste", "takeaway"]):
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


def answer_question(question: str, df: pd.DataFrame, exploded: pd.DataFrame) -> AgentAnswer:
    cleaned = question.strip()
    if not cleaned:
        return AgentAnswer(
            question=question,
            summary="Ask me about recommendations, genre patterns, taste gaps, notes, or the current catalog.",
            detail=_empty_detail(),
            skill="help",
        )
    skill_name = choose_skill(cleaned)
    return SKILLS[skill_name](cleaned, df, exploded)


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
) -> AgentAnswer:
    cleaned = question.strip()
    if not cleaned:
        return answer_question(question, df, exploded)

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        fallback = answer_question(question, df, exploded)
        return AgentAnswer(
            question=fallback.question,
            summary=fallback.summary,
            detail=fallback.detail,
            skill=fallback.skill,
            mode="deterministic fallback",
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
                "Keep answers concise and cite the skill result in plain language."
            ),
        },
        {"role": "user", "content": selected_summary},
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
        fallback = answer_question(cleaned, df, exploded)
        return AgentAnswer(
            question=cleaned,
            summary=text or fallback.summary,
            detail=fallback.detail,
            skill=fallback.skill,
            mode="openai",
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
        last_answer = run_skill(skill_name, cleaned, df, exploded, arguments)
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
        last_answer = answer_question(cleaned, df, exploded)
    return AgentAnswer(
        question=cleaned,
        summary=final_text or last_answer.summary,
        detail=last_answer.detail,
        skill=last_answer.skill,
        mode="openai",
    )
