from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from album_agent import answer_question, answer_question_with_openai, choose_skill
from album_data import load_data
from test_data_validation import valid_row


def sample_data(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = tmp_path / "albums.csv"
    pd.DataFrame(
        [
            valid_row(
                Artist="Beatles",
                Album="A Hard Day's Night",
                Released=1964,
                Rating="5",
                Genres="rock,pop",
                Notes="The music was festive.",
                **{"Global Rating": 3.89},
            ),
            valid_row(
                Artist="The Who",
                Album="Tommy",
                Released=1969,
                Rating="2",
                Genres="rock",
                Notes="Rhythm is pretty boring.",
                **{"Global Rating": 3.31},
            ),
            valid_row(
                Artist="A Tribe Called Quest",
                Album="People's Instinctive Travels",
                Released=1990,
                Rating="4",
                Genres="hip-hop",
                Notes="The beats are enjoyable and groovy.",
                **{"Global Rating": 3.62},
            ),
            valid_row(
                Artist="The Rolling Stones",
                Album="Sticky Fingers",
                Released=1971,
                Rating="4",
                Genres="rock",
                Notes="Loose and confident rock record.",
                **{"Global Rating": 3.76},
            ),
            valid_row(
                Artist="Alice Coltrane",
                Album="Journey in Satchidananda",
                Released=1971,
                Rating="",
                Genres="jazz",
                Notes="Spiritual jazz with harp and drone.",
                **{"Global Rating": 4.11},
            ),
        ]
    ).to_csv(csv_path, index=False)
    return load_data(csv_path)


def answer_context(answer):
    clean = answer.detail.head(8).copy()
    clean = clean.where(pd.notna(clean), None)
    rows = clean.to_dict(orient="records")
    return {
        "last_question": answer.question,
        "last_skill": answer.skill,
        "last_summary": answer.summary,
        "last_rows": rows,
        "selected_album": rows[0] if rows else None,
    }


def test_choose_skill_routes_common_questions() -> None:
    assert choose_skill("What should I listen to next?") == "recommendations"
    assert choose_skill("Build me a 3 album starter pack") == "playlist_builder"
    assert choose_skill("Show me a rock playlist") == "playlist_builder"
    assert choose_skill("Where do I disagree with consensus?") == "taste_gaps"
    assert choose_skill("What genres do I rate highest?") == "genre_analysis"
    assert choose_skill("Find notes that mention boring") == "notes_search"
    assert choose_skill("Give me three capstone-ready insights") == "story_insights"
    assert choose_skill("Create a one-page report on my music taste") == "story_insights"
    assert choose_skill("Walk me through my 1970s rock taste") == "dashboard_walkthrough"
    assert choose_skill("Summarize this catalog") == "catalog_overview"


def test_answer_question_recommends_with_context(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Recommend a rock album", df, exploded)

    assert answer.skill == "recommendations"
    assert "Beatles - A Hard Day's Night" in answer.summary
    assert answer.detail["Album"].tolist()[0] == "A Hard Day's Night"


def test_answer_question_builds_filtered_playlist(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Build me a 3 album rock starter pack", df, exploded)

    assert answer.skill == "playlist_builder"
    assert "3-album" in answer.summary
    assert answer.detail["Slot"].tolist() == [1, 2, 3]
    assert answer.detail["Role"].tolist()[0] == "Gateway"
    assert answer.detail["Album"].tolist() == ["A Hard Day's Night", "Sticky Fingers", "Tommy"]
    assert answer.detail["Reason"].str.contains("personal rating").all()


def test_answer_question_builds_unrated_discovery_playlist(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Build a two album unrated discovery playlist", df, exploded)

    assert answer.skill == "playlist_builder"
    assert "unrated discovery" in answer.summary
    assert answer.detail["Album"].tolist() == ["Journey in Satchidananda"]
    assert pd.isna(answer.detail["RatingNum"].tolist()[0])


def test_followup_can_explain_second_result(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)
    first = answer_question("Recommend rock albums", df, exploded)

    answer = answer_question("Why the second one?", df, exploded, context=answer_context(first))

    assert answer.skill == "context_followup"
    assert "Sticky Fingers" in answer.summary
    assert answer.detail["Album"].tolist() == ["Sticky Fingers"]


def test_followup_finds_similar_albums(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)
    first = answer_question("Recommend a rock album", df, exploded)

    answer = answer_question("Show more like this", df, exploded, context=answer_context(first))

    assert answer.skill == "context_followup"
    assert "Using Beatles - A Hard Day's Night" in answer.summary
    assert "A Hard Day's Night" not in answer.detail["Album"].tolist()
    assert "Sticky Fingers" in answer.detail["Album"].tolist()


def test_followup_compares_against_overall_taste(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)
    first = answer_question("Recommend a rock album", df, exploded)

    answer = answer_question("Compare against my overall taste", df, exploded, context=answer_context(first))

    assert answer.skill == "context_followup"
    assert "current average personal rating" in answer.summary
    assert {"Metric", "Value", "Catalog Average"}.issubset(answer.detail.columns)


def test_answer_question_finds_note_matches(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Find notes that mention boring", df, exploded)

    assert answer.skill == "notes_search"
    assert "1 albums" in answer.summary
    assert answer.detail["Album"].tolist() == ["Tommy"]


def test_answer_question_returns_story_insights(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Give me three capstone-ready insights", df, exploded)

    assert answer.skill == "story_insights"
    assert "story-ready insights" in answer.summary
    assert "Insight" in answer.detail.columns


def test_answer_question_builds_taste_report(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Create a one-page report on my music taste", df, exploded)

    assert answer.skill == "story_insights"
    assert "taste report" in answer.summary
    assert {"Section", "Narrative", "Evidence", "Metric"}.issubset(answer.detail.columns)
    assert "Taste identity" in answer.detail["Section"].tolist()
    assert "Next listening question" in answer.detail["Section"].tolist()


def test_answer_question_can_request_dashboard_filter_action(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)
    rock = df.loc[df["PrimaryGenre"].eq("rock")]
    rock_keys = pd.MultiIndex.from_frame(rock[["Artist", "Album", "Released"]])
    exploded_keys = pd.MultiIndex.from_frame(exploded[["Artist", "Album", "Released"]])
    rock_exploded = exploded.loc[exploded_keys.isin(rock_keys)]

    answer = answer_question(
        "Show me unrated jazz from the 1970s",
        rock,
        rock_exploded,
        filter_df=df,
        filter_exploded=exploded,
    )

    assert answer.skill == "set_dashboard_filters"
    assert answer.dashboard_action is not None
    assert answer.dashboard_action["type"] == "set_filters"
    filters = answer.dashboard_action["filters"]
    assert filters["genres"] == ["jazz"]
    assert filters["decades"] == ["1970s"]
    assert filters["statuses"] == ["unrated"]


def test_answer_question_can_request_filter_reset(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Reset dashboard filters", df, exploded)

    assert answer.skill == "set_dashboard_filters"
    assert answer.dashboard_action == {"type": "set_filters", "clear_existing": True, "filters": {}}


def test_answer_question_builds_dashboard_walkthrough(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Walk me through my 1970s rock taste", df, exploded)

    assert answer.skill == "dashboard_walkthrough"
    assert "guided walkthrough" in answer.summary
    assert {"Step", "Dashboard move", "What to inspect", "Evidence"}.issubset(answer.detail.columns)
    assert answer.dashboard_action is not None
    filters = answer.dashboard_action["filters"]
    assert filters["genres"] == ["rock"]
    assert filters["decades"] == ["1970s"]


def test_openai_agent_falls_back_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    df, exploded = sample_data(tmp_path)

    answer = answer_question_with_openai("Recommend a rock album", df, exploded)

    assert answer.skill == "recommendations"
    assert answer.mode == "deterministic fallback"
