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
        ]
    ).to_csv(csv_path, index=False)
    return load_data(csv_path)


def test_choose_skill_routes_common_questions() -> None:
    assert choose_skill("What should I listen to next?") == "recommendations"
    assert choose_skill("Where do I disagree with consensus?") == "taste_gaps"
    assert choose_skill("What genres do I rate highest?") == "genre_analysis"
    assert choose_skill("Find notes that mention boring") == "notes_search"
    assert choose_skill("Give me three capstone-ready insights") == "story_insights"
    assert choose_skill("Summarize this catalog") == "catalog_overview"


def test_answer_question_recommends_with_context(tmp_path: Path) -> None:
    df, exploded = sample_data(tmp_path)

    answer = answer_question("Recommend a rock album", df, exploded)

    assert answer.skill == "recommendations"
    assert "Beatles - A Hard Day's Night" in answer.summary
    assert answer.detail["Album"].tolist()[0] == "A Hard Day's Night"


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


def test_openai_agent_falls_back_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    df, exploded = sample_data(tmp_path)

    answer = answer_question_with_openai("Recommend a rock album", df, exploded)

    assert answer.skill == "recommendations"
    assert answer.mode == "deterministic fallback"
