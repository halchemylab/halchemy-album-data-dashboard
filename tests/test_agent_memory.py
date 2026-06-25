from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from album_data import load_data
from album_memory import build_agent_memory, ensure_agent_memory, load_agent_memory
from test_data_validation import valid_row


def sample_catalog(tmp_path: Path):
    csv_path = tmp_path / "albums.csv"
    pd.DataFrame(
        [
            valid_row(
                Artist="Beatles",
                Album="A Hard Day's Night",
                Released=1964,
                Rating="5",
                Genres="rock,pop",
                Notes="Festive guitar hooks.",
                **{"Global Rating": 3.89},
            ),
            valid_row(
                Artist="Beatles",
                Album="Revolver",
                Released=1966,
                Rating="5",
                Genres="rock,pop",
                Notes="Inventive studio record.",
                **{"Global Rating": 4.23},
            ),
            valid_row(
                Artist="The Who",
                Album="Tommy",
                Released=1969,
                Rating="2",
                Genres="rock",
                Notes="Boring concept record.",
                **{"Global Rating": 3.31},
            ),
            valid_row(
                Artist="Alice Coltrane",
                Album="Journey in Satchidananda",
                Released=1971,
                Rating="",
                Genres="jazz",
                Notes="",
                **{"Global Rating": 4.11},
            ),
        ]
    ).to_csv(csv_path, index=False)
    return load_data(csv_path)


def test_build_agent_memory_captures_taste_signals(tmp_path: Path) -> None:
    df, exploded = sample_catalog(tmp_path)

    memory = build_agent_memory(df, exploded)

    assert memory["catalog"]["albums"] == 4
    assert memory["catalog"]["rated"] == 3
    assert memory["catalog"]["unrated"] == 1
    assert memory["favorite_genres"][0]["Genre"] == "pop"
    assert memory["reliable_artists"][0]["Artist"] == "Beatles"
    assert memory["unresolved_queue"][0]["Album"] == "Journey in Satchidananda"


def test_ensure_agent_memory_persists_json(tmp_path: Path) -> None:
    df, exploded = sample_catalog(tmp_path)
    memory_path = tmp_path / "agent_memory.json"

    memory = ensure_agent_memory(df, exploded, memory_path)
    loaded = load_agent_memory(memory_path)

    assert loaded is not None
    assert loaded["catalog_signature"] == memory["catalog_signature"]
    assert memory_path.exists()
