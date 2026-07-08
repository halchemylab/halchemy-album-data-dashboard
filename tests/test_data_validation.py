import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from album_data import AlbumDataError, load_data, validate_albums_csv


def valid_row(**overrides):
    row = {
        "Artist": "Beatles",
        "Album": "A Hard Day's Night",
        "Released": 1964,
        "Rating": "5",
        "Notes": "The music was festive.",
        "Global Rating": 3.89,
        "Genres": "rock,pop",
        "Origin": "uk",
        "Generated Date": "2022-12-11T04:00:20.982Z",
    }
    row.update(overrides)
    return row


def validation_errors(**overrides) -> list[str]:
    return validate_albums_csv(pd.DataFrame([valid_row(**overrides)]))


def test_validation_reports_missing_columns() -> None:
    df = pd.DataFrame([valid_row()]).drop(columns=["Generated Date"])

    errors = validate_albums_csv(df)

    assert errors == ["Missing required columns: Generated Date"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("Artist", "", "Artist is blank"),
        ("Album", " ", "Album is blank"),
        ("Released", "nineteen sixty four", "Released must be a whole year"),
        ("Released", 1964.5, "Released must be a whole year"),
        ("Global Rating", "great", "Global Rating must be numeric when present"),
        ("Generated Date", "not a date", "Generated Date must be parseable"),
        ("Rating", "6", "Rating must be 1-5, did-not-listen, or blank"),
        ("Rating", "maybe", "Rating must be 1-5, did-not-listen, or blank"),
    ],
)
def test_validation_reports_bad_values(field: str, value, expected: str) -> None:
    errors = validation_errors(**{field: value})

    assert any(expected in error for error in errors)


@pytest.mark.parametrize("rating", ["", None, "did-not-listen", "1", 2, 3.0, "4.0", "5"])
def test_validation_accepts_supported_ratings(rating) -> None:
    assert validation_errors(Rating=rating) == []


@pytest.mark.parametrize("global_rating", ["", None, 3.89, "4.45"])
def test_validation_accepts_blank_or_numeric_global_rating(global_rating) -> None:
    assert validation_errors(**{"Global Rating": global_rating}) == []


def test_load_data_normalizes_valid_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "albums.csv"
    pd.DataFrame(
        [
            valid_row(Rating=5.0),
            valid_row(Album="Unrated Album", Rating=""),
            valid_row(Album="Skipped Album", Rating="did-not-listen"),
        ]
    ).to_csv(csv_path, index=False)

    df, exploded = load_data(csv_path)

    assert df["Released"].tolist() == [1964, 1964, 1964]
    assert df["RatingStatus"].tolist() == ["5", "unrated", "did-not-listen"]
    assert df["RatingLabel"].tolist() == ["essential", "unrated", "skipped"]
    assert df["RatingNum"].tolist()[0] == 5
    assert pd.isna(df["RatingNum"].tolist()[1])
    assert {"Genre", "RatingDelta", "SearchText"}.issubset(df.columns.union(exploded.columns))


def test_load_data_raises_for_invalid_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "albums.csv"
    pd.DataFrame([valid_row(Released="unknown")]).to_csv(csv_path, index=False)

    with pytest.raises(AlbumDataError, match="Released must be a whole year"):
        load_data(csv_path)
