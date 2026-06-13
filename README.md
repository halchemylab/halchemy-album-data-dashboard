# Halchemy Album Data Dashboard

A Streamlit dashboard for exploring `albums.csv`: ratings, genres, eras, notes, and personal-vs-global taste gaps.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app is CSV-first. Add more rows to `albums.csv`, rerun Streamlit, and the dashboard will recalculate filters, charts, and tables.

## CSV schema

`albums.csv` must include these columns:

| Column | Required value |
| --- | --- |
| `Artist` | Non-empty artist name |
| `Album` | Non-empty album title |
| `Released` | Whole release year, such as `1964` |
| `Rating` | `1` through `5`, `did-not-listen`, or blank |
| `Notes` | Freeform notes; may be blank |
| `Global Rating` | Numeric external rating; may be blank |
| `Genres` | Comma-separated genre names; may be blank |
| `Origin` | Freeform origin code or label; may be blank |
| `Generated Date` | Parseable date or timestamp |

If the CSV has missing columns or invalid values, the app shows the affected row numbers instead of failing with a traceback.

## Test

```powershell
pytest
```
