# Halchemy Album Data Dashboard

A Streamlit dashboard for exploring `albums.csv`: ratings, genres, eras, notes, and personal-vs-global taste gaps.

The dashboard also includes an **Album Agent** tab. It is a skill-based data agent: the chat interface sends questions to an agent layer, the agent chooses an album-analysis skill, the skill runs deterministic pandas analysis over the currently filtered data, and the answer is summarized back in the UI.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app is CSV-first. Add more rows to `albums.csv`, rerun Streamlit, and the dashboard will recalculate filters, charts, and tables.

## Album Agent

The Agent tab can run in two modes:

- **Local fallback mode**: rule-based routing chooses a skill without calling OpenAI. This keeps demos and tests reliable.
- **OpenAI skill router mode**: OpenAI chooses the best skill/tool for the question, the app runs that skill against the filtered DataFrame, and OpenAI writes the final explanation from the tool result.

Supported skills:

| Skill | What it answers |
| --- | --- |
| `catalog_overview` | Counts, averages, and high-level catalog summaries |
| `genre_analysis` | Favorite genres and genre-level rating patterns |
| `recommendations` | Album suggestions from the current filtered catalog |
| `taste_gaps` | Where personal ratings diverge from global ratings |
| `notes_search` | Searches freeform notes for words or phrases |
| `story_insights` | Capstone-ready narrative insights backed by data |

To enable OpenAI mode, set your key before running Streamlit:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL="gpt-5.5"
streamlit run app.py
```

Or create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_MODEL = "gpt-5.5"
```

The secrets file is ignored by git.

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
