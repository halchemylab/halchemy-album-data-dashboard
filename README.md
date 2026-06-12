# Halchemy Album Data Dashboard

A Streamlit dashboard for exploring `albums.csv`: ratings, genres, eras, notes, and personal-vs-global taste gaps.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app is CSV-first. Add more rows to `albums.csv`, rerun Streamlit, and the dashboard will recalculate filters, charts, and tables.
