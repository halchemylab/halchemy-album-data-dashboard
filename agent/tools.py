from __future__ import annotations

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
        "name": "playlist_builder",
        "description": "Build a sequenced listening path, starter pack, revisit queue, or short playlist from the current filtered catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of albums to include, from 2 through 8.",
                },
                "genre": {"type": "string", "description": "Optional genre to narrow the playlist."},
                "decade": {"type": "string", "description": "Optional decade such as 1970s or 1990s."},
                "artist": {"type": "string", "description": "Optional artist name to narrow the playlist."},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "listening_mission",
        "description": "Create an actionable listening mission with a familiar anchor, stretch picks, and progress-ready steps.",
        "parameters": {
            "type": "object",
            "properties": {},
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
            "properties": {"genre": {"type": "string", "description": "Optional specific genre to inspect."}},
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
        "name": "taste_hypotheses",
        "description": "Generate testable hypotheses about the user's music taste, with evidence, counterexamples, and next actions.",
        "parameters": {
            "type": "object",
            "properties": {},
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
                "search": {
                    "type": "string",
                    "description": "Optional search text for artist, album, notes, or genres.",
                },
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
                "search": {
                    "type": "string",
                    "description": "Optional search text for artist, album, notes, or genres.",
                },
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
