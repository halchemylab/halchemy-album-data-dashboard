from __future__ import annotations

import re


def choose_skill(question: str) -> str:
    lowered = question.lower()
    if any(
        phrase in lowered for phrase in ["walk me through", "guide me through", "dashboard walkthrough", "guided tour"]
    ):
        return "dashboard_walkthrough"
    if any(phrase in lowered for phrase in ["clear filters", "reset filters", "reset dashboard", "show everything"]):
        return "set_dashboard_filters"
    if any(
        phrase in lowered
        for phrase in [
            "mission",
            "listening mission",
            "discovery mission",
            "help me discover",
            "what should i explore",
            "next listening goal",
        ]
    ):
        return "listening_mission"
    if any(
        phrase in lowered
        for phrase in [
            "playlist",
            "listening path",
            "listening queue",
            "starter pack",
            "mixtape",
            "sequence",
            "tonight",
            "revisit queue",
            "bridge me",
        ]
    ):
        return "playlist_builder"
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
        "hip hop",
        "folk",
        "soul",
    ]
    if any(verb in lowered for verb in filter_verbs) and (
        any(term in lowered for term in filter_terms) or re.search(r"\b(?:19|20)?\d0'?s\b", lowered)
    ):
        return "set_dashboard_filters"
    if any(
        word in lowered
        for word in [
            "hypothesis",
            "hypotheses",
            "theory",
            "theories",
            "pattern",
            "patterns",
        ]
    ):
        return "taste_hypotheses"
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
    if "genre" in lowered or any(
        word in lowered for word in ["rock", "pop", "jazz", "hip-hop", "hip hop", "folk", "soul"]
    ):
        return "genre_analysis"
    if any(word in lowered for word in ["note", "notes", "mention", "find"]):
        return "notes_search"
    return "catalog_overview"
