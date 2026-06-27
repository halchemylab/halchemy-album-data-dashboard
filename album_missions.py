from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MISSIONS_VERSION = 1
MISSIONS_PATH = Path(__file__).with_name("listening_missions.json")


def load_missions(path: Path = MISSIONS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"version": MISSIONS_VERSION, "missions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": MISSIONS_VERSION, "missions": []}
    if not isinstance(data, dict) or data.get("version") != MISSIONS_VERSION:
        return {"version": MISSIONS_VERSION, "missions": []}
    missions = data.get("missions", [])
    if not isinstance(missions, list):
        missions = []
    return {"version": MISSIONS_VERSION, "missions": missions}


def save_missions(data: dict[str, Any], path: Path = MISSIONS_PATH) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_mission(mission: dict[str, Any], path: Path = MISSIONS_PATH) -> dict[str, Any]:
    data = load_missions(path)
    missions = data["missions"]
    mission_id = str(mission.get("id") or f"mission-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    stored = {
        "id": mission_id,
        "status": str(mission.get("status") or "not started"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **mission,
        "id": mission_id,
    }
    missions.insert(0, stored)
    data["missions"] = missions[:20]
    save_missions(data, path)
    return stored
