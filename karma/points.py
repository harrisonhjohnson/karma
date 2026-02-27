"""Karma points system (KARMA-004).

Points are stored in points.json at the repo root:
  { "total": int, "log": [{ "ts": ISO8601, "event": str, "pts": int }] }

Events and values:
  add_seed   -> +10 (per call)
  new_root   -> +5  (per root; pass count= to award multiple)
  edit_seed  -> +2  (per call)
  edit_root  -> +5  (per new root; pass count= to award multiple)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

KARMA_DIR = Path(__file__).parent.parent
POINTS_FILE = KARMA_DIR / "points.json"

# Points awarded per single event occurrence
POINTS_TABLE: dict[str, int] = {
    "add_seed": 10,
    "new_root": 5,
    "edit_seed": 2,
    "edit_root": 5,
}


def _load(points_file: Path = POINTS_FILE) -> dict:
    """Load points.json, returning default structure if missing or corrupt."""
    if not points_file.exists():
        return {"total": 0, "log": []}
    try:
        data = json.loads(points_file.read_text(encoding="utf-8"))
        if "total" not in data or "log" not in data:
            return {"total": 0, "log": []}
        return data
    except (json.JSONDecodeError, OSError):
        return {"total": 0, "log": []}


def _save(data: dict, points_file: Path = POINTS_FILE) -> None:
    """Persist data dict to points.json."""
    points_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def award(
    event: str,
    count: int = 1,
    *,
    points_file: Path = POINTS_FILE,
) -> int:
    """Award points for an event and persist to points.json.

    Args:
        event: One of 'add_seed', 'new_root', 'edit_seed', 'edit_root'.
        count: How many times the event occurred (e.g. 3 new roots -> count=3).
        points_file: Override path for testing.

    Returns:
        Total points awarded this call (pts_per_event * count).

    Raises:
        ValueError: If event is not in POINTS_TABLE.
    """
    if event not in POINTS_TABLE:
        raise ValueError(f"Unknown event '{event}'. Valid: {list(POINTS_TABLE)}")

    pts_per = POINTS_TABLE[event]
    total_pts = pts_per * count

    data = _load(points_file)
    data["total"] += total_pts
    data["log"].append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "pts": total_pts,
        }
    )
    _save(data, points_file)

    return total_pts


def get_total(*, points_file: Path = POINTS_FILE) -> int:
    """Return current total karma points.

    Args:
        points_file: Override path for testing.
    """
    return _load(points_file)["total"]


def get_log(*, points_file: Path = POINTS_FILE) -> list[dict]:
    """Return the full points log (list of dicts with ts, event, pts).

    Args:
        points_file: Override path for testing.
    """
    return _load(points_file)["log"]
