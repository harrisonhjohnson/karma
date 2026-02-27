"""Tests for karma points system (KARMA-004)."""

import json
from pathlib import Path

import pytest

from karma.points import award, get_total, get_log, POINTS_TABLE


@pytest.fixture()
def points_file(tmp_path) -> Path:
    """Return a temp path for points.json — fresh for each test."""
    return tmp_path / "points.json"


class TestAward:
    def test_award_add_seed_returns_10(self, points_file):
        """award('add_seed') returns 10 points."""
        pts = award("add_seed", points_file=points_file)
        assert pts == 10

    def test_award_new_root_returns_5(self, points_file):
        """award('new_root') returns 5 points."""
        pts = award("new_root", points_file=points_file)
        assert pts == 5

    def test_award_edit_seed_returns_2(self, points_file):
        """award('edit_seed') returns 2 points."""
        pts = award("edit_seed", points_file=points_file)
        assert pts == 2

    def test_award_edit_root_returns_5(self, points_file):
        """award('edit_root') returns 5 points."""
        pts = award("edit_root", points_file=points_file)
        assert pts == 5

    def test_award_count_multiplies_points(self, points_file):
        """award('new_root', count=3) returns 15 points (5 * 3)."""
        pts = award("new_root", count=3, points_file=points_file)
        assert pts == 15

    def test_award_accumulates_total(self, points_file):
        """Multiple award calls accumulate in total."""
        award("add_seed", points_file=points_file)   # +10
        award("new_root", count=2, points_file=points_file)  # +10
        award("edit_seed", points_file=points_file)  # +2
        total = get_total(points_file=points_file)
        assert total == 22

    def test_award_creates_points_file(self, points_file):
        """award() creates points.json if it doesn't exist."""
        assert not points_file.exists()
        award("add_seed", points_file=points_file)
        assert points_file.exists()

    def test_award_invalid_event_raises(self, points_file):
        """award() raises ValueError for unknown event names."""
        with pytest.raises(ValueError, match="Unknown event"):
            award("fly_to_moon", points_file=points_file)

    def test_award_persists_across_calls(self, points_file):
        """Points persist: re-reading after award reflects accumulated total."""
        award("add_seed", points_file=points_file)
        award("add_seed", points_file=points_file)
        # Read raw file to confirm persistence
        data = json.loads(points_file.read_text())
        assert data["total"] == 20

    def test_award_starting_from_missing_file(self, points_file):
        """award() initializes from zero when file is missing."""
        pts = award("add_seed", points_file=points_file)
        assert pts == 10
        assert get_total(points_file=points_file) == 10


class TestGetTotal:
    def test_get_total_zero_when_missing(self, points_file):
        """get_total() returns 0 when points.json doesn't exist."""
        assert get_total(points_file=points_file) == 0

    def test_get_total_after_awards(self, points_file):
        """get_total() reflects all accumulated points."""
        award("add_seed", points_file=points_file)   # +10
        award("new_root", count=3, points_file=points_file)  # +15
        assert get_total(points_file=points_file) == 25

    def test_get_total_reads_existing_file(self, points_file):
        """get_total() correctly reads pre-existing points.json."""
        points_file.write_text(json.dumps({"total": 99, "log": []}))
        assert get_total(points_file=points_file) == 99

    def test_get_total_corrupt_file_returns_zero(self, points_file):
        """get_total() returns 0 if points.json is corrupt JSON."""
        points_file.write_text("not-json{{{")
        assert get_total(points_file=points_file) == 0


class TestGetLog:
    def test_get_log_empty_when_missing(self, points_file):
        """get_log() returns [] when points.json doesn't exist."""
        assert get_log(points_file=points_file) == []

    def test_get_log_entries_have_required_keys(self, points_file):
        """Each log entry has 'ts', 'event', and 'pts' keys."""
        award("add_seed", points_file=points_file)
        log = get_log(points_file=points_file)
        assert len(log) == 1
        entry = log[0]
        assert "ts" in entry
        assert "event" in entry
        assert "pts" in entry

    def test_get_log_records_event_name(self, points_file):
        """Log entry records the correct event name."""
        award("edit_seed", points_file=points_file)
        log = get_log(points_file=points_file)
        assert log[0]["event"] == "edit_seed"

    def test_get_log_records_pts_earned(self, points_file):
        """Log entry records total pts for that call (pts_per * count)."""
        award("new_root", count=4, points_file=points_file)
        log = get_log(points_file=points_file)
        assert log[0]["pts"] == 20  # 5 * 4

    def test_get_log_timestamps_are_iso8601(self, points_file):
        """Log entry timestamps are valid ISO 8601 strings."""
        from datetime import datetime
        award("add_seed", points_file=points_file)
        log = get_log(points_file=points_file)
        ts = log[0]["ts"]
        # Should parse without raising
        dt = datetime.fromisoformat(ts)
        assert dt is not None

    def test_get_log_multiple_entries_in_order(self, points_file):
        """Multiple awards produce multiple ordered log entries."""
        award("add_seed", points_file=points_file)
        award("new_root", count=2, points_file=points_file)
        award("edit_seed", points_file=points_file)
        log = get_log(points_file=points_file)
        assert len(log) == 3
        assert log[0]["event"] == "add_seed"
        assert log[1]["event"] == "new_root"
        assert log[2]["event"] == "edit_seed"

    def test_get_log_corrupt_file_returns_empty(self, points_file):
        """get_log() returns [] if points.json is corrupt."""
        points_file.write_text("bad json")
        assert get_log(points_file=points_file) == []
