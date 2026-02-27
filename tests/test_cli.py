"""Tests for karma CLI commands (KARMA-002)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from karma.cli import cli, SEEDS_DIR, ROOTS_FILE, POINTS_FILE


@pytest.fixture()
def tmp_karma_dir(tmp_path, monkeypatch):
    """Redirect SEEDS_DIR, ROOTS_FILE, POINTS_FILE to a temp location."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    roots = tmp_path / "roots.json"
    points = tmp_path / "points.json"

    monkeypatch.setattr("karma.cli.SEEDS_DIR", seeds)
    monkeypatch.setattr("karma.cli.ROOTS_FILE", roots)
    monkeypatch.setattr("karma.cli.POINTS_FILE", points)

    return {"seeds": seeds, "roots": roots, "points": points, "base": tmp_path}


class TestAdd:
    def test_add_creates_seed_file(self, tmp_karma_dir):
        """karma add creates the correctly slugified .md file."""
        runner = CliRunner()
        seeds_dir = tmp_karma_dir["seeds"]

        # Mock editor to avoid opening a real editor
        with patch("karma.cli.subprocess.run") as mock_run:
            mock_run.return_value = None
            # Simulate editor writing content (file already has frontmatter from _open_editor)
            result = runner.invoke(cli, ["add", "My First Seed"])

        assert result.exit_code == 0, result.output
        seed_file = seeds_dir / "my-first-seed.md"
        assert seed_file.exists(), f"Expected {seed_file} to exist"

    def test_add_creates_correct_slug(self, tmp_karma_dir):
        """Slugification handles spaces, uppercase, and special chars."""
        runner = CliRunner()
        seeds_dir = tmp_karma_dir["seeds"]

        with patch("karma.cli.subprocess.run"):
            runner.invoke(cli, ["add", "Hello World! Test"])

        assert (seeds_dir / "hello-world-test.md").exists()

    def test_add_writes_frontmatter(self, tmp_karma_dir):
        """Saved seed file has YAML frontmatter with title and created_at."""
        runner = CliRunner()
        seeds_dir = tmp_karma_dir["seeds"]

        with patch("karma.cli.subprocess.run"):
            runner.invoke(cli, ["add", "Frontmatter Test"])

        content = (seeds_dir / "frontmatter-test.md").read_text()
        assert "title: Frontmatter Test" in content
        assert "created_at:" in content
        assert "tags:" in content

    def test_add_duplicate_exits_nonzero(self, tmp_karma_dir):
        """Adding a seed that already exists exits with error."""
        runner = CliRunner()
        seeds_dir = tmp_karma_dir["seeds"]
        # Pre-create the file
        (seeds_dir / "existing-seed.md").write_text("---\ntitle: Existing Seed\n---\n")

        result = runner.invoke(cli, ["add", "Existing Seed"])
        assert result.exit_code != 0

    def test_add_prints_saved_message(self, tmp_karma_dir):
        """karma add prints confirmation of saved file path."""
        runner = CliRunner()

        with patch("karma.cli.subprocess.run"):
            result = runner.invoke(cli, ["add", "Saved Message Test"])

        assert "seeds/saved-message-test.md" in result.output


class TestStatus:
    def test_status_empty(self, tmp_karma_dir):
        """status shows zeros when no seeds, roots, or points exist."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Seeds:  0" in result.output
        assert "Roots:  0" in result.output
        assert "Points: 0" in result.output

    def test_status_counts_seeds(self, tmp_karma_dir):
        """status correctly counts .md files in seeds/."""
        seeds_dir = tmp_karma_dir["seeds"]
        (seeds_dir / "seed-one.md").write_text("---\ntitle: Seed One\n---\n")
        (seeds_dir / "seed-two.md").write_text("---\ntitle: Seed Two\n---\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Seeds:  2" in result.output

    def test_status_reads_roots_count(self, tmp_karma_dir):
        """status reads root count from roots.json (counts unique edges)."""
        roots_file = tmp_karma_dir["roots"]
        # Bidirectional graph: a-b and b-a = 1 unique edge
        roots_data = {
            "seed-a": ["seed-b", "seed-c"],
            "seed-b": ["seed-a"],
            "seed-c": ["seed-a"],
        }
        roots_file.write_text(json.dumps(roots_data))

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        # Total edges = 4, divide by 2 = 2 unique edges
        assert "Roots:  2" in result.output

    def test_status_reads_points_total(self, tmp_karma_dir):
        """status reads karma points from points.json."""
        points_file = tmp_karma_dir["points"]
        points_file.write_text(json.dumps({"total": 42, "log": []}))

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Points: 42" in result.output

    def test_status_missing_json_files(self, tmp_karma_dir):
        """status shows 0 for roots and points when JSON files are missing."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Roots:  0" in result.output
        assert "Points: 0" in result.output

    def test_status_shows_header(self, tmp_karma_dir):
        """status output includes 'Karma Status' header."""
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])

        assert "Karma Status" in result.output
