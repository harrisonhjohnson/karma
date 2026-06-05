"""Tests for relationship discovery engine (KARMA-003).

All tests use small fake numpy embeddings — sentence-transformers model
is never downloaded or loaded in this test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from karma.discovery import (
    _cosine_similarity,
    discover_roots,
    load_embeddings,
    load_roots,
    save_embeddings,
    save_roots,
    update_roots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed(directory: Path, slug: str, body: str = "Some content.") -> Path:
    """Create a minimal seed .md file in directory."""
    p = directory / f"{slug}.md"
    p.write_text(
        f"---\ntitle: {slug}\ncreated_at: 2026-01-01T00:00:00Z\ntags: []\n---\n\n{body}",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Cosine similarity unit tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        a = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_partial_similarity(self):
        a = np.array([1.0, 1.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        sim = _cosine_similarity(a, b)
        assert 0.5 < sim < 1.0

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.5])
        assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# Embeddings cache I/O
# ---------------------------------------------------------------------------


class TestEmbeddingsIO:
    def test_load_embeddings_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("karma.discovery.EMBEDDINGS_FILE", tmp_path / "emb.json")
        result = load_embeddings()
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        emb_file = tmp_path / "embeddings.json"
        monkeypatch.setattr("karma.discovery.EMBEDDINGS_FILE", emb_file)
        data = {"seed-a": [0.1, 0.2, 0.3], "seed-b": [0.9, 0.8, 0.7]}
        save_embeddings(data)
        loaded = load_embeddings()
        assert loaded == data

    def test_load_malformed_json_returns_empty(self, tmp_path, monkeypatch):
        emb_file = tmp_path / "embeddings.json"
        emb_file.write_text("not-valid-json")
        monkeypatch.setattr("karma.discovery.EMBEDDINGS_FILE", emb_file)
        assert load_embeddings() == {}


# ---------------------------------------------------------------------------
# Roots I/O
# ---------------------------------------------------------------------------


class TestRootsIO:
    def test_load_roots_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("karma.discovery.ROOTS_FILE", tmp_path / "roots.json")
        assert load_roots() == {}

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        roots_file = tmp_path / "roots.json"
        monkeypatch.setattr("karma.discovery.ROOTS_FILE", roots_file)
        data = {"seed-a": ["seed-b"], "seed-b": ["seed-a"]}
        save_roots(data)
        loaded = load_roots()
        assert loaded == data

    def test_load_malformed_json_returns_empty(self, tmp_path, monkeypatch):
        roots_file = tmp_path / "roots.json"
        roots_file.write_text("{bad json}")
        monkeypatch.setattr("karma.discovery.ROOTS_FILE", roots_file)
        assert load_roots() == {}


# ---------------------------------------------------------------------------
# discover_roots — threshold logic (mocked embeddings, no model)
# ---------------------------------------------------------------------------


class TestDiscoverRoots:
    """All tests pass pre-computed embeddings so no model is loaded."""

    def _make_embeddings(self) -> dict[str, list[float]]:
        """Three seeds: a and b are very similar, c is orthogonal to both."""
        a = [1.0, 0.9, 0.0, 0.0]
        b = [0.95, 0.85, 0.05, 0.0]
        c = [0.0, 0.0, 1.0, 1.0]
        return {"seed-a": a, "seed-b": b, "seed-c": c}

    def test_similar_seeds_discovered_above_threshold(self, tmp_path):
        """seed-a and seed-b are similar; discover_roots(seed-a) returns [seed-b]."""
        embeddings = self._make_embeddings()
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()
        _make_seed(seeds_dir, "seed-a")

        result = discover_roots(
            "seed-a",
            floor=0.45,
            seeds_dir=seeds_dir,
            embeddings_override=embeddings,
        )
        assert "seed-b" in result
        assert "seed-c" not in result
        assert "seed-a" not in result  # self-reference excluded

    def test_dissimilar_seed_not_discovered(self, tmp_path):
        """seed-c is orthogonal; discover_roots(seed-c) returns empty list."""
        embeddings = self._make_embeddings()
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()
        _make_seed(seeds_dir, "seed-c")

        result = discover_roots(
            "seed-c",
            floor=0.45,
            seeds_dir=seeds_dir,
            embeddings_override=embeddings,
        )
        assert result == []

    def test_high_floor_excludes_moderate_matches(self, tmp_path):
        """At floor=0.98, the moderate match seed-b (sim 0.6) is excluded."""
        embeddings = {
            "seed-a": [1.0, 0.0, 0.0, 0.0],
            "seed-b": [0.6, 0.8, 0.0, 0.0],  # sim = 0.6, below 0.98
            "seed-c": [0.0, 0.0, 1.0, 1.0],
        }
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()
        _make_seed(seeds_dir, "seed-a")

        result = discover_roots(
            "seed-a",
            floor=0.98,
            seeds_dir=seeds_dir,
            embeddings_override=embeddings,
        )
        assert result == []

    def test_missing_seed_file_returns_empty(self, tmp_path):
        """If seed file doesn't exist, returns empty list."""
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()

        result = discover_roots(
            "nonexistent-seed",
            seeds_dir=seeds_dir,
            embeddings_override={},
        )
        assert result == []

    def test_no_other_seeds_returns_empty(self, tmp_path):
        """Single seed in embeddings; no roots possible."""
        embeddings = {"seed-alone": [1.0, 0.0, 0.0]}
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()
        _make_seed(seeds_dir, "seed-alone")

        result = discover_roots(
            "seed-alone",
            seeds_dir=seeds_dir,
            embeddings_override=embeddings,
        )
        assert result == []

    def test_top_k_caps_results_strongest_first(self, tmp_path):
        """More candidates above the floor than top_k -> only the strongest
        top_k are returned, in descending similarity order."""
        embeddings = {
            "seed-a": [1.0, 0.0, 0.0, 0.0],
            "near-1": [0.99, 0.14, 0.0, 0.0],
            "near-2": [0.95, 0.31, 0.0, 0.0],
            "near-3": [0.90, 0.44, 0.0, 0.0],
            "near-4": [0.85, 0.53, 0.0, 0.0],  # dropped at top_k=3
        }
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()
        _make_seed(seeds_dir, "seed-a")

        result = discover_roots(
            "seed-a",
            floor=0.3,
            top_k=3,
            seeds_dir=seeds_dir,
            embeddings_override=embeddings,
        )
        assert result == ["near-1", "near-2", "near-3"]

    def test_moderate_match_linked_at_default_floor(self, tmp_path):
        """A moderate match (sim ~0.42) that a 0.45 cutoff would orphan is linked
        at the default floor — the top-K fix for related-but-not-identical notes."""
        embeddings = {
            "seed-a": [1.0, 0.0, 0.0, 0.0],
            "seed-b": [0.42, 0.91, 0.0, 0.0],  # sim ~0.42 (< old 0.45 cutoff)
            "seed-c": [0.0, 0.0, 1.0, 1.0],  # unrelated
        }
        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()
        _make_seed(seeds_dir, "seed-a")

        result = discover_roots(  # default floor=0.35, top_k=5
            "seed-a",
            seeds_dir=seeds_dir,
            embeddings_override=embeddings,
        )
        assert "seed-b" in result
        assert "seed-c" not in result


# ---------------------------------------------------------------------------
# update_roots — bidirectional graph update
# ---------------------------------------------------------------------------


class TestUpdateRoots:
    def test_adds_entry_for_seed(self):
        """Seed slug gets an entry in the updated roots dict."""
        roots = update_roots("seed-a", ["seed-b"], roots_override={})
        assert "seed-a" in roots

    def test_bidirectional_link_created(self):
        """Both directions are stored: seed-a -> seed-b and seed-b -> seed-a."""
        roots = update_roots("seed-a", ["seed-b"], roots_override={})
        assert "seed-b" in roots["seed-a"]
        assert "seed-a" in roots["seed-b"]

    def test_no_duplicates_on_repeated_update(self):
        """Running update_roots twice doesn't create duplicate entries."""
        roots = update_roots("seed-a", ["seed-b"], roots_override={})
        roots = update_roots("seed-a", ["seed-b"], roots_override=roots)
        assert roots["seed-a"].count("seed-b") == 1
        assert roots["seed-b"].count("seed-a") == 1

    def test_multiple_roots_all_added(self):
        """All slugs in new_roots are added bidirectionally."""
        roots = update_roots("seed-a", ["seed-b", "seed-c"], roots_override={})
        assert "seed-b" in roots["seed-a"]
        assert "seed-c" in roots["seed-a"]
        assert "seed-a" in roots["seed-b"]
        assert "seed-a" in roots["seed-c"]

    def test_existing_roots_preserved(self):
        """Pre-existing edges are not removed when new ones are added."""
        existing = {"seed-a": ["seed-z"], "seed-z": ["seed-a"]}
        roots = update_roots("seed-a", ["seed-b"], roots_override=existing)
        assert "seed-z" in roots["seed-a"]
        assert "seed-b" in roots["seed-a"]

    def test_empty_new_roots(self):
        """Passing an empty list doesn't corrupt existing graph."""
        existing = {"seed-a": ["seed-b"], "seed-b": ["seed-a"]}
        roots = update_roots("seed-x", [], roots_override=existing)
        assert "seed-x" in roots
        assert roots["seed-x"] == []
        # Existing edges untouched
        assert "seed-b" in roots["seed-a"]

    def test_returns_updated_dict(self):
        """update_roots returns the updated dict (when using roots_override)."""
        result = update_roots("seed-a", ["seed-b"], roots_override={})
        assert isinstance(result, dict)
        assert len(result) == 2
