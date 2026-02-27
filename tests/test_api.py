"""Tests for karma/api.py — FastAPI endpoints (KARMA-015).

Uses FastAPI TestClient with tmp_path fixtures to override module-level path
variables (SEEDS_DIR, ROOTS_FILE, POINTS_FILE, EMBEDDINGS_FILE). This matches
the override pattern used in test_discovery.py and test_chat.py.

No real model calls or real Anthropic API calls are made.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from karma.api import app
import karma.api as karma_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_seed(seeds_dir: Path, slug: str, title: str, body: str = "") -> Path:
    """Create a seed .md file with frontmatter."""
    now = datetime.now(timezone.utc).isoformat()
    path = seeds_dir / f"{slug}.md"
    path.write_text(
        f"---\ntitle: {title}\ncreated_at: {now}\ntags: []\n---\n\n{body}",
        encoding="utf-8",
    )
    return path


def _write_roots(roots_file: Path, data: dict) -> None:
    roots_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _patch_paths(tmp_path: Path):
    """Return a context-manager-style fixture that patches all path vars."""
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    roots_file = tmp_path / "roots.json"
    points_file = tmp_path / "points.json"
    embeddings_file = tmp_path / "embeddings.json"
    return seeds_dir, roots_file, points_file, embeddings_file


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------


def test_status_empty(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file
    karma_api.POINTS_FILE = points_file
    karma_api.EMBEDDINGS_FILE = embeddings_file

    client = TestClient(app)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"seeds": 0, "roots": 0, "points": 0}


def test_status_with_seeds(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file
    karma_api.POINTS_FILE = points_file
    karma_api.EMBEDDINGS_FILE = embeddings_file

    _write_seed(seeds_dir, "alpha", "Alpha")
    _write_seed(seeds_dir, "beta", "Beta")
    # 1 unique edge (bidirectional)
    _write_roots(roots_file, {"alpha": ["beta"], "beta": ["alpha"]})
    points_file.write_text(json.dumps({"total": 15, "log": []}), encoding="utf-8")

    client = TestClient(app)
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["seeds"] == 2
    assert data["roots"] == 1
    assert data["points"] == 15


# ---------------------------------------------------------------------------
# GET /api/seeds
# ---------------------------------------------------------------------------


def test_list_seeds_empty(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    client = TestClient(app)
    resp = client.get("/api/seeds")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_seeds_returns_summaries(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    _write_seed(seeds_dir, "alpha", "Alpha Seed", "Some body text")
    _write_seed(seeds_dir, "beta", "Beta Seed")
    _write_roots(roots_file, {"alpha": ["beta"], "beta": ["alpha"]})

    client = TestClient(app)
    resp = client.get("/api/seeds")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2

    slugs = {item["slug"] for item in items}
    assert slugs == {"alpha", "beta"}

    alpha = next(i for i in items if i["slug"] == "alpha")
    assert alpha["title"] == "Alpha Seed"
    assert alpha["root_count"] == 1
    assert alpha["created_at"] is not None


def test_list_seeds_no_roots_file(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file  # does not exist

    _write_seed(seeds_dir, "solo", "Solo Seed")

    client = TestClient(app)
    resp = client.get("/api/seeds")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["root_count"] == 0


# ---------------------------------------------------------------------------
# GET /api/seeds/{slug}
# ---------------------------------------------------------------------------


def test_get_seed_not_found(tmp_path):
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    client = TestClient(app)
    resp = client.get("/api/seeds/nonexistent")
    assert resp.status_code == 404
    assert "nonexistent" in resp.json()["detail"]


def test_get_seed_returns_detail(tmp_path):
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    _write_seed(seeds_dir, "my-seed", "My Seed", "Hello world content")
    _write_roots(roots_file, {"my-seed": ["other"], "other": ["my-seed"]})

    client = TestClient(app)
    resp = client.get("/api/seeds/my-seed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "my-seed"
    assert data["title"] == "My Seed"
    assert data["body"] == "Hello world content"
    assert "other" in data["roots"]


def test_get_seed_strips_frontmatter(tmp_path):
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    _write_seed(seeds_dir, "test-seed", "Test Seed", "Body only text here")

    client = TestClient(app)
    resp = client.get("/api/seeds/test-seed")
    assert resp.status_code == 200
    body = resp.json()["body"]
    assert "---" not in body
    assert "title:" not in body
    assert "Body only text here" in body


# ---------------------------------------------------------------------------
# POST /api/seeds
# ---------------------------------------------------------------------------


def test_create_seed(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file
    karma_api.POINTS_FILE = points_file
    karma_api.EMBEDDINGS_FILE = embeddings_file

    # Patch discovery at its source to avoid loading sentence-transformers
    with patch("karma.discovery.discover_roots", return_value=[]), \
         patch("karma.discovery.save_embeddings"):
        client = TestClient(app)
        resp = client.post("/api/seeds", json={"title": "New Seed", "body": "Seed body"})

    assert resp.status_code == 201
    data = resp.json()
    assert data["slug"] == "new-seed"
    assert data["pts_awarded"] >= 10  # at least add_seed points
    assert isinstance(data["new_roots"], list)

    # File should exist on disk
    assert (seeds_dir / "new-seed.md").exists()


def test_create_seed_conflict(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file
    karma_api.POINTS_FILE = points_file
    karma_api.EMBEDDINGS_FILE = embeddings_file

    _write_seed(seeds_dir, "existing", "Existing")

    client = TestClient(app)
    resp = client.post("/api/seeds", json={"title": "Existing", "body": ""})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


def test_create_seed_missing_title(tmp_path):
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir

    client = TestClient(app)
    resp = client.post("/api/seeds", json={"body": "No title"})
    assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# PUT /api/seeds/{slug}
# ---------------------------------------------------------------------------


def test_update_seed_not_found(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file
    karma_api.POINTS_FILE = points_file

    client = TestClient(app)
    resp = client.put("/api/seeds/ghost", json={"body": "new body"})
    assert resp.status_code == 404


def test_update_seed_updates_body(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file
    karma_api.POINTS_FILE = points_file
    karma_api.EMBEDDINGS_FILE = embeddings_file

    _write_seed(seeds_dir, "my-note", "My Note", "Original body")

    with patch("karma.discovery.discover_roots", return_value=[]), \
         patch("karma.discovery.save_embeddings"):
        client = TestClient(app)
        resp = client.put("/api/seeds/my-note", json={"body": "Updated body"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "my-note"
    assert data["pts_awarded"] >= 2  # edit_seed = 2 pts

    # Verify file updated
    content = (seeds_dir / "my-note.md").read_text(encoding="utf-8")
    assert "Updated body" in content
    assert "My Note" in content  # title preserved in frontmatter


def test_update_seed_preserves_frontmatter(tmp_path):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file
    karma_api.POINTS_FILE = points_file
    karma_api.EMBEDDINGS_FILE = embeddings_file

    _write_seed(seeds_dir, "keep-fm", "Keep Frontmatter", "Original")

    with patch("karma.discovery.discover_roots", return_value=[]), \
         patch("karma.discovery.save_embeddings"):
        client = TestClient(app)
        resp = client.put("/api/seeds/keep-fm", json={"body": "New body text"})

    assert resp.status_code == 200
    content = (seeds_dir / "keep-fm.md").read_text(encoding="utf-8")
    assert "title: Keep Frontmatter" in content
    assert "New body text" in content


# ---------------------------------------------------------------------------
# GET /api/graph
# ---------------------------------------------------------------------------


def test_graph_empty(tmp_path):
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    client = TestClient(app)
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"nodes": [], "edges": []}


def test_graph_node_structure(tmp_path):
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    _write_seed(seeds_dir, "alpha", "Alpha")
    _write_seed(seeds_dir, "beta", "Beta")
    _write_roots(roots_file, {"alpha": ["beta"], "beta": ["alpha"]})

    client = TestClient(app)
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["nodes"]) == 2
    node_ids = {n["id"] for n in data["nodes"]}
    assert node_ids == {"alpha", "beta"}

    for node in data["nodes"]:
        assert "id" in node
        assert "label" in node
        assert "color" in node
        assert "value" in node


def test_graph_node_colors(tmp_path):
    """Node color depends on root count: gray=0, blue=1-2, gold=3+."""
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    # iso has 0 roots -> gray
    # connected has 1 root -> blue
    _write_seed(seeds_dir, "iso", "Isolated")
    _write_seed(seeds_dir, "connected", "Connected")
    _write_roots(roots_file, {"connected": ["iso"], "iso": ["connected"]})

    client = TestClient(app)
    resp = client.get("/api/graph")
    data = resp.json()

    nodes = {n["id"]: n for n in data["nodes"]}
    # iso has 1 root (connected -> iso), so blue
    assert nodes["iso"]["color"] == "#4a9eff"
    assert nodes["connected"]["color"] == "#4a9eff"


def test_graph_isolated_node_is_gray(tmp_path):
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    _write_seed(seeds_dir, "lone", "Lone Seed")
    # No roots file -> 0 roots

    client = TestClient(app)
    resp = client.get("/api/graph")
    data = resp.json()
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["color"] == "#aaaaaa"
    assert data["edges"] == []


def test_graph_edges_deduplicated(tmp_path):
    """Bidirectional roots stored in roots.json should yield only 1 edge."""
    seeds_dir, roots_file, *_ = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.ROOTS_FILE = roots_file

    _write_seed(seeds_dir, "a", "A")
    _write_seed(seeds_dir, "b", "B")
    _write_roots(roots_file, {"a": ["b"], "b": ["a"]})

    client = TestClient(app)
    resp = client.get("/api/graph")
    data = resp.json()

    assert len(data["edges"]) == 1
    edge = data["edges"][0]
    assert set([edge["from"], edge["to"]]) == {"a", "b"}


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------


def test_chat_no_api_key(tmp_path, monkeypatch):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.EMBEDDINGS_FILE = embeddings_file

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    client = TestClient(app)
    resp = client.post("/api/chat", json={"query": "What do I know?"})
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_chat_returns_answer(tmp_path, monkeypatch):
    seeds_dir, roots_file, points_file, embeddings_file = _patch_paths(tmp_path)
    karma_api.SEEDS_DIR = seeds_dir
    karma_api.EMBEDDINGS_FILE = embeddings_file

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Mock the chat.answer function to avoid real API + model calls
    mock_result = {"answer": "Here is what I know.", "sources": ["Seed One", "Seed Two"]}

    with patch("karma.chat.answer", return_value=mock_result):
        client = TestClient(app)
        resp = client.post("/api/chat", json={"query": "Tell me everything"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Here is what I know."
    assert "Seed One" in data["sources"]
    assert "Seed Two" in data["sources"]


def test_chat_missing_query(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = TestClient(app)
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 422  # Pydantic validation error
