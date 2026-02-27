"""Tests for karma chat module (KARMA-006).

All tests mock the Claude API — no real API calls are made.
The anthropic client is injected via the anthropic_client= parameter.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from karma.chat import (
    _build_system_prompt,
    _cosine_similarity,
    _load_embeddings,
    _read_seed_body,
    _read_seed_title,
    answer,
    retrieve_top_seeds,
    SYSTEM_PROMPT_TEMPLATE,
    TOP_K,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_vec(dim: int = 4, seed: int = 0) -> list[float]:
    """Generate a deterministic unit vector for testing."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _write_seed(seeds_dir: Path, slug: str, title: str, body: str) -> Path:
    path = seeds_dir / f"{slug}.md"
    path.write_text(
        f"---\ntitle: {title}\ncreated_at: 2026-02-26T00:00:00+00:00\ntags: []\n---\n\n{body}",
        encoding="utf-8",
    )
    return path


def _mock_anthropic_client(response_text: str = "Test answer."):
    """Build a minimal mock Anthropic client that returns response_text."""
    client = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    message = MagicMock()
    message.content = [content_block]
    client.messages.create.return_value = message
    return client


@pytest.fixture()
def tmp_seeds_dir(tmp_path) -> Path:
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    return seeds


@pytest.fixture()
def tmp_embeddings_file(tmp_path) -> Path:
    return tmp_path / "embeddings.json"


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.5], dtype=np.float32)
        assert _cosine_similarity(a, b) == 0.0

    def test_opposite_vectors_return_minus_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6


# ---------------------------------------------------------------------------
# _load_embeddings
# ---------------------------------------------------------------------------


class TestLoadEmbeddings:
    def test_returns_empty_when_file_missing(self, tmp_path):
        ef = tmp_path / "embeddings.json"
        assert _load_embeddings(ef) == {}

    def test_loads_valid_file(self, tmp_path):
        ef = tmp_path / "embeddings.json"
        data = {"seed-a": [0.1, 0.2, 0.3]}
        ef.write_text(json.dumps(data))
        result = _load_embeddings(ef)
        assert result == data

    def test_returns_empty_on_corrupt_json(self, tmp_path):
        ef = tmp_path / "embeddings.json"
        ef.write_text("not-json{{{")
        assert _load_embeddings(ef) == {}


# ---------------------------------------------------------------------------
# _read_seed_title / _read_seed_body
# ---------------------------------------------------------------------------


class TestReadSeedTitle:
    def test_reads_title_from_frontmatter(self, tmp_seeds_dir):
        path = _write_seed(tmp_seeds_dir, "my-seed", "My Seed Title", "body text")
        assert _read_seed_title(path) == "My Seed Title"

    def test_falls_back_to_slug_when_no_frontmatter(self, tmp_seeds_dir):
        path = tmp_seeds_dir / "my-seed.md"
        path.write_text("Just plain text, no frontmatter", encoding="utf-8")
        assert _read_seed_title(path) == "my-seed"

    def test_falls_back_to_slug_when_file_missing(self, tmp_seeds_dir):
        path = tmp_seeds_dir / "ghost-seed.md"
        assert _read_seed_title(path) == "ghost-seed"


class TestReadSeedBody:
    def test_strips_frontmatter(self, tmp_seeds_dir):
        path = _write_seed(tmp_seeds_dir, "my-seed", "Title", "Hello world body.")
        body = _read_seed_body(path)
        assert body == "Hello world body."
        assert "title:" not in body

    def test_returns_full_text_when_no_frontmatter(self, tmp_seeds_dir):
        path = tmp_seeds_dir / "plain.md"
        path.write_text("No frontmatter here.", encoding="utf-8")
        assert _read_seed_body(path) == "No frontmatter here."

    def test_returns_empty_when_file_missing(self, tmp_seeds_dir):
        path = tmp_seeds_dir / "ghost.md"
        assert _read_seed_body(path) == ""


# ---------------------------------------------------------------------------
# retrieve_top_seeds
# ---------------------------------------------------------------------------


class TestRetrieveTopSeeds:
    def test_returns_empty_when_no_embeddings(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        result = retrieve_top_seeds(
            "anything",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
        )
        assert result == []

    def test_retrieves_top_seeds_by_similarity(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """The seed whose embedding is closest to query appears first."""
        # query vector: points along axis 0
        query_vec = [1.0, 0.0, 0.0, 0.0]

        # seed-a is close to query, seed-b is orthogonal
        seed_a_vec = [0.99, 0.01, 0.0, 0.0]
        seed_b_vec = [0.0, 1.0, 0.0, 0.0]

        embeddings = {"seed-a": seed_a_vec, "seed-b": seed_b_vec}
        tmp_embeddings_file.write_text(json.dumps(embeddings))

        _write_seed(tmp_seeds_dir, "seed-a", "Seed A", "Content A")
        _write_seed(tmp_seeds_dir, "seed-b", "Seed B", "Content B")

        results = retrieve_top_seeds(
            "query",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=query_vec,
        )

        assert len(results) == 2
        assert results[0]["slug"] == "seed-a"
        assert results[1]["slug"] == "seed-b"

    def test_respects_top_k_limit(self, tmp_seeds_dir, tmp_embeddings_file):
        """retrieve_top_seeds returns at most top_k results."""
        embeddings = {f"seed-{i}": _make_vec(seed=i) for i in range(10)}
        tmp_embeddings_file.write_text(json.dumps(embeddings))
        # Seeds don't need to exist on disk — slug falls back when missing

        results = retrieve_top_seeds(
            "query",
            top_k=3,
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=_make_vec(seed=99),
        )
        assert len(results) == 3

    def test_result_has_required_keys(self, tmp_seeds_dir, tmp_embeddings_file):
        """Each result dict has slug, title, body, score keys."""
        embeddings = {"my-seed": [1.0, 0.0]}
        tmp_embeddings_file.write_text(json.dumps(embeddings))
        _write_seed(tmp_seeds_dir, "my-seed", "My Seed", "Body text")

        results = retrieve_top_seeds(
            "query",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
        )
        assert len(results) == 1
        r = results[0]
        assert "slug" in r
        assert "title" in r
        assert "body" in r
        assert "score" in r

    def test_result_title_from_frontmatter(self, tmp_seeds_dir, tmp_embeddings_file):
        """Title in result comes from seed's YAML frontmatter."""
        embeddings = {"my-seed": [1.0, 0.0]}
        tmp_embeddings_file.write_text(json.dumps(embeddings))
        _write_seed(tmp_seeds_dir, "my-seed", "Frontmatter Title", "Body")

        results = retrieve_top_seeds(
            "q",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
        )
        assert results[0]["title"] == "Frontmatter Title"

    def test_score_is_float(self, tmp_seeds_dir, tmp_embeddings_file):
        embeddings = {"seed-x": [1.0, 0.0]}
        tmp_embeddings_file.write_text(json.dumps(embeddings))

        results = retrieve_top_seeds(
            "q",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
        )
        assert isinstance(results[0]["score"], float)


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_injects_seed_titles(self):
        seeds = [{"title": "Seed Alpha", "body": "Alpha body text"}]
        prompt = _build_system_prompt(seeds)
        assert "Seed Alpha" in prompt

    def test_injects_seed_bodies(self):
        seeds = [{"title": "Seed Alpha", "body": "Alpha body text"}]
        prompt = _build_system_prompt(seeds)
        assert "Alpha body text" in prompt

    def test_multiple_seeds_all_present(self):
        seeds = [
            {"title": "Seed A", "body": "Body A"},
            {"title": "Seed B", "body": "Body B"},
        ]
        prompt = _build_system_prompt(seeds)
        assert "Seed A" in prompt
        assert "Seed B" in prompt
        assert "Body A" in prompt
        assert "Body B" in prompt

    def test_empty_seeds_shows_placeholder(self):
        prompt = _build_system_prompt([])
        assert "(No seeds available)" in prompt

    def test_contains_knowledge_assistant_instruction(self):
        prompt = _build_system_prompt([])
        assert "knowledge assistant" in prompt.lower()

    def test_contains_cite_instruction(self):
        prompt = _build_system_prompt([])
        assert "cite" in prompt.lower()


# ---------------------------------------------------------------------------
# answer() — Claude API is always mocked
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_returns_dict_with_answer_and_sources(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """answer() returns {'answer': str, 'sources': list}."""
        embeddings = {"seed-a": [1.0, 0.0]}
        tmp_embeddings_file.write_text(json.dumps(embeddings))
        _write_seed(tmp_seeds_dir, "seed-a", "Seed Alpha", "Alpha body")

        client = _mock_anthropic_client("Here is the answer.")

        result = answer(
            "What is alpha?",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
            anthropic_client=client,
        )

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Here is the answer."
        assert isinstance(result["sources"], list)

    def test_sources_list_contains_seed_titles(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """Sources list contains titles (not slugs) of retrieved seeds."""
        embeddings = {"seed-a": [1.0, 0.0]}
        tmp_embeddings_file.write_text(json.dumps(embeddings))
        _write_seed(tmp_seeds_dir, "seed-a", "Seed Alpha", "Alpha body")

        client = _mock_anthropic_client("Answer text.")

        result = answer(
            "query",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
            anthropic_client=client,
        )

        assert "Seed Alpha" in result["sources"]

    def test_claude_called_with_correct_model(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """answer() calls Claude with model='claude-sonnet-4-6'."""
        embeddings = {"seed-a": [1.0, 0.0]}
        tmp_embeddings_file.write_text(json.dumps(embeddings))
        _write_seed(tmp_seeds_dir, "seed-a", "Seed Alpha", "Body")

        client = _mock_anthropic_client()

        answer(
            "query",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
            anthropic_client=client,
        )

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs.get("model") == "claude-sonnet-4-6"

    def test_claude_called_with_system_prompt_containing_seed(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """The system prompt passed to Claude contains the seed body."""
        embeddings = {"seed-a": [1.0, 0.0]}
        tmp_embeddings_file.write_text(json.dumps(embeddings))
        _write_seed(tmp_seeds_dir, "seed-a", "Seed Alpha", "Unique body content XYZ")

        client = _mock_anthropic_client()

        answer(
            "query",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
            anthropic_client=client,
        )

        call_kwargs = client.messages.create.call_args.kwargs
        system_prompt = call_kwargs.get("system", "")
        assert "Unique body content XYZ" in system_prompt

    def test_claude_receives_user_query_as_message(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """The user query is passed as the user message to Claude."""
        embeddings = {"seed-a": [1.0, 0.0]}
        tmp_embeddings_file.write_text(json.dumps(embeddings))

        client = _mock_anthropic_client()

        answer(
            "What is the meaning of alpha?",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
            anthropic_client=client,
        )

        call_kwargs = client.messages.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the meaning of alpha?"

    def test_answer_with_no_seeds_still_calls_claude(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """answer() calls Claude even when no seeds exist (empty KB)."""
        client = _mock_anthropic_client("I don't have seeds to reference.")

        result = answer(
            "query",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
            anthropic_client=client,
        )

        client.messages.create.assert_called_once()
        assert result["sources"] == []

    def test_no_seeds_system_prompt_has_placeholder(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """When no seeds, system prompt contains no-seeds placeholder."""
        client = _mock_anthropic_client("No seeds answer.")

        answer(
            "query",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=[1.0, 0.0],
            anthropic_client=client,
        )

        call_kwargs = client.messages.create.call_args.kwargs
        system_prompt = call_kwargs.get("system", "")
        assert "(No seeds available)" in system_prompt

    def test_multiple_seeds_all_appear_in_sources(
        self, tmp_seeds_dir, tmp_embeddings_file
    ):
        """All retrieved seeds appear in sources list."""
        vecs = {f"seed-{i}": _make_vec(seed=i) for i in range(3)}
        tmp_embeddings_file.write_text(json.dumps(vecs))
        for i in range(3):
            _write_seed(tmp_seeds_dir, f"seed-{i}", f"Seed {i}", f"Body {i}")

        client = _mock_anthropic_client("Multi-seed answer.")

        result = answer(
            "query",
            seeds_dir=tmp_seeds_dir,
            embeddings_file=tmp_embeddings_file,
            query_embedding_override=_make_vec(seed=99),
            anthropic_client=client,
        )

        # All 3 seeds should be in sources (top_k=5, only 3 exist)
        assert len(result["sources"]) == 3
