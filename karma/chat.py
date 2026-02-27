"""Claude-powered RAG chat over the knowledge base (KARMA-006).

Pattern:
  1. Embed the user query with sentence-transformers (all-MiniLM-L6-v2)
  2. Cosine similarity against all cached seed embeddings
  3. Retrieve top-5 seeds by score
  4. Load their markdown content
  5. Call Claude API (claude-sonnet-4-6) with seeds injected in system prompt
  6. Return { "answer": str, "sources": [seed_title, ...] }

ANTHROPIC_API_KEY is read from the environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

KARMA_DIR = Path(__file__).parent.parent
SEEDS_DIR = KARMA_DIR / "seeds"
EMBEDDINGS_FILE = KARMA_DIR / "embeddings.json"

_MODEL = None  # lazy-loaded

SYSTEM_PROMPT_TEMPLATE = (
    "You are a knowledge assistant for Harrison's personal knowledge base. "
    "Answer questions using only the provided seeds. "
    "If the answer isn't in the seeds, say so. "
    "Cite which seeds you used.\n\n"
    "Seeds:\n{seed_content}"
)

TOP_K = 5


def _get_model():
    """Lazy-load sentence-transformers model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _load_embeddings(embeddings_file: Path = EMBEDDINGS_FILE) -> dict[str, list[float]]:
    if not embeddings_file.exists():
        return {}
    try:
        return json.loads(embeddings_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_seed_title(seed_path: Path) -> str:
    """Extract title from YAML frontmatter or fall back to slug."""
    try:
        text = seed_path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", maxsplit=2)
            if len(parts) >= 2:
                for line in parts[1].splitlines():
                    if line.startswith("title:"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return seed_path.stem


def _read_seed_body(seed_path: Path) -> str:
    """Return seed content (strips YAML frontmatter)."""
    try:
        text = seed_path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", maxsplit=2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text.strip()
    except OSError:
        return ""


def retrieve_top_seeds(
    query: str,
    top_k: int = TOP_K,
    *,
    seeds_dir: Path = SEEDS_DIR,
    embeddings_file: Path = EMBEDDINGS_FILE,
    query_embedding_override: list[float] | None = None,
) -> list[dict]:
    """Retrieve top-k seeds most similar to query.

    Args:
        query: User's question string.
        top_k: How many seeds to return.
        seeds_dir: Override for testing.
        embeddings_file: Override for testing.
        query_embedding_override: Pre-computed embedding for testing (skips model).

    Returns:
        List of dicts: [{ "slug": str, "title": str, "body": str, "score": float }, ...]
        Ordered by descending similarity score.
    """
    embeddings = _load_embeddings(embeddings_file)

    if not embeddings:
        return []

    # Embed the query
    if query_embedding_override is not None:
        query_vec = np.array(query_embedding_override, dtype=np.float32)
    else:
        model = _get_model()
        query_vec = np.array(
            model.encode(query, show_progress_bar=False), dtype=np.float32
        )

    # Score all seeds
    scored: list[tuple[str, float]] = []
    for slug, vec_list in embeddings.items():
        vec = np.array(vec_list, dtype=np.float32)
        if vec.size == 0:
            continue
        score = _cosine_similarity(query_vec, vec)
        scored.append((slug, score))

    # Sort descending, take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    results: list[dict] = []
    for slug, score in top:
        seed_path = seeds_dir / f"{slug}.md"
        title = _read_seed_title(seed_path) if seed_path.exists() else slug
        body = _read_seed_body(seed_path) if seed_path.exists() else ""
        results.append({"slug": slug, "title": title, "body": body, "score": score})

    return results


def _build_system_prompt(seeds: list[dict]) -> str:
    """Build the system prompt by injecting seed content."""
    if not seeds:
        seed_content = "(No seeds available)"
    else:
        parts = []
        for seed in seeds:
            parts.append(f"--- Seed: {seed['title']} ---\n{seed['body']}")
        seed_content = "\n\n".join(parts)
    return SYSTEM_PROMPT_TEMPLATE.format(seed_content=seed_content)


def answer(
    query: str,
    *,
    seeds_dir: Path = SEEDS_DIR,
    embeddings_file: Path = EMBEDDINGS_FILE,
    query_embedding_override: list[float] | None = None,
    anthropic_client=None,
) -> dict:
    """Answer a question using RAG over the knowledge base.

    Args:
        query: User's question.
        seeds_dir: Override for testing.
        embeddings_file: Override for testing.
        query_embedding_override: Pre-computed query embedding for testing.
        anthropic_client: Injected Anthropic client (for mocking in tests).

    Returns:
        { "answer": str, "sources": [seed_title, ...] }
    """
    # Retrieve top seeds
    seeds = retrieve_top_seeds(
        query,
        seeds_dir=seeds_dir,
        embeddings_file=embeddings_file,
        query_embedding_override=query_embedding_override,
    )

    system_prompt = _build_system_prompt(seeds)
    source_titles = [s["title"] for s in seeds]

    # Call Claude
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )

    response_text = message.content[0].text

    return {"answer": response_text, "sources": source_titles}
