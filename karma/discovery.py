"""Relationship discovery engine — semantic similarity via sentence-transformers.

KARMA-003: Embeds seeds using all-MiniLM-L6-v2, computes cosine similarity,
maintains roots.json adjacency list (bidirectional), caches embeddings in
embeddings.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

KARMA_DIR = Path(__file__).parent.parent
SEEDS_DIR = KARMA_DIR / "seeds"
ROOTS_FILE = KARMA_DIR / "roots.json"
EMBEDDINGS_FILE = KARMA_DIR / "embeddings.json"

_MODEL = None  # lazy-loaded to avoid slow import at CLI startup


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_embeddings() -> dict[str, list[float]]:
    """Load cached embeddings from embeddings.json.

    Returns a dict mapping seed slug -> embedding vector (as list of floats).
    Returns empty dict if file missing or malformed.
    """
    if not EMBEDDINGS_FILE.exists():
        return {}
    try:
        return json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Persist embeddings dict to embeddings.json."""
    EMBEDDINGS_FILE.write_text(
        json.dumps(embeddings, separators=(",", ":")), encoding="utf-8"
    )


def embed_seed(path: Path) -> np.ndarray:
    """Embed the full text of a seed file.

    Strips YAML frontmatter before embedding so metadata doesn't skew similarity.
    Returns a 1-D numpy float32 array.
    """
    text = path.read_text(encoding="utf-8")

    # Strip YAML frontmatter (--- ... ---)
    if text.startswith("---"):
        parts = text.split("---", maxsplit=2)
        if len(parts) >= 3:
            text = parts[2].strip()

    model = _get_model()
    vector = model.encode(text, show_progress_bar=False)
    return np.array(vector, dtype=np.float32)


def discover_roots(
    seed_slug: str,
    threshold: float = 0.45,
    *,
    seeds_dir: Path | None = None,
    embeddings_override: dict[str, list[float]] | None = None,
) -> list[str]:
    """Find seeds semantically similar to `seed_slug` above `threshold`.

    1. Embeds the target seed (or uses embeddings_override for testing).
    2. Compares against all cached embeddings.
    3. Returns list of slugs whose cosine similarity >= threshold (excluding self).
    4. Updates embeddings.json with the new seed's embedding.

    Args:
        seed_slug: Slug of the seed to find roots for.
        threshold: Cosine similarity threshold (default 0.45).
        seeds_dir: Override seeds directory (for testing).
        embeddings_override: Pre-loaded embeddings dict (for testing, skips model).

    Returns:
        List of related seed slugs (may be empty).
    """
    _seeds_dir = seeds_dir or SEEDS_DIR
    seed_path = _seeds_dir / f"{seed_slug}.md"

    if not seed_path.exists():
        return []

    # Load existing embeddings
    if embeddings_override is not None:
        embeddings = dict(embeddings_override)
    else:
        embeddings = load_embeddings()

    # Embed the new/updated seed
    if embeddings_override is not None:
        # In test mode, use provided embedding for the seed itself
        target_vector = np.array(embeddings.get(seed_slug, []), dtype=np.float32)
    else:
        target_vector = embed_seed(seed_path)
        embeddings[seed_slug] = target_vector.tolist()
        save_embeddings(embeddings)

    if target_vector.size == 0:
        return []

    related: list[str] = []
    for slug, vec_list in embeddings.items():
        if slug == seed_slug:
            continue
        other_vector = np.array(vec_list, dtype=np.float32)
        if other_vector.size == 0:
            continue
        sim = _cosine_similarity(target_vector, other_vector)
        if sim >= threshold:
            related.append(slug)

    return related


def load_roots() -> dict[str, list[str]]:
    """Load roots.json adjacency list.

    Returns dict mapping seed slug -> list of related slugs.
    Returns {} if file missing or malformed.
    """
    if not ROOTS_FILE.exists():
        return {}
    try:
        return json.loads(ROOTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_roots(roots: dict[str, list[str]]) -> None:
    """Persist roots adjacency list to roots.json."""
    ROOTS_FILE.write_text(
        json.dumps(roots, indent=2, sort_keys=True), encoding="utf-8"
    )


def update_roots(
    seed_slug: str,
    new_roots: list[str],
    *,
    roots_override: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Merge new_roots into the roots.json adjacency list (bidirectional).

    For each slug in new_roots:
    - Adds slug to seed_slug's neighbor list (if not already present).
    - Adds seed_slug to slug's neighbor list (if not already present).

    Args:
        seed_slug: The seed whose roots were just discovered.
        new_roots: List of related seed slugs.
        roots_override: Pre-loaded roots dict (for testing).

    Returns:
        Updated roots dict.
    """
    if roots_override is not None:
        roots = {k: list(v) for k, v in roots_override.items()}
    else:
        roots = load_roots()

    # Ensure the seed itself has an entry
    if seed_slug not in roots:
        roots[seed_slug] = []

    for related_slug in new_roots:
        # Add related_slug to seed's list
        if related_slug not in roots[seed_slug]:
            roots[seed_slug].append(related_slug)

        # Add seed to related_slug's list (bidirectional)
        if related_slug not in roots:
            roots[related_slug] = []
        if seed_slug not in roots[related_slug]:
            roots[related_slug].append(seed_slug)

    if roots_override is None:
        save_roots(roots)

    return roots
