"""Graph view for Karma Streamlit UI (KARMA-005).

Reads roots.json and seeds/ directory. Renders an interactive pyvis network
where nodes are seeds and edges are roots (semantic relationships).

Node coloring by root count:
  0 roots  -> gray  (#aaaaaa)
  1-2 roots -> blue  (#4a9eff)
  3+ roots -> gold  (#f0a500)

Node size is proportional to degree (number of connections), clamped to
a readable range [15, 50].
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

KARMA_DIR = Path(__file__).parent.parent.parent
SEEDS_DIR = KARMA_DIR / "seeds"
ROOTS_FILE = KARMA_DIR / "roots.json"


def _color_for_degree(degree: int) -> str:
    if degree == 0:
        return "#aaaaaa"
    if degree <= 2:
        return "#4a9eff"
    return "#f0a500"


def _size_for_degree(degree: int, min_size: int = 15, max_size: int = 50) -> int:
    """Scale node size: degree 0 -> min_size, degree 10+ -> max_size."""
    clamped = min(degree, 10)
    return min_size + int((max_size - min_size) * (clamped / 10))


def _read_title(seed_path: Path) -> str:
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
    # Fall back to slug (filename without extension)
    return seed_path.stem


def _load_roots(roots_file: Path = ROOTS_FILE) -> dict[str, list[str]]:
    if not roots_file.exists():
        return {}
    try:
        data = json.loads(roots_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _load_seed_titles(seeds_dir: Path = SEEDS_DIR) -> dict[str, str]:
    """Return mapping: slug -> display title."""
    titles: dict[str, str] = {}
    if not seeds_dir.exists():
        return titles
    for path in seeds_dir.glob("*.md"):
        slug = path.stem
        titles[slug] = _read_title(path)
    return titles


def render_graph_view(
    roots_file: Path = ROOTS_FILE,
    seeds_dir: Path = SEEDS_DIR,
) -> None:
    """Render the interactive pyvis graph inside Streamlit."""
    from pyvis.network import Network  # type: ignore

    roots = _load_roots(roots_file)
    titles = _load_seed_titles(seeds_dir)

    # Merge: a seed with no roots still deserves a node if it exists on disk
    all_slugs: set[str] = set(roots.keys()) | set(titles.keys())

    if not all_slugs:
        st.info('Add seeds with `karma add` to grow your graph.')
        return

    # Build degree map
    degree: dict[str, int] = {slug: len(roots.get(slug, [])) for slug in all_slugs}

    # Node / edge counts for header
    node_count = len(all_slugs)
    # Count unique edges (each stored bidirectionally)
    raw_edge_count = sum(len(v) for v in roots.values())
    edge_count = raw_edge_count // 2

    st.markdown(f"**{node_count} seeds** | **{edge_count} roots**")
    st.caption(
        "Seeds are your notes. Roots are auto-discovered connections between them."
    )
    st.caption(
        "Node colors: gray = isolated, blue = 1-2 roots, gold = 3+ roots"
    )

    # Build pyvis network
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        notebook=False,
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    # Add nodes
    for slug in all_slugs:
        label = titles.get(slug, slug)
        d = degree[slug]
        net.add_node(
            slug,
            label=label,
            color=_color_for_degree(d),
            size=_size_for_degree(d),
            title=f"{label}\n{d} root(s)",
        )

    # Add edges (only once per pair)
    seen_edges: set[frozenset[str]] = set()
    for slug, neighbors in roots.items():
        for neighbor in neighbors:
            edge_key = frozenset([slug, neighbor])
            if edge_key not in seen_edges:
                net.add_edge(slug, neighbor)
                seen_edges.add(edge_key)

    # Generate HTML and embed via st.components
    html = net.generate_html()
    components.html(html, height=620, scrolling=False)
