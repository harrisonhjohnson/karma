"""Karma FastAPI backend (KARMA-012).

Exposes the karma knowledge base over a REST API consumed by the Phase 2
web frontend (karma/web/). All heavy lifting delegates to existing modules:
  discovery.py — relationship discovery and roots.json management
  points.py    — karma points tracking
  chat.py      — Claude RAG chat

Path overrides (SEEDS_DIR, ROOTS_FILE, etc.) are module-level variables so
tests can monkeypatch them without importing the full module.

Endpoints:
  GET  /api/seeds              — list all seeds
  GET  /api/seeds/{slug}       — get one seed
  POST /api/seeds              — create seed (runs discovery + awards points)
  PUT  /api/seeds/{slug}       — update seed body
  GET  /api/graph              — vis.js-compatible graph data
  POST /api/chat               — RAG chat query
  GET  /api/status             — seed/root/points summary
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slugify import slugify

# ---------------------------------------------------------------------------
# Path configuration — monkeypatch these in tests
# ---------------------------------------------------------------------------

KARMA_DIR = Path(__file__).parent.parent
SEEDS_DIR = KARMA_DIR / "seeds"
ROOTS_FILE = KARMA_DIR / "roots.json"
POINTS_FILE = KARMA_DIR / "points.json"
EMBEDDINGS_FILE = KARMA_DIR / "embeddings.json"
WEB_DIR = Path(__file__).parent / "web"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SeedSummary(BaseModel):
    slug: str
    title: str
    created_at: Optional[str] = None
    root_count: int


class SeedDetail(BaseModel):
    slug: str
    title: str
    body: str
    roots: list[str]


class CreateSeedRequest(BaseModel):
    title: str
    body: str = ""


class UpdateSeedRequest(BaseModel):
    body: str


class SeedWriteResponse(BaseModel):
    slug: str
    pts_awarded: int
    new_roots: list[str]


class GraphNode(BaseModel):
    id: str
    label: str
    color: str
    value: int  # degree, used for node sizing


class GraphEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = {"populate_by_name": True}


class GraphData(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class StatusResponse(BaseModel):
    seeds: int
    roots: int
    points: int


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _read_frontmatter(seed_path: Path) -> dict:
    """Parse YAML frontmatter from a seed file. Returns {} on failure."""
    try:
        text = seed_path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", maxsplit=2)
            if len(parts) >= 2:
                fm: dict = {}
                for line in parts[1].splitlines():
                    if ":" in line:
                        key, _, val = line.partition(":")
                        fm[key.strip()] = val.strip()
                return fm
    except OSError:
        pass
    return {}


def _read_seed_title(seed_path: Path) -> str:
    """Extract title from frontmatter or fall back to slug."""
    fm = _read_frontmatter(seed_path)
    return fm.get("title", seed_path.stem)


def _read_seed_body(seed_path: Path) -> str:
    """Return seed content with YAML frontmatter stripped."""
    try:
        text = seed_path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", maxsplit=2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text.strip()
    except OSError:
        return ""


def _load_roots() -> dict[str, list[str]]:
    if not ROOTS_FILE.exists():
        return {}
    try:
        data = json.loads(ROOTS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _count_unique_roots(roots: dict[str, list[str]]) -> int:
    total = sum(len(v) for v in roots.values())
    return total // 2


def _color_for_degree(degree: int) -> str:
    if degree == 0:
        return "#aaaaaa"
    if degree <= 2:
        return "#4a9eff"
    return "#f0a500"


def _build_frontmatter(title: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    return f"---\ntitle: {title}\ncreated_at: {now}\ntags: []\n---\n\n"


def _ensure_seeds_dir() -> None:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Karma API",
    description="REST backend for the Karma knowledge graph",
    version="0.2.0",
)


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------


@app.get("/api/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Return seed count, root count, and total karma points."""
    _ensure_seeds_dir()
    seed_count = len(list(SEEDS_DIR.glob("*.md")))
    roots = _load_roots()
    root_count = _count_unique_roots(roots)

    # Read points total
    points_total = 0
    if POINTS_FILE.exists():
        try:
            data = json.loads(POINTS_FILE.read_text(encoding="utf-8"))
            points_total = data.get("total", 0)
        except (json.JSONDecodeError, OSError):
            pass

    return StatusResponse(seeds=seed_count, roots=root_count, points=points_total)


# ---------------------------------------------------------------------------
# GET /api/seeds
# ---------------------------------------------------------------------------


@app.get("/api/seeds", response_model=list[SeedSummary])
def list_seeds() -> list[SeedSummary]:
    """List all seeds with slug, title, created_at, root_count."""
    _ensure_seeds_dir()
    roots = _load_roots()
    results: list[SeedSummary] = []
    for path in sorted(SEEDS_DIR.glob("*.md")):
        slug = path.stem
        fm = _read_frontmatter(path)
        title = fm.get("title", slug)
        created_at = fm.get("created_at")
        root_count = len(roots.get(slug, []))
        results.append(
            SeedSummary(slug=slug, title=title, created_at=created_at, root_count=root_count)
        )
    return results


# ---------------------------------------------------------------------------
# GET /api/seeds/{slug}
# ---------------------------------------------------------------------------


@app.get("/api/seeds/{slug}", response_model=SeedDetail)
def get_seed(slug: str) -> SeedDetail:
    """Get a single seed by slug. Returns 404 if not found."""
    seed_path = SEEDS_DIR / f"{slug}.md"
    if not seed_path.exists():
        raise HTTPException(status_code=404, detail=f"Seed '{slug}' not found")

    title = _read_seed_title(seed_path)
    body = _read_seed_body(seed_path)
    roots = _load_roots()
    seed_roots = roots.get(slug, [])

    return SeedDetail(slug=slug, title=title, body=body, roots=seed_roots)


# ---------------------------------------------------------------------------
# POST /api/seeds
# ---------------------------------------------------------------------------


@app.post("/api/seeds", response_model=SeedWriteResponse, status_code=201)
def create_seed(req: CreateSeedRequest) -> SeedWriteResponse:
    """Create a new seed. Runs relationship discovery and awards points."""
    _ensure_seeds_dir()
    slug = slugify(req.title)
    seed_path = SEEDS_DIR / f"{slug}.md"

    if seed_path.exists():
        raise HTTPException(status_code=409, detail=f"Seed '{slug}' already exists")

    # Write seed file
    content = _build_frontmatter(req.title)
    if req.body:
        content += req.body
    seed_path.write_text(content, encoding="utf-8")

    # Run relationship discovery
    new_roots: list[str] = []
    try:
        from karma.discovery import discover_roots, update_roots

        new_roots = discover_roots(
            slug,
            seeds_dir=SEEDS_DIR,
        )
        if new_roots:
            update_roots(slug, new_roots)
    except Exception:  # noqa: BLE001
        pass

    # Award points
    pts_awarded = 0
    try:
        from karma.points import award

        pts_awarded += award("add_seed", points_file=POINTS_FILE)
        if new_roots:
            pts_awarded += award("new_root", count=len(new_roots), points_file=POINTS_FILE)
    except Exception:  # noqa: BLE001
        pass

    return SeedWriteResponse(slug=slug, pts_awarded=pts_awarded, new_roots=new_roots)


# ---------------------------------------------------------------------------
# PUT /api/seeds/{slug}
# ---------------------------------------------------------------------------


@app.put("/api/seeds/{slug}", response_model=SeedWriteResponse)
def update_seed(slug: str, req: UpdateSeedRequest) -> SeedWriteResponse:
    """Update a seed's body. Runs discovery and awards edit points. Returns 404 if not found."""
    seed_path = SEEDS_DIR / f"{slug}.md"
    if not seed_path.exists():
        raise HTTPException(status_code=404, detail=f"Seed '{slug}' not found")

    # Preserve frontmatter, replace body
    existing = seed_path.read_text(encoding="utf-8")
    if existing.startswith("---"):
        parts = existing.split("---", maxsplit=2)
        if len(parts) >= 3:
            new_content = f"---{parts[1]}---\n\n{req.body}"
        else:
            new_content = req.body
    else:
        new_content = req.body
    seed_path.write_text(new_content, encoding="utf-8")

    # Run relationship discovery
    new_roots: list[str] = []
    try:
        from karma.discovery import discover_roots, update_roots

        new_roots = discover_roots(slug, seeds_dir=SEEDS_DIR)
        if new_roots:
            update_roots(slug, new_roots)
    except Exception:  # noqa: BLE001
        pass

    # Award points
    pts_awarded = 0
    try:
        from karma.points import award

        pts_awarded += award("edit_seed", points_file=POINTS_FILE)
        if new_roots:
            pts_awarded += award("edit_root", count=len(new_roots), points_file=POINTS_FILE)
    except Exception:  # noqa: BLE001
        pass

    return SeedWriteResponse(slug=slug, pts_awarded=pts_awarded, new_roots=new_roots)


# ---------------------------------------------------------------------------
# GET /api/graph
# ---------------------------------------------------------------------------


@app.get("/api/graph", response_model=GraphData)
def get_graph() -> GraphData:
    """Return vis.js-compatible graph data: nodes and deduplicated edges."""
    _ensure_seeds_dir()
    roots = _load_roots()

    # All slugs: those in roots.json + those on disk without roots
    disk_slugs: set[str] = {p.stem for p in SEEDS_DIR.glob("*.md")}
    all_slugs: set[str] = set(roots.keys()) | disk_slugs

    if not all_slugs:
        return GraphData(nodes=[], edges=[])

    # Build degree map
    degree: dict[str, int] = {slug: len(roots.get(slug, [])) for slug in all_slugs}

    # Build nodes
    nodes: list[GraphNode] = []
    for slug in sorted(all_slugs):
        seed_path = SEEDS_DIR / f"{slug}.md"
        label = _read_seed_title(seed_path) if seed_path.exists() else slug
        d = degree[slug]
        nodes.append(
            GraphNode(
                id=slug,
                label=label,
                color=_color_for_degree(d),
                value=d,
            )
        )

    # Build edges (deduplicated)
    edges: list[GraphEdge] = []
    seen_edges: set[frozenset[str]] = set()
    for slug, neighbors in roots.items():
        for neighbor in neighbors:
            edge_key = frozenset([slug, neighbor])
            if edge_key not in seen_edges:
                edges.append(GraphEdge(**{"from": slug, "to": neighbor}))
                seen_edges.add(edge_key)

    return GraphData(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a question using RAG over the knowledge base."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set. Chat is unavailable.",
        )

    try:
        from karma.chat import answer as chat_answer

        result = chat_answer(
            req.query,
            seeds_dir=SEEDS_DIR,
            embeddings_file=EMBEDDINGS_FILE,
        )
        return ChatResponse(answer=result["answer"], sources=result.get("sources", []))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Static file serving (mounted last so /api/* routes take priority)
# ---------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    """Catch-all: serve index.html for any non-API path (SPA routing)."""
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    # Web UI not built yet — return a plain JSON message
    from fastapi.responses import JSONResponse
    return JSONResponse({"message": "Karma Web UI not found. Run `karma web` after building the frontend."}, status_code=404)
