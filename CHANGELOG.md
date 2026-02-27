# Changelog

All notable changes to Karma will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.0] - 2026-02-27

### Added

- `karma web` CLI command: starts FastAPI server on `localhost:8765` and opens
  the single-page web UI in the browser; `--port` and `--no-browser` flags
  available (KARMA-013)
- `karma/api.py`: FastAPI REST backend exposing 7 endpoints (`GET /api/seeds`,
  `GET /api/seeds/{slug}`, `POST /api/seeds`, `PUT /api/seeds/{slug}`,
  `GET /api/graph`, `POST /api/chat`, `GET /api/status`); all endpoints use
  Pydantic models for request/response validation (KARMA-012)
- `karma/web/`: single-page web frontend served by FastAPI
  - `index.html`: three-panel shell (Graph, Editor, Chat)
  - `style.css`: dark theme matching Streamlit palette; fully responsive column
    layout with sticky top bar
  - `app.js`: vis.js interactive graph (click node opens seed in Editor),
    seed editor with live markdown preview via marked.js, chat panel with
    context seed attribution (KARMA-014)
- Karma score badge always visible in the web UI top bar; updates after each
  seed save (KARMA-014)
- `tests/test_api.py`: 22 pytest tests covering all 7 API endpoints using
  FastAPI TestClient and tmp_path path overrides; no real model or API calls
  made in any test (KARMA-015)
- Dependencies: `fastapi>=0.100.0`, `uvicorn[standard]>=0.23.0`,
  `httpx>=0.24.0` added to `pyproject.toml` (KARMA-012)

### Fixed

- Graph view: added onboarding captions explaining seeds/roots jargon and node
  color legend below the seeds/roots metric header (KARMA-009)
- Chat view: renamed "Sources:" label to "Context seeds:" to accurately
  describe that the listed seeds were passed as context, not necessarily cited
  by Claude (KARMA-010)
- Chat view: added empty-state guard — if `seeds/` has no `.md` files, shows
  an info message and hides the chat input instead of allowing a confusing
  "no seeds available" error from Claude (KARMA-011)

## [0.1.0] - 2026-02-26

### Added

- Project structure, `pyproject.toml`, CLI entry points, and test scaffold (KARMA-001)
- `karma add <title>` CLI command: opens `$EDITOR`, saves seed to `seeds/<slug>.md`
  with YAML frontmatter (`title`, `created_at`, `tags`), runs relationship discovery,
  and awards karma points on exit (KARMA-002)
- `karma status` CLI command: prints seed count, root count, and total karma points
  (KARMA-002)
- Relationship discovery engine using `sentence-transformers` (model: `all-MiniLM-L6-v2`,
  runs locally with no API calls); computes cosine similarity between seed embeddings;
  pairs above threshold 0.45 become bidirectional roots; adjacency list persisted to
  `roots.json`; embeddings cached in `embeddings.json` to avoid recomputation (KARMA-003)
- Karma points system: awards points for `add_seed` (+10), `new_root` (+5 per root),
  `edit_seed` (+2), and `edit_root` (+5 per new root triggered); full event log
  persisted to `points.json` (KARMA-004)
- Streamlit graph view: interactive pyvis network rendering seeds as nodes and roots as
  edges; degree-based node coloring (gray = 0 roots, blue = 1-2 roots, gold = 3+);
  node size scales with degree; empty state placeholder shown when no seeds exist
  (KARMA-005)
- `karma chat` CLI command: launches Streamlit UI at localhost (KARMA-005)
- Claude-powered chat interface using RAG pattern: embeds user query, retrieves top-5
  semantically similar seeds, injects seed content into Claude system prompt, cites
  source seed titles in every response (KARMA-006)
- `karma/ui/chat_view.py`: Streamlit chat UI with `st.chat_input` / `st.chat_message`,
  session history, and per-message source attribution (KARMA-006)
