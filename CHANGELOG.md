# Changelog

All notable changes to Karma will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
