# Karma

[![CI](https://github.com/harrisonhjohnson/karma/actions/workflows/ci.yml/badge.svg)](https://github.com/harrisonhjohnson/karma/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Personal knowledge graph with auto-discovered relationships.

Karma stores your notes as markdown seeds and automatically discovers semantic
relationships between them — no manual linking required. As you add seeds, the
graph grows richer on its own, surfacing connections you might not have noticed.

## Installation

```
pip install -e .
```

Requires Python 3.11+. Set your Anthropic API key for the chat interface:

```
export ANTHROPIC_API_KEY=your-key-here
```

## Quickstart

Add a seed (opens `$EDITOR`):

```
karma add "Machine Learning Basics"
# Seed saved: seeds/machine-learning-basics.md
#   Roots discovered: 2
#   +10 karma points (add_seed)
#   +10 karma points (2 new roots)
```

Check your graph status:

```
karma status
# Karma Status
#   Seeds:  7
#   Roots:  4
#   Points: 120
```

Launch the Streamlit UI (graph + chat):

```
karma chat
# Opens Streamlit in your browser at http://localhost:8501
```

Launch the full Web UI (recommended):

```
karma web
# Karma Web UI starting at http://localhost:8765
# Opens browser automatically
```

## How Discovery Works

When you add or edit a seed, Karma reads its full text and computes a semantic
fingerprint (an embedding) using a local language model that runs entirely on
your machine with no API calls. It then compares that fingerprint against all
your existing seeds. Any two seeds whose content is semantically similar enough
automatically become connected — these connections are called roots.

Roots are stored bidirectionally in `roots.json`. When you open the Graph view,
every seed appears as a node and every root appears as an edge. Seeds with more
connections appear larger and change color: gray (isolated), blue (1-2 roots),
gold (3+ roots). The graph updates each time you add a seed.

## Karma Points

Karma awards points as you grow your knowledge graph:

| Event | Points |
|-------|--------|
| Add new seed | +10 |
| New root discovered (per root) | +5 |
| Edit existing seed | +2 |
| Edit that triggers new root | +5 per new root |

Points accumulate in `points.json` with a full log of every event.

## Web UI

`karma web` launches a standalone single-page app on `localhost:8765`. It has
three panels side by side:

- **Graph** — vis.js interactive network of all seeds and roots. Click any node
  to open that seed in the Editor.
- **Editor** — title field and markdown textarea with a live rendered preview
  on the right. Save creates a new seed or updates an existing one, runs
  relationship discovery, and refreshes the graph automatically.
- **Chat** — ask questions about your knowledge base. Answers show which seeds
  were used as context below each response.

The karma score is always visible in the top bar and updates after each save.

Streamlit (`karma chat`) remains available as a fallback.

## File Layout

```
karma/
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── karma/
│   ├── cli.py           # karma add, karma chat, karma status, karma web
│   ├── api.py           # FastAPI backend: /api/* endpoints
│   ├── discovery.py     # semantic similarity, roots.json management
│   ├── points.py        # karma points logic
│   ├── chat.py          # Claude RAG chat backend
│   ├── ui/              # Streamlit UI (karma chat)
│   │   ├── app.py
│   │   ├── graph_view.py
│   │   └── chat_view.py
│   └── web/             # Single-page web UI (karma web)
│       ├── index.html
│       ├── style.css
│       └── app.js
├── seeds/               # your markdown seeds (one file per seed)
├── roots.json           # auto-generated relationship graph
├── points.json          # karma points log
├── embeddings.json      # cached seed embeddings (auto-generated)
└── tests/
    ├── test_api.py
    ├── test_discovery.py
    ├── test_points.py
    └── test_cli.py
```
