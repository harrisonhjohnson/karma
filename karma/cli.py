"""Karma CLI entry points."""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import click
from slugify import slugify

KARMA_DIR = Path(__file__).parent.parent
SEEDS_DIR = KARMA_DIR / "seeds"
ROOTS_FILE = KARMA_DIR / "roots.json"
POINTS_FILE = KARMA_DIR / "points.json"


def _ensure_seeds_dir() -> None:
    SEEDS_DIR.mkdir(exist_ok=True)


def _seed_path(slug: str) -> Path:
    return SEEDS_DIR / f"{slug}.md"


def _build_frontmatter(title: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    return f"---\ntitle: {title}\ncreated_at: {now}\ntags: []\n---\n\n"


def _open_editor(filepath: Path, content: str) -> str:
    """Write content to file, open $EDITOR, return final file content."""
    filepath.write_text(content, encoding="utf-8")
    editor = os.environ.get("EDITOR", "vi")
    try:
        subprocess.run([editor, str(filepath)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return filepath.read_text(encoding="utf-8")


def _read_roots_count() -> int:
    if not ROOTS_FILE.exists():
        return 0
    try:
        data = json.loads(ROOTS_FILE.read_text(encoding="utf-8"))
        # Count unique edges (each edge stored bidirectionally, so divide total by 2)
        total = sum(len(v) for v in data.values())
        return total // 2
    except (json.JSONDecodeError, AttributeError):
        return 0


def _read_points_total() -> int:
    if not POINTS_FILE.exists():
        return 0
    try:
        data = json.loads(POINTS_FILE.read_text(encoding="utf-8"))
        return data.get("total", 0)
    except (json.JSONDecodeError, AttributeError):
        return 0


@click.group()
def cli():
    """Karma — personal knowledge graph."""
    pass


@cli.command()
@click.argument("title")
def add(title: str) -> None:
    """Add a new seed. Opens $EDITOR to write content."""
    _ensure_seeds_dir()
    slug = slugify(title)
    filepath = _seed_path(slug)

    if filepath.exists():
        click.echo(f"Seed already exists: {filepath}")
        sys.exit(1)

    content = _build_frontmatter(title)
    final_content = _open_editor(filepath, content)

    click.echo(f"Seed saved: seeds/{slug}.md")

    # Hook for KARMA-003: run relationship discovery
    try:
        from karma.discovery import discover_roots, update_roots

        new_roots = discover_roots(slug)
        if new_roots:
            update_roots(slug, new_roots)
            click.echo(f"  Roots discovered: {len(new_roots)}")
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        click.echo(f"  Discovery skipped: {exc}", err=True)

    # Hook for KARMA-004: award points
    try:
        from karma.points import award

        pts = award("add_seed")
        click.echo(f"  +{pts} karma points (add_seed)")

        if new_roots:
            root_pts = award("new_root", count=len(new_roots))
            click.echo(f"  +{root_pts} karma points ({len(new_roots)} new roots)")
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        click.echo(f"  Points skipped: {exc}", err=True)


@cli.command()
def status() -> None:
    """Show seed count, root count, and karma points."""
    _ensure_seeds_dir()
    seed_count = len(list(SEEDS_DIR.glob("*.md")))
    root_count = _read_roots_count()
    points_total = _read_points_total()

    click.echo("Karma Status")
    click.echo(f"  Seeds:  {seed_count}")
    click.echo(f"  Roots:  {root_count}")
    click.echo(f"  Points: {points_total}")


@cli.command()
def chat() -> None:
    """Launch the Streamlit knowledge graph UI."""
    import subprocess as sp

    app_path = Path(__file__).parent / "ui" / "app.py"
    sp.run(["streamlit", "run", str(app_path)], check=False)
