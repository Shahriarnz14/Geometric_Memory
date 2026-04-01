"""Path helpers for tiny-graph notebooks and utilities."""

from __future__ import annotations

from pathlib import Path


def get_tiny_graphs_root() -> Path:
    """Returns the package root for `tiny_graphs_notebooks`."""
    return Path(__file__).resolve().parents[1]


def resolve_from_tiny_graphs_root(path_like: str | Path) -> Path:
    """Resolves relative paths against the tiny-graphs package root."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (get_tiny_graphs_root() / path).resolve()


def resolve_from_cwd(path_like: str | Path) -> Path:
    """Resolves relative paths against the current working directory."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path.cwd().resolve() / path).resolve()
