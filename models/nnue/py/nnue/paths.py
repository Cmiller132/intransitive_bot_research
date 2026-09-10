"""Workspace locations: the root is the nearest ancestor of this package that
holds Cargo.toml and Project.md; runs and datasets live under it whatever the
working directory is."""

from __future__ import annotations

from pathlib import Path


def workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (parent / "Project.md").is_file():
            return parent
    raise FileNotFoundError(f"no workspace root (Cargo.toml and Project.md) above {__file__}")


def run_dir(name: str) -> Path:
    """`<workspace root>/runs/<name>`."""
    return workspace_root() / "runs" / name


def data_dir(name: str) -> Path:
    """`<workspace root>/runs/nnue_data/<name>`: one dataset (DESIGN item 18)."""
    return workspace_root() / "runs" / "nnue_data" / name
