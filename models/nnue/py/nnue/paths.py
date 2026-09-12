"""Workspace locations: the root is the nearest ancestor of this package that
holds Cargo.toml and Project.md; runs and datasets live under it whatever the
working directory is."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def bot_binary() -> Path:
    """The release `bot`, or the build named by NNUE_BOT (a frozen copy, so a
    running match never holds the file the next `cargo build` replaces)."""
    if os.environ.get("NNUE_BOT"):
        path = Path(os.environ["NNUE_BOT"])
        if not path.is_file():
            raise FileNotFoundError(f"NNUE_BOT={path} is not a file")
        return path.resolve()
    for name in ("bot.exe", "bot"):
        path = workspace_root() / "target" / "release" / name
        if path.is_file():
            return path
    raise FileNotFoundError("build the release `bot` first (cargo build --release -p cli)")
