"""Player directories: specs, validation and uploads."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import stat
import tarfile
from pathlib import Path
from typing import Any

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
KINDS = ("sq", "conv", "rpsi")
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")


def split_spec(spec: Any) -> tuple[str, str] | None:
    """Split a player spec into (kind, rest), or None when it is malformed."""
    if not isinstance(spec, str) or ":" not in spec:
        return None
    kind, rest = spec.split(":", 1)
    if kind not in KINDS or not rest.strip():
        return None
    return kind, rest.strip()


def resolve_spec(spec: str, directory: str | os.PathLike[str]) -> str:
    """Turn a stored spec into the command-line spec with absolute paths."""
    parts = split_spec(spec)
    if parts is None:
        raise ValueError(f"invalid player spec {spec!r}")
    kind, rest = parts
    directory = Path(directory)
    if kind == "rpsi":
        tokens = rest.split()
        if not os.path.isabs(tokens[0]):
            tokens[0] = str((directory / tokens[0]).resolve())
        return "rpsi:" + " ".join(tokens)
    return f"{kind}:{(directory / rest).resolve()}"


def spec_targets(spec: str, directory: str | os.PathLike[str]) -> list[Path]:
    """The files a spec requires to exist inside the player directory."""
    parts = split_spec(spec)
    if parts is None:
        raise ValueError(f"invalid player spec {spec!r}")
    kind, rest = parts
    directory = Path(directory)
    if kind == "rpsi":
        first = rest.split()[0]
        return [] if os.path.isabs(first) else [directory / first]
    path = directory / rest
    return [path, path.with_name(path.name + ".json")]


def read_player_json(directory: Path) -> dict[str, str] | None:
    """Read and validate one player.json, returning its fields or None."""
    try:
        meta = json.loads((directory / "player.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name, spec, added = meta.get("name"), meta.get("spec"), meta.get("added")
    if not isinstance(name, str) or not NAME_RE.match(name):
        return None
    if split_spec(spec) is None or not isinstance(added, str):
        return None
    return {"name": name, "spec": spec, "added": added}


def scan_players(players_dir: Path) -> list[dict[str, str]]:
    """Every valid player directory under the players root."""
    found = []
    if not players_dir.is_dir():
        return found
    for entry in sorted(os.listdir(players_dir)):
        directory = players_dir / entry
        if not directory.is_dir() or not NAME_RE.match(entry):
            continue
        meta = read_player_json(directory)
        if meta is not None and meta["name"] == entry:
            found.append(meta)
    return found


def extract_archive(data: bytes, directory: Path) -> None:
    """Extract a gzipped tar into the directory, rejecting escaping members."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = member.name.replace("\\", "/").split("/")
            if member.name.startswith("/") or ".." in parts:
                raise ValueError(f"unsafe archive member {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"archive member {member.name!r} is not a file")
        for member in archive.getmembers():
            archive.extract(member, directory)


def install_upload(players_dir: Path, name: str, spec: str, files: list[tuple[str, bytes]], added: str) -> str | None:
    """Write an uploaded player into players/<name>; returns an error or None."""
    directory = players_dir / name
    directory.mkdir(parents=True)
    try:
        for filename, data in files:
            base = os.path.basename(filename.replace("\\", "/"))
            if not base:
                raise ValueError("file without a name")
            if base.endswith(ARCHIVE_SUFFIXES):
                extract_archive(data, directory)
            else:
                (directory / base).write_bytes(data)
        for target in spec_targets(spec, directory):
            if not target.exists():
                raise ValueError(f"spec references missing file {target.name}")
        kind, rest = split_spec(spec)
        if kind == "rpsi":
            first = directory / rest.split()[0]
            if first.is_file():
                mode = first.stat().st_mode
                first.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        meta = {"name": name, "spec": spec, "added": added}
        (directory / "player.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    except (OSError, ValueError, tarfile.TarError) as error:
        shutil.rmtree(directory, ignore_errors=True)
        return str(error)
    return None
