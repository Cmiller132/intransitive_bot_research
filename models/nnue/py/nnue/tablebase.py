"""The endgame-tablebase probe contract, shared by `nnue.importer selfplay
--tablebase-labels` and the prober being built on the Rust side. Both sides
point at this docstring.

    bot tb-probe --data <dir> --input <jsonl> --output <jsonl>

`--data` is the tablebase's data directory, `--input` a file of positions and
`--output` the file of values to write. The prober reads one JSON object per
line,

    {"board": [81 integers], "since_capture": n}

and writes one JSON object per input line, in the same order,

    {"value": 1 | 0 | -1 | null}

1 the side to move wins with best play, -1 loses, 0 draws, null the position
lies outside the tablebase or is unknown. Exactly one answer per position: the
importer refuses a shorter or longer file, and a value that is none of the
four. Blank lines in either file are skipped.

`board` is the dataset's cell encoding (`nnue.data` FIELDS["board"], the
mover's frame), 81 cells rank-major with square = 9 * rank + file: square 0 is
the mover's own home corner, square 80 the goal the mover advances toward (the
opponent advances toward square 0). 0 is an empty cell, 1, 2 and 3 the mover's
rock, paper and scissors, 4, 5 and 6 the opponent's rock, paper and scissors.
Rock beats scissors, scissors beats paper, paper beats rock, so an enemy cell
`e` beats a mover's cell `o` exactly when `(e - o) % 3 == 1`. `since_capture`
is the plies since the last capture (the dataset's column of that name); the
game's capture clock is the dataset's, not the line's, so a prober that needs
it takes it from the set's provenance.

The importer probes only the rows it may: at most `MAX_PIECES` pieces on the
board and a position the mover can still move in. A non-null value becomes the
row's target (+1, 0 or -1) with kind PROOF and weight 1; a null leaves the
row's search label alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .data import KIND_PROOF
from .paths import bot_binary, sha256

MAX_PIECES = 5  # the tablebase holds positions of at most this many pieces
DEFAULT_COMMAND = "bot tb-probe"
DATA_ENV = "NNUE_TABLEBASE_DATA"  # the data directory when no --tablebase-data is given
INDEX_NAMES = ("manifest.json", "index.json", "index")  # what a tablebase build may name its index
VALUES = (-1, 0, 1)


def default_data() -> Path | None:
    """The data directory of the environment, or None: a tablebase lives
    wherever its owner built it, so this package holds no path of its own."""
    return Path(os.environ[DATA_ENV]) if os.environ.get(DATA_ENV) else None


def split_command(command: str) -> list[str]:
    """A prober command line as argv. Splitting keeps Windows backslashes (a
    quoted word may hold spaces), and a leading `bot` is the release binary
    `paths.bot_binary` resolves, so the default command runs this workspace's
    build and a test can name its own script instead."""
    parts = [p for p in shlex.split(command, posix=False) if p]
    parts = [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]
    if not parts:
        raise ValueError("the prober command is empty")
    if parts[0].lower() in ("bot", "bot.exe"):
        parts[0] = str(bot_binary())
    return parts


def data_identity(directory: Path) -> dict:
    """What the values came from: the hash of the directory's index or
    manifest when it has one, else a hash over its files' relative names and
    sizes. These rows become exact labels, so the set that produced them is
    pinned in the provenance the way a teacher's network is."""
    for name in INDEX_NAMES:
        index = directory / name
        if index.is_file():
            return {"kind": name, "sha256": sha256(index)}
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in files:
        digest.update(f"{path.relative_to(directory).as_posix()}\0{path.stat().st_size}\n".encode())
    return {"kind": "names and sizes", "files": len(files), "sha256": digest.hexdigest()}


def resolve(command: str = DEFAULT_COMMAND, data_dir: Path | None = None) -> dict:
    """The prober as argv with its binary and its data directory pinned, or a
    clear error: the caller resolves before it reads a record, so a missing
    binary or directory costs nothing. The returned block is the provenance's."""
    argv = split_command(command)
    binary = Path(argv[0]) if Path(argv[0]).is_file() else None
    if binary is None:
        found = shutil.which(argv[0])
        binary = Path(found) if found else None
    if binary is None:
        raise FileNotFoundError(f"the tablebase prober {argv[0]!r} is neither a file nor on PATH")
    if data_dir is None:
        raise FileNotFoundError(f"the tablebase needs a data directory (--tablebase-data or {DATA_ENV})")
    directory = Path(data_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"the tablebase data directory {directory} does not exist")
    return {
        "command": command,
        "argv": argv,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "data": str(directory),
        "data_identity": data_identity(directory),
    }


def eligible(board: np.ndarray) -> np.ndarray:
    """Mask of the rows the tablebase can be asked about: at most `MAX_PIECES`
    pieces and a position the mover can still move in (a terminal position is
    not a tablebase entry)."""
    import engine

    board = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    small = np.count_nonzero(board, axis=1) <= MAX_PIECES
    for row in np.flatnonzero(small).tolist():
        if not any(engine.legal_mask(board[row].tolist())):
            small[row] = False
    return small


def probe(
    board: np.ndarray,
    since_capture: np.ndarray,
    command: str | list[str] = DEFAULT_COMMAND,
    data_dir: Path | None = None,
) -> list[int | None]:
    """The prober's value for every position, in order (the module's contract).
    Raises when the prober fails, writes no file, answers a different number of
    positions or answers something other than 1, 0, -1 or null."""
    argv = split_command(command) if isinstance(command, str) else list(command)
    board = np.asarray(board, dtype=np.uint8).reshape(-1, 81)
    since = np.asarray(since_capture).reshape(-1)
    with tempfile.TemporaryDirectory(prefix="nnue_tb_") as scratch:
        positions, answers = Path(scratch) / "positions.jsonl", Path(scratch) / "values.jsonl"
        with positions.open("w", encoding="utf-8") as sink:
            for cells, plies in zip(board.tolist(), since.tolist(), strict=True):
                sink.write(json.dumps({"board": [int(c) for c in cells], "since_capture": int(plies)}) + "\n")
        argv = [*argv, "--data", str(data_dir), "--input", str(positions), "--output", str(answers)]
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"{argv[0]} exited with {result.returncode}: {result.stderr.strip()[-300:]}")
        if not answers.is_file():
            raise RuntimeError(f"{argv[0]} wrote no answers: {result.stderr.strip()[-300:]}")
        values: list[int | None] = []
        for line in answers.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get("value")
            if value is not None and int(value) not in VALUES:
                raise ValueError(f"the prober answered {value!r}, not 1, 0, -1 or null")
            values.append(None if value is None else int(value))
    if len(values) != len(board):
        raise ValueError(f"the prober answered {len(values)} of {len(board)} positions")
    return values


def relabel(rows: dict[str, np.ndarray], prober: dict) -> dict:
    """Give every row the tablebase can answer its exact value: the target +1,
    0 or -1 from the mover's view, kind PROOF and weight 1. A row answered null
    keeps its search label, and a set without an eligible row never starts the
    prober. `prober` is a `resolve` block, so the binary and the data directory
    are known good and pinned. The columns are relabelled in place; the
    provenance block is returned."""
    index = np.flatnonzero(eligible(rows["board"]))
    stats = {
        **prober,
        "max_pieces": MAX_PIECES,
        "eligible": int(len(index)),
        "relabelled": {"1": 0, "0": 0, "-1": 0},
        "unknown": 0,
        "contract": "nnue.tablebase: one {board, since_capture} per line in, one {value} per line out; a "
        "non-null value is the row's target with kind PROOF and weight 1",
    }
    if not len(index):
        return stats
    values = probe(rows["board"][index], rows["since_capture"][index], prober["argv"], prober["data"])
    for row, value in zip(index.tolist(), values, strict=True):
        if value is None:
            stats["unknown"] += 1
            continue
        rows["target"][row] = float(value)
        rows["kind"][row] = KIND_PROOF
        rows["weight"][row] = 1.0
        stats["relabelled"][str(value)] += 1
    return stats
