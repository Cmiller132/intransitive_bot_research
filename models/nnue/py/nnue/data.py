"""Datasets (DESIGN item 18): one directory of columnar NumPy arrays plus
`provenance.json`, memory-mapped when read, and the batch sampler that mixes
several datasets by declared shares.

Every row is one position in the mover's frame with an exact clock context and
one label. Kinds: 0 a played root labelled with its lambda return, 1 the child
of a visited candidate labelled minus the candidate's completed Q, 2 an exact
proof (win 1 / loss -1 / draw 0), 3 a teacher search value, 4 a raw network
value, 5 a prototype replay row whose counters are averages (source domain
only). Splits: 0 train, 1 validation, 2 test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIELDS: dict[str, np.dtype] = {
    "board": np.dtype(np.uint8),  # (N, 81) cell codes, mover's frame
    "since_capture": np.dtype(np.uint16),
    "ply": np.dtype(np.uint16),
    "capture_clock": np.dtype(np.uint16),  # the game's clock, plies without a capture that draw
    "target": np.dtype(np.float32),  # value in [-1, 1] from the mover's view
    "weight": np.dtype(np.float32),
    "kind": np.dtype(np.uint8),
    "outcome": np.dtype(np.int8),  # final result from the mover's view, valid when outcome_ok
    "outcome_ok": np.dtype(np.bool_),
    "source": np.dtype(np.uint8),  # producer id recorded in provenance.json
    "game": np.dtype(np.uint32),  # episode id within the dataset
    "orbit": np.dtype(np.uint64),  # symmetry-orbit hash of the board
    "split": np.dtype(np.uint8),
}
KIND_RETURN, KIND_CHILD, KIND_PROOF, KIND_TEACHER, KIND_RAW, KIND_AVERAGED = range(6)
TRAIN, VALIDATION, TEST = range(3)
# Producers: conv windows, the prototype's replay and searched arrays, the
# student's own games (nnue.collect) and the site's human games; labels keep
# the producer.
SOURCE_CONV, SOURCE_PROTOTYPE_REPLAY, SOURCE_PROTOTYPE_SEARCHED, SOURCE_STUDENT, SOURCE_HUMAN = 1, 2, 3, 4, 5


IDS_FILE = "ids.npy"  # optional cache: the feature ids of every row, (N, 2, 42) int16


def encode_ids(directory: Path, chunk: int = 1 << 18) -> Path:
    """Write the feature-id cache of a dataset (`python -m nnue.data encode <set>`):
    the trainer then gathers ids instead of encoding boards per batch."""
    from .features import feature_ids

    directory = Path(directory)
    board = np.load(directory / "board.npy", mmap_mode="r")
    since = np.load(directory / "since_capture.npy", mmap_mode="r")
    clock = np.load(directory / "capture_clock.npy", mmap_mode="r")
    n = len(board)
    tmp = directory / (IDS_FILE + ".tmp")
    ids = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=(n, 2, 42))
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        ids[start:end] = feature_ids(
            np.asarray(board[start:end]), np.asarray(since[start:end]), np.asarray(clock[start:end])
        ).astype(np.int16)
    ids.flush()
    del ids
    tmp.replace(directory / IDS_FILE)
    return directory / IDS_FILE


def write(directory: Path, rows: dict[str, np.ndarray], provenance: dict) -> None:
    """Write one dataset; refuses to overwrite an existing one."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"{directory} exists; datasets are immutable")
    n = len(rows["board"])
    for name, dtype in FIELDS.items():
        array = np.ascontiguousarray(rows[name], dtype=dtype)
        if name == "board":
            if array.shape != (n, 81):
                raise ValueError("board must be (N, 81)")
        elif array.shape != (n,):
            raise ValueError(f"{name} must be (N,)")
        rows[name] = array
    if np.any(np.abs(rows["target"]) > 1.0001) or np.any(rows["capture_clock"] == 0):
        raise ValueError("targets must lie in [-1, 1] and every row needs a clock")
    tmp = directory.with_name(directory.name + ".tmp")
    tmp.mkdir(parents=True)
    for name in FIELDS:
        np.save(tmp / f"{name}.npy", rows[name], allow_pickle=False)
    (tmp / "provenance.json").write_text(
        json.dumps({**provenance, "rows": n, "fields": {k: str(v) for k, v in FIELDS.items()}}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(directory)


def mover_can_capture(board: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """(N,) bool: an enemy piece stands next to an own piece that beats it."""
    from .features import attacked

    out = np.zeros(len(board), dtype=bool)
    for start in range(0, len(board), chunk):
        part = np.asarray(board[start : start + chunk])
        out[start : start + chunk] = (attacked(part) & (part >= 4)).any(1)
    return out


@dataclass
class Dataset:
    """One dataset's arrays, memory-mapped, and the row indices of one split."""

    directory: Path
    rows: dict[str, np.ndarray]
    provenance: dict
    index: np.ndarray

    @classmethod
    def open(
        cls, directory: Path, split: int = TRAIN, kinds: tuple[int, ...] | None = None, quiet: bool = False
    ) -> Dataset:
        """`quiet` keeps only rows where the mover has no capture (the
        positions the search evaluates after quiescence, Stockfish's rule of
        skipping positions whose best move is a capture)."""
        directory = Path(directory)
        rows = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in FIELDS}
        if (directory / IDS_FILE).is_file():
            rows["ids"] = np.load(directory / IDS_FILE, mmap_mode="r")
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        keep = rows["split"] == split
        if kinds is not None:
            keep &= np.isin(rows["kind"], kinds)
        if quiet:
            keep &= ~mover_can_capture(rows["board"])
        return cls(directory, rows, provenance, np.flatnonzero(keep))

    def __len__(self) -> int:
        return len(self.index)

    def take(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        """The rows at `positions` (indices into this split), sorted for the memory map."""
        idx = self.index[np.sort(positions)]
        return {name: np.asarray(array[idx]) for name, array in self.rows.items()}

    def all(self, limit: int | None = None) -> dict[str, np.ndarray]:
        idx = self.index if limit is None else self.index[:limit]
        return {name: np.asarray(array[idx]) for name, array in self.rows.items()}


class Mixture:
    """Batches drawn from several datasets by fixed shares: `parts` is
    `[(dataset, share), ...]` and every batch holds `round(batch * share)`
    rows of each, sampled uniformly with replacement (the streaming regime
    of DESIGN item 23)."""

    def __init__(self, parts: list[tuple[Dataset, float]], batch: int, seed: int):
        total = sum(share for _, share in parts)
        self.parts = [(d, share / total) for d, share in parts if len(d)]
        self.batch = batch
        self.rng = np.random.default_rng(seed)

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def restore(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]

    def next(self) -> dict[str, np.ndarray]:
        pieces = []
        counts = [max(1, round(self.batch * share)) for _, share in self.parts]
        counts[-1] = max(1, self.batch - sum(counts[:-1]))
        for (dataset, _), count in zip(self.parts, counts, strict=True):
            pieces.append(dataset.take(self.rng.integers(len(dataset), size=count)))
        names = list(FIELDS) + (["ids"] if all("ids" in p for p in pieces) else [])
        return {name: np.concatenate([p[name] for p in pieces]) for name in names}


if __name__ == "__main__":
    import sys

    from .paths import data_dir

    if len(sys.argv) < 3 or sys.argv[1] != "encode":
        raise SystemExit("usage: python -m nnue.data encode <set> [<set> ...]")
    for name in sys.argv[2:]:
        path = encode_ids(data_dir(name))
        print(json.dumps({"event": "encoded", "file": str(path)}))
