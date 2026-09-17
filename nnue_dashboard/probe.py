"""Questions only the nnue package can answer, asked by the dashboard's runner.

The dashboard itself is standard library only and may run under another interpreter, so the runner starts this file
with the training interpreter from the workspace root and reads one JSON object from its last output line:

    python probe.py net <file.nnue>            width, format and hash of a network
    python probe.py sets <set>...              per dataset: exists, rows (searched-root rows), a valid id cache
    python probe.py caps                       what this checkout's trainer and importer support
    python probe.py train-state <train args>   fresh | resume | mismatch | foreign, as quickstart.sh decides
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def net(path: str) -> dict:
    from nnue import export
    from nnue.paths import sha256

    info = export.read(Path(path))
    return {"hidden": info["hidden"], "version": info["version"], "sha256": sha256(Path(path))}


def sets(names: list[str]) -> dict:
    from nnue.data import IDS_FILE, IDS_META, cache_identity
    from nnue.paths import data_dir

    out = {}
    for name in dict.fromkeys(names):
        directory = data_dir(name)
        try:
            prov = json.loads((directory / "provenance.json").read_text())
        except (OSError, ValueError):
            out[name] = {"exists": False, "rows": 0, "cache_ok": False}
            continue
        try:
            cache_ok = (directory / IDS_FILE).is_file() and json.loads((directory / IDS_META).read_text()) == cache_identity(directory)
        except (OSError, ValueError):
            cache_ok = False
        out[name] = {"exists": True, "rows": int(prov.get("rows") or 0), "cache_ok": cache_ok}
    return out


def caps() -> dict:
    import inspect

    from nnue import features, importer, train

    return {"ema": "ema" in train.Config.__dataclass_fields__,
            "quiet": "quiet" in inspect.signature(importer.import_selfplay).parameters,
            "version9": 9 in features.LAYOUTS}


def train_state(argv: list[str]) -> dict:
    from dataclasses import asdict, fields

    from nnue.paths import data_dir, run_dir, sha256
    from nnue.train import Config, parse

    run, parts, config = parse(argv)
    out = run_dir(run)
    if not out.exists() or not any(out.iterdir()):
        return {"state": "fresh", "epochs_done": 0}
    if not (out / "config.json").is_file():
        return {"state": "foreign", "epochs_done": 0}
    epochs_done = 0
    try:
        with (out / "log.csv").open() as f:
            epochs_done = max(0, sum(1 for _ in f) - 1)
    except OSError:
        pass
    if not (out / "latest.pt").is_file():
        return {"state": "mismatch", "epochs_done": epochs_done, "why": "no checkpoint to resume from"}
    saved = json.loads((out / "config.json").read_text())
    defaults = {f.name: f.default for f in fields(Config)}
    volatile = {"resume": "", "stop_epoch": 0}
    inputs = {
        "data": [list(part) for part in parts],
        "datasets": {name: sha256(data_dir(name) / "provenance.json") for name, _ in parts},
        "init_sha256": sha256(Path(config.init)) if config.init else None,
    }
    mine, theirs = {**asdict(config), **volatile}, {**defaults, **saved["config"], **volatile}
    changed = sorted(k for k in mine if mine.get(k) != theirs.get(k))
    changed += [k for k, v in inputs.items() if saved.get(k) != v]
    if changed:
        return {"state": "mismatch", "epochs_done": epochs_done, "why": "changed: " + ", ".join(changed)}
    return {"state": "resume", "epochs_done": epochs_done}


def main() -> None:
    command, args = sys.argv[1], sys.argv[2:]
    result = {"net": lambda: net(args[0]), "sets": lambda: sets(args), "caps": caps,
              "train-state": lambda: train_state(args)}[command]()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
