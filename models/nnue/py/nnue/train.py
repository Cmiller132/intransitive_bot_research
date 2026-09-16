"""The training loop (DESIGN items 9, 10, 23): batches from a mixture of
datasets, sixfold symmetry augmentation with a paired consistency loss,
quantisation-aware training, validation by dataset and stratum under all six
symmetries, atomic checkpoints with exact resume, and the integer export of
the best network.

    python -m nnue.train --run <name> --data <set>:<share> [--data <set>:<share> ...] [--<field> value ...]

Every field of `Config` is a flag; `runs/<name>/config.json` records the values
used. A run holds `latest.pt`, `best.pt`, `best.nnue` (+ `.json`) and
`log.csv` with one row per epoch.

With `--ema d` (0 < d < 1) an exponential moving average of the weights, begun
at the end of the warmup and moved a little toward the live weights after
every step, is the network of record: it is validated, saved as the
checkpoint's `model` and exported; the live weights ride along as `live` so a
resume continues the descent exactly. Leela's and KataGo's published networks
are averaged weights; the average is the cheapest variance reduction there is.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from . import export as export_module  # noqa: E402
from .data import KIND_RETURN, TRAIN, VALIDATION, Dataset, Mixture  # noqa: E402
from .features import LAYOUTS, Layout, feature_ids, goal_ids, symmetry_table  # noqa: E402
from .model import NNUE  # noqa: E402
from .paths import data_dir, run_dir, sha256  # noqa: E402


@dataclass
class Config:
    hidden: int = 512
    batch: int = 2048
    steps_per_epoch: int = 500
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 200
    lr_floor: float = 0.15  # the cosine schedule ends at this share of lr
    qat_start_epoch: int = 0  # fake quantisation from this epoch on
    raw_weight: float = 0.02  # weight of the bounded logit term beside the tanh mean squared error
    symmetry_weight: float = 0.2
    version: int = 8  # the file format: 8 (27 opponent-material contexts), 6 (the incumbent's layout) or
    # 9 (8 plus the goal-corner rows and the eight piece-count output heads)
    threads: int = 4
    device: str = "cpu"  # or cuda when the GPU is free; the file and the checkpoints are device-free
    seed: int = 0
    val_rows: int = 8192
    grad_clip: float = 5.0
    patience: int = 6
    init: str = ""  # a .nnue file or a .pt checkpoint to start from
    resume: str = ""  # a latest.pt to continue exactly
    stop_epoch: int = 0  # stop after this many epochs (0: run to `epochs`); the schedule is unchanged
    widen_outgoing: float = 0.0  # widening from a narrower init: the new readout and dense columns are +-this
    ema: float = 0.0  # decay per step of the weight average that is the network of record (0: off; 0.999 = a
    # window of about a thousand steps, one epoch of the H512 recipe)


def parse(argv: list[str]) -> tuple[str, list[tuple[str, float]], Config]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", action="append", default=[], help="<dataset>:<share>, repeatable")
    for field in fields(Config):
        kind = flag if isinstance(field.default, bool) else type(field.default)
        parser.add_argument(f"--{field.name}", type=kind, default=field.default)
    args = parser.parse_args(argv)
    parts = []
    for spec in args.data:
        name, _, share = spec.partition(":")
        parts.append((name, float(share) if share else 1.0))
    config = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    return args.run, parts, config


def flag(text: str) -> bool:
    if text.lower() in ("1", "true", "yes", "on"):
        return True
    if text.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def value_loss(raw: torch.Tensor, target: torch.Tensor, config: Config) -> torch.Tensor:
    """Per-row value loss (DESIGN item 9): tanh mean squared error plus the bounded logit term."""
    logit = torch.atanh(target.clamp(-0.98, 0.98))
    return (raw.tanh() - target).square() + config.raw_weight * F.smooth_l1_loss(raw, logit, reduction="none")


def encode(rows: dict[str, np.ndarray], layout: Layout) -> torch.Tensor:
    """Feature ids in `layout` of a batch, from the dataset's id cache when it
    has one (`python -m nnue.data encode <set>`), else from the boards. A
    format 9 batch also carries the boards, whose goal corners are two rows the
    cache (a format 8 encoding) does not hold."""
    if "ids" in rows:
        ids = rows["ids"].astype(np.int64)
    else:
        ids = feature_ids(rows["board"], rows["since_capture"], rows["capture_clock"])
    ids = layout.rows(ids)
    if layout.goal:
        ids = np.concatenate([ids, goal_ids(rows["board"])], axis=2)
    return torch.from_numpy(ids)


def step_loss(
    model: NNUE, rows: dict[str, np.ndarray], table: torch.Tensor, config: Config, qat: bool, rng: torch.Generator
) -> tuple[torch.Tensor, dict[str, float]]:
    """The batch under two different random symmetries: the mean of both
    value losses and the paired consistency term."""
    device = table.device
    ids = encode(rows, model.layout).to(device)
    n = len(ids)
    first = torch.randint(6, (n, 1, 1), generator=rng).to(device)
    second = (first + torch.randint(1, 6, (n, 1, 1), generator=rng).to(device)) % 6
    raw = model(torch.cat([table[first, ids], table[second, ids]]), qat=qat)
    a, b = raw.chunk(2)
    target = torch.from_numpy(rows["target"]).float().to(device)
    weight = torch.from_numpy(rows["weight"]).float().to(device)
    norm = weight.sum().clamp_min(1e-8)
    value = ((value_loss(a, target, config) + value_loss(b, target, config)) * 0.5 * weight).sum() / norm
    consistency = (a.tanh() - b.tanh()).square().mean()
    loss = value + config.symmetry_weight * consistency
    return loss, {"value": float(value.detach()), "consistency": float(consistency.detach())}


@torch.no_grad()
def evaluate(model: NNUE, dataset: Dataset, table: torch.Tensor, limit: int, batch: int = 2048) -> dict:
    """Validation under all six symmetries at qat: error statistics overall
    and by stratum (pieces at most 4, 5-12, over 12; since_capture over 100)."""
    rows = dataset.all(limit)
    values = []
    for start in range(0, len(rows["board"]), batch):
        part = {k: v[start : start + batch] for k, v in rows.items()}
        ids = encode(part, model.layout).to(table.device)
        views = [model(table[k, ids], qat=True).tanh() for k in range(6)]
        values.append(torch.stack(views, 1).cpu().numpy())
    values = np.concatenate(values) if values else np.zeros((0, 6))
    target, weight = rows["target"][:, None], rows["weight"][:, None]
    error = values - target

    def stats(mask: np.ndarray) -> dict | None:
        if not mask.any():
            return None
        w, e = weight[mask], error[mask]
        return {
            "rows": int(mask.sum()),
            "mse": float((e * e * w).sum() / (6 * w.sum())),
            "mae": float((np.abs(e) * w).sum() / (6 * w.sum())),
            "bias": float((e * w).sum() / (6 * w.sum())),
        }

    pieces = np.count_nonzero(rows["board"], axis=1)
    spread = np.ptp(values, axis=1) if len(values) else np.zeros(0)
    return {
        **(stats(np.ones(len(target), dtype=bool)) or {"rows": 0, "mse": 0.0, "mae": 0.0, "bias": 0.0}),
        "symmetry_range_mean": float(spread.mean()) if len(spread) else 0.0,
        "symmetry_range_p95": float(np.quantile(spread, 0.95)) if len(spread) else 0.0,
        "sparse": stats(pieces <= 4),
        "middle": stats((pieces >= 5) & (pieces <= 12)),
        "full": stats(pieces > 12),
        "long_clock": stats(rows["since_capture"] > 100),
        # Selection ignores outcome-labelled rows (kind 0): a game result is a
        # +-1 draw from the value, so its error floor would swamp the rest.
        "selection": stats(rows["kind"] != KIND_RETURN),
    }


def learning_rate(config: Config, step: int, total: int) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = min(1.0, (step - config.warmup_steps) / max(1, total - config.warmup_steps))
    return config.lr * (config.lr_floor + (1 - config.lr_floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def initial_model(config: Config) -> NNUE:
    if config.init.endswith(".nnue"):
        model = export_module.load(Path(config.init))
        if model.hidden < config.hidden:
            model = model.widen_hidden(config.hidden, config.seed, config.widen_outgoing)
        elif model.hidden != config.hidden:
            raise ValueError(f"{config.init} has hidden {model.hidden}, config says {config.hidden}")
        # 6 -> 8 repeats the piece rows into every context, 8 -> 9 adds the zero
        # goal rows and eight copies of the one head; both keep the evaluation.
        if model.version == 6 and config.version > 6:
            model = model.widen_contexts()
        if model.version == 8 and config.version == 9:
            model = model.widen_buckets()
        if model.version != config.version:
            raise ValueError(f"{config.init} is format {model.version}, config says {config.version}")
    else:
        model = NNUE(config.hidden, config.version)
        if config.init:
            model.load_state_dict(torch.load(config.init, map_location="cpu", weights_only=False)["model"])
    return model


def ema_update(average: NNUE, live: NNUE, decay: float) -> None:
    """The average moves toward the live weights by (1 - decay); integer buffers are copied."""
    with torch.no_grad():
        for a, b in zip(average.parameters(), live.parameters(), strict=True):
            a.mul_(decay).add_(b, alpha=1.0 - decay)
        for a, b in zip(average.buffers(), live.buffers(), strict=True):
            if a.is_floating_point():
                a.mul_(decay).add_(b, alpha=1.0 - decay)
            else:
                a.copy_(b)


def truncate_log(path: Path, epochs: int) -> None:
    """Keeps the header and the first `epochs` rows: a row written before a
    crash that lost its checkpoint goes, the checkpoint stays the authority."""
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if len(lines) > epochs + 1:
            path.write_text("".join(lines[: epochs + 1]), encoding="utf-8")


def train(run: str, parts: list[tuple[str, float]], config: Config) -> Path:
    if config.version not in LAYOUTS:
        raise ValueError(f"version must be one of {sorted(LAYOUTS)}")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    out = run_dir(run)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [(Dataset.open(data_dir(name), TRAIN), share) for name, share in parts]
    validation = [(name, Dataset.open(data_dir(name), VALIDATION), share) for name, share in parts]
    # With id caches everywhere a step needs two small columns and the ids; gathering every column (the
    # 81-byte board above all) from the memory maps made the trainer page instead of compute. Format 9
    # adds the board, whose goal corners the format 8 id cache does not encode.
    lean = all("ids" in dataset.rows for dataset, _ in datasets)
    columns = ("target", "weight") + (("board",) if LAYOUTS[config.version].goal else ())
    mixture = Mixture(datasets, config.batch, config.seed, columns if lean else None)
    device = torch.device(config.device)
    rng = torch.Generator().manual_seed(config.seed + 1)
    model = initial_model(config).to(device)
    table = torch.from_numpy(symmetry_table(model.layout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * config.steps_per_epoch
    step, first_epoch, best, stale = 0, 0, float("inf"), 0
    average: NNUE | None = None  # the weight average (config.ema > 0), the model of record once it has begun
    ema_started: int | None = None
    provenance = {name: sha256(data_dir(name) / "provenance.json") for name, _ in parts}
    # A run is bound to its datasets' provenance and its init file; a resume must present the same.
    inputs = {"data": parts, "datasets": provenance, "init_sha256": sha256(Path(config.init)) if config.init else None}
    if config.resume:
        saved = torch.load(config.resume, map_location="cpu", weights_only=False)
        volatile = {"resume": "", "stop_epoch": 0}
        defaults = {
            f.name: f.default for f in fields(Config)
        }  # a field added since the run started reads as its default
        same = {**defaults, **saved["config"], **volatile} == {**asdict(config), **volatile}
        if not same or any(saved.get(k) != v for k, v in inputs.items()):
            raise ValueError("resume with the settings, init and datasets the run was started with")
        model.load_state_dict(saved["live"] if "live" in saved else saved["model"])
        if "live" in saved:  # the checkpoint's model is the average; the live weights continue the descent
            average = copy.deepcopy(model)
            average.load_state_dict(saved["model"])
            ema_started = saved.get("ema_started")
        optimizer.load_state_dict(saved["optimizer"])
        step, first_epoch, best, stale = saved["step"], saved["epoch"] + 1, saved["best"], saved["stale"]
        torch.set_rng_state(saved["torch_rng"])
        rng.set_state(saved["aug_rng"])
        mixture.restore(saved["sampler"])
        truncate_log(out / "log.csv", first_epoch)
    else:
        (out / "config.json").write_text(json.dumps({"config": asdict(config), **inputs}, indent=2))
    print(json.dumps({"event": "start", "run": run, "first_epoch": first_epoch, "rows": [len(d) for d, _ in datasets]}))
    for epoch in range(first_epoch, config.epochs):
        started = time.perf_counter()
        qat = epoch >= config.qat_start_epoch
        model.train()
        sums: dict[str, float] = {}
        for _ in range(config.steps_per_epoch):
            lr = learning_rate(config, step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, parts_of_loss = step_loss(model, mixture.next(), table, config, qat, rng)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise RuntimeError("non-finite loss or gradient")
            optimizer.step()
            model.constrain()
            if config.ema > 0:
                if average is None and step >= config.warmup_steps:
                    average, ema_started = copy.deepcopy(model), step  # begun from the live weights, not the init
                elif average is not None:
                    ema_update(average, model, config.ema)
            step += 1
            for k, v in {"loss": float(loss.detach()), **parts_of_loss}.items():
                sums[k] = sums.get(k, 0.0) + v
        model.eval()
        record_model = model if average is None else average  # what is validated, checkpointed and exported
        record_model.eval()
        metrics = {name: evaluate(record_model, d, table, config.val_rows) for name, d, _ in validation}
        shares = sum(share for _, _, share in validation)
        objective = (
            sum((metrics[name]["selection"] or metrics[name])["mse"] * share for name, _, share in validation) / shares
        )
        improved = objective < best
        best, stale = (objective, 0) if improved else (best, stale + 1)
        record = {
            "epoch": epoch,
            "step": step,
            "qat": qat,
            "lr": lr,
            **{k: v / config.steps_per_epoch for k, v in sums.items()},
            "objective": objective,
            "best": best,
            "improved": improved,
            "seconds": time.perf_counter() - started,
            "validation": metrics,
        }
        print(json.dumps(record), flush=True)
        log_row(out / "log.csv", record)
        payload = {
            "model": record_model.state_dict(),
            **({"live": model.state_dict(), "ema_started": ema_started} if average is not None else {}),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "torch_rng": torch.get_rng_state(),
            "aug_rng": rng.get_state(),
            "sampler": mixture.state(),
            "config": asdict(config),
            **inputs,
        }
        save(out / "latest.pt", payload)
        if improved:
            save(out / "best.pt", payload)
            export_module.export(
                copy.deepcopy(record_model).cpu(),
                out / "best.nnue",
                {"run": run, "epoch": epoch, "objective": objective, "config": asdict(config), **inputs},
            )
        if stale >= config.patience or (config.stop_epoch and epoch + 1 >= config.stop_epoch):
            break
    print(json.dumps({"event": "done", "run": run, "best": best}))
    return out


def log_row(path: Path, record: dict) -> None:
    """One flat CSV row per epoch; nested validation metrics become `<set>.<metric>` columns."""
    flat: dict[str, object] = {}
    for k, v in record.items():
        if k != "validation":
            flat[k] = v
    for name, metrics in record["validation"].items():
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{name}.{k}.{kk}"] = vv
            elif v is not None:
                flat[f"{name}.{k}"] = v
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        if new:
            writer.writeheader()
        writer.writerow(flat)


if __name__ == "__main__":
    train(*parse(sys.argv[1:]))
