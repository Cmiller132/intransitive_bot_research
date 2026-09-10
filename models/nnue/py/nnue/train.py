"""The training loop (DESIGN items 9, 10, 23): batches from a mixture of
datasets, sixfold symmetry augmentation with a paired consistency loss,
quantisation-aware training, validation by dataset and stratum under all six
symmetries, atomic checkpoints with exact resume, and the integer export of
the best network.

    python -m nnue.train --run <name> --data <set>:<share> [--data <set>:<share> ...] [--<field> value ...]

Every field of `Config` is a flag; `runs/<name>/config.json` records the values
used. A run holds `latest.pt`, `best.pt`, `best.nnue` (+ `.json`) and
`log.csv` with one row per epoch.
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
from .features import (  # noqa: E402
    ELAPSED_BASE,
    FEATURES,
    FORMAT6_FEATURES,
    PAD,
    RACE_BASE,
    feature_ids,
    piece_bucket,
    symmetry_table,
)
from .model import NNUE  # noqa: E402
from .paths import data_dir, run_dir  # noqa: E402


@dataclass
class Config:
    hidden: int = 512
    buckets: int = 1
    batch: int = 2048
    steps_per_epoch: int = 500
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 200
    lr_floor: float = 0.15  # the cosine schedule ends at this share of lr
    qat_start_epoch: int = 0  # fake quantisation from this epoch on
    loss: str = "control"  # control: tanh MSE + bounded logit term; bce: soft cross-entropy on 2 raw
    raw_weight: float = 0.02
    outcome_weight: float = 0.0
    symmetry_weight: float = 0.2
    clock: bool = True  # false: the 32 clock rows stay zero and never fire (the ablation of item 5)
    race: bool = False  # true: the 18 race rows of format 7 train (a format 6 init is widened with zero rows)
    quiet: bool = False  # true: train and validate only on rows where the mover has no capture
    threads: int = 4
    device: str = "cpu"  # or cuda when the GPU is free; the file and the checkpoints are device-free
    seed: int = 0
    val_rows: int = 8192
    grad_clip: float = 5.0
    patience: int = 6
    init: str = ""  # a .nnue file or a .pt checkpoint to start from
    resume: str = ""  # a latest.pt to continue exactly
    stop_epoch: int = 0  # stop after this many epochs (0: run to `epochs`); the schedule is unchanged


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
    """Per-row value loss (DESIGN item 9)."""
    if config.loss == "bce":
        return F.binary_cross_entropy_with_logits(2 * raw, (target + 1) / 2, reduction="none")
    if config.loss != "control":
        raise ValueError(f"unknown loss {config.loss}")
    logit = torch.atanh(target.clamp(-0.98, 0.98))
    return (raw.tanh() - target).square() + config.raw_weight * F.smooth_l1_loss(raw, logit, reduction="none")


def encode(rows: dict[str, np.ndarray], clock: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature ids and output buckets of a batch, from the dataset's id cache
    when it has one (`python -m nnue.data encode <set>`), else from the boards."""
    if "ids" in rows:
        ids = rows["ids"].astype(np.int64)
    else:
        ids = feature_ids(rows["board"], rows["since_capture"], rows["capture_clock"])
    if not clock:
        ids[(ids >= ELAPSED_BASE) & (ids < RACE_BASE)] = PAD
    # A batch without boards comes from the lean mixture of a one-bucket model, whose head ignores the bucket.
    bucket = piece_bucket(rows["board"]) if "board" in rows else np.zeros(len(ids), dtype=np.int64)
    return torch.from_numpy(ids), torch.from_numpy(bucket)


def step_loss(
    model: NNUE, rows: dict[str, np.ndarray], table: torch.Tensor, config: Config, qat: bool, rng: torch.Generator
) -> tuple[torch.Tensor, dict[str, float]]:
    """The batch under two different random symmetries: the mean of both
    value losses, an outcome term where the result is known, and the paired
    consistency term."""
    device = table.device
    ids, bucket = encode(rows, config.clock)
    ids, bucket = ids.to(device), bucket.to(device)
    n = len(ids)
    first = torch.randint(6, (n, 1, 1), generator=rng).to(device)
    second = (first + torch.randint(1, 6, (n, 1, 1), generator=rng).to(device)) % 6
    raw = model(torch.cat([table[first, ids], table[second, ids]]), torch.cat([bucket, bucket]), qat=qat)
    a, b = raw.chunk(2)
    target = torch.from_numpy(rows["target"]).float().to(device)
    weight = torch.from_numpy(rows["weight"]).float().to(device)
    norm = weight.sum().clamp_min(1e-8)
    value = ((value_loss(a, target, config) + value_loss(b, target, config)) * 0.5 * weight).sum() / norm
    consistency = (a.tanh() - b.tanh()).square().mean()
    loss = value + config.symmetry_weight * consistency
    parts = {"value": float(value.detach()), "consistency": float(consistency.detach())}
    if config.outcome_weight > 0:
        mask = torch.from_numpy(rows["outcome_ok"]).float().to(device)
        outcome = torch.from_numpy(rows["outcome"]).float().to(device)
        term = ((a.tanh() - outcome).square() + (b.tanh() - outcome).square()) * 0.5 * mask
        outcome_loss = term.sum() / mask.sum().clamp_min(1)
        loss = loss + config.outcome_weight * outcome_loss
        parts["outcome"] = float(outcome_loss.detach())
    return loss, parts


@torch.no_grad()
def evaluate(
    model: NNUE, dataset: Dataset, table: torch.Tensor, limit: int, clock: bool = True, batch: int = 2048
) -> dict:
    """Validation under all six symmetries at qat: error statistics overall
    and by stratum (pieces at most 4, 5-12, over 12; since_capture over 100)."""
    rows = dataset.all(limit)
    values = []
    for start in range(0, len(rows["board"]), batch):
        part = {k: v[start : start + batch] for k, v in rows.items()}
        ids, bucket = encode(part, clock)
        ids, bucket = ids.to(table.device), bucket.to(table.device)
        views = [model(table[k, ids], bucket, qat=True).tanh() for k in range(6)]
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
            model = model.widen_hidden(config.hidden)
        elif model.hidden != config.hidden:
            raise ValueError(f"{config.init} has hidden {model.hidden}, config says {config.hidden}")
        if model.buckets == 1 and config.buckets > 1:
            model = model.widen_buckets()
        elif model.buckets != config.buckets:
            raise ValueError(f"{config.init} has {model.buckets} buckets, config says {config.buckets}")
        if config.race and model.features < FEATURES:
            model = model.widen_features()
        elif not config.race and model.features == FEATURES:
            raise ValueError(f"{config.init} has the race rows, pass --race true")
    else:
        model = NNUE(config.hidden, config.buckets, FEATURES if config.race else FORMAT6_FEATURES)
        if config.init:
            load_state(model, torch.load(config.init, map_location="cpu", weights_only=False)["model"])
    if not config.clock:
        with torch.no_grad():
            model.embedding.weight[ELAPSED_BASE:RACE_BASE].zero_()
    return model


def load_state(model: NNUE, state: dict) -> None:
    """Load a checkpoint's parameters; a checkpoint written before the race
    rows (a 1,005-row table) fills the first rows and leaves the rest zero."""
    weight = state["embedding.weight"]
    if weight.shape[0] < model.embedding.weight.shape[0]:
        padded = torch.zeros_like(model.embedding.weight)
        padded[: weight.shape[0] - 1] = weight[:-1]  # the old padding row is dropped
        state = {**state, "embedding.weight": padded}
    model.load_state_dict(state)


def train(run: str, parts: list[tuple[str, float]], config: Config) -> Path:
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    out = run_dir(run)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [(Dataset.open(data_dir(name), TRAIN, quiet=config.quiet), share) for name, share in parts]
    validation = [(name, Dataset.open(data_dir(name), VALIDATION, quiet=config.quiet), share) for name, share in parts]
    # With id caches everywhere and one head, a step needs three small columns and the ids; gathering every
    # column (the 81-byte board above all) from the memory maps made the trainer page instead of compute.
    lean = config.buckets == 1 and all("ids" in dataset.rows for dataset, _ in datasets)
    mixture = Mixture(
        datasets, config.batch, config.seed, ("target", "weight", "outcome", "outcome_ok") if lean else None
    )
    device = torch.device(config.device)
    table = torch.from_numpy(symmetry_table()).to(device)
    rng = torch.Generator().manual_seed(config.seed + 1)
    model = initial_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * config.steps_per_epoch
    step, first_epoch, best, stale = 0, 0, float("inf"), 0
    if config.resume:
        saved = torch.load(config.resume, map_location="cpu", weights_only=False)
        volatile = {"resume": "", "stop_epoch": 0}
        if {**saved["config"], **volatile} != {**asdict(config), **volatile} or saved["data"] != parts:
            raise ValueError("resume with the settings and datasets the run was started with")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, first_epoch, best, stale = saved["step"], saved["epoch"] + 1, saved["best"], saved["stale"]
        torch.set_rng_state(saved["torch_rng"])
        rng.set_state(saved["aug_rng"])
        mixture.restore(saved["sampler"])
    else:
        (out / "config.json").write_text(json.dumps({"config": asdict(config), "data": parts}, indent=2))
    provenance = {name: export_module.sha256(data_dir(name) / "provenance.json") for name, _ in parts}
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
            step += 1
            for k, v in {"loss": float(loss.detach()), **parts_of_loss}.items():
                sums[k] = sums.get(k, 0.0) + v
        model.eval()
        metrics = {name: evaluate(model, d, table, config.val_rows, config.clock) for name, d, _ in validation}
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
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "torch_rng": torch.get_rng_state(),
            "aug_rng": rng.get_state(),
            "sampler": mixture.state(),
            "config": asdict(config),
            "data": parts,
        }
        save(out / "latest.pt", payload)
        if improved:
            save(out / "best.pt", payload)
            export_module.export(
                copy.deepcopy(model).cpu(),
                out / "best.nnue",
                {"run": run, "epoch": epoch, "objective": objective, "config": asdict(config), "datasets": provenance},
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
