"""Widening preflight bound to an experiment, its inputs and this implementation.

    python -m nnue.widen_check <new-width-experiment>/experiment.json

Writes immutable initial/transition exports, channel evidence and report.json.
The CPU probe runs 1,000 float updates and one QAT backward without an update.
It never writes an arm checkpoint or changes the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import export
from .data import TRAIN, Dataset, Mixture
from .features import PIECE_ROWS, SLOTS, feature_ids, symmetry_table
from .model import NNUE, QA, QB, accumulate, fake_quant
from .paths import data_dir, sha256, workspace_root
from .train import Config, initial_model, learning_rate, step_loss

UPDATES = 1000
FIXTURE = "models/nnue/tests/fixtures/format8_positions.jsonl"


def absolute(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else workspace_root() / path


def configs(manifest: dict) -> tuple[Config, Config, list]:
    """Accept a matched width comparison whose first QAT batch follows the probe."""
    arms = sorted(manifest["arms"], key=lambda a: a["train"]["config"]["hidden"])
    if len(arms) != 2:
        raise ValueError("need exactly two training arms")
    specs = [arm["train"] for arm in arms]
    low, high = [Config(**{**s["config"], "init": str(absolute(s["init"]))}) for s in specs]
    if (
        replace(high, hidden=low.hidden) != low
        or specs[0]["data"] != specs[1]["data"]
        or (low.hidden, high.hidden) != (512, 1024)
        or low.version not in (6, 8)
        or low.device != "cpu"
        or low.resume
        or low.stop_epoch
        or low.qat_start_epoch != 1
        or low.qat_start_epoch * low.steps_per_epoch != UPDATES
        or low.epochs * low.steps_per_epoch <= UPDATES
    ):
        raise ValueError("need matched H512/H1024 CPU format-6/8 arms, one 1000-update float epoch, no resume")
    return low, high, specs[0]["data"]


def outgoing(model: NNUE, old: int, gradient: bool = False) -> torch.Tensor:
    """One row per new channel, both perspectives and both downstream heads."""
    cols = torch.cat([torch.arange(old, model.hidden), torch.arange(model.hidden + old, 2 * model.hidden)])
    rows = []
    for layer in (model.output, model.dense):
        weight = layer.weight.grad if gradient else layer.weight
        if weight is None:
            weight = torch.zeros_like(layer.weight)
        rows.append(weight[:, cols].reshape(weight.shape[0], 2, -1).permute(2, 1, 0).flatten(1))
    return torch.cat(rows, 1).detach()


def incoming(model: NNUE, old: int, gradient: bool = False) -> torch.Tensor:
    rows = []
    for parameter in (model.piece, model.attack, model.clock):
        value = parameter.grad if gradient else parameter
        if value is None:
            value = torch.zeros_like(parameter)
        rows.append(value.reshape(-1, model.hidden)[:, old:])
    return torch.cat(rows).T.detach()


def vectors(value: torch.Tensor) -> dict:
    array = value.detach().cpu().numpy()
    return {
        "channels": len(array),
        "duplicates": len(array) - len(np.unique(array, axis=0)),
        "norms": np.linalg.norm(array, axis=1).tolist(),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


@torch.no_grad()
def activations(model: NNUE, ids: torch.Tensor, old: int) -> torch.Tensor:
    weight = torch.cat([model.rows(True), model.bias.new_zeros(1, model.hidden)])
    acc = accumulate(ids.reshape(-1, SLOTS), weight) + fake_quant(model.bias, QA)
    return acc[:, old:].clamp(0, 1).square().T.contiguous()


def activation_record(value: torch.Tensor) -> dict:
    return {
        **vectors(value),
        "variance": value.var(dim=1, unbiased=False).tolist(),
        "dead_fraction": (value == 0).float().mean(dim=1).tolist(),
        "saturated_fraction": (value == 1).float().mean(dim=1).tolist(),
    }


def preserved(base: NNUE, wide: NNUE) -> bool:
    old, new = base.hidden, wide.hidden
    return all(
        torch.equal(getattr(base, name), getattr(wide, name)[..., :old])
        for name in (
            ("piece", "attack", "clock", "bias", "context")
            if base.context is not None
            else ("piece", "attack", "clock", "bias")
        )
    ) and all(
        torch.equal(a, b)
        for a, b in [
            (base.output.weight, torch.cat([wide.output.weight[:, :old], wide.output.weight[:, new : new + old]], 1)),
            (base.dense.weight, torch.cat([wide.dense.weight[:, :old], wide.dense.weight[:, new : new + old]], 1)),
            (base.output.bias, wide.output.bias),
            (base.dense.bias, wide.dense.bias),
            (base.delta.weight, wide.delta.weight),
        ]
    )


def initial_checks(base: NNUE, wide: NNUE, config: Config, ids: torch.Tensor) -> tuple[dict, dict]:
    same = initial_model(config)
    other = initial_model(replace(config, seed=config.seed + 1))
    columns = incoming(wide, base.hidden)
    act = activations(wide, ids, base.hidden)
    records = {"columns": vectors(columns), "activations": activation_record(act)}
    checks = {
        "preserved": preserved(base, wide),
        "zero_outgoing": not bool(outgoing(wide, base.hidden).count_nonzero()),
        "zero_new_residuals": wide.context is None or not bool(wide.context[..., base.hidden :].count_nonzero()),
        "same_seed": all(torch.equal(v, same.state_dict()[k]) for k, v in wide.state_dict().items()),
        "other_seed_preserves": preserved(base, other)
        and torch.equal(wide.bias, other.bias)
        and (wide.context is None or torch.equal(wide.context, other.context))
        and all(torch.equal(v, other.state_dict()[k]) for k, v in wide.state_dict().items() if "." in k),
        "other_seed_changes_columns": bool(torch.all(torch.any(columns != incoming(other, base.hidden), dim=1))),
        "distinct_nonzero_columns": records["columns"]["duplicates"] == 0
        and all(n > 0 for n in records["columns"]["norms"]),
        "distinct_varying_activations": records["activations"]["duplicates"] == 0
        and all(v > 0 for v in records["activations"]["variance"]),
    }
    return {"checks": checks, **records}, {"initial_columns": columns.numpy(), "initial_activations": act.numpy()}


def residuals(model: NNUE, old: int, gradient: bool = False) -> torch.Tensor:
    """Context, new channel, piece row; format 6 has no residual table."""
    if model.context is None:
        return model.piece.new_zeros(0, model.hidden - old, PIECE_ROWS)
    value = model.context.grad if gradient else model.context
    if value is None:
        value = torch.zeros_like(model.context)
    return value[..., old:].permute(0, 2, 1).detach().clone()


def context_visits(model: NNUE, ids: torch.Tensor) -> torch.Tensor:
    """Perspective counts from actual forward inputs; empty perspectives have no piece row."""
    first = ids[..., :20].amin(dim=-1).flatten()
    first = first[first < model.layout.attack_base] // PIECE_ROWS
    return torch.bincount(first, minlength=model.layout.contexts)


def served_counts(model: NNUE, old: int) -> dict:
    weights = torch.round(outgoing(model, old) * QB)
    # Each channel contains two perspectives, first for output then for dense.
    direct, dense = weights[:, :2], weights[:, 2:]
    return {"direct": direct.count_nonzero(dim=1).tolist(), "dense": dense.count_nonzero(dim=1).tolist()}


def probe(model: NNUE, old: int, config: Config, mixture: Mixture, ids: torch.Tensor) -> tuple[dict, dict]:
    """Task gradients precede clipping/decay; the final QAT backward never updates weights."""
    if config.qat_start_epoch != 1 or config.steps_per_epoch != UPDATES:
        raise ValueError("probe needs one 1000-update float epoch before QAT")
    rng = torch.Generator().manual_seed(config.seed + 1)
    table = torch.from_numpy(symmetry_table(model.layout))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    digest = hashlib.sha256()
    first_crossing = first_float_incoming = None
    max_shared = torch.zeros(model.hidden - old)
    max_residual = torch.zeros(model.layout.contexts if model.context is not None else 0, model.hidden - old)
    visits = torch.zeros(model.layout.contexts, dtype=torch.int64)
    last_visits = visits.clone()

    def observe(module, args):
        nonlocal last_visits
        last_visits = context_visits(module, args[0])
        visits.add_(last_visits)

    samples = []
    with model.register_forward_pre_hook(observe):
        for step in range(UPDATES + 1):
            qat = step == UPDATES
            rows = mixture.next()
            for key, array in sorted(rows.items()):
                digest.update(key.encode())
                digest.update(str((array.shape, array.dtype)).encode())
                digest.update(array.tobytes())
            lr = learning_rate(config, step, config.epochs * config.steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, _ = step_loss(model, rows, table, config, qat, rng)
            loss.backward()
            shared = incoming(model, old, True)
            residual = residuals(model, old, True)
            shared_norm, residual_norm = shared.norm(dim=1), residual.norm(dim=2)
            max_shared = torch.maximum(max_shared, shared_norm)
            max_residual = torch.maximum(max_residual, residual_norm)
            if step == 0:
                initial_gradient = outgoing(model, old, True).clone()
            if not qat and first_float_incoming is None and bool(shared_norm.any() or residual_norm.any()):
                first_float_incoming = step + 1
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise RuntimeError("non-finite probe loss or gradient")
            if not qat:
                optimizer.step()
                model.constrain()
            served = served_counts(model, old)
            nonzero = sum(served["direct"]) + sum(served["dense"])
            if nonzero and first_crossing is None:
                first_crossing = step + 1
            samples.append(
                {
                    "batch": step + 1,
                    "qat": qat,
                    "updated": not qat,
                    "lr": lr,
                    "loss": float(loss.detach()),
                    "quantized_outgoing_nonzero": nonzero,
                }
            )
    columns = incoming(model, old)
    act = activations(model, ids, old)
    initial, final = vectors(initial_gradient), vectors(columns)
    act_record = activation_record(act)
    checks = {
        "initial_outgoing_gradient": any(n > 0 for n in initial["norms"])
        and initial["duplicates"] < len(initial["norms"]) - 1,
        "served_readout_at_transition": nonzero > 0,
        "qat_incoming_task_gradient": bool(shared_norm.any() or residual_norm.any()),
        "final_distinct_columns": final["duplicates"] == 0 and all(n > 0 for n in final["norms"]),
        "final_distinct_varying_activations": act_record["duplicates"] == 0
        and all(v > 0 for v in act_record["variance"]),
    }
    return {
        "updates": UPDATES,
        "batches": UPDATES + 1,
        "qat_backward_batch": UPDATES + 1,
        "batch_sha256": digest.hexdigest(),
        "first_quantized_outgoing_update": first_crossing,
        "first_float_incoming_backward": first_float_incoming,
        "initial_outgoing_gradient": initial,
        "qat_shared_task_gradient": vectors(shared),
        "qat_residual_task_gradient_norms": residual_norm.tolist(),
        "max_shared_task_gradient_norm": max_shared.tolist(),
        "max_residual_task_gradient_norms": max_residual.tolist(),
        "served_readout_per_channel": served,
        "context_perspective_visits": visits.tolist(),
        "qat_context_perspective_visits": last_visits.tolist(),
        "final_columns": final,
        "final_activations": act_record,
        "samples": samples,
        "checks": checks,
    }, {
        "initial_outgoing_gradient": initial_gradient.numpy(),
        "qat_shared_task_gradient": shared.numpy(),
        "qat_residual_task_gradient": residual.numpy(),
        "final_columns": columns.numpy(),
        "final_residual_columns": residuals(model, old).numpy(),
        "final_activations": act.numpy(),
    }


def rust_values(bot: Path, net: Path, positions: Path, out: Path, count: int) -> np.ndarray:
    command = [str(bot), "nnue", "validate", "--model", str(net), "--input", str(positions)]
    with out.open("w", encoding="utf-8") as stream, out.with_suffix(".err").open("w", encoding="utf-8") as err:
        subprocess.run(command, stdout=stream, stderr=err, check=True, timeout=300)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    if len(rows) != count or [r["row"] for r in rows] != list(range(count)):
        raise ValueError("incomplete or reordered Rust validation")
    values = np.array([r["raw"] for r in rows])
    if not np.isfinite(values).all():
        raise ValueError("non-finite Rust validation")
    return values


def binding(manifest_path: Path, manifest: dict, parts: list) -> dict:
    files = [
        manifest_path,
        absolute(manifest["parent"]),
        absolute(manifest["parent_record"]["path"]),
        absolute(manifest["bot"]),
        absolute(FIXTURE),
    ]
    files.extend(sorted(Path(__file__).parent.glob("*.py")))
    expected = {
        absolute(manifest["parent"]): manifest["init_sha256"],
        absolute(manifest["parent_record"]["path"]): manifest["parent_record"]["sha256"],
        absolute(manifest["bot"]): manifest["bot_sha256"],
    }
    inventory = {item["dataset"]: item for item in manifest["data_inventory"]}
    if set(inventory) != {name for name, _ in parts}:
        raise ValueError("data inventory differs from the arms")
    for name, share in parts:
        item = inventory[name]
        if item["share"] != share:
            raise ValueError("data share differs from inventory")
        for filename, key in [("provenance.json", "provenance_sha256"), ("ids8.json", "cache_meta_sha256")]:
            path = data_dir(name) / filename
            files.append(path)
            expected[path] = item[key]
    hashes = {str(p.resolve()): sha256(p) for p in files}
    for path, digest in expected.items():
        if hashes[str(path.resolve())] != digest:
            raise ValueError(f"input identity differs: {path}")
    return hashes


def run(manifest_path: Path) -> dict:
    manifest_path = absolute(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    low, high, parts = configs(manifest)
    if absolute(manifest["parent"]) != Path(low.init):
        raise ValueError("arm init differs from the bound parent")
    before = binding(manifest_path, manifest, parts)
    if sha256(manifest_path) != manifest_hash:
        raise ValueError("manifest changed while reading")
    directory = manifest_path.parent / "preflight"
    directory.mkdir(exist_ok=False)
    (directory / "experiment.json").write_bytes(manifest_bytes)
    report = {
        "schema": 2,
        "passed": False,
        "bindings": before,
        "seed": high.seed,
        "device": "cpu",
        "threads": 4,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "updates": UPDATES,
        "widths": [low.hidden, high.hidden],
        "version": high.version,
        "manifest_sha256": manifest_hash,
        "errors": [],
    }
    try:
        torch.set_num_threads(4)
        torch.manual_seed(high.seed)
        base, wide = initial_model(low), initial_model(high)
        parent = export.load(Path(low.init))
        if (parent.hidden, parent.version) != (low.hidden, low.version):
            raise ValueError("parent must already have the control width and format")
        del parent
        positions = [json.loads(line) for line in absolute(FIXTURE).read_text().splitlines()]
        if len(positions) != 512:
            raise ValueError("need all 512 shared fixture positions")
        board = np.array([p["board"] for p in positions], dtype=np.uint8)
        since = np.array([p["since_capture"] for p in positions])
        clock = np.array([p["clock"] for p in positions])
        ids = torch.from_numpy(wide.layout.rows(feature_ids(board, since, clock)))
        report["initial"], arrays = initial_checks(base, wide, high, ids)
        report["fixture_context_perspective_visits"] = context_visits(wide, ids).tolist()
        arrays["initial_residual_columns"] = residuals(wide, low.hidden).numpy()
        with torch.no_grad():
            report["float_max_difference_descriptive"] = float((base(ids) - wide(ids)).abs().max())
        np.savez(directory / "channels.npz", **arrays)
        values, rust = [], []
        for model in (base, wide):
            net = directory / f"h{model.hidden}_initial.nnue"
            export.export(model, net)
            raw = export.integer_eval(net, board, since, clock)
            values.append(raw)
            request = directory / f"h{model.hidden}_positions.jsonl"
            request.write_text(
                "".join(
                    json.dumps(
                        {
                            **{k: p[k] for k in ("board", "ply", "clock", "since_capture")},
                            ("raw" if model.version == 8 else "raw6"): float(value),
                        }
                    )
                    + "\n"
                    for p, value in zip(positions, raw, strict=True)
                ),
                encoding="utf-8",
            )
            rust.append(
                rust_values(
                    absolute(manifest["bot"]), net, request, directory / f"h{model.hidden}_rust.jsonl", len(positions)
                )
            )
        report["integer"] = {
            "parent_export_exact": sha256(directory / f"h{base.hidden}_initial.nnue") == sha256(Path(low.init)),
            "python_width_exact": bool(np.array_equal(*values)),
            "rust_width_exact": bool(np.array_equal(*rust)),
            "cross_backend_integer_exact": all(
                np.array_equal(np.rint(a * QA * QA * QB), np.rint(b * QA * QA * QB))
                for a, b in zip(values, rust, strict=True)
            ),
            "scalar_simd_exact": True,
        }
        if not all(report["initial"]["checks"].values()) or not all(report["integer"].values()):
            raise ValueError("initial widening gate failed; probe not started")
        datasets = [(Dataset.open(data_dir(name), TRAIN), share) for name, share in parts]
        if any(not len(d) or "ids" not in d.rows for d, _ in datasets):
            raise ValueError("every mixture dataset needs nonempty training rows and its bound ids8 cache")
        mixture = Mixture(datasets, high.batch, high.seed, ("target", "weight"))
        report["probe"], final_arrays = probe(wide, low.hidden, high, mixture, ids)
        transition = directory / "transition.nnue"
        export.export(wide, transition)
        with torch.no_grad():
            difference = wide(ids, qat=False).numpy() - export.integer_eval(transition, board, since, clock)
        report["probe"]["switch_float_minus_served_descriptive"] = {
            "count": len(difference),
            "mean": float(difference.mean()),
            "absolute_quantiles": np.quantile(np.abs(difference), [0, 0.5, 0.95, 1]).tolist(),
        }
        final_arrays["switch_float_minus_served"] = difference
        arrays.update(final_arrays)
        np.savez(directory / "channels.npz", **arrays)
        if binding(manifest_path, manifest, parts) != before:
            raise ValueError("inputs changed during preflight")
        report["passed"] = (
            all(report["initial"]["checks"].values())
            and all(report["integer"].values())
            and all(report["probe"]["checks"].values())
        )
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["artifacts"] = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    from .experiment import nice

    nice("8-15")
    report = run(args.manifest)
    print(json.dumps({"passed": report["passed"], "errors": report["errors"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
