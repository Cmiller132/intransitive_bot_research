"""Cache exact frozen control-head inputs from a completed execution parity study.

The cache contains development features and prediction checks, never outcome
labels or confirmation data. CUDA extraction uses the recorded compiled actor
and preserves each full-pool batch context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from .probe_control import atomic_json, replay_candidate_pool, state_key
from .probe_parity import (
    SOURCES,
    CudaRunner,
    file_digest,
    load_execution,
    pack_arrays,
    unpack_arrays,
    worker_lock,
)

VERSION = 1
BATCH = 1024
FEATURES = 256
ROLE = "development features, no labels/confirmation"


def source_data(parity: dict, checkpoint_hash: str, cfg) -> tuple[dict, dict]:
    """Validate frozen evidence and retain only states, occurrences and prediction references."""
    load_execution(parity, checkpoint_hash, cfg)
    if parity.get("error") is not None:
        raise ValueError("source parity artifact contains an execution error")
    request = parity["request"]
    if cfg.net.state_head != "spatial" or cfg.net.value_hidden != FEATURES:
        raise ValueError("feature extraction requires the 256-dimensional spatial state encoder")
    if request["variants"]["C"] != "compiled_production_tf32_on":
        raise ValueError("source lacks the declared compiled production variant")
    setups = [s for s in parity["setups"] if s["variant"] == "C"]
    if not setups or any(
        s.get("cuda_matmul_tf32") is not True or s.get("cudnn_tf32") is not True or s.get("matmul_precision") != "high"
        for s in setups
    ):
        raise ValueError("source compiled backend settings do not match")
    pools = {
        identity: replay_candidate_pool(metadata["snapshot"], request["pilot_seed"], identity)
        for identity, metadata in request["pools"].items()
    }
    if not pools:
        raise ValueError("source has no complete occurrence pools")
    states, indices, mappings, plan = [], {}, {}, []
    for identity, pool in pools.items():
        local = []
        for state in pool.states:
            if any(not 0 <= v <= 6 for v in state["board"]) or not (
                0 <= state["since_capture"] < state["clock"] == cfg.rules.clock_max
                and 0 <= state["ply"] < cfg.rules.max_plies
            ):
                raise ValueError("pool contains an invalid or terminal state")
            key = state_key(state)
            if key not in indices:
                indices[key] = len(states)
                states.append(dict(state_key=key, state=state))
            local.append(indices[key])
        mappings[identity] = dict(
            family=identity,
            state_indices=local,
            occurrences=pool.occurrences,
            occurrence_columns=["local_state", "source", "step", "node", "recorded_rank", "recorded_regret"],
            batches=pool.batches,
            source_payload_sha256=pool.snapshot()["payload"]["sha256"],
        )
        for start in range(0, len(local), BATCH):
            stop = min(start + BATCH, len(local))
            spec = dict(
                id=f"C/pool/{identity}/{start}", variant="C", kind="pool", pool=identity, start=start, stop=stop
            )
            if spec not in request["plan"]:
                raise ValueError("source lacks the exact complete compiled pool-batch plan")
            unit = parity["units"][spec["id"]]
            if len(unit["result"]["arrays"]) != 1:
                raise ValueError("compiled pool batch has unexpected evidence")
            arrays = unpack_arrays(unit["result"]["arrays"][0])
            if set(arrays) != {"scores"} or arrays["scores"].shape != (stop - start, 2):
                raise ValueError("compiled pool scores have an invalid shape")
            rows = local[start:stop]
            plan.append(
                dict(
                    id=spec["id"],
                    pool=identity,
                    start=start,
                    stop=stop,
                    state_indices=rows + [rows[-1]] * (BATCH - len(rows)),
                    valid_rows=len(rows),
                    reference_scores=arrays["scores"].tolist(),
                )
            )
    saved_plan = [s["id"] for s in request["plan"] if s["variant"] == "C" and s["kind"] == "pool"]
    if saved_plan != [s["id"] for s in plan]:
        raise ValueError("compiled pool evidence order or coverage differs from the frozen pools")
    return dict(states=states, pools=mappings, plan=plan), pools


@contextmanager
def capture_input(head, buffer: torch.Tensor):
    """Copy the actual head input into a lossless, fixed-shape capture buffer."""

    def copy_input(module, args):
        value = args[0]
        if value.shape != buffer.shape or value.dtype != buffer.dtype or value.device != buffer.device:
            raise ValueError(
                f"head input/capture mismatch: actual {value.shape} {value.dtype} {value.device}; "
                f"buffer {buffer.shape} {buffer.dtype} {buffer.device}"
            )
        buffer.copy_(value)

    handle = head.register_forward_pre_hook(copy_input)
    try:
        yield
    finally:
        handle.remove()


def comparison(reference: np.ndarray, actual: np.ndarray, state_indices: list[int]) -> dict:
    """Report exact agreement and numerical error, including the worst element's state."""
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or reference.shape[0] != len(state_indices):
        raise ValueError("prediction comparison shape or state mapping mismatch")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite prediction evidence")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    mismatch = reference != actual
    worst = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    return dict(
        shape=list(reference.shape),
        reference_dtype=str(reference.dtype),
        actual_dtype=str(actual.dtype),
        exact_elements=int((~mismatch).sum()),
        differing_elements=int(mismatch.sum()),
        max_abs=float(np.abs(delta).max()),
        mean_abs=float(np.abs(delta).mean()),
        rmse=float(np.sqrt(np.mean(delta * delta))),
        worst=dict(
            state_index=state_indices[worst[0]],
            element=[int(i) for i in worst],
            reference=float(reference[worst]),
            actual=float(actual[worst]),
        ),
    )


def feature_array(buffer: torch.Tensor) -> np.ndarray:
    """Preserve BF16 bits; CPU fixtures retain their actual FP32 values."""
    value = buffer.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().copy()
    if value.dtype == torch.float32:
        return value.numpy().copy()
    raise ValueError(f"unsupported capture dtype {value.dtype}")


def feature_tensor(value: np.ndarray, dtype: torch.dtype, device) -> torch.Tensor:
    expected = np.uint16 if dtype == torch.bfloat16 else np.float32
    if value.dtype != expected or dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("feature encoding does not match its declared dtype")
    result = torch.from_numpy(value.copy())
    if dtype == torch.bfloat16:
        result = result.view(torch.bfloat16)
    return result.to(device)


@torch.no_grad()
def extract(runner, data: dict, compile_head, dtype, device, report: dict) -> None:
    """Compare unhooked, captured and reconstructed predictions in identical batch contexts."""
    head = runner.net.regret
    if not isinstance(head, torch.nn.Sequential) or head[0].in_features != FEATURES or head[-1].out_features != 2:
        raise ValueError("unexpected control-head architecture")
    runner.net.requires_grad_(False)
    states = [row["state"] for row in data["states"]]
    reference = {}
    report["chunks"] = {}
    for spec in data["plan"]:
        output = runner.forward([states[i] for i in spec["state_indices"]])
        reference[spec["id"]] = output
        scores = np.column_stack((output["rank"], output["regret"]))
        report["chunks"][spec["id"]] = dict(
            comparisons=dict(
                saved_C=comparison(
                    np.asarray(spec["reference_scores"]),
                    scores[: spec["valid_rows"]],
                    spec["state_indices"][: spec["valid_rows"]],
                )
            ),
            arrays=pack_arrays(dict(unhooked_scores=scores)),
        )
    buffer = torch.full((BATCH, FEATURES), torch.nan, dtype=dtype, device=device)
    torch._dynamo.mark_dynamic(buffer, 0)
    with capture_input(head, buffer):
        for spec in data["plan"]:
            buffer.fill_(torch.nan)
            output = runner.forward([states[i] for i in spec["state_indices"]])
            if not torch.isfinite(buffer).all():
                raise ValueError("capture buffer was not completely written with finite head inputs")
            before = reference.pop(spec["id"])
            if set(output) != set(before):
                raise ValueError("hook changed evaluator output fields")
            chunk = report["chunks"][spec["id"]]
            chunk["comparisons"]["hooked"] = {
                name: comparison(before[name], output[name], spec["state_indices"]) for name in before
            }
            arrays = unpack_arrays(chunk["arrays"])
            arrays.update(
                features=feature_array(buffer), hooked_scores=np.column_stack((output["rank"], output["regret"]))
            )
            chunk["arrays"] = pack_arrays(arrays)
            print(f"features captured {spec['id']}", flush=True)
    reconstruct = compile_head(head)
    for spec in data["plan"]:
        chunk = report["chunks"][spec["id"]]
        arrays = unpack_arrays(chunk["arrays"])
        inputs = feature_tensor(arrays["features"], dtype, device)
        torch._dynamo.mark_dynamic(inputs, 0)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if inputs.is_cuda else nullcontext()
        with context:
            scores = reconstruct(inputs).float().cpu().numpy().copy()
        chunk["comparisons"]["reconstructed"] = comparison(arrays["unhooked_scores"], scores, spec["state_indices"])
        arrays["reconstructed_scores"] = scores
        chunk["arrays"] = pack_arrays(arrays)


def numerical_failures(report: dict) -> list[str]:
    failures = []
    for identity, chunk in report["chunks"].items():
        values = chunk["comparisons"]
        for name, result in (
            ("saved_C", values["saved_C"]),
            *((f"hooked/{name}", result) for name, result in values["hooked"].items()),
            ("reconstructed", values["reconstructed"]),
        ):
            if result["differing_elements"]:
                failures.append(f"{identity}/{name}")
    return failures


def runtime_identity() -> dict:
    import triton

    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        triton=triton.__version__,
        numpy=np.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--parity", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"ckpt_\d{6}\.pt", args.ckpt.name):
        parser.error("use an immutable ckpt_NNNNNN.pt")
    if args.out.exists() or args.out.resolve() in (args.ckpt.resolve(), args.parity.resolve()):
        parser.error("feature output must be a new path separate from its inputs")
    if os.environ.get("TRITON_INTERPRET", "0") != "0" or os.environ.get("TORCH_COMPILE_DISABLE", "0") != "0":
        parser.error("CUDA feature extraction requires real Triton and enabled compilation")
    torch.set_num_threads(1)
    with worker_lock():
        from .model import config_of
        from .train import compile_net

        with args.ckpt.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        cfg = config_of(checkpoint)
        if any(not torch.isfinite(value).all() for value in checkpoint["ema"].values()):
            raise ValueError("nonfinite frozen EMA weights")
        parity_bytes = args.parity.read_bytes()
        parity = json.loads(parity_bytes)
        data, pools = source_data(parity, checkpoint_hash, cfg)
        runtime = runtime_identity()
        if runtime != parity["request"]["runtime"]:
            raise ValueError("runtime or CUDA hardware differs from the completed parity study")
        report = dict(
            schema=VERSION,
            status="running",
            role=ROLE,
            scope="Frozen development features and numerical reconstruction only; no utility labels, head fitting, "
            "confirmation, population inference or strength claim.",
            provenance=dict(
                checkpoint_sha256=checkpoint_hash,
                weights="ema",
                parity_sha256=hashlib.sha256(parity_bytes).hexdigest(),
                parity_source_sha256=parity["request"]["source_sha256"],
                source_sha256={
                    name: file_digest(Path(__file__).with_name(name)) for name in (*SOURCES, "fit_control.py")
                },
                runtime=runtime,
            ),
            settings=dict(
                execution="compiled_production_tf32_on",
                precision="bf16_autocast",
                capture_dtype="torch.bfloat16",
                feature_encoding="bfloat16_uint16_bits",
                batch=BATCH,
                feature_dimension=FEATURES,
                padding="repeat the last valid state, matching the source C full-pool batches",
                numerical_gate="exact saved-C, unhooked/hooked and original-head reconstruction equality",
                fullgraph=False,
                error_on_graph_break=True,
                skip_nnmodule_hook_guards=False,
                dynamic_dimensions=dict(capture_buffer=[0], reconstruction_input=[0]),
                unhooked_batch_context="automatic dynamic batch after the source C warm-up at 1 then 1024",
                allow_ignore_mark_dynamic=False,
                graph_break_policy="raise during tracing; verify recorded counters; no eager fallback",
                occurrence_scores="recorded model predictions, not outcome or utility labels",
            ),
            data=data,
        )
        atomic_json(args.out, report, create=True)
        runner = None
        before_breaks = None
        started = time.perf_counter()
        try:
            import torch._dynamo.config as dynamo_config
            from torch._dynamo import error_on_graph_break
            from torch._dynamo.utils import counters

            before_breaks = dict(counters["graph_break"])
            with (
                dynamo_config.patch(
                    suppress_errors=False, skip_nnmodule_hook_guards=False, allow_ignore_mark_dynamic=False
                ),
                error_on_graph_break(True),
            ):
                runner = CudaRunner(
                    cfg,
                    checkpoint,
                    "C",
                    parity["request"]["panel"],
                    parity["request"]["batch"],
                    pools,
                    parity["request"]["pilot_seed"],
                )
                runner.net.requires_grad_(False)
                report["setup"] = runner.warmup()
                extract(runner, data, lambda head: compile_net(head, cfg), torch.bfloat16, "cuda", report)
            report["graph_breaks"] = {
                reason: count - before_breaks.get(reason, 0)
                for reason, count in counters["graph_break"].items()
                if count > before_breaks.get(reason, 0)
            }
            report["numerical_failures"] = numerical_failures(report)
            if report["graph_breaks"] or report["numerical_failures"]:
                raise ValueError("compiled feature validation failed; see graph_breaks and per-chunk comparisons")
            report["status"] = "complete"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["elapsed_seconds"] = time.perf_counter() - started
            if before_breaks is not None:
                report["graph_breaks"] = {
                    reason: count - before_breaks.get(reason, 0)
                    for reason, count in counters["graph_break"].items()
                    if count > before_breaks.get(reason, 0)
                }
            try:
                atomic_json(args.out, report)
            finally:
                if runner is not None:
                    runner.close()
        print(args.out)


if __name__ == "__main__":
    main()
