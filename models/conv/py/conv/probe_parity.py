"""Frozen EMA evaluator, search and candidate-selection parity against a completed utility probe.

The CUDA-only command compares eager numerical references and compiled actor
execution against a protocol-4 compiled-B1 utility probe. Comparisons are not an
acceptance test for an RGSC target or a claim of trajectory equivalence.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from compression import zstd

from .control import State
from .planes import N_ACTIONS, initial_board
from .probe_control import (
    Game,
    atomic_json,
    replay_candidate_pool,
    seed_for,
    state_from,
    state_key,
    state_record,
    validate_return_support,
)

VERSION = 3
VARIANTS = {"A": "eager_reference_tf32_off", "B": "eager_reference_tf32_on", "C": "compiled_production_tf32_on"}
SOURCES = (
    "probe_parity.py",
    "probe_control.py",
    "train.py",
    "model.py",
    "search.py",
    "env.py",
    "kernels.py",
    "planes.py",
    "config.py",
    "control.py",
)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pack_arrays(arrays: dict[str, np.ndarray]) -> dict:
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    for value in arrays.values():
        if value.dtype.hasobject or (value.dtype.kind in "fc" and not np.isfinite(value).all()):
            raise ValueError("nonfinite or object array in parity evidence")
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    raw = stream.getvalue()
    return dict(
        sha256=hashlib.sha256(raw).hexdigest(), size=len(raw), data=base64.b64encode(zstd.compress(raw)).decode()
    )


def unpack_arrays(payload: dict) -> dict[str, np.ndarray]:
    compressed = base64.b64decode(payload["data"], validate=True)
    if zstd.get_frame_info(compressed).decompressed_size != payload["size"]:
        raise ValueError("array payload size mismatch")
    raw = zstd.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != payload["sha256"]:
        raise ValueError("array payload checksum mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(v.dtype.hasobject or (v.dtype.kind in "fc" and not np.isfinite(v).all()) for v in arrays.values()):
        raise ValueError("nonfinite or object array in parity evidence")
    return arrays


def validate_config(cfg) -> None:
    validate_return_support(cfg.rules.clock_penalty)
    if not cfg.learn.compile or cfg.learn.envs != 1024 or not (cfg.net.regret and cfg.net.wdl):
        raise ValueError("parity requires the compiled 1024-board actor with both control and WDL heads")
    if (cfg.search.sims, cfg.search.candidates, cfg.search.cheap_sims, cfg.search.cheap_candidates) != (128, 16, 16, 4):
        raise ValueError("unsupported full/cheap search budgets")
    if cfg.search.contempt_source != "wdl" or cfg.rules.clock_min != cfg.rules.clock_max:
        raise ValueError("parity requires WDL contempt and the fixed-clock pilot")
    if any(not math.isfinite(value) for value in asdict(cfg.search).values() if isinstance(value, float)):
        raise ValueError("nonfinite search configuration")


def validate_probe_definition(definition: dict, checkpoint_hash: str, cfg) -> None:
    """Bind the current compiled-B1 behavior, checkpoint and source implementation."""
    if definition["version"] != 4:
        raise ValueError("unsupported utility-probe definition version")
    if definition["checkpoint_sha256"] != checkpoint_hash or definition["weights"] != "ema":
        raise ValueError("pilot does not use this frozen EMA checkpoint")
    if (
        definition["inference_execution"] != "compiled"
        or definition["inference_precision"] != "bf16_autocast"
        or definition["inference_batch"] != 1
        or definition["execution_scope"] != "development_learnability"
    ):
        raise ValueError("unsupported pilot evaluator")
    if definition["device"] != "cuda" or definition["runtime"]["cuda_matmul_tf32"] is not True:
        raise ValueError("pilot must use CUDA compiled BF16 B1 with matmul TF32 on")
    if any(
        definition["runtime"][name] != expected
        for name, expected in dict(
            matmul_precision="high",
            cudnn_tf32=True,
            cudnn_benchmark=False,
            cudnn_deterministic=False,
            deterministic_algorithms=False,
        ).items()
    ):
        raise ValueError("unsupported pilot backend flags")
    if definition["search"] != asdict(cfg.search) or definition["rules"] != asdict(cfg.rules):
        raise ValueError("checkpoint search/rules disagree with pilot")
    if definition["draw_kernel_width"] != cfg.play.draw_kernel_width:
        raise ValueError("checkpoint draw settings disagree with pilot")
    validate_config(cfg)
    for name, expected in definition["source_sha256"].items():
        if name not in SOURCES or file_digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"utility probe source changed: {name}")
    if set(definition["source_sha256"]) != set(SOURCES) - {"probe_parity.py"}:
        raise ValueError("incomplete utility probe source identity")


def load_pools(pilot: dict, checkpoint_hash: str, cfg) -> dict:
    """Validate the completed source and reproduce its complete occurrence pools."""
    if pilot.get("status") != "complete" or pilot.get("schema") != 4 or pilot.get("error") is not None:
        raise ValueError("parity requires a completed protocol-4 utility probe")
    request, definition = pilot["request"], pilot["definition"]
    protocol = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    if pilot["protocol"] != protocol or request["protocol"] != protocol or request["seed"] != pilot["seed"]:
        raise ValueError("pilot protocol or selection-seed identity mismatch")
    if (
        request["version"] != 4
        or type(request["games"]) is not int
        or request["games"] < 1
        or len(pilot["games"]) != request["games"]
        or request["definition"] != definition
        or request["selection"] not in ("selectors", "uniform_corpus")
    ):
        raise ValueError("invalid current utility-probe request")
    validate_probe_definition(definition, checkpoint_hash, cfg)
    episodes = {e["episode_id"]: e for e in pilot["episodes"]}
    if len(episodes) != len(pilot["episodes"]):
        raise ValueError("duplicate source episode identity")
    pools = {}
    for game in pilot["games"]:
        identity = game["game"]
        if identity in pools:
            raise ValueError("duplicate discovery identity")
        episode = episodes[identity]
        if game["discovery_censored"] or episode["censored"]:
            raise ValueError("parity requires complete discovery pools")
        pool = replay_candidate_pool(episode["candidate_pool"], pilot["seed"], identity)
        if game["candidates"] != pool.selections():
            raise ValueError("source selection summary disagrees with its complete pool")
        for state in pool.states:
            if any(not 0 <= value <= 6 for value in state["board"]):
                raise ValueError("invalid board encoding in source pool")
            if (
                not 0 <= state["since_capture"] < state["clock"] == cfg.rules.clock_max
                or state["ply"] >= cfg.rules.max_plies
            ):
                raise ValueError("source pool contains a terminal or incompatible state")
        pools[identity] = pool
    return pools


def freeze_panel(pools: dict, clock: int) -> tuple[list[dict], list[dict]]:
    """Keep selections from the first two discoveries, then fixed strata over all pools."""
    panel, chosen = [], {}

    def add(state, reason, shared_reason=False):
        key = state_key(state)
        if key in chosen:
            if shared_reason:
                chosen[key]["reasons"].append(reason)
            return False
        entry = dict(state=state, state_key=key, reasons=[reason])
        chosen[key] = entry
        panel.append(entry)
        return True

    occurrences = []
    all_states = {}
    for identity, pool in list(pools.items())[:2]:
        for method, selection in pool.selections()["selected"].items():
            add(selection["state"], f"{identity}/{method}", True)
    for identity, pool in pools.items():
        for state in pool.states:
            all_states.setdefault(state_key(state), state)
        for index, row in enumerate(pool.occurrences):
            occurrences.append((identity, index, pool.states[row[0]], row))
    add(state_record(State(np.asarray(initial_board(), np.int8), 0, 0, clock)), "initial_position", True)
    strata = (
        ("played_early", "played", lambda x: x[2]["ply"]),
        ("played_late", "played", lambda x: -x[2]["ply"]),
        ("tree_early", "tree", lambda x: x[2]["ply"]),
        ("tree_late", "tree", lambda x: -x[2]["ply"]),
        ("near_clock", None, lambda x: x[2]["clock"] - x[2]["since_capture"]),
        ("rank_max", None, lambda x: -x[3][4]),
        ("rank_min", None, lambda x: x[3][4]),
    )
    for reason, source, criterion in strata:
        if len(panel) == 16:
            break
        for identity, index, state, row in sorted(occurrences, key=lambda x: (criterion(x), x[0], x[1])):
            if (source is None or row[1] == source) and add(state, f"{reason}/{identity}/{index}"):
                break
    for key, state in sorted(all_states.items()):
        if len(panel) == 16:
            break
        if key not in chosen:
            add(state, "distinct_fill")
    if len(panel) != 16:
        raise ValueError("source pools cannot supply 16 distinct panel states")
    remaining = [state for key, state in all_states.items() if key not in chosen]
    if len(remaining) < 1008:
        raise ValueError("source pools cannot supply a heterogeneous 1024-state batch")
    indices = np.linspace(0, len(remaining) - 1, 1008, dtype=np.int64)
    return panel, [e["state"] for e in panel] + [remaining[i] for i in indices]


def make_plan(pools: dict) -> list[dict]:
    plan = []
    for variant in VARIANTS:
        for batch in (1, 1024):
            plan.append(dict(id=f"{variant}/eval/{batch}", variant=variant, kind="eval", batch=batch))
        if variant == "B":
            continue
        for state in range(16):
            for seed in (0, 1):
                for full in (False, True):
                    plan.append(
                        dict(
                            id=f"{variant}/root/{state}/{seed}/{int(full)}",
                            variant=variant,
                            kind="root",
                            state=state,
                            seed=seed,
                            full=full,
                            steps=1,
                        )
                    )
        for full in (False, True):
            plan.append(
                dict(
                    id=f"{variant}/repeat/{int(full)}",
                    variant=variant,
                    kind="root",
                    state=0,
                    seed=0,
                    full=full,
                    steps=1,
                )
            )
            plan.append(
                dict(id=f"{variant}/batch/{int(full)}", variant=variant, kind="batch", seed=0, full=full, steps=1)
            )
        for state in range(4):
            plan.append(
                dict(
                    id=f"{variant}/switch/{state}",
                    variant=variant,
                    kind="root",
                    state=state,
                    seed=0,
                    full=bool(state % 2),
                    steps=2,
                )
            )
        if variant == "C":
            for identity, pool in pools.items():
                for start in range(0, len(pool.states), 1024):
                    plan.append(
                        dict(
                            id=f"C/pool/{identity}/{start}",
                            variant=variant,
                            kind="pool",
                            pool=identity,
                            start=start,
                            stop=min(start + 1024, len(pool.states)),
                        )
                    )
    return plan


def load_execution(artifact: dict, checkpoint_hash: str, cfg) -> None:
    """Validate a completed current execution artifact for feature extraction."""
    validate_config(cfg)
    request = artifact["request"]
    if (
        artifact.get("status") != "complete"
        or artifact.get("schema") != VERSION
        or request["version"] != VERSION
        or artifact.get("error") is not None
    ):
        raise ValueError("requires a completed current execution parity artifact")
    if request["utility_schema"] != 4:
        raise ValueError("execution source is not a current utility probe")
    validate_probe_definition(request["utility_definition"], checkpoint_hash, cfg)
    if request["checkpoint_sha256"] != checkpoint_hash or request["variants"] != VARIANTS:
        raise ValueError("execution checkpoint or variant mismatch")
    if request["config"] != asdict(cfg) or len(request["batch"]) != 1024 or len(request["panel"]) != 16:
        raise ValueError("unsupported parity configuration or state panel")
    planned = {spec["id"]: spec for spec in request["plan"]}
    if len(planned) != len(request["plan"]) or set(artifact["units"]) != set(planned):
        raise ValueError("parity has duplicate or incomplete units")
    for identity, unit in artifact["units"].items():
        if unit["spec_sha256"] != digest(planned[identity]) or unit["sha256"] != digest(unit["result"]):
            raise ValueError("parity unit identity or checksum mismatch")
        for payload in unit["result"]["arrays"]:
            unpack_arrays(payload)
    if set(request["source_sha256"]) != set(SOURCES):
        raise ValueError("parity source identity is incomplete")
    for name, expected in request["source_sha256"].items():
        if file_digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"parity source changed: {name}")


@contextmanager
def worker_lock():
    """One Windows parity worker; the operating system releases the mutex after a crash."""
    if os.name != "nt":
        raise RuntimeError("this parity command currently supports the Windows CUDA runtime only")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.CreateMutexW(None, False, "Local\\conv_production_actor_parity")
    if not handle:
        raise OSError(ctypes.get_last_error(), "could not create parity worker mutex")
    acquired = kernel.WaitForSingleObject(handle, 0) in (0, 0x80)
    try:
        if not acquired:
            raise RuntimeError("another parity worker is active")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


class Progress:
    """Atomic completed-unit evidence with an immutable request and explicit remaining work."""

    def __init__(self, path: Path, request: dict, resume: bool):
        if request["version"] != VERSION:
            raise ValueError("parity requires a current request")
        self.path, self.request = path, request
        self.units, self.setups, self.comparisons = {}, [], {}
        self.elapsed, self.started = 0.0, time.perf_counter()
        if resume:
            saved = json.loads(path.read_text())
            if (
                saved["schema"] != VERSION
                or saved["request"] != request
                or saved["status"] not in ("running", "complete")
            ):
                raise ValueError("parity progress request or status mismatch")
            self.units, self.setups, self.elapsed = saved["units"], saved["setups"], saved["elapsed_seconds"]
            planned = {s["id"]: s for s in request["plan"]}
            for identity, unit in self.units.items():
                if (
                    identity not in planned
                    or unit["spec_sha256"] != digest(planned[identity])
                    or unit["sha256"] != digest(unit["result"])
                ):
                    raise ValueError("parity progress unit identity or checksum mismatch")
                for payload in unit["result"].get("arrays", []):
                    unpack_arrays(payload)
            if saved["status"] == "complete" and set(self.units) != set(planned):
                raise ValueError("complete parity progress has unrun units")
        else:
            self.save(create=True)

    def save(self, create=False, error=None):
        remaining = [s["id"] for s in self.request["plan"] if s["id"] not in self.units]
        report = dict(
            schema=VERSION,
            status="running" if remaining else "complete",
            request=self.request,
            units=self.units,
            setups=self.setups,
            elapsed_seconds=self.elapsed + time.perf_counter() - self.started,
            remaining=remaining,
            comparisons=self.comparisons,
            scope="Frozen numerical references and execution-context comparisons; the protocol-4 source uses "
            "compiled B1. These are not utility or strength acceptance tests.",
            coupling="Same-shape searches share explicit modes and search seeds. "
            "Whole-actor master seeds and cross-batch RNG prefixes are not assumed equivalent.",
            timing="Network warm-up/compile is recorded per process setup. Each root unit separately records "
            "graph warm-up and measured search. Interrupted in-flight work is rerun "
            "and is absent from recorded unit costs.",
            error=error,
        )
        atomic_json(self.path, report, create=create)
        return report

    def record(self, spec, result):
        if spec["id"] in self.units:
            raise ValueError("parity unit already recorded")
        digest(result)
        self.units[spec["id"]] = dict(spec_sha256=digest(spec), sha256=digest(result), result=result)
        self.comparisons.update(summarize(self.units, self.request, set(self.comparisons)))
        self.save()


def error_stats(first, second) -> dict:
    difference = np.abs(np.asarray(first, np.float64) - np.asarray(second, np.float64))
    if not difference.size or not np.isfinite(difference).all():
        raise ValueError("invalid numerical comparison")
    return dict(
        mean=float(difference.mean()), p99=float(np.quantile(difference, 0.99)), maximum=float(difference.max())
    )


def probabilities(logits, legal=None):
    values = np.asarray(logits, np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite score")
    mask = np.ones_like(values, bool) if legal is None else np.asarray(legal, bool)
    if not mask.any(axis=-1).all():
        raise ValueError("empty legal or candidate set")
    values = np.where(mask, values, -np.inf)
    mass = np.exp(values - values.max(axis=-1, keepdims=True))
    return mass / mass.sum(axis=-1, keepdims=True)


def evaluator_comparison(a, b):
    if set(a) != set(b) or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("evaluator evidence shape mismatch")
    legal_equal = bool(np.array_equal(a["legal"], b["legal"]))
    result = dict(
        rows=len(a["v"]), legal_exact=legal_equal, legal_count_exact=bool(np.array_equal(a["count"], b["count"]))
    )
    for name in ("q", "v", "rank", "regret", "draw"):
        result[name] = error_stats(a[name], b[name])
    if legal_equal:
        tv = np.abs(probabilities(a["logits"], a["legal"]) - probabilities(b["logits"], b["legal"])).sum(-1) / 2
        result["policy_tv"] = error_stats(tv, np.zeros_like(tv))
    return result


def root_comparison(a, b):
    same_state = a["state_key"] == b["state_key"]
    result = dict(rows=len(same_state), matched_state_rows=int(same_state.sum()))
    if not same_state.any():
        return result
    a, b = ({k: v[same_state] for k, v in side.items()} for side in (a, b))
    for name in (
        "action",
        "moves",
        "tactics",
        "initial_effective_candidates",
        "initial_physical_candidates",
        "candidates",
        "candidate_states_sha256",
        "tree_sha256",
    ):
        if name in a and name in b:
            result[f"{name}_exact_rows"] = int(np.all(a[name] == b[name], axis=tuple(range(1, a[name].ndim))).sum())
    result["tree_context_hash_available"] = "tree_sha256" in a and "tree_sha256" in b
    for name in ("played_q", "value", "rank", "regret"):
        result[name] = error_stats(a[name], b[name])
    same_moves = np.all(a["moves"] == b["moves"], axis=-1)
    result["aligned_move_rows"] = int(same_moves.sum())
    if same_moves.any():
        result["q"] = error_stats(a["q"][same_moves], b["q"][same_moves])
        tv = np.abs(a["target"][same_moves] - b["target"][same_moves]).sum(-1) / 2
        result["target_tv"] = error_stats(tv, np.zeros_like(tv))
    return result


def summarize(units, request, skip=None):
    comparisons = {}
    skip = skip or set()
    cache = {}

    def arrays(identity):
        if identity not in cache:
            cache[identity] = [unpack_arrays(v) for v in units[identity]["result"]["arrays"]]
        return cache[identity]

    for batch in (1, 1024):
        for first, second in (("A", "B"), ("B", "C"), ("A", "C")):
            a, b = f"{first}/eval/{batch}", f"{second}/eval/{batch}"
            if a in units and b in units and f"{a}:{b}" not in skip:
                comparisons[f"{a}:{b}"] = evaluator_comparison(arrays(a)[0], arrays(b)[0])
    for variant in VARIANTS:
        a, b = f"{variant}/eval/1", f"{variant}/eval/1024"
        if a in units and b in units and f"{variant}/batch_shape" not in skip:
            comparisons[f"{variant}/batch_shape"] = evaluator_comparison(
                arrays(a)[0], {k: v[:16] for k, v in arrays(b)[0].items()}
            )
    for identity in units:
        if identity.startswith("A/") and any(f"/{kind}/" in identity for kind in ("root", "batch", "switch")):
            other = "C/" + identity[2:]
            if other in units and f"{identity}:{other}" not in skip:
                comparisons[f"{identity}:{other}"] = [
                    root_comparison(a, b) for a, b in zip(arrays(identity), arrays(other), strict=False)
                ]
                comparisons[f"{identity}:{other}/recorded_steps"] = [len(arrays(identity)), len(arrays(other))]
    for variant in ("A", "C"):
        for full in (0, 1):
            a, b = f"{variant}/root/0/0/{full}", f"{variant}/repeat/{full}"
            if a in units and b in units and f"{variant}/repeat/{full}" not in skip:
                comparisons[f"{variant}/repeat/{full}"] = root_comparison(arrays(a)[0], arrays(b)[0])
    for identity, metadata in request["pools"].items():
        planned = [s for s in request["plan"] if s["kind"] == "pool" and s["pool"] == identity]
        if planned and all(s["id"] in units for s in planned) and f"C/pool/{identity}" not in skip:
            scores = np.concatenate([arrays(s["id"])[0]["scores"] for s in planned])
            comparisons[f"C/pool/{identity}"] = pool_comparison(
                metadata["snapshot"], scores, request["pilot_seed"], identity
            )
    return comparisons


def pool_comparison(snapshot, scores, master, identity):
    pool = replay_candidate_pool(snapshot, master, identity)
    if scores.shape != (len(pool.states), 2) or not np.isfinite(scores).all():
        raise ValueError("incomplete or nonfinite full-pool rescore")
    from .probe_control import CandidatePool

    altered = CandidatePool(master, identity)
    old, new, start = [], [], 0
    for size in pool.batches:
        batch = []
        for state_id, source, step, node, rank, regret in pool.occurrences[start : start + size]:
            replacement = scores[state_id]
            old.append([rank, regret])
            new.append(replacement)
            batch.append(
                dict(
                    state=pool.states[state_id],
                    source=source,
                    occurrence=[step, node],
                    rank=float(replacement[0]),
                    regret=float(replacement[1]),
                )
            )
        altered.offer(batch)
        start += size
    old, new = np.asarray(old), np.asarray(new)
    a, b = pool.selections()["selected"], altered.selections()["selected"]
    return dict(
        occurrences=len(old),
        distinct_states=len(scores),
        rank=error_stats(old[:, 0], new[:, 0]),
        regret=error_stats(old[:, 1], new[:, 1]),
        rank_probability_tv=float(np.abs(probabilities(old[:, 0]) - probabilities(new[:, 0])).sum() / 2),
        selectors={
            method: dict(
                same_state=state_key(a[method]["state"]) == state_key(b[method]["state"]),
                same_occurrence=a[method]["occurrence_id"] == b[method]["occurrence_id"],
                original=a[method],
                rescored=b[method],
            )
            for method in a
        },
        source_execution="recorded compiled B1 occurrence predictions",
        rescored_execution="compiled B1024 full-pool chunks",
        scope="The same saved occurrence pools, multiplicities, offer boundaries and selector RNGs; "
        "this is an execution-context comparison, not a utility or strength test. "
        "Changes in tree availability are examined separately by root comparisons.",
    )


class CudaRunner:
    """One EMA/evaluator and one temporary search arena at a time."""

    def __init__(self, cfg, checkpoint, variant, panel, batch, pools, master):
        from .model import build
        from .train import compile_net, student_evaluator

        self.cfg, self.panel, self.batch, self.pools = cfg, panel, batch, pools
        self.master = master
        torch.backends.cuda.matmul.allow_tf32 = variant != "A"
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        self.net = build(cfg, checkpoint, "cuda", "ema").eval()
        if getattr(self.net, "fresh", []):
            raise ValueError("parity requires complete frozen EMA heads")
        forward = compile_net(self.net, cfg) if variant == "C" else self.net
        self.evaluate = student_evaluator(forward, cfg.play.draw_kernel_width, cfg.search.contempt_source)

    def tensors(self, states):
        return tuple(
            torch.tensor(values, device="cuda", dtype=dtype)
            for values, dtype in (
                ([s["board"] for s in states], torch.int8),
                ([s["since_capture"] for s in states], torch.int32),
                ([s["ply"] for s in states], torch.int32),
                ([s["clock"] for s in states], torch.int32),
            )
        )

    @torch.no_grad()
    def forward(self, states):
        output = self.evaluate(*self.tensors(states))
        names = ("logits", "q", "v", "draw", "legal", "count", "rank", "regret")
        if len(output) != len(names):
            raise ValueError("evaluator lacks required control heads")
        arrays = {name: value.detach().float().cpu().numpy() for name, value in zip(names, output, strict=True)}
        legal = arrays["legal"].astype(bool)
        if (
            legal.shape != (len(states), N_ACTIONS)
            or not legal.any(-1).all()
            or not np.array_equal(legal.sum(-1), arrays["count"])
        ):
            raise ValueError("invalid evaluator legality")
        arrays["legal"] = legal
        expected = {
            name: (len(states), N_ACTIONS) if name in ("logits", "q", "legal") else (len(states),) for name in names
        }
        if any(value.shape != expected[name] for name, value in arrays.items()):
            raise ValueError("unexpected evaluator output shape")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("nonfinite evaluator output")
        return arrays

    def warmup(self):
        times = {}
        for batch in (1, 1024):
            start = time.perf_counter()
            for _ in range(3):
                self.forward(self.batch[:batch])
            torch.cuda.synchronize()
            times[str(batch)] = time.perf_counter() - start
        return dict(
            seconds=times,
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
        )

    @torch.no_grad()
    def root(self, spec):
        from .env import Env
        from .search import LOSING_RANK, GumbelSearch, RepetitionHistory

        states = self.batch if spec["kind"] == "batch" else [self.panel[spec["state"]]["state"]]
        seed = seed_for(self.master, "parity_root", spec["seed"], spec.get("state", "batch"), spec["full"])
        if len(states) == 1:
            game = Game(self.cfg, self.evaluate, state_from(states[0]), seed, "cuda", context_seed=seed)
        else:
            env = Env(
                len(states), "cuda", self.cfg.rules.max_plies, self.cfg.rules.clock_min, self.cfg.rules.clock_max, seed
            )
            history = RepetitionHistory(len(states), self.cfg.rules.clock_max, "cuda")
            search = GumbelSearch(
                self.cfg.search,
                len(states),
                "cuda",
                env.clock,
                self.cfg.rules.clock_penalty,
                self.cfg.rules.max_plies,
                seed,
                history,
            )
            game = SimpleNamespace(env=env, history=history, search=search)
        env, history, search = game.env, game.history, game.search

        def reset(env, history, search):
            for destination, value in zip(
                (env.board, env.since_capture, env.ply, env.clock), self.tensors(states), strict=True
            ):
                destination.copy_(value)
            env.derive()
            history.reset(env.board, env.ply)
            search.forget()
            search.rng.manual_seed(seed)

        reset(env, history, search)
        start = time.perf_counter()
        search(env.board, env.since_capture, env.ply, env.legal, self.evaluate, spec["full"])
        torch.cuda.synchronize()
        warmup = time.perf_counter() - start
        reset(env, history, search)
        arrays, ends, measured = [], [], 0.0
        for step in range(spec["steps"]):
            full = spec["full"] if step == 0 else not spec["full"]
            start = time.perf_counter()
            root = search(env.board, env.since_capture, env.ply, env.legal, self.evaluate, full)
            torch.cuda.synchronize()
            measured += time.perf_counter() - start
            rows = torch.arange(len(states), device="cuda")
            if not env.legal[rows, root.action].all():
                raise ValueError("search selected an illegal action")
            output = {
                name: getattr(root, name).detach().cpu().numpy()
                for name in ("moves", "target", "value", "action", "candidates", "tactics", "played_q", "rank")
            }
            output["q"] = search._completed(torch.zeros(len(states), device="cuda", dtype=torch.long)).cpu().numpy()
            output["regret"] = search.regret[0].cpu().numpy()
            qplay = (
                search._completed(torch.zeros(len(states), device="cuda", dtype=torch.long))
                - self.cfg.search.contempt * search.root_draw
            )
            final_score = search._root_rank(search.score0 + search._sigma(search.count[0]) * qplay).gather(
                1, search.survivors
            )
            final_score = final_score.masked_fill(search.slots >= search.previous_m[:, None], -torch.inf)
            for name, value in (("score0", search.score0), ("final_selection_score", final_score)):
                if torch.isnan(value).any():
                    raise ValueError("NaN in root selection score")
                output[name] = torch.where(torch.isfinite(value), value, 0).cpu().numpy()
                output[f"{name}_finite"] = torch.isfinite(value).cpu().numpy()
                output[f"{name}_infinity"] = (
                    (torch.isposinf(value).to(torch.int8) - torch.isneginf(value).to(torch.int8)).cpu().numpy()
                )
            output["root_logp"] = torch.where(torch.isfinite(search.root_logp), search.root_logp, 0).cpu().numpy()
            output["sigma"] = search._sigma(search.count[0]).cpu().numpy()
            output["visits"] = search.count[0].cpu().numpy()
            output["active_survivors"] = search.previous_m.cpu().numpy()
            output["root_win1"] = search.root_win1.cpu().numpy()
            admitted = search._root_rank(search.score0, LOSING_RANK).topk(self.cfg.search.candidates, -1).indices
            admission_count = env.legal_count.clamp(
                max=self.cfg.search.candidates if full else self.cfg.search.cheap_candidates
            )
            slots = torch.arange(self.cfg.search.candidates, device="cuda")
            output["initial_effective_candidates"] = (
                search.actions[0]
                .gather(1, admitted)
                .masked_fill(slots[None] >= admission_count[:, None], -1)
                .cpu()
                .numpy()
            )
            output["initial_physical_candidates"] = search.actions[0].gather(1, admitted).cpu().numpy()
            output["candidates"] = (
                root.candidates.masked_fill(slots[None] >= search.previous_m[:, None], -1).cpu().numpy()
            )
            boards, since, plies, clocks = (v.cpu().numpy() for v in (env.board, env.since_capture, env.ply, env.clock))
            output["state_key"] = np.asarray(
                [State(boards[i], int(since[i]), int(plies[i]), int(clocks[i])).key().hex() for i in range(len(states))]
            )
            tree_boards, tree_since, tree_plies, terminal, count = (
                v.cpu().numpy() for v in (search.boards, search.since, search.plies, search.terminal, search.node_count)
            )
            hashes = []
            for row in range(len(states)):
                keys = [
                    State(
                        tree_boards[node, row], int(tree_since[node, row]), int(tree_plies[node, row]), int(clocks[row])
                    )
                    .key()
                    .hex()
                    for node in range(1, int(count[row]))
                    if not terminal[node, row]
                ]
                hashes.append(digest(sorted(keys)))
            output["candidate_states_sha256"] = np.asarray(hashes)
            if len(states) == 1:
                output["tree_sha256"] = np.asarray([game.tree_hash()])
            arrays.append(pack_arrays(output))
            if step + 1 < spec["steps"]:
                env.step(root.action.to(torch.int32))
                ends = env.end_reason.cpu().tolist()
                if env.done.any():
                    break
                history.push(env.board, env.since_capture, env.ply, env.done)
                search.advance(env.done)
        del game, env, history, search
        gc.collect()
        torch.cuda.empty_cache()
        return dict(
            arrays=arrays,
            warmup_seconds=warmup,
            measured_seconds=measured,
            end_reasons=ends,
            requested_steps=spec["steps"],
            search_seed=seed,
            initial_full=spec["full"],
            selection_score_scope="Final survivor scores with explicit infinity masks; score0 retains policy plus "
            "temperature-scaled root Gumbels. Earlier elimination margins are not recorded.",
        )

    def run(self, spec):
        if spec["kind"] in ("root", "batch"):
            return self.root(spec)
        start = time.perf_counter()
        if spec["kind"] == "pool":
            states = self.pools[spec["pool"]].states[spec["start"] : spec["stop"]]
            size = len(states)
            output = self.forward(states + [states[-1]] * (1024 - size))
            arrays = dict(scores=np.column_stack((output["rank"][:size], output["regret"][:size])))
        elif spec["batch"] == 1:
            values = [self.forward([entry["state"]]) for entry in self.panel]
            arrays = {name: np.concatenate([v[name] for v in values]) for name in values[0]}
        else:
            arrays = self.forward(self.batch)
        return dict(arrays=[pack_arrays(arrays)], measured_seconds=time.perf_counter() - start)

    def close(self):
        del self.evaluate, self.net
        gc.collect()
        torch.cuda.empty_cache()


def execute(progress, factory):
    try:
        progress.comparisons = summarize(progress.units, progress.request)
        for variant in VARIANTS:
            pending = [s for s in progress.request["plan"] if s["variant"] == variant and s["id"] not in progress.units]
            if not pending:
                continue
            start = time.perf_counter()
            runner = factory(variant)
            try:
                construction = time.perf_counter() - start
                progress.setups.append(dict(variant=variant, construction_seconds=construction, **runner.warmup()))
                progress.save()
                for spec in pending:
                    progress.record(spec, runner.run(spec))
                    print(f"parity {len(progress.units)}/{len(progress.request['plan'])}: {spec['id']}", flush=True)
            finally:
                runner.close()
                del runner
    except Exception as error:
        progress.save(error=f"{type(error).__name__}: {error}")
        raise
    return progress.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"ckpt_\d{6}\.pt", args.ckpt.name):
        parser.error("use an immutable ckpt_NNNNNN.pt")
    if args.out.resolve() in (args.ckpt.resolve(), args.pilot.resolve()):
        parser.error("output must be separate from checkpoint and pilot")
    if args.out.exists() != args.resume:
        parser.error("new output must not exist; --resume requires an existing artifact")
    torch.set_num_threads(1)
    with worker_lock():
        from .model import config_of

        with args.ckpt.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        if any(not torch.isfinite(value).all() for value in checkpoint["ema"].values()):
            raise ValueError("nonfinite frozen EMA weights")
        cfg = config_of(checkpoint)
        source_bytes = args.pilot.read_bytes()
        artifact = json.loads(source_bytes)
        pools = load_pools(artifact, checkpoint_hash, cfg)
        panel, batch = freeze_panel(pools, cfg.rules.clock_max)
        master = artifact["seed"]
        import triton

        runtime = dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            triton=triton.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count,
        )
        source_runtime = artifact["definition"]["runtime"]
        aliases = dict(device="cuda_device", capability="cuda_capability", multiprocessors="cuda_multiprocessors")
        if any(source_runtime[aliases.get(name, name)] != value for name, value in runtime.items()):
            raise ValueError("runtime or CUDA hardware differs from the source pilot")
        request = dict(
            version=VERSION,
            checkpoint_sha256=checkpoint_hash,
            utility_probe_sha256=hashlib.sha256(source_bytes).hexdigest(),
            utility_schema=artifact["schema"],
            utility_definition=artifact["definition"],
            pilot_seed=master,
            source_sha256={name: file_digest(Path(__file__).with_name(name)) for name in SOURCES},
            runtime=runtime,
            config=asdict(cfg),
            variants=VARIANTS,
            panel=panel,
            batch=batch,
            pools={identity: dict(snapshot=pool.snapshot()) for identity, pool in pools.items()},
            plan=make_plan(pools),
        )
        progress = Progress(args.out, request, args.resume)
        execute(progress, lambda variant: CudaRunner(cfg, checkpoint, variant, panel, batch, pools, master))
        print(args.out)


if __name__ == "__main__":
    main()
