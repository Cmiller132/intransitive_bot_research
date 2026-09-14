"""Frozen-checkpoint RGSC probe with independent continuations and recorded candidate alternatives.

This diagnostic reuses the game environment and search without training or a
restart buffer. CPU tests supply a tiny evaluator; real checkpoint inference
requires an explicit device. Results describe prediction error, not strength.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import re
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from compression import zstd

from .control import State
from .diagnose_control import interval
from .planes import N_SQUARES

PROTOCOL_VERSION = 4
CANDIDATE_ENCODING = "json-zstd-base64-v1"
ESTIMANDS = {
    "state_policy": "Squared mean residual over the state's stochastic search and continuation policy.",
    "root_context": "Mean squared conditional residual given a root context; includes root-search variability.",
    "action_mode": "Mean squared conditional residual given the on-policy action and full/cheap root-search mode.",
}


def seed_for(master: int, *parts) -> int:
    """Separate reproducible streams for discovery, selection, anchors, and confirmations."""
    data = json.dumps([master, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") & ((1 << 63) - 1)


def state_record(state: State) -> dict:
    return dict(board=state.board.tolist(), since_capture=state.since_capture, ply=state.ply, clock=state.clock)


def state_from(record: dict) -> State:
    return State(
        np.array(record["board"], dtype=np.int8), int(record["since_capture"]), int(record["ply"]), int(record["clock"])
    )


def state_key(record: dict) -> str:
    return state_from(record).key().hex()


class CandidatePool:
    """Streaming selections over eligible node occurrences, using RNGs independent of play."""

    def __init__(self, master: int, game_id: str):
        self.rank_rng = np.random.default_rng(seed_for(master, game_id, "rank_selection"))
        self.uniform_rng = np.random.default_rng(seed_for(master, game_id, "uniform_selection"))
        self.count = 0
        self.state_ids: dict[str, int] = {}
        self.states: list[dict] = []
        self.occurrences: list[list] = []
        self.batches: list[int] = []
        self._payload = None
        self.log_mass = -math.inf
        self.best: dict[str, tuple[float, dict]] = {}
        self.sources = {"played": 0, "tree": 0}

    def offer(self, candidates: list[dict]) -> None:
        if not candidates:
            return
        scores = np.array([c["rank"] for c in candidates], dtype=np.float64)
        predicted = np.array([c["regret"] for c in candidates], dtype=np.float64)
        if not np.isfinite(scores).all() or not np.isfinite(predicted).all():
            raise ValueError("nonfinite candidate prediction")
        self.log_mass = float(np.logaddexp(self.log_mass, np.logaddexp.reduce(scores)))
        ranking = scores + self.rank_rng.gumbel(size=len(candidates))
        uniform = self.uniform_rng.random(len(candidates))
        for method, criterion in (
            ("rank_sample", ranking),
            ("rank_argmax", scores),
            ("regret_argmax", predicted),
            ("uniform", uniform),
        ):
            i = int(criterion.argmax())
            value = float(criterion[i])
            if method not in self.best or value > self.best[method][0]:
                self.best[method] = value, {**candidates[i], "occurrence_id": self.count + i}
        self.count += len(candidates)
        self.batches.append(len(candidates))
        self._payload = None
        for c in candidates:
            self.sources[c["source"]] += 1
            key = state_key(c["state"])
            if key not in self.state_ids:
                self.state_ids[key] = len(self.states)
                self.states.append({**c["state"], "board": c["state"]["board"].copy()})
            self.occurrences.append([self.state_ids[key], c["source"], *c["occurrence"], c["rank"], c["regret"]])

    def selections(self) -> dict:
        if not self.count:
            raise ValueError("empty candidate set")
        result = {}
        for method, (_, candidate) in self.best.items():
            probability = (
                math.exp(candidate["rank"] - self.log_mass)
                if method == "rank_sample"
                else 1 / self.count
                if method == "uniform"
                else 1.0
            )
            result[method] = {**candidate, "occurrence_probability": probability}
        return {
            "occurrences": self.count,
            "distinct_states": len(self.state_ids),
            "sources": self.sources.copy(),
            "selected": result,
            "baseline": "uniform eligible occurrences; repeated states retain multiplicity",
        }

    def snapshot(self) -> dict:
        """Selections and a compressed pool cached at discovery completion."""
        if self._payload is None:
            raw = json.dumps(
                dict(states=self.states, occurrences=self.occurrences, batches=self.batches),
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            self._payload = dict(
                encoding=CANDIDATE_ENCODING,
                raw_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b64encode(zstd.compress(raw)).decode("ascii"),
            )
        return {**self.selections(), "payload": self._payload}


def candidate_batches(snapshot: dict):
    """Decode ordered occurrence batches, preserving repeated states and reduction boundaries."""
    try:
        payload = snapshot["payload"]
        if set(payload) != {"encoding", "raw_bytes", "sha256", "data"}:
            raise ValueError("unexpected payload fields")
        if payload["encoding"] != CANDIDATE_ENCODING or type(payload["raw_bytes"]) is not int:
            raise ValueError("unsupported encoding or size")
        compressed = base64.b64decode(payload["data"], validate=True)
        if (
            payload["raw_bytes"] <= 0
            or zstd.get_frame_info(compressed).decompressed_size != payload["raw_bytes"]
            or zstd.get_frame_size(compressed) != len(compressed)
        ):
            raise ValueError("compressed size mismatch")
        raw = zstd.decompress(compressed)
        if len(raw) != payload["raw_bytes"] or hashlib.sha256(raw).hexdigest() != payload["sha256"]:
            raise ValueError("checksum mismatch")
        data = json.loads(raw)
        if set(data) != {"states", "occurrences", "batches"}:
            raise ValueError("unexpected table fields")
        states, rows, batches = (data[k] for k in ("states", "occurrences", "batches"))
        if not all(isinstance(v, list) and v for v in (states, rows, batches)):
            raise ValueError("empty or invalid tables")
        keys = set()
        for state in states:
            if set(state) != {"board", "since_capture", "ply", "clock"}:
                raise ValueError("invalid state fields")
            board = state["board"]
            if not isinstance(board, list) or len(board) != N_SQUARES:
                raise ValueError("invalid board shape")
            if any(type(v) is not int or not -128 <= v < 128 for v in board):
                raise ValueError("invalid board encoding")
            if any(type(state[k]) is not int or not 0 <= state[k] < 2**31 for k in ("since_capture", "ply", "clock")):
                raise ValueError("invalid state counters")
            keys.add(state_key(state))
        if len(keys) != len(states) or len(states) != snapshot["distinct_states"]:
            raise ValueError("distinct state count mismatch")
        if any(type(n) is not int or n < 1 for n in batches) or sum(batches) != len(rows):
            raise ValueError("invalid batch boundaries")
        sources = {"played": 0, "tree": 0}
        used = set()
        for row in rows:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError("invalid occurrence row")
            state_id, source, step, node, rank, regret = row
            if type(state_id) is not int or not 0 <= state_id < len(states) or source not in sources:
                raise ValueError("invalid state reference or source")
            if any(type(v) is not int or v < 0 for v in (step, node)):
                raise ValueError("invalid occurrence location")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (rank, regret)):
                raise ValueError("invalid prediction")
            sources[source] += 1
            used.add(state_id)
        if len(rows) != snapshot["occurrences"] or sources != snapshot["sources"] or len(used) != len(states):
            raise ValueError("occurrence summary mismatch")
    except (KeyError, TypeError, ValueError, OverflowError, zstd.ZstdError) as error:
        raise ValueError(f"invalid candidate payload: {error}") from error
    offset = 0
    for size in batches:
        yield [
            dict(
                state={**states[i], "board": states[i]["board"].copy()},
                source=source,
                occurrence=[step, node],
                rank=rank,
                regret=regret,
            )
            for i, source, step, node, rank, regret in rows[offset : offset + size]
        ]
        offset += size


def replay_candidate_pool(snapshot: dict, master: int, game_id: str) -> CandidatePool:
    """Reproduce and validate saved selections without search or network inference."""
    pool = CandidatePool(master, game_id)
    for batch in candidate_batches(snapshot):
        pool.offer(batch)
    if pool.selections() != {k: v for k, v in snapshot.items() if k != "payload"}:
        raise ValueError("candidate selections do not match persisted occurrences")
    pool._payload = snapshot["payload"]
    return pool


def selection_manifest(snapshot: dict, master: int, game: str, selection: str, samples: int | None) -> dict:
    """Bind sampled occurrence slots to shared within-game state measurements before labeling."""
    pool = replay_candidate_pool(snapshot, master, game)
    seed = seed_for(master, game, "corpus_selection") if selection == "uniform_corpus" else None
    if selection == "uniform_corpus":
        if type(samples) is not int or not 1 <= samples <= pool.count:
            raise ValueError("uniform corpus requires enough pool occurrences for its requested sample")
        chosen = [(None, int(i)) for i in np.random.default_rng(seed).choice(pool.count, samples, replace=False)]
    elif selection == "selectors" and samples is None:
        chosen = [(method, value["occurrence_id"]) for method, value in pool.selections()["selected"].items()]
    else:
        raise ValueError("invalid selection mode or sample count")
    measurements, slots, multiplicities = {}, [], {}
    for slot, (method, index) in enumerate(chosen):
        key = state_key(pool.states[pool.occurrences[index][0]])
        if key not in measurements:
            measurements[key] = f"{game}/candidate{len(measurements)}"
        identity = measurements[key]
        multiplicities[identity] = multiplicities.get(identity, 0) + 1
        entry = dict(slot=slot, occurrence_id=index, state_key=key, measurement_id=identity)
        if method is not None:
            entry["selector"] = method
        slots.append(entry)
    return dict(
        mode=selection,
        seed=seed,
        source_payload_sha256=snapshot["payload"]["sha256"],
        population_occurrences=pool.count,
        requested_slots=len(chosen),
        inclusion_probability=len(chosen) / pool.count if selection == "uniform_corpus" else None,
        slot_weight=1 / len(chosen) if selection == "uniform_corpus" else None,
        replacement=False if selection == "uniform_corpus" else None,
        slots=slots,
        multiplicities=multiplicities,
    )


def trajectory_excess(episode: dict) -> float | None:
    """Validate recorded played roots and average their terminal-return calibration excess."""
    trajectory = episode.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != episode["plies"] or not trajectory:
        raise ValueError("recorded trajectory length does not match its episode")
    context = episode["context"]
    for index, row in enumerate(trajectory):
        if set(row) != {"state", "action", "played_q", "full"}:
            raise ValueError("recorded trajectory lacks root telemetry")
        if (
            type(row["action"]) is not int
            or not 0 <= row["action"] < 648
            or type(row["full"]) is not bool
            or type(row["played_q"]) not in (int, float)
            or not math.isfinite(row["played_q"])
            or not -1 <= row["played_q"] <= 1
            or row["state"]["ply"] != episode["initial"]["ply"] + index
            or row["state"]["clock"] != episode["initial"]["clock"]
        ):
            raise ValueError("invalid recorded root telemetry")
        if index == 0 and (
            row["state"] != episode["initial"]
            or row["action"] != episode["first_action"]
            or row["played_q"] != episode["first_q"]
            or row["full"] != context["full"]
        ):
            raise ValueError("recorded first root disagrees with its episode context")
    outcome = episode["outcome"]
    if episode["censored"]:
        if outcome is not None:
            raise ValueError("censored trajectory cannot carry a terminal return")
        return None
    if type(outcome) not in (int, float) or not math.isfinite(outcome) or not -1 <= outcome <= 1:
        raise ValueError("trajectory lacks a bounded numeric terminal return")
    return float(np.mean([
        2 * row["played_q"] * (row["played_q"] - outcome * (-1 if index % 2 else 1))
        for index, row in enumerate(trajectory)
    ]))


class Game:
    """A fresh root search followed by independently seeded continuation search and play."""

    def __init__(self, cfg, evaluate, state: State, seed: int, device: str, context_seed: int | None = None):
        from .env import Env
        from .search import GumbelSearch, RepetitionHistory

        self.cfg, self.evaluate, self.seed = cfg, evaluate, seed
        self.context_seed = seed_for(seed, "root_context") if context_seed is None else context_seed
        self.streams = {k: seed_for(seed, "future", k) for k in ("search", "mode", "environment")}
        self.root_streams = {k: seed_for(self.context_seed, "root", k) for k in ("search", "mode")}
        self.mode_rng = np.random.default_rng(self.streams["mode"])
        self.root = self.context = None
        self.played = False
        self.env = Env(1, device, cfg.rules.max_plies, state.clock, state.clock, self.streams["environment"])
        self.env.board.copy_(torch.tensor(state.board[None], device=device, dtype=torch.int8))
        self.env.since_capture.fill_(state.since_capture)
        self.env.ply.fill_(state.ply)
        self.env.clock.fill_(state.clock)
        self.env.derive()
        self.history = RepetitionHistory(1, cfg.rules.clock_max, device)
        self.history.reset(self.env.board, self.env.ply)
        self.search = GumbelSearch(
            cfg.search,
            1,
            device,
            self.env.clock,
            cfg.rules.clock_penalty,
            cfg.rules.max_plies,
            self.root_streams["search"],
            self.history,
        )

    @torch.no_grad()
    def prepare(self, context_id: str, protocol: str, full: bool | None = None) -> dict:
        """Run one root search, retaining its tree and selected action for continuation."""
        if self.root is not None:
            raise ValueError("root context already prepared")
        env = self.env
        if (
            int(env.legal_count[0]) == 0
            or int(env.ply[0]) >= self.cfg.rules.max_plies
            or int(env.since_capture[0]) >= int(env.clock[0])
        ):
            raise ValueError("probe requires a nonterminal opening below the ply cap")
        full = (
            bool(np.random.default_rng(self.root_streams["mode"]).random() < self.cfg.search.full_fraction)
            if full is None
            else bool(full)
        )
        start = time.perf_counter()
        self.root = self.search(env.board, env.since_capture, env.ply, env.legal, self.evaluate, full)
        initial = self.snapshot()
        self.context = dict(
            context_id=context_id,
            protocol=protocol,
            state_key=state_key(initial),
            initial=initial,
            seed=self.context_seed,
            streams=self.root_streams.copy(),
            full=full,
            action=int(self.root.action[0]),
            q=float(self.root.played_q[0]),
            tree_sha256=self.tree_hash(),
            scheduled_simulations=self.cfg.search.sims if full else self.cfg.search.cheap_sims,
            seconds=time.perf_counter() - start,
        )
        if not np.isfinite(self.context["q"]):
            raise ValueError("nonfinite root prediction")
        return self.context

    def tree_hash(self) -> str:
        """Fingerprint the root tree retained by search.advance and the repetition history."""
        digest = hashlib.sha256()
        count = int(self.search.node_count[0])
        for name in (
            "boards",
            "since",
            "plies",
            "actions",
            "n_legal",
            "prior",
            "net_q",
            "wins",
            "losses",
            "count",
            "total",
            "child",
            "terminal",
            "value",
            "node_draw",
            "hash",
            "rank",
            "regret",
        ):
            tensor = getattr(self.search, name)[:count].contiguous().view(torch.uint8).cpu()
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
        for tensor in (
            self.search.node_count,
            self.search.played_slot,
            self.history.hashes,
            self.history.size,
            self.history.next,
        ):
            digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return digest.hexdigest()

    def snapshot(self) -> dict:
        env = self.env
        return state_record(
            State(env.board[0].cpu().numpy().copy(), int(env.since_capture[0]), int(env.ply[0]), int(env.clock[0]))
        )

    def candidates(self, root, step: int) -> list[dict]:
        search = self.search
        records = [
            dict(
                state=self.snapshot(),
                source="played",
                occurrence=[step, 0],
                rank=float(root.rank[0]),
                regret=float(search.regret[0, 0]),
            )
        ]
        count = int(search.node_count[0])
        for i in range(1, count):
            if not bool(search.terminal[i, 0]):
                state = State(
                    search.boards[i, 0].cpu().numpy().copy(),
                    int(search.since[i, 0]),
                    int(search.plies[i, 0]),
                    int(self.env.clock[0]),
                )
                records.append(
                    dict(
                        state=state_record(state),
                        source="tree",
                        occurrence=[step, i],
                        rank=float(search.rank[i, 0]),
                        regret=float(search.regret[i, 0]),
                    )
                )
        return records

    @torch.no_grad()
    def play(
        self,
        episode_id: str,
        protocol: str,
        pool: CandidatePool | None = None,
        record: bool = False,
        expected_context: dict | None = None,
    ) -> dict:
        from .env import END_CLOCK, END_MAX_PLIES, END_WIN, material_lead

        env, search = self.env, self.search
        if self.played:
            raise ValueError("episode already played")
        self.played = True
        start = time.perf_counter()
        if self.root is None:
            self.prepare(
                expected_context["context_id"] if expected_context else f"{episode_id}/root",
                protocol,
                expected_context["full"] if expected_context else None,
            )
        if expected_context is not None:
            for field in (
                "context_id",
                "protocol",
                "state_key",
                "seed",
                "streams",
                "full",
                "action",
                "q",
                "tree_sha256",
            ):
                if self.context[field] != expected_context[field]:
                    raise ValueError(f"root context did not reproduce {field}")
        initial = self.context["initial"]
        first_q, first_action = self.context["q"], self.context["action"]
        # The conditioned root tree is retained; subsequent random choices are independent.
        search.rng.manual_seed(self.streams["search"])
        trajectory = []
        simulations = self.context["scheduled_simulations"]
        limit = self.cfg.rules.max_plies - int(env.ply[0])
        for step in range(limit):
            if step == 0:
                root = self.root
                full = self.context["full"]
            else:
                full = bool(self.mode_rng.random() < self.cfg.search.full_fraction)
                root = search(env.board, env.since_capture, env.ply, env.legal, self.evaluate, full)
                simulations += self.cfg.search.sims if full else self.cfg.search.cheap_sims
            if record:
                trajectory.append(dict(
                    state=self.snapshot(), action=int(root.action[0]), played_q=float(root.played_q[0]), full=full
                ))
            if pool is not None:
                pool.offer(self.candidates(root, step))
            # The terminal clock label uses the last mover's pre-move material lead.
            clock_return = -self.cfg.rules.clock_penalty * float(material_lead(env.board)[0])
            env.step(root.action.to(torch.int32))
            reason = int(env.end_reason[0])
            if reason:
                terminal = 1.0 if reason == END_WIN else clock_return if reason == END_CLOCK else None
                outcome = None if terminal is None else terminal * (-1 if step % 2 else 1)
                result = dict(
                    episode_id=episode_id,
                    protocol=protocol,
                    seed=self.seed,
                    streams=self.streams.copy(),
                    context=self.context.copy(),
                    state_key=state_key(initial),
                    initial=initial,
                    first_q=first_q,
                    first_action=first_action,
                    outcome=outcome,
                    residual=None if outcome is None else first_q - outcome,
                    plies=step + 1,
                    end_reason=reason,
                    censored=reason == END_MAX_PLIES,
                    scheduled_simulations=simulations,
                    seconds=time.perf_counter() - start,
                    trajectory=trajectory if record else None,
                )
                if record:
                    result["current_excess"] = trajectory_excess(result)
                return result
            self.history.push(env.board, env.since_capture, env.ply, env.done)
            search.advance(env.done)
        raise RuntimeError("environment did not stop at its ply cap")


def paired_observation(first: dict, second: dict, estimand: str, forbidden: set[str] | None = None) -> float:
    """Validate shared conditioning and independent futures, then return a signed residual product."""
    if estimand not in ESTIMANDS:
        raise ValueError("unknown paired estimand")
    for field in ("state_key", "protocol"):
        if first[field] != second[field]:
            raise ValueError(f"paired continuations disagree on {field}")
    if first["episode_id"] == second["episode_id"] or first["seed"] == second["seed"]:
        raise ValueError("paired continuations must have different episodes and RNG streams")
    if set(first["streams"].values()) & set(second["streams"].values()):
        raise ValueError("paired continuations share a random stream")
    for episode in (first, second):
        if episode["episode_id"] in (forbidden or set()):
            raise ValueError("confirmation reuses a discovery or base episode")
        if episode["censored"] or episode["residual"] is None or not np.isfinite(episode["residual"]):
            raise ValueError("censored or nonfinite outcome cannot label a pair")
        if set(episode["streams"]) != {"search", "mode", "environment"} or len(set(episode["streams"].values())) != 3:
            raise ValueError("search, mode and environment streams must be separate")
        context = episode["context"]
        if set(context["streams"]) != {"search", "mode"} or len(set(context["streams"].values())) != 2:
            raise ValueError("root search and mode streams must be separate")
        if context["state_key"] != episode["state_key"] or context["protocol"] != episode["protocol"]:
            raise ValueError("episode disagrees with its root context")
        if context["action"] != episode["first_action"] or context["q"] != episode["first_q"]:
            raise ValueError("episode changed the root action or prediction")
        if not np.isfinite(episode["outcome"]) or not np.isclose(
            episode["residual"], episode["first_q"] - episode["outcome"], rtol=0, atol=1e-12
        ):
            raise ValueError("residual does not match prediction and terminal outcome")
    a, b = first["context"], second["context"]
    future_streams = set(first["streams"].values()) | set(second["streams"].values())
    if future_streams & (set(a["streams"].values()) | set(b["streams"].values())):
        raise ValueError("root and future random streams overlap")
    if estimand == "root_context":
        if any(a[k] != b[k] for k in ("context_id", "seed", "streams", "full", "action", "q", "tree_sha256")):
            raise ValueError("fine-context pairs require the identical root search")
    else:
        if (
            a["seed"] == b["seed"]
            or a["context_id"] == b["context_id"]
            or set(a["streams"].values()) & set(b["streams"].values())
        ):
            raise ValueError("this estimand requires independent root searches")
        if estimand == "action_mode" and (a["action"] != b["action"] or a["full"] != b["full"]):
            raise ValueError("coarse-context pairs must match action and search mode")
    return float(first["residual"] * second["residual"])


def confirmation_completion_range(confirmation: dict, identity: str, pairs: int) -> np.ndarray:
    """Complete explicitly missing signed pair observations within [-4, 4]."""
    if confirmation.get("measurement_id") != identity or len(confirmation.get("pairs", [])) != pairs:
        raise ValueError("incomplete confirmation or incorrect measurement identity")
    total, missing = 0.0, 0
    for index, pair in enumerate(confirmation["pairs"]):
        status = pair.get("status")
        if status not in ("complete", "attempt_cap", "outcome_censored") or "product" not in pair:
            raise ValueError("pair must have an explicit completion or censoring status and product")
        expected = [] if status == "attempt_cap" else [f"{identity}/pair{index}/{side}" for side in (0, 1)]
        if pair.get("episodes") != expected:
            raise ValueError("pair episode identities do not match the planned confirmation")
        product = pair["product"]
        if status == "complete":
            if type(product) not in (int, float) or not math.isfinite(product) or not -4 <= product <= 4:
                raise ValueError("completed pair product must be finite and within [-4, 4]")
            total += product
        else:
            if product is not None:
                raise ValueError("censored pair cannot contain a measured product")
            missing += 1
    mean = confirmation.get("mean", "absent")
    if (missing and mean is not None) or (
        not missing and (type(mean) not in (int, float) or not math.isclose(mean, total / pairs, abs_tol=1e-12))
    ):
        raise ValueError("confirmation mean does not match its signed products")
    return np.array([total - 4 * missing, total + 4 * missing]) / pairs


def complete_mean(values: list) -> float | None:
    """Average a fixed nonempty grid only when every requested value is observed."""
    return float(np.mean(values)) if values and all(value is not None for value in values) else None


def check_mean(record: dict, field: str, expected: float | None) -> None:
    actual = record.get(field, "absent")
    if (expected is None and actual is not None) or (
        expected is not None
        and (type(actual) not in (int, float) or not math.isclose(actual, expected, abs_tol=1e-12))
    ):
        raise ValueError(f"{field} does not match its planned measurements")


def measurement_completion_ranges(
    value: dict, identity: str, pairs: int, anchors: int, bases: int, endpoint: str
) -> dict:
    """Validate one selected state's nested grid and bound its requested endpoints."""
    if value.get("measurement_id") != identity or endpoint not in ("future", "both"):
        raise ValueError("invalid measurement identity or endpoint")
    result = {}
    if endpoint == "both":
        result["opening"] = confirmation_completion_range(value["opening"], f"{identity}/opening", pairs).tolist()
    elif "opening" in value:
        raise ValueError("future-only measurement contains an opening confirmation")
    rows = value.get("bases")
    if not isinstance(rows, list) or len(rows) != bases:
        raise ValueError("base count does not match the requested grid")
    ranges, means, excesses = [], [], []
    for b, base in enumerate(rows):
        base_id = f"{identity}/base{b}"
        if base.get("base_episode") != base_id or type(base.get("base_censored")) is not bool:
            raise ValueError("incomplete base status or incorrect base episode identity")
        future = base.get("future_anchors")
        expected = 0 if base["base_censored"] else anchors
        if not isinstance(future, list) or len(future) != expected:
            raise ValueError("future anchor count does not match the request and base status")
        future_ranges, anchor_means = [], []
        for a, anchor in enumerate(future):
            if anchor.get("base_episode") != base_id:
                raise ValueError("anchor refers to a different base episode")
            future_ranges.append(confirmation_completion_range(anchor["confirmation"], f"{base_id}/anchor{a}", pairs))
            anchor_means.append(anchor["confirmation"]["mean"])
        mean = complete_mean(anchor_means)
        check_mean(base, "future_mean", mean)
        means.append(mean)
        ranges.append(np.mean(future_ranges, axis=0) if future_ranges else np.array([-4.0, 4.0]))
        excess = base.get("current_excess", "absent")
        if (base["base_censored"] and excess is not None) or (
            not base["base_censored"] and (type(excess) not in (int, float) or not math.isfinite(excess))
        ):
            raise ValueError("base lacks its explicit terminal-return excess status")
        excesses.append(excess)
    check_mean(value, "future_mean", complete_mean(means))
    check_mean(value, "current_excess_mean", complete_mean(excesses))
    result["future"] = np.mean(ranges, axis=0).tolist()
    return result


def finite_sample_completion_range(
    records: list[dict], games: int, pairs: int, anchors: int, bases: int = 1, endpoint: str = "both"
) -> dict:
    """Bound planned selector contrasts while retaining censored games, bases, anchors and pairs."""
    if any(type(n) is not int or n < 1 for n in (games, pairs, anchors, bases)) or len(records) != games:
        raise ValueError("completion ranges require every requested discovery game and positive counts")
    if endpoint not in ("future", "both"):
        raise ValueError("invalid endpoint")
    methods = ("rank_sample", "rank_argmax", "regret_argmax")
    targets = ("opening", "future") if endpoint == "both" else ("future",)
    bounds = {f"{target}/{method}": np.zeros(2) for target in targets for method in methods}
    for index, record in enumerate(records):
        identity = f"discovery{index}"
        if record.get("game") != identity or type(record.get("discovery_censored")) is not bool:
            raise ValueError("incomplete discovery status or incorrect discovery identity")
        measured = record.get("measurements")
        if not isinstance(measured, dict):
            raise ValueError("discovery lacks its measurement records")
        if record["discovery_censored"]:
            if measured:
                raise ValueError("censored discovery cannot have selected-state confirmations")
            for bound in bounds.values():
                bound += [-8, 8]
            continue
        selected = record["candidates"]["selected"]
        if set(selected) != {*methods, "uniform"}:
            raise ValueError("completed discovery lacks its planned selections")
        keys = {method: state_key(candidate["state"]) for method, candidate in selected.items()}
        unique = list(dict.fromkeys(keys.values()))
        if set(measured) != set(unique):
            raise ValueError("selected states do not match recorded measurements")
        candidate_bounds = {
            key: measurement_completion_ranges(measured[key], f"{identity}/candidate{number}", pairs, anchors, bases, endpoint)
            for number, key in enumerate(unique)
        }
        for target in targets:
            reference = candidate_bounds[keys["uniform"]][target]
            for method in methods:
                if keys[method] != keys["uniform"]:
                    chosen = candidate_bounds[keys[method]][target]
                    bounds[f"{target}/{method}"] += [chosen[0] - reference[1], chosen[1] - reference[0]]
    return {comparison: (value / games).tolist() for comparison, value in bounds.items()}


def atomic_json(path: Path, payload: dict, create: bool = False) -> int:
    """Persist a complete progress snapshot; initial creation refuses an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".rgsc_probe_", suffix=".tmp", encoding="utf-8", delete=False
        ) as f:
            name = f.name
            json.dump(payload, f, separators=(",", ":"), allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if create:
            os.link(name, path)
        else:
            os.replace(name, path)
        return path.stat().st_size
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


class Progress:
    """Caches completed episodes and root searches in one atomically replaced artifact."""

    def __init__(self, path: Path | None, request: dict, resume: bool):
        self.path, self.request = path, request
        self.started = time.perf_counter()
        self.elapsed = 0.0
        self.episodes: dict[str, dict] = {}
        self.roots: dict[str, dict] = {}
        self.selection_manifests: dict[str, dict] = {}
        self.persistence = dict(completed_writes=0, bytes_written=0, seconds=0.0, max_seconds=0.0)
        self.completed = None
        if resume:
            if path is None:
                raise ValueError("resume requires a progress path")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("request") != request:
                raise ValueError("progress protocol, settings, or initial states do not match")
            self.episodes = {e["episode_id"]: e for e in payload["episodes"]}
            self.roots = {c["context_id"]: c for c in payload["root_searches"]}
            self.selection_manifests = payload["selection_manifests"]
            self.persistence = {key: payload["persistence"][key] for key in self.persistence}
            if len(self.episodes) != len(payload["episodes"]) or len(self.roots) != len(payload["root_searches"]):
                raise ValueError("duplicate identities in progress artifact")
            for identity, episode in self.episodes.items():
                if re.fullmatch(r"discovery\d+", identity):
                    if "candidate_pool" not in episode:
                        raise ValueError("cached discovery lacks candidate payload")
                    replay_candidate_pool(episode["candidate_pool"], request["seed"], identity)
            self.elapsed = float(payload["elapsed_seconds"])
            if payload["status"] == "complete":
                for game in payload["games"]:
                    candidates = self.episodes[game["game"]]["candidate_pool"]
                    if game["candidates"] != {k: v for k, v in candidates.items() if k != "payload"}:
                        raise ValueError("candidate report does not match saved discovery")
                validate_completed_payload(payload)
                self.completed = payload
            elif payload["status"] != "running":
                raise ValueError("unknown progress status")
        elif path is not None:
            self.save(create=True)

    def save(self, report: dict | None = None, create: bool = False) -> dict:
        payload = {
            **(report or {}),
            "status": "complete" if report is not None else "running",
            "request": self.request,
            "elapsed_seconds": self.elapsed + time.perf_counter() - self.started,
            "episodes": list(self.episodes.values()),
            "root_searches": list(self.roots.values()),
            "selection_manifests": self.selection_manifests.copy(),
            "persistence": {
                **self.persistence,
                "scope": "Completed writes before this snapshot; excludes this snapshot's own write and in-flight work.",
            },
        }
        if self.path is not None:
            start = time.perf_counter()
            size = atomic_json(self.path, payload, create=create)
            elapsed = time.perf_counter() - start
            self.persistence["completed_writes"] += 1
            self.persistence["bytes_written"] += size
            self.persistence["seconds"] += elapsed
            self.persistence["max_seconds"] = max(self.persistence["max_seconds"], elapsed)
        return payload


def validate_completed_payload(payload: dict) -> None:
    """Reproduce completed selections, trajectory labels and pair products without inference."""
    request = payload["request"]
    episodes = {e["episode_id"]: e for e in payload["episodes"]}
    if len(payload["games"]) != request["games"]:
        raise ValueError("completed report lacks requested discovery games")
    expected_manifests = {}
    for index, game in enumerate(payload["games"]):
        identity = f"discovery{index}"
        if game["game"] != identity or game["split"] != request["split_manifest"][identity]:
            raise ValueError("completed discovery or split identity mismatch")
        discovery = episodes[identity]
        if game["discovery_censored"] != discovery["censored"]:
            raise ValueError("completed discovery censoring mismatch")
        if discovery["censored"]:
            if game["selection_manifest"] is not None or game["measurements"]:
                raise ValueError("censored discovery cannot have sampled labels")
            continue
        manifest = selection_manifest(
            discovery["candidate_pool"], request["seed"], identity, request["selection"], request["samples_per_game"]
        )
        expected_manifests[identity] = manifest
        if game["selection_manifest"] != manifest:
            raise ValueError("completed selection manifest differs from the frozen full pool")
        selected = {slot["state_key"]: slot["measurement_id"] for slot in manifest["slots"]}
        if set(game["measurements"]) != set(selected):
            raise ValueError("completed sampled states differ from their measurements")
        for key, measurement_id in selected.items():
            value = game["measurements"][key]
            ranges = measurement_completion_ranges(
                value, measurement_id, request["pairs"], request["anchors"], request["bases"], request["endpoint"]
            )
            if value["finite_sample_completion_ranges"] != ranges:
                raise ValueError("completed measurement ranges differ from their observations")
            confirmations = [(value["opening"], key)] if request["endpoint"] == "both" else []
            forbidden = {identity, *(base["base_episode"] for base in value["bases"])}
            for base in value["bases"]:
                episode = episodes[base["base_episode"]]
                if episode["state_key"] != key or base["base_censored"] != episode["censored"]:
                    raise ValueError("completed base state or censoring mismatch")
                excess = trajectory_excess(episode)
                check_mean(episode, "current_excess", excess)
                check_mean(base, "current_excess", excess)
                for a, anchor in enumerate(base["future_anchors"]):
                    seed = seed_for(request["seed"], base["base_episode"], "anchor", a)
                    length = len(episode["trajectory"])
                    index = int(np.random.default_rng(seed).integers(length))
                    if (
                        anchor["seed"] != seed or anchor["index"] != index
                        or anchor["state_key"] != state_key(episode["trajectory"][index]["state"])
                        or anchor["inclusion_probability"] != 1 / length
                    ):
                        raise ValueError("completed anchor differs from its independent uniform draw")
                    confirmations.append((anchor["confirmation"], anchor["state_key"]))
            for confirmation, expected_state in confirmations:
                for pair in confirmation["pairs"]:
                    if pair["status"] == "attempt_cap":
                        continue
                    first, second = [episodes[e] for e in pair["episodes"]]
                    if first["state_key"] != expected_state or second["state_key"] != expected_state:
                        raise ValueError("completed confirmation uses a different state")
                    censored = first["censored"] or second["censored"]
                    if (pair["status"] == "outcome_censored") != censored:
                        raise ValueError("completed pair censoring disagrees with its outcomes")
                    if not censored:
                        product = paired_observation(first, second, request["estimand"], forbidden)
                        if not math.isclose(product, pair["product"], abs_tol=1e-12):
                            raise ValueError("completed pair product differs from its independent episodes")
    if payload["selection_manifests"] != expected_manifests:
        raise ValueError("persisted selection manifests differ from completed full pools")


def validate_return_support(clock_penalty: float) -> None:
    """Require bounded clock returns for the probe's signed-product support."""
    if not math.isfinite(clock_penalty) or abs(clock_penalty) > 1:
        raise ValueError("probe requires a finite clock_penalty within [-1, 1]")


def family_splits(games: int, selection: str, samples: int | None, counts: tuple[int, int, int] | None) -> dict:
    """Assign independent discovery families to fixed data roles before collecting outcomes."""
    if selection == "selectors":
        if samples is not None or counts is not None:
            raise ValueError("selector diagnostics do not accept corpus sample or split counts")
        return {f"discovery{i}": "development" for i in range(games)}
    if selection != "uniform_corpus" or type(samples) is not int or samples < 1:
        raise ValueError("uniform corpus requires a positive sample count")
    if counts is None or len(counts) != 3 or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != games:
        raise ValueError("corpus split counts must be nonnegative and sum to the requested games")
    roles = [role for role, count in zip(("train", "validation", "locked_test"), counts, strict=True) for _ in range(count)]
    return {f"discovery{i}": role for i, role in enumerate(roles)}


def probe(
    cfg,
    evaluate,
    protocol: str,
    master: int,
    games: int,
    pairs: int,
    anchors: int,
    device: str,
    initial_states: list[State] | None = None,
    *,
    estimand: str,
    bases: int = 1,
    endpoint: str = "both",
    selection: str = "selectors",
    samples_per_game: int | None = None,
    split_counts: tuple[int, int, int] | None = None,
    max_root_attempts: int = 128,
    progress_path: Path | None = None,
    resume: bool = False,
    definition: dict | None = None,
) -> dict:
    """Measure explicit paired estimands at openings and independently sampled future anchors."""
    from .planes import initial_board

    validate_return_support(cfg.rules.clock_penalty)
    if any(type(n) is not int or n < 1 for n in (games, pairs, anchors, bases, max_root_attempts)):
        raise ValueError("games, pairs, anchors, bases and root-attempt limit must be positive")
    if estimand not in ESTIMANDS:
        raise ValueError("choose an explicit supported estimand")
    if initial_states is not None and len(initial_states) != games:
        raise ValueError("one initial state per discovery game is required")
    if endpoint not in ("future", "both"):
        raise ValueError("endpoint must be future or both")
    splits = family_splits(games, selection, samples_per_game, split_counts)
    request = dict(
        version=PROTOCOL_VERSION,
        protocol=protocol,
        definition=definition,
        estimand=estimand,
        seed=master,
        games=games,
        pairs=pairs,
        anchors=anchors,
        bases=bases,
        endpoint=endpoint,
        selection=selection,
        samples_per_game=samples_per_game,
        split_manifest=splits,
        device=device,
        max_root_attempts=max_root_attempts,
        search=asdict(cfg.search),
        rules=asdict(cfg.rules),
        initial_states=None if initial_states is None else [state_record(s) for s in initial_states],
    )
    progress = Progress(progress_path, request, resume)
    if progress.completed is not None:
        return progress.completed
    records, confirmations = [], []

    def run(state, identity, pool=None, record=False, context=None):
        if identity in progress.episodes:
            cached = progress.episodes[identity]
            expected = dict(
                episode_id=identity,
                protocol=protocol,
                seed=seed_for(master, identity),
                state_key=state.key().hex(),
                initial=state_record(state),
            )
            if any(cached.get(k) != v for k, v in expected.items()):
                raise ValueError("cached episode does not match requested continuation")
            if cached["streams"] != {
                k: seed_for(cached["seed"], "future", k) for k in ("search", "mode", "environment")
            }:
                raise ValueError("cached episode has incompatible future streams")
            root_seed = seed_for(cached["seed"], "root_context") if context is None else context["seed"]
            expected_root = dict(
                protocol=protocol,
                state_key=state.key().hex(),
                seed=root_seed,
                streams={k: seed_for(root_seed, "root", k) for k in ("search", "mode")},
            )
            if context is not None:
                expected_root.update({k: context[k] for k in ("context_id", "full", "action", "q", "tree_sha256")})
            if any(cached["context"].get(k) != v for k, v in expected_root.items()):
                raise ValueError("cached episode does not match requested root context")
            if (record and not cached["trajectory"]) or (pool is not None and "candidate_pool" not in cached):
                raise ValueError("cached episode lacks requested trajectory or candidates")
            if record:
                check_mean(cached, "current_excess", trajectory_excess(cached))
            return cached
        game = Game(
            cfg, evaluate, state, seed_for(master, identity), device, None if context is None else context["seed"]
        )
        result = game.play(identity, protocol, pool, record, expected_context=context)
        if record:
            result["current_excess"] = trajectory_excess(result)
        if pool is not None:
            result["candidate_pool"] = pool.snapshot()
        progress.episodes[identity] = result
        progress.save()
        return result

    def root_context(state, identity, full=None):
        if identity in progress.roots:
            cached = progress.roots[identity]
            root_seed = seed_for(master, identity, "context")
            streams = {k: seed_for(root_seed, "root", k) for k in ("search", "mode")}
            expected = dict(
                context_id=identity,
                protocol=protocol,
                state_key=state.key().hex(),
                initial=state_record(state),
                seed=root_seed,
                streams=streams,
                full=bool(np.random.default_rng(streams["mode"]).random() < cfg.search.full_fraction)
                if full is None
                else bool(full),
                mode_sampling="on_policy" if full is None else "conditioned",
            )
            if any(cached.get(k) != v for k, v in expected.items()):
                raise ValueError("cached root does not match requested search")
            return cached
        game = Game(
            cfg, evaluate, state, seed_for(master, identity, "owner"), device, seed_for(master, identity, "context")
        )
        context = game.prepare(identity, protocol, full)
        context["mode_sampling"] = "on_policy" if full is None else "conditioned"
        progress.roots[identity] = context
        progress.save()
        return context

    def confirm(state, identity, forbidden):
        measurements = []
        for p in range(pairs):
            prefix = f"{identity}/pair{p}"
            contexts = [None, None]
            attempts = 0
            if estimand == "root_context":
                shared = root_context(state, f"{prefix}/shared_root")
                contexts = [shared, shared]
            elif estimand == "action_mode":
                outer = root_context(state, f"{prefix}/outer_root")
                contexts[0] = outer
                for attempts in range(1, max_root_attempts + 1):
                    proposal = root_context(state, f"{prefix}/matched_root{attempts}", outer["full"])
                    if proposal["action"] == outer["action"]:
                        contexts[1] = proposal
                        break
                if contexts[1] is None:
                    measurements.append(
                        dict(
                            status="attempt_cap",
                            product=None,
                            episodes=[],
                            contexts=[outer["context_id"]],
                            root_attempts=attempts,
                            action=outer["action"],
                            full=outer["full"],
                        )
                    )
                    continue
            a, b = [run(state, f"{prefix}/{side}", context=contexts[side]) for side in (0, 1)]
            censored = a["censored"] or b["censored"]
            value = None if censored else paired_observation(a, b, estimand, forbidden)
            measurements.append(
                dict(
                    status="outcome_censored" if censored else "complete",
                    product=value,
                    episodes=[a["episode_id"], b["episode_id"]],
                    contexts=[a["context"]["context_id"], b["context"]["context_id"]],
                    root_attempts=attempts,
                    action=a["first_action"],
                    full=a["context"]["full"],
                )
            )
        result = dict(
            measurement_id=identity,
            pairs=measurements,
            mean=None
            if any(p["product"] is None for p in measurements)
            else float(np.mean([p["product"] for p in measurements])),
        )
        confirmations.append(result)
        return result

    for game in range(games):
        identity = f"discovery{game}"
        if initial_states is None:
            rng = np.random.default_rng(seed_for(master, identity, "clock"))
            state = State(
                np.array(initial_board(), dtype=np.int8),
                0,
                0,
                int(rng.integers(cfg.rules.clock_min, cfg.rules.clock_max + 1)),
            )
        else:
            state = initial_states[game]
        discovery = run(state, identity, pool=CandidatePool(master, identity))
        choices = {k: v for k, v in discovery["candidate_pool"].items() if k != "payload"}
        measured = {}
        record = dict(
            game=identity, split=splits[identity], discovery_censored=discovery["censored"],
            planned_slots=samples_per_game if selection == "uniform_corpus" else 4,
            candidates=choices, selection_manifest=None, measurements=measured,
        )
        records.append(record)
        if discovery["censored"]:
            print(f"probe {game + 1}/{games}: discovery censored; no confirmations", flush=True)
            continue
        manifest = selection_manifest(discovery["candidate_pool"], master, identity, selection, samples_per_game)
        if identity in progress.selection_manifests:
            if progress.selection_manifests[identity] != manifest:
                raise ValueError("cached selection manifest differs from the requested full-pool sample")
        else:
            progress.selection_manifests[identity] = manifest
            progress.save()
        record["selection_manifest"] = manifest
        pool = replay_candidate_pool(discovery["candidate_pool"], master, identity)
        states = {state_key(s): s for s in pool.states}
        for slot in manifest["slots"]:
            key = slot["state_key"]
            if key in measured:
                continue
            prefix = slot["measurement_id"]
            opening = state_from(states[key])
            base_ids = [f"{prefix}/base{b}" for b in range(bases)]
            forbidden = {identity, *base_ids}
            value = dict(measurement_id=prefix, bases=[])
            if endpoint == "both":
                value["opening"] = confirm(opening, f"{prefix}/opening", forbidden)
            for base_id in base_ids:
                base = run(opening, base_id, record=True)
                future = []
                if not base["censored"]:
                    for a in range(anchors):
                        anchor_seed = seed_for(master, base_id, "anchor", a)
                        index = int(np.random.default_rng(anchor_seed).integers(len(base["trajectory"])))
                        anchor = state_from(base["trajectory"][index]["state"])
                        confirmation = confirm(anchor, f"{base_id}/anchor{a}", forbidden)
                        future.append(dict(
                            index=index, seed=anchor_seed, base_episode=base_id,
                            inclusion_probability=1 / len(base["trajectory"]), state_key=anchor.key().hex(),
                            confirmation=confirmation,
                        ))
                value["bases"].append(dict(
                    base_episode=base_id, base_censored=base["censored"], current_excess=base["current_excess"],
                    future_anchors=future, future_mean=complete_mean([a["confirmation"]["mean"] for a in future]),
                ))
            value["future_mean"] = complete_mean([base["future_mean"] for base in value["bases"]])
            value["current_excess_mean"] = complete_mean([base["current_excess"] for base in value["bases"]])
            value["finite_sample_completion_ranges"] = measurement_completion_ranges(
                value, prefix, pairs, anchors, bases, endpoint
            )
            measured[key] = value
        print(
            f"probe {game + 1}/{games}: {choices['occurrences']} occurrences, {len(progress.episodes)} episodes",
            flush=True,
        )

    comparisons = {}
    if selection == "selectors":
        completion_ranges = finite_sample_completion_range(records, games, pairs, anchors, bases, endpoint)
        rng = np.random.default_rng(seed_for(master, "bootstrap"))
        for target in (("opening", "future") if endpoint == "both" else ("future",)):
            for method in ("rank_sample", "rank_argmax", "regret_argmax"):
                differences = []
                for record in records:
                    if record["discovery_censored"]:
                        continue
                    values = []
                    for choice in (method, "uniform"):
                        key = state_key(record["candidates"]["selected"][choice]["state"])
                        value = record["measurements"][key]
                        values.append(value["opening"]["mean"] if target == "opening" else value["future_mean"])
                    if None not in values:
                        differences.append(values[0] - values[1])
                d = np.array(differences)
                comparisons[f"{target}/{method}"] = dict(
                    complete_discovery_games=len(d), censored_or_missing_games=games - len(d),
                    mean_lift=float(d.mean()) if len(d) else None,
                    game_bootstrap_ci95=interval(d[rng.integers(len(d), size=(1000, len(d)))].mean(1)) if len(d) >= 2 else None,
                    finite_sample_completion_range=completion_ranges[f"{target}/{method}"],
                )
    pair_records = [p for c in confirmations for p in c["pairs"]]
    complete = sum(p["status"] == "complete" for p in pair_records)
    episodes, roots = list(progress.episodes.values()), list(progress.roots.values())
    coverage = coverage_counts(records, games, samples_per_game if selection == "uniform_corpus" else 4, bases, anchors, pairs, endpoint)
    report = dict(
        schema=PROTOCOL_VERSION,
        protocol=protocol,
        definition=definition,
        seed=master,
        estimand=estimand,
        target_definition=ESTIMANDS[estimand],
        future_definition="Expected mean of the fresh-restart opening quantity at uniformly sampled states on an "
        "independent base trajectory. Each anchor resets search and repetition history before receiving new "
        "root contexts and independent future outcomes; the base trajectory's retained context is not preserved.",
        scope="Batch-one development/learnability measurements under a fixed behavior protocol; not production "
        "batch-1024 equivalence, optimal-play error, learning gain or strength. Locked-test outcomes remain "
        "reserved for the frozen analysis plan; corpus mode emits no selector-effect summaries.",
        independence="Separate future search/mode/environment streams per episode. Fine-root pairs intentionally "
        "share a reproduced root context; coarse action/mode pairs use independent root searches "
        "matched by rejection. No discovery/base outcomes label confirmation pairs.",
        censoring="Ply caps and coarse root-attempt caps are missing measurements. Capped discoveries generate "
        "no confirmations. Complete-case comparisons can be biased; observed matching coverage "
        "does not bound the utility of omitted actions.",
        uncertainty="Intervals resample originating discovery games. Pilot intervals with few games "
        "are not acceptance evidence.",
        finite_sample_completion_range_definition=dict(
            signed_pair_support=[-4, 4],
            averaging="All requested discovery games, bases, anchors and pairs per logical selection slot. "
            "Repeated sampled states retain their slot weight while sharing within-game measurements. "
            "Only explicitly censored work is completed as unknown; a shared recorded selection cancels "
            "only after a completed discovery.",
            scope="Deterministic ranges for completion of the planned finite sample, not confidence intervals. "
            "They omit population, Monte Carlo, selection and repeated-look uncertainty and are not adoption evidence.",
        ),
        resume_scope="Completed episodes and root-search attempts are reused. An interrupted in-flight unit is "
        "rerun; its incomplete work is absent from recorded costs. Use only one process per artifact.",
        coverage=dict(
            **coverage,
            launched_pairs=len(pair_records),
            complete_pairs=complete,
            complete_launched_pair_fraction=complete / len(pair_records) if pair_records else None,
            attempt_capped_pairs=sum(p["status"] == "attempt_cap" for p in pair_records),
            outcome_censored_pairs=sum(p["status"] == "outcome_censored" for p in pair_records),
            matched_root_attempts=sum(p["root_attempts"] for p in pair_records),
        ),
        comparisons=comparisons,
        games=records,
        cost=dict(
            episodes=len(episodes),
            root_searches=len(roots),
            censored_episodes=sum(e["censored"] for e in episodes),
            censored_discoveries=sum(r["discovery_censored"] for r in records),
            plies=sum(e["plies"] for e in episodes),
            scheduled_simulations=sum(e["scheduled_simulations"] for e in episodes)
            + sum(c["scheduled_simulations"] for c in roots),
            search_play_seconds=sum(e["seconds"] for e in episodes) + sum(c["seconds"] for c in roots),
        ),
    )
    return progress.save(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--estimand", choices=tuple(ESTIMANDS), required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="JSON progress and final report; use --resume to continue it"
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse completed work with identical protocol and settings"
    )
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--anchors", type=int, default=1)
    parser.add_argument(
        "--max-root-attempts",
        type=int,
        default=128,
        help="coarse action/mode matching attempts before recording a missing pair",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.games, args.pairs, args.anchors, args.max_root_attempts) < 1:
        parser.error("games, pairs, anchors and max-root-attempts must be positive")
    if not re.fullmatch(r"ckpt_\d{6}\.pt", args.ckpt.name):
        parser.error("use an immutable ckpt_NNNNNN.pt, not latest.pt")
    if args.out.exists() and not args.resume:
        parser.error("output already exists")
    if args.resume and not args.out.is_file():
        parser.error("resume requires an existing progress artifact")
    if args.device == "cpu":
        os.environ["TRITON_INTERPRET"] = "1"
    torch.set_num_threads(1)
    import triton

    from .model import build, config_of
    from .train import student_evaluator

    with args.ckpt.open("rb") as f:
        checkpoint_hash = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)
    cfg = config_of(checkpoint)
    validate_return_support(cfg.rules.clock_penalty)
    definition = dict(
        version=PROTOCOL_VERSION,
        checkpoint_sha256=checkpoint_hash,
        weights="ema",
        device=args.device,
        inference_execution="eager",
        inference_precision="bf16_autocast" if args.device == "cuda" else "fp32",
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            triton=triton.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cuda_device=torch.cuda.get_device_name() if args.device == "cuda" else None,
            cuda_capability=list(torch.cuda.get_device_capability()) if args.device == "cuda" else None,
            cuda_multiprocessors=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            if args.device == "cuda"
            else None,
        ),
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "probe_control.py",
                "search.py",
                "env.py",
                "kernels.py",
                "model.py",
                "planes.py",
                "train.py",
                "config.py",
                "control.py",
            )
        },
        search=asdict(cfg.search),
        draw_kernel_width=cfg.play.draw_kernel_width,
        rules=asdict(cfg.rules),
        history="fresh_restart",
        estimand=args.estimand,
    )
    protocol = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    kwargs = dict(
        estimand=args.estimand,
        max_root_attempts=args.max_root_attempts,
        progress_path=args.out,
        resume=args.resume,
        definition=definition,
    )
    if args.resume and json.loads(args.out.read_text(encoding="utf-8"))["status"] == "complete":
        probe(cfg, None, protocol, args.seed, args.games, args.pairs, args.anchors, args.device, **kwargs)
        print(args.out)
        return
    net = build(cfg, checkpoint, args.device, "ema").eval()
    if getattr(net, "fresh", []):
        raise ValueError("probe requires all checkpoint heads to be present")
    evaluator = student_evaluator(net, cfg.play.draw_kernel_width, cfg.search.contempt_source)
    probe(cfg, evaluator, protocol, args.seed, args.games, args.pairs, args.anchors, args.device, **kwargs)
    print(args.out)


if __name__ == "__main__":
    main()
