"""Bias-excess regret-guided search control (DESIGN.md item 33).

Fresh played rows are labelled by the suffix mean of calibration excess
2Q(Q-z). During readiness the largest measured label supplies each game's
candidate. After two windows in which rank-softmax selection beats uniform,
Gumbel-max samples a candidate from the ranking distribution over the game's
played rows and sampled search-tree nodes. A played candidate carries its
measured excess and an observation; a tree candidate carries predicted excess.

The prioritised buffer uses clamped excess before an opening has two outcomes,
then the clamped persistent squared residual rbar^2 - s_r^2/n. Replays fold
before new candidates enter. Restarted games only observe their opening and
never propose descendants."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import Control as ControlConfig
from .planes import N_SQUARES

REGRET_FLOOR = 1e-6
# Q and Q-domain outcomes lie in [-1, 1], so calibration excess is at most 4.
REGRET_MAX = 4.0
# Observations of an opening before its empirical bias replaces its entry regret.
SCORED_AT = 2


@dataclass
class State:
    """One restart position in its mover's frame with its game's capture clock."""

    board: np.ndarray  # (81,) int8
    since_capture: int
    ply: int
    clock: int

    def key(self) -> bytes:
        tail = np.array([self.since_capture, self.ply, self.clock], dtype=np.int32).view(np.int8)
        return np.concatenate([self.board.astype(np.int8), tail]).tobytes()


@dataclass
class Entry:
    """One opening with provisional excess and running residual statistics."""

    state: State
    regret: float
    mean_q: float = 0.0
    mean_z: float = 0.0
    count: int = 0
    r_mean: float = 0.0
    r_m2: float = 0.0
    tree: bool = False
    predicted: float = 0.0
    drift: list = field(default_factory=list)  # |played Q - mean before| of this iteration's replays

    def priority(self) -> float:
        if self.count < SCORED_AT:
            return max(0.0, self.regret)
        sample_variance = self.r_m2 / (self.count - 1)
        return max(0.0, self.r_mean**2 - sample_variance / self.count)

    def observe(self, q: float, z: float) -> None:
        """One more game from this opening: its played Q and outcome."""
        if self.count > 0:
            self.drift.append(abs(q - self.mean_q))
        residual = q - z
        self.count += 1
        self.mean_q += (q - self.mean_q) / self.count
        self.mean_z += (z - self.mean_z) / self.count
        delta = residual - self.r_mean
        self.r_mean += delta / self.count
        self.r_m2 += delta * (residual - self.r_mean)


def clamp_regret(regret: float) -> float | None:
    """A finite regret within the domain, else None."""
    if not np.isfinite(regret):
        return None
    return float(min(max(regret, 0.0), REGRET_MAX))


class Buffer:
    """The prioritised regret buffer: at most `capacity` openings keyed by
    position, each with its entry and priority; a candidate already present
    is one more observation of it instead of a duplicate."""

    def __init__(self, capacity: int, ema: float):
        self.capacity, self.ema = capacity, ema
        self.entries: dict[bytes, Entry] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, key: bytes) -> bool:
        return key in self.entries

    def priorities(self) -> np.ndarray:
        return np.array([e.priority() for e in self.entries.values()], np.float64)

    def lowest(self) -> bytes:
        return min(self.entries, key=lambda k: self.entries[k].priority())

    def insert(self, state: State, regret: float, observation: tuple[float, float] | None = None) -> bool:
        """Admit a candidate with clamped calibration excess: kept when there is room
        or it beats the lowest priority, which it then replaces. A played row
        brings its game as the first `observation` (played Q, outcome); a
        search node brings none. Returns whether the opening is in the buffer."""
        regret = clamp_regret(regret)
        if regret is None:
            return False
        key = state.key()
        if key in self.entries:
            self.replay(key, [regret], [observation] if observation is not None else [])
            return True
        if len(self.entries) >= self.capacity:
            lowest = self.lowest()
            if regret <= self.entries[lowest].priority():
                return False
            self.drop(lowest)
        entry = Entry(state, regret, tree=observation is None, predicted=regret)
        if observation is not None:
            entry.observe(*observation)
        self.entries[key] = entry
        return True

    def replay(self, key: bytes, regrets: list[float], observations: list[tuple[float, float]]) -> bool:
        """Fold the games replayed from an opening still present: its measured
        excess takes one EMA step per game, folded order-independently as
        (1-a)^m R + (1-(1-a)^m) mean(r) for the m games of a window, and its
        running means take their (played Q, outcome). Returns whether the
        opening was present; an evicted one is not resurrected."""
        entry = self.entries.get(key)
        if entry is None:
            return False
        regrets = [r for r in (clamp_regret(r) for r in regrets) if r is not None]
        if regrets:
            keep = (1.0 - self.ema) ** len(regrets)
            entry.regret = keep * entry.regret + (1.0 - keep) * float(np.mean(regrets))
        for q, z in observations:
            entry.observe(q, z)
        return True

    def drop(self, key: bytes) -> bool:
        """Remove an opening; returns whether it was present."""
        return self.entries.pop(key, None) is not None

    def state(self) -> dict:
        entries = list(self.entries.values())
        return {
            "board": np.array([e.state.board for e in entries], np.int8).reshape(-1, N_SQUARES),
            "since_capture": np.array([e.state.since_capture for e in entries], np.int32),
            "ply": np.array([e.state.ply for e in entries], np.int32),
            "clock": np.array([e.state.clock for e in entries], np.int32),
            "regret": np.array([e.regret for e in entries], np.float64),
            "mean_q": np.array([e.mean_q for e in entries], np.float64),
            "mean_z": np.array([e.mean_z for e in entries], np.float64),
            "count": np.array([e.count for e in entries], np.int32),
            "r_mean": np.array([e.r_mean for e in entries], np.float64),
            "r_m2": np.array([e.r_m2 for e in entries], np.float64),
            "tree": np.array([e.tree for e in entries], bool),
            "predicted": np.array([e.predicted for e in entries], np.float64),
        }

    def load(self, state: dict) -> None:
        """Restore a checkpointed buffer, defaulting absent running fields to zero."""
        self.entries.clear()
        n = len(state["regret"])
        zeros = np.zeros(n)
        for i in range(n):
            s = State(
                np.asarray(state["board"][i], np.int8),
                int(state["since_capture"][i]),
                int(state["ply"][i]),
                int(state["clock"][i]),
            )
            self.entries[s.key()] = Entry(
                s,
                float(state["regret"][i]),
                float(state.get("mean_q", zeros)[i]),
                float(state.get("mean_z", zeros)[i]),
                int(state.get("count", np.zeros(n, np.int32))[i]),
                float(state.get("r_mean", zeros)[i]),
                float(state.get("r_m2", zeros)[i]),
                bool(state.get("tree", np.zeros(n, bool))[i]),
                float(state.get("predicted", state["regret"])[i]),
            )
        # A run resumed with a smaller capacity keeps the highest priorities.
        while len(self.entries) > self.capacity:
            self.drop(self.lowest())


def restart_probabilities(priority: np.ndarray, temperature: float, share: float) -> np.ndarray:
    """Draw probabilities proportional to priority ^ (1 / temperature), as a
    softmax of the log so nothing overflows, with no entry above `share` of
    the mass once the buffer holds enough entries for that: the excess of the
    entries at the cap goes to the others in proportion."""
    log_priority = np.log(np.maximum(priority.astype(np.float64), REGRET_FLOOR)) / temperature
    prob = np.exp(log_priority - log_priority.max())
    prob /= prob.sum()
    if share * len(prob) < 1.0:
        return prob
    capped = np.zeros(len(prob), dtype=bool)
    while (over := (prob > share) & ~capped).any():
        capped |= over
        rest = prob[~capped].sum()
        prob[capped] = share
        prob[~capped] *= (1.0 - share * capped.sum()) / rest if rest > 0 else 0.0
    return prob


class SearchControl:
    """Restarts, sampled row/tree candidates, readiness, and the buffer."""

    def __init__(self, cfg: ControlConfig, envs: int, steps: int, device, seed: int, warmup: int):
        self.cfg, self.envs, self.steps = cfg, envs, steps
        self.device = torch.device(device)
        self.buffer = Buffer(cfg.capacity, cfg.ema)
        self.warmup_left = warmup
        self.heads_active = False
        self.rank_lift: list[float] = []
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        # The device copy the environment draws from: filled entries first,
        # the rest at probability zero.
        K = cfg.capacity
        self.dev_board = torch.zeros(K, N_SQUARES, dtype=torch.int8, device=self.device)
        self.dev_since = torch.zeros(K, dtype=torch.int32, device=self.device)
        self.dev_ply = torch.zeros(K, dtype=torch.int32, device=self.device)
        self.dev_clock = torch.zeros(K, dtype=torch.int32, device=self.device)
        self.dev_prob = torch.zeros(K, dtype=torch.float32, device=self.device)
        self.prob = np.zeros(0)  # the host copy of the draw probabilities of the last upload
        # Per board, whether its game started from the buffer and the best
        # Gumbel-perturbed tree sample carried across that game's searches.
        self.restarted = np.zeros(envs, dtype=bool)
        self.tree_score = np.full(envs, -np.inf, dtype=np.float32)
        self.tree_regret = np.zeros(envs, dtype=np.float32)
        self.tree_state = np.zeros((envs, N_SQUARES + 3), dtype=np.int32)
        # Games that ended in the last observed window, keyed by (board, step).
        self.ended: dict[tuple[int, int], tuple[bool, float, float, np.ndarray]] = {}
        # The window's replays by the opening they started from: the opening,
        # the replays' regrets and their (played Q, outcome) observations.
        self.replays: dict[bytes, tuple[State, list[float], list[tuple[float, float]]]] = {}
        # Fresh-game candidates, admitted in finish order after replay folds.
        self.pending: list[tuple[str, State, float, tuple[float, float] | None]] = []
        self.replayed = (0, 0)  # distinct openings replayed in the last window, and the most replays of one
        self.finished = self.restarts = self.dropped = self.lost = 0
        self.candidates = {"row": 0, "tree": 0}
        self.admitted = {"row": 0, "tree": 0}
        self.candidate_plies: list[int] = []
        self.lift_ranks: list[float] = []
        self.lift_labels: list[float] = []
        self.tree_errors: list[float] = []
        self.replay_priorities: list[float] = []  # priorities of the replayed openings after the fold

    # ------------------------------------------------------------------
    def active(self) -> bool:
        return self.warmup_left <= 0 and len(self.buffer) > 0

    def upload(self) -> None:
        """Fold the window's replays into their openings, admit fresh-game
        candidates in finish order, then copy the buffer to the device for the
        coming window's draws."""
        for key, (_, regrets, observations) in self.replays.items():
            entry = self.buffer.entries.get(key)
            if entry is not None and entry.tree and entry.count == 0 and regrets:
                self.tree_errors.append(entry.predicted - float(np.mean(regrets)))
            if self.buffer.replay(key, regrets, observations):
                self.replay_priorities.append(self.buffer.entries[key].priority())
            else:
                self.lost += 1
        self.replayed = (len(self.replays), max((len(r) for _, r, _ in self.replays.values()), default=0))
        self.replays.clear()
        for source, state, regret, observation in self.pending:
            self.candidates[source] += 1
            self.candidate_plies.append(state.ply)
            self.admitted[source] += self.buffer.insert(state, regret, observation)
        self.pending.clear()
        state = self.buffer.state()
        n = len(state["regret"])
        self.dev_prob.zero_()
        self.prob = np.zeros(0)
        if n == 0:
            return
        self.dev_board[:n].copy_(torch.from_numpy(state["board"]))
        self.dev_since[:n].copy_(torch.from_numpy(state["since_capture"]))
        self.dev_ply[:n].copy_(torch.from_numpy(state["ply"]))
        self.dev_clock[:n].copy_(torch.from_numpy(state["clock"]))
        self.prob = restart_probabilities(self.buffer.priorities(), self.cfg.temperature, self.cfg.share)
        self.dev_prob[:n].copy_(torch.from_numpy(self.prob.astype(np.float32)))

    @torch.no_grad()
    def draw(self, env) -> None:
        """Choose where each board starts its next game, should its game end
        at the coming step: a buffer opening with probability `restart` while
        the control is active, else the initial position."""
        if not self.active():
            env.restart_initial(torch.ones_like(env.done))
            return
        u = torch.rand(self.envs, generator=self.rng, device=self.device)
        use = u < self.cfg.restart
        idx = torch.multinomial(self.dev_prob, self.envs, replacement=True, generator=self.rng)
        env.restart_from(
            use,
            self.dev_board[idx],
            self.dev_since[idx],
            self.dev_ply[idx],
            self.dev_clock[idx],
            idx.to(torch.int32),
        )
        env.restart_initial(~use)

    # ------------------------------------------------------------------
    def observe(self, rollout, origin_now: torch.Tensor) -> None:
        """Carry each game's best sampled tree candidate and record its ends."""
        T = self.steps
        done = rollout.done.cpu().numpy()
        origin = rollout.origin.cpu().numpy()
        score = rollout.tree_score.cpu().numpy()
        regret = rollout.tree_regret.cpu().numpy()
        board = rollout.tree_board.cpu().numpy()
        since = rollout.tree_since.cpu().numpy()
        ply = rollout.tree_ply.cpu().numpy()
        clock = rollout.clock.cpu().numpy()
        origin_now = origin_now.cpu().numpy()
        self.ended.clear()
        for t in range(T):
            better = score[t] > self.tree_score
            self.tree_score = np.where(better, score[t], self.tree_score)
            self.tree_regret = np.where(better, regret[t], self.tree_regret)
            rows = np.nonzero(better)[0]
            self.tree_state[rows, :N_SQUARES] = board[t, rows]
            self.tree_state[rows, N_SQUARES] = since[t, rows]
            self.tree_state[rows, N_SQUARES + 1] = ply[t, rows]
            self.tree_state[rows, N_SQUARES + 2] = clock[t, rows]
            for n in np.nonzero(done[t])[0]:
                n = int(n)
                self.ended[(n, t)] = (
                    bool(self.restarted[n]),
                    float(self.tree_score[n]),
                    float(self.tree_regret[n]),
                    self.tree_state[n].copy(),
                )
                self.tree_score[n] = -np.inf
                # Whether the next game came from the buffer, as the environment recorded it after the reset.
                self.restarted[n] = (origin[t + 1, n] if t + 1 < T else origin_now[n]) >= 0

    def finish(self, n: int, t: int, result: int | None, complete: bool, rows: list) -> None:
        """A game of board `n` ended at step `t`: `rows` are its
        `(window, a, b)` slices oldest first (every one present when
        `complete`), carrying the regret, played Q and return labels of its
        rows."""
        restarted, tree_score, tree_regret, tree_state = self.ended.pop((n, t))
        self.finished += 1
        self.restarts += restarted
        if not complete or not rows:
            return
        T, N = self.steps, self.envs

        def column(w, name, a, b):
            f = getattr(w, name)
            return f.view(T, N, *f.shape[1:])[a : b + 1, n].numpy()

        def state_at(w, i):
            return State(
                column(w, "board", i, i)[0].astype(np.int8),
                int(column(w, "since_capture", i, i)[0]),
                int(column(w, "ply", i, i)[0]),
                int(column(w, "clock", i, i)[0]),
            )

        # The game's outcome in the Q domain (the terminal row's return: +-1 for
        # a win, the clock penalty for a clock draw) seen from any row: the
        # sign flips with every ply between the row and the end.
        last_w, _, last_b = rows[-1]
        end_ret = float(column(last_w, "ret", last_b, last_b)[0]) if result is not None else 0.0
        lengths = [b - a + 1 for _, a, b in rows]
        total = sum(lengths)

        def observation(k, i):
            """(played Q, outcome) of row `i` of slice `k`."""
            w, a, _ = rows[k]
            before_end = total - 1 - (sum(lengths[:k]) + i - a)
            return float(column(w, "played_q", i, i)[0]), end_ret * (1.0 if before_end % 2 == 0 else -1.0)

        first_w, first_a, _ = rows[0]
        if restarted:
            # The game replayed the opening its first row holds; it observes
            # that opening at the next upload and proposes nothing itself. A
            # replay cut by the ply cap taught nothing, so the opening leaves.
            first = state_at(first_w, first_a)
            if result is None:
                self.dropped += self.buffer.drop(first.key())
                return
            regret = float(column(first_w, "regret", first_a, first_a)[0])
            _, regrets, observations = self.replays.setdefault(first.key(), (first, [], []))
            regrets.append(regret)
            observations.append(observation(0, first_a))
            return
        if result is None:
            return
        for w, a, b in rows:
            self.lift_ranks.extend(column(w, "rank", a, b).astype(np.float64).tolist())
            self.lift_labels.extend(column(w, "regret", a, b).astype(np.float64).tolist())

        # During readiness use maximum measured excess. Once ready, Gumbel-max
        # samples from the rank distribution over played rows and the carried
        # sampled tree node.
        best, best_score = None, -np.inf
        for k, (w, a, b) in enumerate(rows):
            scores = column(w, "regret" if not self.heads_active else "rank", a, b).astype(np.float64)
            if self.heads_active:
                u = torch.rand(len(scores), generator=self.rng, device=self.device).clamp_(1e-12, 1.0)
                scores = scores + (-(-u.log()).log()).cpu().numpy()
            i = int(scores.argmax())
            if scores[i] > best_score:
                best_score, best = float(scores[i]), (k, a + i)
        if self.heads_active and tree_score > best_score and np.isfinite(tree_score):
            since, ply, clock = (int(v) for v in tree_state[N_SQUARES:])
            state = State(tree_state[:N_SQUARES].astype(np.int8), since, ply, clock)
            self.pending.append(("tree", state, tree_regret, None))
        elif best is not None:
            k, i = best
            w = rows[k][0]
            self.pending.append(("row", state_at(w, i), float(column(w, "regret", i, i)[0]), observation(k, i)))

    def end_iteration(self) -> dict:
        """Per-window metrics, readiness update, and warm-up countdown."""
        if self.lift_labels:
            ranks = np.asarray(self.lift_ranks, np.float64)
            labels = np.asarray(self.lift_labels, np.float64)
            weights = np.exp(ranks - ranks.max())
            lift = float(np.dot(weights, labels) / weights.sum() - labels.mean())
        else:
            lift = 0.0
        self.rank_lift = (self.rank_lift + [lift])[-2:]
        if len(self.rank_lift) == 2 and all(value > 0.0 for value in self.rank_lift):
            self.heads_active = True
        entries = list(self.buffer.entries.values())
        n = len(entries)
        priority = np.array([e.priority() for e in entries]) if n else np.zeros(0)
        regret = np.array([e.regret for e in entries]) if n else np.zeros(0)
        clocks = np.array([e.state.clock for e in entries]) if n else np.zeros(0, np.int32)
        drift = [d for e in entries for d in e.drift]
        for e in entries:
            e.drift.clear()
        top = np.sort(self.prob)[::-1] if len(self.prob) else np.zeros(0)
        scored_priority = [e.priority() for e in entries if e.count >= SCORED_AT]
        unscored_priority = [e.priority() for e in entries if e.count < SCORED_AT]
        metrics = {
            "buffer_size": n,
            "buffer_regret_mean": float(regret.mean()) if n else 0.0,
            "buffer_regret_max": float(regret.max()) if n else 0.0,
            "buffer_priority_mean": float(priority.mean()) if n else 0.0,
            "buffer_priority_max": float(priority.max()) if n else 0.0,
            "buffer_scored_frac": float(np.mean([e.count >= SCORED_AT for e in entries])) if n else 0.0,
            "buffer_tree_frac": float(np.mean([e.tree for e in entries])) if n else 0.0,
            "buffer_board_distinct": len({e.state.board.tobytes() for e in entries}),
            "buffer_priority_scored_mean": float(np.mean(scored_priority)) if scored_priority else 0.0,
            "buffer_priority_unscored_mean": float(np.mean(unscored_priority)) if unscored_priority else 0.0,
            "buffer_ply_mean": float(np.mean([e.state.ply for e in entries])) if n else 0.0,
            "buffer_clock_left": float(np.mean([e.state.clock - e.state.since_capture for e in entries])) if n else 0.0,
            "buffer_clock_max_frac": float(np.bincount(clocks).max() / n) if n else 0.0,
            "buffer_candidates_row": self.candidates["row"],
            "buffer_candidates_tree": self.candidates["tree"],
            "buffer_inserted": self.admitted["row"] + self.admitted["tree"],
            "buffer_inserted_tree": self.admitted["tree"],
            "buffer_dropped": self.dropped,
            "buffer_lost": self.lost,
            "draw_ess": float(1.0 / np.square(self.prob).sum()) if len(self.prob) else 0.0,
            "draw_top16": float(top[:16].sum()) if len(top) else 0.0,
            "tree_regret_error": float(np.mean(self.tree_errors)) if self.tree_errors else 0.0,
            "replay_priority_mean": float(np.mean(self.replay_priorities)) if self.replay_priorities else 0.0,
            "replay_q_drift": float(np.mean(drift)) if drift else 0.0,
            "restart_frac": self.restarts / max(self.finished, 1),
            "restart_distinct": self.replayed[0],
            "restart_max": self.replayed[1],
            "control_warmup": int(self.warmup_left > 0),
            "rank_lift": lift,
            "heads_active": int(self.heads_active),
            "candidate_ply_mean": float(np.mean(self.candidate_plies)) if self.candidate_plies else 0.0,
        }
        self.warmup_left = max(self.warmup_left - 1, 0)
        self.finished = self.restarts = self.dropped = self.lost = 0
        self.candidates = {"row": 0, "tree": 0}
        self.admitted = {"row": 0, "tree": 0}
        self.candidate_plies.clear()
        self.lift_ranks.clear()
        self.lift_labels.clear()
        self.tree_errors.clear()
        self.replay_priorities.clear()
        self.replayed = (0, 0)
        return metrics

    def state(self) -> dict:
        return {
            "buffer": self.buffer.state(),
            "warmup_left": self.warmup_left,
            "heads_active": self.heads_active,
            "rank_lift": self.rank_lift,
        }

    def load(self, state: dict) -> None:
        self.buffer.load(state["buffer"])
        self.warmup_left = int(state.get("warmup_left", 0))
        self.heads_active = bool(state.get("heads_active", False))
        self.rank_lift = [float(value) for value in state.get("rank_lift", [])][-2:]
