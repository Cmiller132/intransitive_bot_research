"""Rollout windows and replay (DESIGN.md items 17 to 25). A window is one
collection of `steps x envs` positions with every label the learner needs,
computed once on the device, then stored immutably on the host and persisted to
disk so a resumed run keeps its recent data."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from compression import zstd

from .env import END_CLOCK, END_WIN, material_lead
from .planes import N_SQUARES, action_targets, flip_actions
from .search import SLOTS

SCHEMA = 2
REGRET_VERSION = 2


@dataclass
class Rollout:
    """One collection on the device: step-major `(T, N, ...)` tensors filled by the actor."""

    board: torch.Tensor  # (T, N, 81) int8, position before the move
    since_capture: torch.Tensor  # (T, N) int16
    ply: torch.Tensor  # (T, N) int16
    clock: torch.Tensor  # (T, N) int16, the game's capture clock
    action: torch.Tensor  # (T, N) int16
    moves: torch.Tensor  # (T, N, SLOTS) int16 legal moves of the root in slot order (search.Root)
    target: torch.Tensor  # (T, N, SLOTS) bf16 policy target over the moves
    target_ok: torch.Tensor  # (T, N) bool: full search
    tactics: torch.Tensor  # (T, N, SLOTS) uint8 packed exact labels of the moves (search.pack_tactics)
    value: torch.Tensor  # (T + 1, N) float32 search root values, last row the bootstrap
    reward: torch.Tensor  # (T, N) float32 for the mover
    done: torch.Tensor  # (T, N) bool
    end_reason: torch.Tensor  # (T, N) uint8
    candidates: torch.Tensor  # (T, N, C) int16
    candidate_q: torch.Tensor  # (T, N, C) float32
    candidate_visited: torch.Tensor  # (T, N, C) bool
    # The search control's (DESIGN.md item 33): the completed Q of the played
    # move, the regret head's ranking score of the root, one tree node sampled
    # from its ranking distribution, and the buffer entry the row's game
    # started from (-1 for the initial position).
    played_q: torch.Tensor  # (T, N) float32
    rank: torch.Tensor  # (T, N) float32
    tree_score: torch.Tensor  # (T, N) float32 rank + Gumbel, -inf when no node is eligible
    tree_regret: torch.Tensor  # (T, N) float32 predicted calibration excess
    tree_board: torch.Tensor  # (T, N, 81) int8
    tree_since: torch.Tensor  # (T, N) int16
    tree_ply: torch.Tensor  # (T, N) int16
    origin: torch.Tensor  # (T, N) int32

    @staticmethod
    def allocate(steps: int, envs: int, candidates: int, device) -> Rollout:
        T, N, C = steps, envs, candidates

        def z(*shape, dtype):
            return torch.zeros(*shape, dtype=dtype, device=device)

        return Rollout(
            board=z(T, N, N_SQUARES, dtype=torch.int8),
            since_capture=z(T, N, dtype=torch.int16),
            ply=z(T, N, dtype=torch.int16),
            clock=z(T, N, dtype=torch.int16),
            action=z(T, N, dtype=torch.int16),
            moves=z(T, N, SLOTS, dtype=torch.int16),
            target=z(T, N, SLOTS, dtype=torch.bfloat16),
            target_ok=z(T, N, dtype=torch.bool),
            tactics=z(T, N, SLOTS, dtype=torch.uint8),
            value=z(T + 1, N, dtype=torch.float32),
            reward=z(T, N, dtype=torch.float32),
            done=z(T, N, dtype=torch.bool),
            end_reason=z(T, N, dtype=torch.uint8),
            candidates=z(T, N, C, dtype=torch.int16),
            candidate_q=z(T, N, C, dtype=torch.float32),
            candidate_visited=z(T, N, C, dtype=torch.bool),
            played_q=z(T, N, dtype=torch.float32),
            rank=z(T, N, dtype=torch.float32),
            tree_score=torch.full((T, N), -torch.inf, dtype=torch.float32, device=device),
            tree_regret=z(T, N, dtype=torch.float32),
            tree_board=z(T, N, N_SQUARES, dtype=torch.int8),
            tree_since=z(T, N, dtype=torch.int16),
            tree_ply=z(T, N, dtype=torch.int16),
            origin=z(T, N, dtype=torch.int32),
        )


@dataclass
class Window:
    """Immutable labels for one collection. Every field is `(steps * envs, ...)` on the host."""

    board: torch.Tensor
    since_capture: torch.Tensor
    ply: torch.Tensor
    clock: torch.Tensor
    action: torch.Tensor
    moves: torch.Tensor
    target: torch.Tensor
    target_ok: torch.Tensor
    tactics: torch.Tensor
    ret: torch.Tensor
    candidates: torch.Tensor
    candidate_q: torch.Tensor
    candidate_visited: torch.Tensor
    occupancy_ok: torch.Tensor  # the board `occupancy_horizon` rows later is the same game
    plies_to_end: torch.Tensor  # plies until a decisive end, -1 when unknown
    material: torch.Tensor  # (S, 2) int8 final own/enemy counts
    material_ok: torch.Tensor  # (S,) bool: the game ended by a win or the capture clock
    outcome: torch.Tensor  # (S,) int8 final result from the mover's view, filled when the game ends (Outcomes)
    outcome_ok: torch.Tensor  # (S,) bool: the outcome is known
    reply: torch.Tensor  # (S,) int16 the opponent's next move in the mover's frame
    reply_ok: torch.Tensor  # (S,) bool: the next row is the same game
    played_q: torch.Tensor  # (S,) float32 completed Q of the played move
    rank: torch.Tensor  # (S,) float32 the regret head's ranking score of the row
    regret: torch.Tensor  # (S,) float32 mean calibration excess 2Q(Q-z_Q) over the rest of the game
    regret_ok: torch.Tensor  # (S,) bool: the regret is known (the game ended decisively or by the clock)
    replayed: torch.Tensor  # (S,) bool: the row's game started from the search control's buffer
    envs: int = 0
    steps: int = 0
    iteration: int = -1

    def __len__(self) -> int:
        return self.board.shape[0]

    def occupancy(self, idx: torch.Tensor, horizon: int) -> torch.Tensor:
        """Boards `horizon` plies later for rows where that is the same game, else the row's own board."""
        later = torch.where(self.occupancy_ok[idx], idx + horizon * self.envs, idx)
        return self.board[later]


def lambda_returns(
    reward: torch.Tensor, done: torch.Tensor, truncated: torch.Tensor, value: torch.Tensor, lam: float
) -> torch.Tensor:
    """Two-player lambda returns from the mover's view at each ply: reward plus
    the negated mixture of the next value and the next return; undiscounted,
    bootstrapped on `value[T]` after the last step. A step truncated by the ply
    cap keeps its own search value (DESIGN item 19)."""
    T = reward.shape[0]
    out = torch.empty_like(reward)
    nxt = -value[T]
    for t in range(T - 1, -1, -1):
        ended = torch.where(truncated[t], value[t], reward[t])
        out[t] = torch.where(done[t], ended, reward[t] + nxt)
        nxt = -((1.0 - lam) * value[t] + lam * out[t])
    return out


def plies_to_end(done: torch.Tensor, terminal: torch.Tensor) -> torch.Tensor:
    """Plies until an end in `terminal`; -1 when the episode ends otherwise or
    runs past the window."""
    T = done.shape[0]
    out = torch.empty(done.shape, dtype=torch.int32, device=done.device)
    nxt = torch.full_like(out[0], -1)
    for t in range(T - 1, -1, -1):
        at_end = torch.where(terminal[t], 1, -1)
        cur = torch.where(done[t], at_end, torch.where(nxt > 0, nxt + 1, nxt))
        out[t] = cur
        nxt = cur
    return out


def same_game_later(done: torch.Tensor, k: int) -> torch.Tensor:
    """(T, N) True where the row k steps later belongs to the same game."""
    T = done.shape[0]
    cumulative = torch.cat([torch.zeros_like(done[:1], dtype=torch.int32), done.int().cumsum(0)])
    ok = torch.zeros_like(done)
    if T > k:
        ok[: T - k] = (cumulative[k:T] - cumulative[: T - k]) == 0
    return ok


def capture_squares(board: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """(T, N) destination of every stored action when it captures, else -1."""
    targets = action_targets().to(board.device)
    to = targets[action.long()]
    hit = board.gather(-1, to.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return torch.where(hit >= 4, to, torch.full_like(to, -1)).to(torch.int16)


def material_after(board: torch.Tensor, capture: torch.Tensor) -> torch.Tensor:
    """(..., 2) own and enemy piece counts after the stored move, in the mover's frame."""
    own = ((board >= 1) & (board <= 3)).sum(-1).to(torch.int8)
    enemy = (board >= 4).sum(-1).to(torch.int8) - (capture >= 0).to(torch.int8)
    return torch.stack([own, enemy], -1)


def final_material(
    done: torch.Tensor, terminal: torch.Tensor, board: torch.Tensor, capture: torch.Tensor
) -> torch.Tensor:
    """(T, N, 2) piece counts at a labelled end of each row's game in that
    row's frame (own and enemy swap every ply back); zeros when unknown."""
    T, N = done.shape
    end = material_after(board, capture)
    out = torch.empty(T, N, 2, dtype=torch.int8, device=done.device)
    nxt = torch.zeros(N, 2, dtype=torch.int8, device=done.device)
    for t in range(T - 1, -1, -1):
        observed = torch.where(terminal[t][:, None], end[t], 0)
        cur = torch.where(done[t][:, None], observed, nxt.flip(-1))
        out[t] = cur
        nxt = cur
    return out


def build_window(rollout: Rollout, cfg, iteration: int) -> Window:
    """All labels from one collection: clock penalties, lambda returns, plies to
    a decisive end, occupancy validity, final material. Returns host tensors."""
    board, action, done, reason = rollout.board, rollout.action, rollout.done, rollout.end_reason
    win = reason == END_WIN
    capture = capture_squares(board, action)
    reward = rollout.reward.clone()
    # The ply cap outranks the clock: a game stopped there keeps its own search value.
    truncated = (rollout.ply.int() + 1 >= cfg.rules.max_plies) & ~win
    clock = (reason == END_CLOCK) & ~truncated
    # The last move before a clock ending never captures, so the mover's lead
    # is read off the board before the move.
    reward = torch.where(clock, -cfg.rules.clock_penalty * material_lead(board), reward)
    ret = lambda_returns(reward, done, truncated, rollout.value, cfg.learn.lam)
    tte = plies_to_end(done, win)
    material_tte = plies_to_end(done, win | clock)
    material = final_material(done, win | clock, board, capture)
    # The opponent's reply is the next row's move, seen from this row's side.
    reply_ok = same_game_later(done, 1)
    reply = torch.where(reply_ok, flip_actions(action.roll(-1, 0)), 0).to(torch.int16)

    def host(x):
        return x.detach().flatten(0, 1).to("cpu", copy=True)

    return Window(
        board=host(board),
        since_capture=host(rollout.since_capture),
        ply=host(rollout.ply),
        clock=host(rollout.clock),
        action=host(action),
        moves=host(rollout.moves),
        target=host(rollout.target),
        target_ok=host(rollout.target_ok),
        tactics=host(rollout.tactics),
        ret=host(ret),
        candidates=host(rollout.candidates),
        candidate_q=host(rollout.candidate_q),
        candidate_visited=host(rollout.candidate_visited),
        occupancy_ok=host(same_game_later(done, cfg.learn.occupancy_horizon)),
        plies_to_end=host(tte),
        material=host(material),
        material_ok=host(material_tte > 0),
        outcome=torch.zeros(done.numel(), dtype=torch.int8),
        outcome_ok=torch.zeros(done.numel(), dtype=torch.bool),
        reply=host(reply),
        reply_ok=host(reply_ok),
        played_q=host(rollout.played_q),
        rank=host(rollout.rank),
        regret=torch.zeros(done.numel(), dtype=torch.float32),
        regret_ok=torch.zeros(done.numel(), dtype=torch.bool),
        replayed=host(rollout.origin >= 0),
        envs=done.shape[1],
        steps=done.shape[0],
        iteration=iteration,
    )


class Outcomes:
    """Final-outcome labels (DESIGN.md item 31): once a game ends, every row
    of it, back to its first row in an earlier window, learns the result from
    its mover's view: +1 for the side that made the last move of a won game,
    -1 for the other side, 0 for a clock draw; a game cut by the ply cap
    stays unlabelled. Rows of games still running stay unlabelled until they
    end, so a window's labels keep arriving after it was written. The same
    walk labels each row's regret (item 33): the mean over the rest of the
    game of calibration excess 2Q(Q-z_Q), where z_Q is the parity-adjusted
    terminal return in the value learner's Q domain."""

    def __init__(self, envs: int, steps: int):
        self.envs, self.steps = envs, steps
        self.start_it = np.full(envs, -1, dtype=np.int64)  # window of each env's current game's first row
        self.start_t = np.zeros(envs, dtype=np.int64)  # and its step there

    def label(
        self, done: torch.Tensor, end_reason: torch.Tensor, window: Window, replay: Replay, control=None
    ) -> set[int]:
        """Label the games that ended in `window`, handing each to the search
        control; returns the earlier iterations whose windows received labels."""
        T, N, it = self.steps, self.envs, window.iteration
        if (self.start_it < 0).any():
            self.start_it[self.start_it < 0] = it
        done = done.cpu().numpy()
        reason = end_reason.cpu().numpy()
        touched: set[int] = set()
        # The previous window's last rows learn their replies from this window's first moves.
        previous = replay.find(it - 1)
        continued = torch.from_numpy(self.start_it < it)
        if previous is not None and continued.any():
            first = flip_actions(window.action.view(T, N)[0])
            previous.reply.view(T, N)[T - 1] = torch.where(continued, first, previous.reply.view(T, N)[T - 1])
            previous.reply_ok.view(T, N)[T - 1] |= continued
            touched.add(it - 1)
        for n in range(N):
            for t in np.nonzero(done[:, n])[0]:
                t = int(t)
                result = 1 if reason[t, n] == END_WIN else (0 if reason[t, n] == END_CLOCK else None)
                s_it, s_t = int(self.start_it[n]), int(self.start_t[n])
                rows, complete = [], True
                for k in range(s_it, it + 1):
                    w = window if k == it else replay.find(k)
                    if w is None:
                        complete = False
                        continue
                    rows.append((w, s_t if k == s_it else 0, t if k == it else T - 1))
                if result is not None:
                    total, count = 0.0, 0
                    last_w, _, last_b = rows[-1]
                    end_ret = float(last_w.ret.view(T, N)[last_b, n])
                    for w, a, b in reversed(rows):
                        before_end = (it - w.iteration) * T + t - np.arange(a, b + 1)
                        values = result * (1 - 2 * (before_end % 2))
                        w.outcome.view(T, N)[a : b + 1, n] = torch.from_numpy(values.astype(np.int8))
                        w.outcome_ok.view(T, N)[a : b + 1, n] = True
                        z_q = end_ret * (1 - 2 * (before_end % 2))
                        q = w.played_q.view(T, N)[a : b + 1, n].numpy().astype(np.float64)
                        excess = 2.0 * q * (q - z_q)
                        suffix = np.cumsum(excess[::-1])[::-1] + total
                        lengths = np.arange(len(excess), 0, -1) + count
                        w.regret.view(T, N)[a : b + 1, n] = torch.from_numpy((suffix / lengths).astype(np.float32))
                        w.regret_ok.view(T, N)[a : b + 1, n] = True
                        total, count = total + float(excess.sum()), count + len(excess)
                        if w.iteration != it:
                            touched.add(w.iteration)
                if control is not None:
                    control.finish(n, t, result, complete, rows)
                self.start_it[n], self.start_t[n] = (it, t + 1) if t + 1 < T else (it + 1, 0)
        return touched


class Replay:
    """The windows of the run, every one persisted compressed as
    `window_NNNNNN.pt.zst` in `directory`, and a sampling window over the
    newest of them that grows as a power law of the windows produced so far
    (`Learn.window_*`). Each iteration yields `Learn.batches` batches, each
    drawn uniformly from one window of the sampling window."""

    LEVEL = 3  # zstd level: 3.4x smaller in 0.2 s per window

    def __init__(self, learn, directory: str, seed: int):
        self.learn = learn
        self.batch = learn.batch
        self.directory = Path(directory)
        self.windows: list[Window] = []  # oldest first, at most `window_size()` of them
        self.cursors: dict[int, tuple[torch.Tensor, int]] = {}  # iteration -> (permutation, next row)
        self.generated = 0  # windows produced so far, counting from iteration 0
        self.rng = torch.Generator().manual_seed(seed)

    def path(self, iteration: int) -> Path:
        return self.directory / f"window_{iteration:06d}.pt.zst"

    def window_size(self) -> int:
        """Windows in the sampling window after `generated` windows: KataGo's
        shifted and scaled power law, which equals `window_min` at
        `window_min` windows and grows by `window_per_iter` per window there."""
        L, n, m, e = self.learn, self.generated, self.learn.window_min, self.learn.window_exponent
        if n <= m:
            return max(n, 1)
        grown = m + (n**e - m**e) / (e * m ** (e - 1)) * L.window_per_iter
        return int(min(n, L.window_max, round(grown)))

    def persist(self, window: Window) -> None:
        """Write (or rewrite, after new outcome labels) one window, compressed."""
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "regret_version": REGRET_VERSION,
            **{f.name: getattr(window, f.name) for f in fields(window)},
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".window_", suffix=".tmp", delete=False) as f:
            f.write(zstd.compress(buffer.getvalue(), self.LEVEL))
            tmp = f.name
        os.replace(tmp, self.path(window.iteration))

    def find(self, iteration: int) -> Window | None:
        """The window of `iteration` if it is still in the sampling window."""
        for w in self.windows:
            if w.iteration == iteration:
                return w
        return None

    def add(self, window: Window) -> None:
        """Persist compressed, append, and drop what the sampling window no longer covers."""
        self.persist(window)
        self.windows.append(window)
        self.generated = max(self.generated, window.iteration + 1)
        self._trim()

    def _trim(self) -> None:
        keep = self.window_size()
        for old in self.windows[:-keep]:
            self.cursors.pop(old.iteration, None)
        del self.windows[:-keep]

    def load(self, iteration: int) -> Window | None:
        """One persisted window, compressed or (older runs) plain; None when absent."""
        path = self.path(iteration)
        if path.is_file():
            raw = zstd.decompress(path.read_bytes())
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        elif path.with_suffix("").is_file():
            payload = torch.load(path.with_suffix(""), map_location="cpu", weights_only=True)
        else:
            return None
        if payload.get("schema") != SCHEMA:
            return None
        rows = payload["board"].shape[0]
        # Windows written before the outcome labels existed carry none.
        payload.setdefault("outcome", torch.zeros(rows, dtype=torch.int8))
        payload.setdefault("outcome_ok", torch.zeros(rows, dtype=torch.bool))
        # Decisive ends were the only material labels in older windows.
        payload.setdefault("material_ok", payload["plies_to_end"] > 0)
        if "reply" not in payload:
            # Before the reply labels, every game started at ply 0, so a ply
            # that continues the previous row's marks the same game.
            T, N = payload["steps"], payload["envs"]
            action, ply = payload["action"].view(T, N), payload["ply"].view(T, N)
            same = torch.zeros(T, N, dtype=torch.bool)
            same[:-1] = ply[1:] == ply[:-1] + 1
            payload["reply"] = torch.where(same, flip_actions(action.roll(-1, 0)), 0).to(torch.int16).flatten()
            payload["reply_ok"] = same.flatten()
        # Windows written before the search control carry no rank or regret.
        for name in ("played_q", "rank", "regret"):
            payload.setdefault(name, torch.zeros(rows, dtype=torch.float32))
        payload.setdefault("regret_ok", torch.zeros(rows, dtype=torch.bool))
        if payload.get("regret_version", 0) < REGRET_VERSION:
            payload["regret_ok"] = torch.zeros(rows, dtype=torch.bool)
        payload.setdefault("replayed", torch.zeros(rows, dtype=torch.bool))
        return Window(**{f.name: payload[f.name] for f in fields(Window)})

    def restore(self, iteration: int) -> int:
        """Load the windows the sampling window covers up to `iteration`; returns how many."""
        self.windows.clear()
        self.cursors.clear()
        self.generated = iteration + 1
        for it in range(max(0, iteration - self.window_size() + 1), iteration + 1):
            window = self.load(it)
            if window is not None:
                self.windows.append(window)
        return len(self.windows)

    def prune(self, iteration: int) -> None:
        """Every window stays on disk; only stale temp files go."""
        for path in self.directory.iterdir():
            if path.name.startswith(".window_") and path.suffix == ".tmp":
                path.unlink(missing_ok=True)

    def positions(self) -> int:
        return sum(len(w) for w in self.windows)

    def _rows(self, window: Window) -> torch.Tensor:
        """The next `batch` rows of `window` from its shuffled stream, without
        replacement until the stream is exhausted and reshuffled."""
        perm, at = self.cursors.get(window.iteration, (None, 0))
        if perm is None:
            perm, at = torch.randperm(len(window), generator=self.rng), 0
        idx = perm[at : at + self.batch]
        at += self.batch
        while len(idx) < self.batch:
            perm, at = torch.randperm(len(window), generator=self.rng), 0
            short = self.batch - len(idx)
            idx = torch.cat([idx, perm[:short]])
            at = short
        self.cursors[window.iteration] = (perm, at)
        return idx

    def batches(self):
        """`(window, row indices)` for one iteration: `Learn.batches` batches,
        each from a window chosen in proportion to its size."""
        if not self.windows:
            return
        sizes = torch.tensor([float(len(w)) for w in self.windows])
        choice = torch.multinomial(sizes, self.learn.batches, replacement=True, generator=self.rng)
        for wi in choice.tolist():
            window = self.windows[wi]
            yield window, self._rows(window)
