"""Rollout windows and replay (DESIGN.md item 15). A window is one collection of
`steps x envs` positions with every label the learner needs, computed once on
the device, then stored immutably on the host and persisted to disk so a
resumed run keeps its recent data."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from .env import END_CLOCK, END_WIN
from .planes import ANTI_PERM, N_ACTIONS, N_SQUARES, action_targets

SCHEMA = 1


@dataclass
class Rollout:
    """One collection on the device: step-major `(T, N, ...)` tensors filled by the actor."""

    board: torch.Tensor  # (T, N, 81) int8, position before the move
    since_capture: torch.Tensor  # (T, N) int16
    ply: torch.Tensor  # (T, N) int16
    action: torch.Tensor  # (T, N) int16
    target: torch.Tensor  # (T, N, 648) bf16 policy target
    target_ok: torch.Tensor  # (T, N) bool: full search
    value: torch.Tensor  # (T + 1, N) float32 search root values, last row the bootstrap
    reward: torch.Tensor  # (T, N) float32 for the mover
    done: torch.Tensor  # (T, N) bool
    end_reason: torch.Tensor  # (T, N) uint8
    candidates: torch.Tensor  # (T, N, C) int16
    candidate_q: torch.Tensor  # (T, N, C) float32
    candidate_visited: torch.Tensor  # (T, N, C) bool

    @staticmethod
    def allocate(steps: int, envs: int, candidates: int, device) -> Rollout:
        T, N, C = steps, envs, candidates

        def z(*shape, dtype):
            return torch.zeros(*shape, dtype=dtype, device=device)

        return Rollout(
            board=z(T, N, N_SQUARES, dtype=torch.int8),
            since_capture=z(T, N, dtype=torch.int16),
            ply=z(T, N, dtype=torch.int16),
            action=z(T, N, dtype=torch.int16),
            target=z(T, N, N_ACTIONS, dtype=torch.bfloat16),
            target_ok=z(T, N, dtype=torch.bool),
            value=z(T + 1, N, dtype=torch.float32),
            reward=z(T, N, dtype=torch.float32),
            done=z(T, N, dtype=torch.bool),
            end_reason=z(T, N, dtype=torch.uint8),
            candidates=z(T, N, C, dtype=torch.int16),
            candidate_q=z(T, N, C, dtype=torch.float32),
            candidate_visited=z(T, N, C, dtype=torch.bool),
        )


@dataclass
class Window:
    """Immutable labels for one collection. Every field is `(steps * envs, ...)` on the host."""

    board: torch.Tensor
    since_capture: torch.Tensor
    ply: torch.Tensor
    action: torch.Tensor
    target: torch.Tensor
    target_ok: torch.Tensor
    ret: torch.Tensor
    candidates: torch.Tensor
    candidate_q: torch.Tensor
    candidate_visited: torch.Tensor
    occupancy_ok: torch.Tensor  # the board `occupancy_horizon` rows later is the same game
    plies_to_end: torch.Tensor  # plies until a decisive end, -1 when unknown
    reply: torch.Tensor
    reply_ok: torch.Tensor
    danger: torch.Tensor  # (S, 81) bool
    danger_ok: torch.Tensor
    material: torch.Tensor  # (S, 2) int8 final own/enemy counts
    envs: int = 0
    steps: int = 0
    iteration: int = -1

    def __len__(self) -> int:
        return self.board.shape[0]

    def occupancy(self, idx: torch.Tensor, horizon: int) -> torch.Tensor:
        """Boards `horizon` plies later for rows where that is the same game, else the row's own board."""
        later = torch.where(self.occupancy_ok[idx], idx + horizon * self.envs, idx)
        return self.board[later]


def lambda_returns(reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor, lam: float) -> torch.Tensor:
    """Two-player lambda returns from the mover's view at each ply: reward plus
    the negated mixture of the next value and the next return; undiscounted,
    bootstrapped on `value[T]` after the last step."""
    T = reward.shape[0]
    out = torch.empty_like(reward)
    nxt = -value[T]
    for t in range(T - 1, -1, -1):
        out[t] = torch.where(done[t], reward[t], reward[t] + nxt)
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
    """(T, N, 2) piece counts at the decisive end of each row's game in that
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


def danger_targets(capture: torch.Tensor, plies_to_end: torch.Tensor, boundary: torch.Tensor, k: int):
    """(T, N, 81) squares captured on within the next `k` plies of the same
    game, in each row's frame, and (T, N) whether the window is fully known."""
    T, N = capture.shape
    anti = torch.tensor(ANTI_PERM, device=capture.device)
    horizon = torch.where(boundary > 0, boundary, T - torch.arange(T, device=capture.device)[:, None])
    target = torch.zeros(T, N, N_SQUARES + 1, device=capture.device)
    padded = torch.cat([capture.long(), torch.full((k, N), -1, dtype=torch.long, device=capture.device)])
    for j in range(k):
        square = padded[j : j + T]
        ok = (j < horizon) & (square >= 0)
        square = square.clamp_min(0)
        square = anti[square] if j % 2 == 1 else square
        square = torch.where(ok, square, N_SQUARES)
        target.scatter_(2, square[..., None], 1.0)
    return target[..., :N_SQUARES].bool(), (plies_to_end > 0) | (horizon >= k)


def build_window(rollout: Rollout, cfg, iteration: int) -> Window:
    """All labels from one collection: clock penalties, lambda returns, plies to
    a decisive end, occupancy validity, opponent reply, capture-danger map,
    final material. Returns host tensors."""
    T, N = rollout.done.shape
    board, action, done, reason = rollout.board, rollout.action, rollout.done, rollout.end_reason
    win = reason == END_WIN
    capture = capture_squares(board, action)
    reward = rollout.reward.clone()
    clock = (reason == END_CLOCK) & (rollout.ply.int() + 1 < cfg.rules.max_plies)
    balance = material_after(board, capture).float()
    reward = torch.where(clock, -cfg.rules.clock_penalty * (balance[..., 0] - balance[..., 1]).sign(), reward)
    ret = lambda_returns(reward, done, rollout.value, cfg.learn.lam)
    tte = plies_to_end(done, win)
    boundary = plies_to_end(done, done)
    reply_ok = ~done & (torch.arange(T, device=done.device)[:, None] < T - 1)
    reply = torch.cat([action[1:], action[:1]], 0)
    danger, danger_ok = danger_targets(capture, tte, boundary, cfg.learn.danger_horizon)
    material = final_material(done, win, board, capture)

    def host(x):
        return x.detach().flatten(0, 1).to("cpu", copy=True)

    return Window(
        board=host(board),
        since_capture=host(rollout.since_capture),
        ply=host(rollout.ply),
        action=host(action),
        target=host(rollout.target),
        target_ok=host(rollout.target_ok),
        ret=host(ret),
        candidates=host(rollout.candidates),
        candidate_q=host(rollout.candidate_q),
        candidate_visited=host(rollout.candidate_visited),
        occupancy_ok=host(same_game_later(done, cfg.learn.occupancy_horizon)),
        plies_to_end=host(tte),
        reply=host(reply),
        reply_ok=host(reply_ok),
        danger=host(danger),
        danger_ok=host(danger_ok),
        material=host(material),
        envs=N,
        steps=T,
        iteration=iteration,
    )


class Replay:
    """The newest `windows` windows; yields shuffled batches for `epochs`
    passes over each window per iteration. Windows are persisted as
    `window_NNNNNN.pt` in `directory` so a resumed run continues on the same data."""

    def __init__(self, windows: int, epochs: int, batch: int, directory: str, seed: int):
        self.capacity, self.epochs, self.batch = windows, epochs, batch
        self.directory = Path(directory)
        self.windows: list[Window] = []
        self.rng = torch.Generator().manual_seed(seed)

    def path(self, iteration: int) -> Path:
        return self.directory / f"window_{iteration:06d}.pt"

    def add(self, window: Window) -> None:
        """Persist, append, drop the oldest beyond the capacity."""
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA, **{f.name: getattr(window, f.name) for f in fields(window)}}
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".window_", suffix=".tmp", delete=False) as f:
            torch.save(payload, f)
            tmp = f.name
        os.replace(tmp, self.path(window.iteration))
        self.windows.append(window)
        del self.windows[: -self.capacity]

    def restore(self, iteration: int) -> int:
        """Load the windows persisted up to `iteration`; returns how many."""
        self.windows.clear()
        for it in range(max(0, iteration - self.capacity + 1), iteration + 1):
            path = self.path(it)
            if not path.is_file():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("schema") != SCHEMA:
                continue
            self.windows.append(Window(**{f.name: payload[f.name] for f in fields(Window)}))
        return len(self.windows)

    def prune(self, iteration: int) -> None:
        """Delete window files older than the capacity, and stale temp files."""
        low = iteration - self.capacity + 1
        for path in self.directory.iterdir():
            match = re.fullmatch(r"window_(\d+)\.pt", path.name)
            if (match and int(match[1]) < low) or (path.name.startswith(".window_") and path.suffix == ".tmp"):
                path.unlink(missing_ok=True)

    def positions(self) -> int:
        return sum(len(w) for w in self.windows)

    def batches(self):
        """`(window, row indices)` for one iteration: every window `epochs`
        times, shuffled; the last batch of a window repeats rows to stay full."""
        for _ in range(self.epochs):
            for wi in torch.randperm(len(self.windows), generator=self.rng).tolist():
                window = self.windows[wi]
                size = len(window)
                perm = torch.randperm(size, generator=self.rng)
                for start in range(0, size, self.batch):
                    idx = perm[start : start + self.batch]
                    if len(idx) < self.batch:
                        short = self.batch - len(idx)
                        idx = torch.cat([idx, perm.repeat(short // size + 1)[:short]])
                    yield window, idx
