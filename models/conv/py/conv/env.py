"""Batched GPU environment: N boards stepped in lockstep with auto-reset. All
state lives in static tensors so a step can be captured in a CUDA graph.
Built on `kernels`; used by `search` (as its transition model) and `train`."""

from __future__ import annotations

import torch

from . import kernels
from .planes import N_ACTIONS, N_PLANES, N_SQUARES, initial_board

END_NONE, END_WIN, END_CLOCK, END_MAX_PLIES = 0, 1, 2, 3


def clock_reached(since_capture: torch.Tensor, clock: torch.Tensor) -> torch.Tensor:
    """Boards that have gone `clock` plies without a capture, per board."""
    return since_capture >= clock


def material_lead(board: torch.Tensor) -> torch.Tensor:
    """(...,) +1 / 0 / -1: whether the mover is ahead on `board` (mover's frame).

    Material is cyclic, so counts alone mislead: a side whose pieces cannot be
    captured is ahead however few it has. A piece is safe when the enemy has
    no piece of the type that beats it (type t is beaten by type t % 3 + 1).
    The side with more safe pieces leads; with equal safe counts, the side
    with more pieces leads (DESIGN.md item 19)."""
    own = torch.stack([(board == 1 + t).sum(-1) for t in range(3)], -1)
    enemy = torch.stack([(board == 4 + t).sum(-1) for t in range(3)], -1)
    # The type that beats type i sits at index i + 1 (a roll, not an index
    # list: this runs inside the search's CUDA graph capture).
    safe_own = (own * (enemy.roll(-1, -1) == 0)).sum(-1)
    safe_enemy = (enemy * (own.roll(-1, -1) == 0)).sum(-1)
    lead = (safe_own - safe_enemy).sign()
    return torch.where(lead != 0, lead, (own.sum(-1) - enemy.sum(-1)).sign()).float()


class Env:
    def __init__(self, n: int, device, max_plies: int, clock_min: int, clock_max: int, seed: int):
        """Each board draws its own capture clock uniformly from
        [`clock_min`, `clock_max`] plies (DESIGN.md item 19), redrawn whenever
        the board resets."""
        self.n = n
        self.device = torch.device(device)
        self.max_plies = max_plies
        self.clock_min = clock_min
        self.clock_max = clock_max
        d = self.device
        self.generator = torch.Generator(device=d)
        self.generator.manual_seed(seed)
        self.board = torch.zeros(n, N_SQUARES, dtype=torch.int8, device=d)
        self.since_capture = torch.zeros(n, dtype=torch.int32, device=d)
        self.ply = torch.zeros(n, dtype=torch.int32, device=d)
        self.clock = torch.zeros(n, dtype=torch.int32, device=d)
        self.legal_i8 = torch.zeros(n, N_ACTIONS, dtype=torch.int8, device=d)
        self.legal_count = torch.zeros(n, dtype=torch.int32, device=d)
        self.planes = torch.zeros(n, N_PLANES, N_SQUARES, dtype=torch.bfloat16, device=d)
        self.scratch = torch.zeros(n, kernels.DERIVE_SCRATCH, dtype=torch.int32, device=d)
        self.goal = torch.zeros(n, dtype=torch.int8, device=d)
        self.eliminated = torch.zeros(n, dtype=torch.int8, device=d)
        self.captured = torch.zeros(n, dtype=torch.int8, device=d)
        self.reward = torch.zeros(n, dtype=torch.float32, device=d)
        self.done = torch.zeros(n, dtype=torch.bool, device=d)
        self.end_reason = torch.zeros(n, dtype=torch.uint8, device=d)
        self.start_board = torch.tensor(initial_board(), dtype=torch.int8, device=d)
        # Where each board's next game starts (DESIGN.md item 33): the initial
        # position with a fresh clock (restart_clock 0) unless the search
        # control stored another; `origin` is the buffer entry the game in
        # play started from, -1 for the initial position.
        self.restart_board = torch.zeros(n, N_SQUARES, dtype=torch.int8, device=d)
        self.restart_since = torch.zeros(n, dtype=torch.int32, device=d)
        self.restart_ply = torch.zeros(n, dtype=torch.int32, device=d)
        self.restart_clock = torch.zeros(n, dtype=torch.int32, device=d)
        self.restart_origin = torch.full((n,), -1, dtype=torch.int32, device=d)
        self.origin = torch.full((n,), -1, dtype=torch.int32, device=d)
        self.reset_all()

    @property
    def legal(self) -> torch.Tensor:
        return self.legal_i8.bool()

    def draw_clocks(self, where: torch.Tensor) -> None:
        """Give the marked boards a fresh capture clock."""
        drawn = torch.randint(
            self.clock_min,
            self.clock_max + 1,
            (self.n,),
            generator=self.generator,
            dtype=torch.int32,
            device=self.device,
        )
        self.clock.copy_(torch.where(where, drawn, self.clock))

    def reset_all(self) -> None:
        self.board.copy_(self.start_board[None].expand(self.n, -1))
        self.since_capture.zero_()
        self.ply.zero_()
        self.draw_clocks(torch.ones_like(self.done))
        self.restart_initial(torch.ones_like(self.done))
        self.origin.fill_(-1)
        self.derive()

    def restart_initial(self, where: torch.Tensor) -> None:
        """The marked boards restart their next game from the initial position."""
        self.restart_board.copy_(torch.where(where[:, None], self.start_board[None], self.restart_board))
        self.restart_since.masked_fill_(where, 0)
        self.restart_ply.masked_fill_(where, 0)
        self.restart_clock.masked_fill_(where, 0)
        self.restart_origin.masked_fill_(where, -1)

    def restart_from(self, where, board, since_capture, ply, clock, origin) -> None:
        """The marked boards restart their next game from the given positions
        (mover's frame) with their capture clocks; `origin` names the buffer
        entry each came from (DESIGN.md item 33)."""
        self.restart_board.copy_(torch.where(where[:, None], board, self.restart_board))
        self.restart_since.copy_(torch.where(where, since_capture, self.restart_since))
        self.restart_ply.copy_(torch.where(where, ply, self.restart_ply))
        self.restart_clock.copy_(torch.where(where, clock, self.restart_clock))
        self.restart_origin.copy_(torch.where(where, origin, self.restart_origin))

    def derive(self) -> None:
        kernels.derive(
            self.board,
            self.since_capture,
            self.ply,
            self.clock,
            self.legal_i8,
            self.legal_count,
            self.planes,
            self.scratch,
        )

    def step(self, action: torch.Tensor) -> None:
        """Apply one action per board, derive the next observation, record
        reward (mover's view), done and end reason, and reset finished boards."""
        kernels.apply(self.board, action, self.since_capture, self.ply, self.goal, self.eliminated, self.captured)
        self.derive()
        win = (self.goal | self.eliminated).bool() | (self.legal_count == 0)
        clock = clock_reached(self.since_capture, self.clock)
        capped = self.ply >= self.max_plies
        # The ply cap outranks the clock, as the labels and the search treat it.
        reason = torch.where(win, END_WIN, torch.where(capped, END_MAX_PLIES, torch.where(clock, END_CLOCK, END_NONE)))
        self.end_reason.copy_(reason.to(torch.uint8))
        self.done.copy_(reason != END_NONE)
        self.reward.copy_(win.float())
        done = self.done
        self.board.copy_(torch.where(done[:, None], self.restart_board, self.board))
        self.since_capture.copy_(torch.where(done, self.restart_since, self.since_capture))
        self.ply.copy_(torch.where(done, self.restart_ply, self.ply))
        # A restart from a stored position keeps that game's clock; the initial position draws one.
        self.draw_clocks(done & (self.restart_clock == 0))
        self.clock.copy_(torch.where(done & (self.restart_clock > 0), self.restart_clock, self.clock))
        self.origin.copy_(torch.where(done, self.restart_origin, self.origin))
        self.derive()
