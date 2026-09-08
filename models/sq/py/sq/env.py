"""Batched GPU environment: N boards stepped in lockstep with auto-reset. All
state lives in static tensors so a step can be captured in a CUDA graph.
Built on `kernels`; used by `search` (as its transition model) and `train`."""

from __future__ import annotations

import torch

from . import kernels
from .planes import N_ACTIONS, N_PLANES, N_SQUARES, initial_board

END_NONE, END_WIN, END_CLOCK, END_MAX_PLIES = 0, 1, 2, 3


def clock_reached(since_capture: torch.Tensor, capture_clock: torch.Tensor) -> torch.Tensor:
    """`capture_clock` is a device scalar; zero disables the clock."""
    return (capture_clock > 0) & (since_capture >= capture_clock)


class Env:
    def __init__(self, n: int, device, max_plies: int, capture_clock: torch.Tensor):
        """`capture_clock` is a device scalar shared with the search so the
        training curriculum can change it without recapturing graphs."""
        self.n = n
        self.device = torch.device(device)
        self.max_plies = max_plies
        self.capture_clock = capture_clock
        d = self.device
        self.board = torch.zeros(n, N_SQUARES, dtype=torch.int8, device=d)
        self.since_capture = torch.zeros(n, dtype=torch.int32, device=d)
        self.ply = torch.zeros(n, dtype=torch.int32, device=d)
        self.legal_i8 = torch.zeros(n, N_ACTIONS, dtype=torch.int8, device=d)
        self.legal_count = torch.zeros(n, dtype=torch.int32, device=d)
        self.planes = torch.zeros(n, N_PLANES, N_SQUARES, dtype=torch.bfloat16, device=d)
        self.goal = torch.zeros(n, dtype=torch.int8, device=d)
        self.eliminated = torch.zeros(n, dtype=torch.int8, device=d)
        self.captured = torch.zeros(n, dtype=torch.int8, device=d)
        self.reward = torch.zeros(n, dtype=torch.float32, device=d)
        self.done = torch.zeros(n, dtype=torch.bool, device=d)
        self.end_reason = torch.zeros(n, dtype=torch.uint8, device=d)
        self.start_board = torch.tensor(initial_board(), dtype=torch.int8, device=d)
        zero = torch.zeros(1, dtype=torch.int32, device=d)
        self.start_legal = torch.zeros(1, N_ACTIONS, dtype=torch.int8, device=d)
        self.start_planes = torch.zeros(1, N_PLANES, N_SQUARES, dtype=torch.bfloat16, device=d)
        kernels.derive(
            self.start_board[None].clone(), zero, zero.clone(), self.start_legal, zero.clone(), self.start_planes
        )
        self.reset_all()

    @property
    def legal(self) -> torch.Tensor:
        return self.legal_i8.bool()

    def reset_all(self) -> None:
        self.board.copy_(self.start_board[None].expand(self.n, -1))
        self.since_capture.zero_()
        self.ply.zero_()
        self.derive()

    def derive(self) -> None:
        kernels.derive(self.board, self.since_capture, self.ply, self.legal_i8, self.legal_count, self.planes)

    def step(self, action: torch.Tensor) -> None:
        """Apply one action per board, derive the next observation, record
        reward (mover's view), done and end reason, and reset finished boards."""
        kernels.apply(self.board, action, self.since_capture, self.ply, self.goal, self.eliminated, self.captured)
        self.derive()
        win = (self.goal | self.eliminated).bool() | (self.legal_count == 0)
        clock = clock_reached(self.since_capture, self.capture_clock)
        capped = self.ply >= self.max_plies
        reason = torch.where(win, END_WIN, torch.where(clock, END_CLOCK, torch.where(capped, END_MAX_PLIES, END_NONE)))
        self.end_reason.copy_(reason.to(torch.uint8))
        self.done.copy_(reason != END_NONE)
        self.reward.copy_(win.float())
        done = self.done[:, None]
        self.board.copy_(torch.where(done, self.start_board[None], self.board))
        self.since_capture.masked_fill_(self.done, 0)
        self.ply.masked_fill_(self.done, 0)
        self.legal_i8.copy_(torch.where(done, self.start_legal, self.legal_i8))
        self.planes.copy_(torch.where(done[:, None], self.start_planes, self.planes))
