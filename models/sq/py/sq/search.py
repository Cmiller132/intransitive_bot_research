"""Batched Gumbel search on the GPU (DESIGN.md items 8 to 11): one tree per
board in a static arena, sequential halving at the root, Triton descent and
backup kernels, one simulation captured as a CUDA graph and replayed, and the
played move's subtree kept for the next search.

Returns the policy target, the root value and the move to play for every
board. The play-time search lives in the Rust `search` crate; the two share
their semantics, not their code."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from . import kernels
from .config import Search as SearchConfig
from .env import clock_reached
from .planes import N_ACTIONS, N_SQUARES, initial_board

BLOCK_A = 1024
# A legal quiet move of the start position, played on rows that expand nothing.
SAFE_ACTION = 28


def halving_plan(sims: int, candidates: int) -> list[tuple[int, int]]:
    """`(survivors, slot)` per simulation: every survivor is visited in turn,
    the field halves when its share of the remaining budget is spent."""
    plan = []
    m = candidates
    while m and len(plan) < sims:
        per = max(1, (sims - len(plan)) // (max(1, math.ceil(math.log2(m))) * m))
        for _ in range(per):
            for slot in range(m):
                if len(plan) < sims:
                    plan.append((m, slot))
        m = max(1, m // 2)
    return plan


def raw_policy(logits: torch.Tensor, legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(probabilities, log probabilities)` over legal actions; rows without a
    legal action get all their mass on action 0 so nothing is NaN."""
    safe = legal | ((torch.arange(N_ACTIONS, device=legal.device) == 0) & ~legal.any(-1)[:, None])
    logp = logits.float().masked_fill(~safe, -torch.inf).log_softmax(-1)
    return logp.exp().masked_fill(~legal, 0.0), logp


@triton.jit
def walk_kernel(
    PRIOR,
    Q,
    LEGAL,
    EXACT,
    COUNT,
    TOTAL,
    CHILD,
    TERMINAL,
    VALUE,
    ACTION,
    LIVE,
    PN,
    PA,
    LENGTH,
    FN,
    FA,
    EXPAND,
    LEAF_VALUE,
    B: tl.constexpr,
    A: tl.constexpr,
    CV: tl.constexpr,
    CS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Descend one tree from its root along deficit selection until an
    immediate win, an unexpanded edge or a terminal node, recording the path.
    One program per board."""
    row = tl.program_id(0)
    edges = tl.arange(0, BLOCK)
    node = tl.full((), 0, tl.int32)
    action = tl.load(ACTION + row).to(tl.int32)
    active = tl.load(LIVE + row)
    depth = tl.full((), 0, tl.int32)
    expand = tl.full((), 0, tl.int1)
    value = tl.full((), 0.0, tl.float32)
    frontier_node = tl.full((), 0, tl.int32) + node
    frontier_action = tl.full((), 0, tl.int32) + action
    while active:
        base = (node * B + row) * A
        if depth > 0:
            n = tl.load(COUNT + base + edges, edges < A, 0).to(tl.int32)
            total = tl.load(TOTAL + base + edges, edges < A, 0.0)
            q = tl.load(Q + base + edges, edges < A, 0.0).to(tl.float32)
            exact = tl.load(EXACT + base + edges, edges < A, 0)
            legal = tl.load(LEGAL + base + edges, edges < A, 0)
            prior = tl.load(PRIOR + base + edges, edges < A, -float("inf")).to(tl.float32)
            completed = tl.where(exact, 1.0, tl.where(n > 0, total / tl.maximum(n, 1), q))
            z = tl.where(legal, prior + (CV + tl.max(n, 0)) * CS * completed, -float("inf"))
            p = tl.exp(z - tl.max(z, 0))
            p = p / tl.sum(p, 0)
            deficit = tl.where(legal, p - n / (1.0 + tl.sum(n, 0)), -float("inf"))
            action = tl.min(tl.where(deficit == tl.max(deficit, 0), edges, 2147483647), 0)
        tl.store(PN + depth * B + row, node)
        tl.store(PA + depth * B + row, action)
        depth += 1
        exact_edge = tl.load(EXACT + base + action)
        child = tl.load(CHILD + base + action).to(tl.int32)
        frontier_node = node + tl.full((), 0, tl.int32)
        frontier_action = action + tl.full((), 0, tl.int32)
        if exact_edge:
            value = -1.0
            active = False
        elif child < 0:
            expand = True
            active = False
        else:
            if tl.load(TERMINAL + child * B + row):
                value = tl.load(VALUE + child * B + row)
                active = False
            else:
                node = child
    tl.store(LENGTH + row, depth)
    tl.store(FN + row, frontier_node)
    tl.store(FA + row, frontier_action)
    tl.store(EXPAND + row, expand)
    tl.store(LEAF_VALUE + row, value)


@triton.jit
def backup_kernel(COUNT, TOTAL, PN, PA, LENGTH, VALUE, B: tl.constexpr, A: tl.constexpr):
    """Add the leaf value along the recorded path with alternating sign."""
    row = tl.program_id(0)
    depth = tl.load(LENGTH + row)
    value = tl.load(VALUE + row)
    while depth > 0:
        depth -= 1
        node = tl.load(PN + depth * B + row)
        action = tl.load(PA + depth * B + row)
        address = (node * B + row) * A + action
        value = -value
        tl.store(COUNT + address, tl.load(COUNT + address) + 1)
        tl.store(TOTAL + address, tl.load(TOTAL + address) + value)


@triton.jit
def compact_kernel(
    BOARDS,
    SINCE,
    PLY,
    PRIOR,
    Q,
    LEGAL,
    EXACT,
    COUNT,
    TOTAL,
    CHILD,
    TERM,
    VALUE,
    HASH,
    ACTION,
    DONE,
    CURRENT,
    CURRENT_SINCE,
    CURRENT_PLY,
    LIVE,
    MAP,
    NODES,
    KEPT,
    B: tl.constexpr,
    C: tl.constexpr,
    A: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Re-root one tree on the played child: the reachable nodes, at most R of
    them in id order, are renumbered in place; everything else is cleared.
    Parents always have smaller ids than their children, so copying in id
    order never overwrites an unread source."""
    row = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    for node in range(C):
        tl.store(MAP + node * B + row, -1)
    action = tl.load(ACTION + row)
    root = tl.load(CHILD + row * A + action).to(tl.int32)
    safe_root = tl.maximum(root, 0)
    old_board = tl.load(BOARDS + (safe_root * B + row) * 81 + x, x < 81, 0)
    new_board = tl.load(CURRENT + row * 81 + x, x < 81, 0)
    valid = (root >= 0) & ~tl.load(DONE + row) & tl.load(LIVE + row)
    valid = valid & ~tl.load(TERM + safe_root * B + row)
    valid = valid & (tl.sum((old_board != new_board).to(tl.int32), 0) == 0)
    valid = valid & (tl.load(SINCE + safe_root * B + row) == tl.load(CURRENT_SINCE + row))
    valid = valid & (tl.load(PLY + safe_root * B + row) == tl.load(CURRENT_PLY + row))
    if valid:
        tl.store(MAP + root * B + row, -2)
    tl.debug_barrier()
    size = tl.full((), 0, tl.int32)
    for node in range(C):
        state = tl.load(MAP + node * B + row)
        if state == -2:
            if size < R:
                tl.store(MAP + node * B + row, size)
                size += 1
                child = tl.load(CHILD + (node * B + row) * A + x, x < A, -1).to(tl.int32)
                tl.store(MAP + child * B + row, -2, (x < A) & (child >= 0))
        tl.debug_barrier()
    for node in range(C):
        dest = tl.load(MAP + node * B + row)
        if dest >= 0:
            src = (node * B + row) * A + x
            dst = (dest * B + row) * A + x
            tl.store(PRIOR + dst, tl.load(PRIOR + src, x < A, 0.0), x < A)
            tl.store(Q + dst, tl.load(Q + src, x < A, 0.0), x < A)
            tl.store(LEGAL + dst, tl.load(LEGAL + src, x < A, 0), x < A)
            tl.store(EXACT + dst, tl.load(EXACT + src, x < A, 0), x < A)
            tl.store(COUNT + dst, tl.load(COUNT + src, x < A, 0), x < A)
            tl.store(TOTAL + dst, tl.load(TOTAL + src, x < A, 0.0), x < A)
            child = tl.load(CHILD + src, x < A, -1).to(tl.int32)
            mapped = tl.load(MAP + child * B + row, (x < A) & (child >= 0), -1)
            tl.store(CHILD + dst, tl.maximum(mapped, -1), x < A)
            tl.store(BOARDS + (dest * B + row) * 81 + x, tl.load(BOARDS + (node * B + row) * 81 + x, x < 81, 0), x < 81)
            tl.store(SINCE + dest * B + row, tl.load(SINCE + node * B + row))
            tl.store(PLY + dest * B + row, tl.load(PLY + node * B + row))
            tl.store(TERM + dest * B + row, tl.load(TERM + node * B + row))
            tl.store(VALUE + dest * B + row, tl.load(VALUE + node * B + row))
            tl.store(HASH + dest * B + row, tl.load(HASH + node * B + row))
        tl.debug_barrier()
    for node in range(C):
        if node >= size:
            edge = (node * B + row) * A + x
            tl.store(COUNT + edge, 0, x < A)
            tl.store(TOTAL + edge, 0.0, x < A)
            tl.store(CHILD + edge, -1, x < A)
            tl.store(TERM + node * B + row, False)
    tl.store(NODES + row, tl.maximum(size, 1))
    tl.store(KEPT + row, valid)


class PositionHasher:
    """SplitMix64 keys per (square, cell) and one for the side to move;
    additive hashing in int64 with wrap-around."""

    def __init__(self, device):
        x = torch.arange(81 * 7 + 1, dtype=torch.int64, device=device)
        x = (x + 20260908) * -7046029254386353131
        x = (x ^ ((x >> 30) & ((1 << 34) - 1))) * -4658895280553007687
        x = (x ^ ((x >> 27) & ((1 << 37) - 1))) * -7723592293110705685
        x = x ^ ((x >> 31) & ((1 << 33) - 1))
        self.keys = x[:-1].reshape(81, 7)
        self.parity_key = x[-1:]
        self.squares = torch.arange(81, device=device)

    def __call__(self, board: torch.Tensor, ply: torch.Tensor) -> torch.Tensor:
        return self.keys[self.squares, board.long()].sum(-1) + (ply.long() & 1) * self.parity_key


class RepetitionHistory:
    """Per-board ring of position hashes since the last capture, including the
    current position, so the search recognises a repeat of a position already
    played in the game."""

    def __init__(self, n: int, capacity: int, device):
        self.capacity = max(capacity, 1)
        self.hasher = PositionHasher(device)
        self.hashes = torch.zeros(self.capacity, n, dtype=torch.int64, device=device)
        self.size = torch.zeros(n, dtype=torch.long, device=device)
        self.next = torch.zeros_like(self.size)
        self.rows = torch.arange(n, device=device)
        self.slots = torch.arange(self.capacity, device=device)[:, None]

    def reset(self, board, ply) -> None:
        self.size.zero_()
        self.next.zero_()
        self._push(self.hasher(board, ply))

    def _push(self, value) -> None:
        self.hashes[self.next, self.rows] = value
        self.next.copy_((self.next + 1) % self.capacity)
        self.size.copy_((self.size + 1).clamp_max(self.capacity))

    def push(self, board, since_capture, ply, done) -> None:
        """After the environment stepped: a capture or a reset empties the ring."""
        clear = done.bool() | (since_capture == 0)
        self.size.masked_fill_(clear, 0)
        self.next.masked_fill_(clear, 0)
        self._push(self.hasher(board, ply))

    def occurrences(self, value) -> torch.Tensor:
        return ((self.hashes == value[None]) & (self.slots < self.size[None])).sum(0)


class GumbelSearch:
    def __init__(
        self,
        cfg: SearchConfig,
        n: int,
        device,
        capture_clock: torch.Tensor,
        clock_penalty: float,
        max_plies: int,
        seed: int,
        history: RepetitionHistory | None,
    ):
        self.cfg, self.n = cfg, n
        self.device = torch.device(device)
        self.capture_clock, self.clock_penalty, self.max_plies = capture_clock, clock_penalty, max_plies
        self.history = history
        d = self.device
        sims, cands = cfg.sims, cfg.candidates
        assert cfg.cheap_sims <= sims and cfg.cheap_candidates <= cands
        self.capacity = cfg.reuse_nodes + sims + 1
        C = self.capacity
        self.rows = torch.arange(n, device=d)
        self.slots = torch.arange(cands, device=d)[None]
        self.rng = torch.Generator(device=d).manual_seed(seed)
        self.hasher = history.hasher if history is not None else PositionHasher(d)
        # Halving schedules for both search sizes, indexed by the searchable candidate count.
        self.plans = {}
        for mode, (mode_sims, mode_cands) in (
            ("full", (sims, cands)),
            ("cheap", (cfg.cheap_sims, cfg.cheap_candidates)),
        ):
            table = torch.zeros(cands + 1, sims, 2, dtype=torch.long)
            for m in range(mode_cands + 1):
                plan = halving_plan(mode_sims, m)
                if plan:
                    table[m, : len(plan)] = torch.tensor(plan)
            self.plans[mode] = table.to(d)
        self.sims = {"full": sims, "cheap": cfg.cheap_sims}
        self.cands = {"full": cands, "cheap": cfg.cheap_candidates}
        # Node arena.
        self.boards = torch.zeros(C, n, N_SQUARES, dtype=torch.int8, device=d)
        self.since = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.plies = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.prior = torch.zeros(C, n, N_ACTIONS, dtype=torch.float16, device=d)
        self.net_q = torch.zeros(C, n, N_ACTIONS, dtype=torch.bfloat16, device=d)
        self.legal = torch.zeros(C, n, N_ACTIONS, dtype=torch.bool, device=d)
        self.exact = torch.zeros(C, n, N_ACTIONS, dtype=torch.bool, device=d)
        self.count = torch.zeros(C, n, N_ACTIONS, dtype=torch.int32, device=d)
        self.total = torch.zeros(C, n, N_ACTIONS, dtype=torch.float32, device=d)
        self.child = torch.full((C, n, N_ACTIONS), -1, dtype=torch.int16, device=d)
        self.terminal = torch.zeros(C, n, dtype=torch.bool, device=d)
        self.value = torch.zeros(C, n, dtype=torch.float32, device=d)
        self.hash = torch.zeros(C, n, dtype=torch.int64, device=d)
        self.root_q = torch.zeros(n, N_ACTIONS, dtype=torch.float32, device=d)
        # Per-simulation scratch.
        self.path_node = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.path_action = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.path_depth = torch.arange(C, device=d)[:, None]
        self.path_length = torch.zeros(n, dtype=torch.int32, device=d)
        self.frontier_node = torch.zeros(n, dtype=torch.int32, device=d)
        self.frontier_action = torch.zeros(n, dtype=torch.int32, device=d)
        self.expand = torch.zeros(n, dtype=torch.bool, device=d)
        self.leaf_value = torch.zeros(n, dtype=torch.float32, device=d)
        self.node_count = torch.ones(n, dtype=torch.int32, device=d)
        self.exact_scratch = torch.zeros(n, N_SQUARES, dtype=torch.int32, device=d)
        self.flags = torch.zeros(3, n, dtype=torch.int8, device=d)
        self.safe_board = torch.tensor(initial_board(), dtype=torch.int8, device=d)
        # Root state.
        self.live = torch.zeros(n, dtype=torch.bool, device=d)
        self.score0 = torch.zeros(n, N_ACTIONS, dtype=torch.float32, device=d)
        self.survivors = torch.zeros(n, cands, dtype=torch.long, device=d)
        self.previous_m = torch.zeros(n, dtype=torch.long, device=d)
        self.schedule = torch.zeros(n, sims, 2, dtype=torch.long, device=d)
        self.sim_index = torch.zeros((), dtype=torch.long, device=d)
        # Reuse.
        self.mapping = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.kept = torch.zeros(n, dtype=torch.bool, device=d)
        self.played_action = torch.zeros(n, dtype=torch.long, device=d)
        self.played_done = torch.ones(n, dtype=torch.bool, device=d)
        self.pending = False
        self.stats = torch.zeros(4, dtype=torch.int64, device=d)  # kept, calls, repetition draws, expansions
        self.graph = None
        self.forward = None

    # ------------------------------------------------------------------
    def advance(self, action: torch.Tensor, done: torch.Tensor) -> None:
        """The move played on every board; the next call re-roots on it."""
        self.played_action.copy_(action.long())
        self.played_done.copy_(done)
        self.pending = True

    def forget(self) -> None:
        """Drop every kept tree (a new network or changed rules)."""
        self.pending = False

    def metrics(self) -> dict:
        kept, calls, draws, expansions = self.stats.tolist()
        self.stats.zero_()
        return {"reuse_kept_frac": kept / max(calls, 1), "repetition_draw_frac": draws / max(expansions, 1)}

    def _completed(self, node: torch.Tensor) -> torch.Tensor:
        n, total = self.count[node, self.rows], self.total[node, self.rows]
        network = torch.where((node == 0)[:, None], self.root_q, self.net_q[node, self.rows].float())
        q = torch.where(n > 0, total / n.clamp_min(1), network)
        return torch.where(self.exact[node, self.rows], 1.0, q)

    def _sigma(self, visits: torch.Tensor) -> torch.Tensor:
        return (self.cfg.c_visit + visits.amax(-1, keepdim=True)) * self.cfg.c_scale

    def _run(self, planes: torch.Tensor):
        if planes.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.forward(planes)
        return self.forward(planes.float())

    def _store(self, ids: torch.Tensor, mask: torch.Tensor, **fields) -> None:
        """Masked per-row write of node fields at `ids`."""
        rows = self.rows
        for name, value in fields.items():
            arena = getattr(self, name)
            old = arena[ids, rows]
            m = mask.view(-1, *([1] * (value.ndim - 1)))
            arena[ids, rows] = torch.where(m, value.to(arena.dtype), old)

    def _start(self, board, since, ply, legal, logits, q, gumbels, mode: str) -> None:
        root = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.live.copy_(legal.any(-1))
        if self.pending:
            compact_kernel[(self.n,)](
                self.boards,
                self.since,
                self.plies,
                self.prior,
                self.net_q,
                self.legal,
                self.exact,
                self.count,
                self.total,
                self.child,
                self.terminal,
                self.value,
                self.hash,
                self.played_action,
                self.played_done,
                board,
                since,
                ply,
                self.live,
                self.mapping,
                self.node_count,
                self.kept,
                self.n,
                self.capacity,
                N_ACTIONS,
                self.cfg.reuse_nodes,
                BLOCK_A,
            )
        else:
            self.count.zero_()
            self.total.zero_()
            self.child.fill_(-1)
            self.terminal.zero_()
            self.node_count.fill_(1)
            self.kept.zero_()
        self.pending = False
        self.stats[0].add_(self.kept.sum())
        self.stats[1].add_(self.n)
        pi, logp = raw_policy(logits, legal)
        exact = torch.zeros_like(legal)
        kernels.exact_wins(board, legal, exact, self.exact_scratch)
        self.boards[0].copy_(board)
        self.since[0].copy_(since)
        self.plies[0].copy_(ply)
        self.prior[0].copy_(logp)
        self.net_q[0].copy_(q)
        self.root_q.copy_(q.float())
        self.legal[0].copy_(legal)
        self.exact[0].copy_(exact)
        self.terminal[0].zero_()
        self.value[0].copy_(torch.where(exact.any(-1), 1.0, (pi * q.float()).sum(-1)))
        self.hash[0].copy_(self.hasher(board, ply))
        temperature = torch.where(ply < self.cfg.temperature_plies, self.cfg.temperature, 1.0)
        self.score0.copy_((logp + temperature[:, None] * gumbels).masked_fill(~legal, -torch.inf))
        admission = self.score0.masked_fill(exact, torch.inf)
        self.survivors.copy_(admission.topk(self.cfg.candidates, -1).indices)
        self.previous_m.copy_(legal.sum(-1).clamp(max=self.cands[mode]))
        self.schedule.copy_(self.plans[mode][self.previous_m])
        self.sim_index.zero_()
        del root

    def _simulate(self) -> None:
        rows = self.rows
        root = torch.zeros_like(rows)
        sim = self.sim_index.expand_as(rows)
        m, slot = self.schedule[rows, sim].unbind(-1)
        score = self.score0 + self._sigma(self.count[0]) * self._completed(root)
        rank = score.gather(1, self.survivors).masked_fill(self.slots >= self.previous_m[:, None], -torch.inf)
        order = rank.argsort(dim=-1, descending=True, stable=True)
        self.survivors.copy_(
            torch.where((m < self.previous_m)[:, None], self.survivors.gather(1, order), self.survivors)
        )
        self.previous_m.copy_(m)
        action = self.survivors.gather(1, slot[:, None]).squeeze(1)
        walk_kernel[(self.n,)](
            self.prior,
            self.net_q,
            self.legal,
            self.exact,
            self.count,
            self.total,
            self.child,
            self.terminal,
            self.value,
            action,
            self.live,
            self.path_node,
            self.path_action,
            self.path_length,
            self.frontier_node,
            self.frontier_action,
            self.expand,
            self.leaf_value,
            self.n,
            N_ACTIONS,
            self.cfg.c_visit,
            self.cfg.c_scale,
            BLOCK_A,
        )
        fn = self.frontier_node.long()
        active = self.expand
        board = torch.where(active[:, None], self.boards[fn, rows], self.safe_board).contiguous()
        since = torch.where(active, self.since[fn, rows], 0).contiguous()
        ply = torch.where(active, self.plies[fn, rows], 0).contiguous()
        act = torch.where(active, self.frontier_action, SAFE_ACTION).contiguous()
        kernels.apply(board, act, since, ply, self.flags[0], self.flags[1], self.flags[2])
        planes, legal, legal_count = kernels.derive_batch(board, since, ply)
        logits, q = self._run(planes)
        pi, logp = raw_policy(logits, legal)
        exact = torch.zeros_like(legal)
        kernels.exact_wins(board, legal, exact, self.exact_scratch)
        has_move = legal_count > 0
        value = torch.where(exact.any(-1), 1.0, (pi * q.float()).sum(-1))
        clock = clock_reached(since, self.capture_clock) & has_move & (ply < self.max_plies)
        balance = ((board >= 1) & (board <= 3)).sum(-1) - (board >= 4).sum(-1)
        value = torch.where(clock, -self.clock_penalty * balance.float().sign(), value)
        value = torch.where(ply >= self.max_plies, 0.0, value)
        value = torch.where(~has_move, -1.0, value)
        terminal = ~has_move | (ply >= self.max_plies) | clock
        # Twofold repetition over the game history plus the path is a draw.
        leaf_hash = self.hasher(board, ply)
        on_path = self.path_depth < self.path_length[None]
        path_ids = torch.where(on_path, self.path_node, 0).long()
        matches = ((self.hash[path_ids, rows] == leaf_hash[None]) & on_path & (self.path_depth > 0)).sum(0)
        earlier = matches + (self.history.occurrences(leaf_hash) if self.history is not None else 0)
        draw = self.cfg.repetition_draw & active & ~terminal & (earlier >= 1)
        self.stats[2].add_(draw.sum())
        self.stats[3].add_(active.sum())
        terminal = terminal | draw
        value = torch.where(draw, 0.0, value)
        new_id = self.node_count.long()
        cached = active & (new_id < self.capacity)
        safe_id = new_id.clamp(max=self.capacity - 1)
        self._store(
            safe_id,
            cached,
            boards=board,
            since=since,
            plies=ply,
            prior=logp,
            net_q=q,
            legal=legal,
            exact=exact,
            terminal=terminal,
            value=value,
            hash=leaf_hash,
        )
        self.node_count.add_(cached.int())
        link = self.child[fn, rows, self.frontier_action.long()]
        self.child[fn, rows, self.frontier_action.long()] = torch.where(cached, safe_id.to(self.child.dtype), link)
        self.leaf_value.copy_(torch.where(active, value, self.leaf_value))
        backup_kernel[(self.n,)](
            self.count,
            self.total,
            self.path_node,
            self.path_action,
            self.path_length,
            self.leaf_value,
            self.n,
            N_ACTIONS,
        )
        self.sim_index.add_(1)

    @torch.no_grad()
    def __call__(self, board, since_capture, ply, legal, planes, forward, full: bool = True):
        """One search on every board.

        Returns `(target, value, action, candidates, candidate_q, visited)`:
        the policy target over 648 actions, the root value, the move to play,
        the root candidate actions, their completed Q and their visited flags."""
        self.forward = forward
        mode = "full" if full else "cheap"
        legal = legal.bool()
        safe_planes = torch.where(legal.any(-1)[:, None, None], planes, planes[:1] * 0)
        logits, q = self._run(safe_planes)
        u = torch.rand(logits.shape, generator=self.rng, device=self.device).clamp_(1e-12, 1.0)
        gumbels = -(-u.log()).log()
        use_graph = self.device.type == "cuda"
        if use_graph and self.graph is None:
            self.forget()
            self._start(board, since_capture, ply, legal, logits, q, gumbels, mode)
            self._simulate()
            torch.cuda.synchronize()
            self._start(board, since_capture, ply, legal, logits, q, gumbels, mode)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self._simulate()
        self._start(board, since_capture, ply, legal, logits, q, gumbels, mode)
        for _ in range(self.sims[mode]):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._simulate()
        root = torch.zeros(self.n, dtype=torch.long, device=self.device)
        qbar = self._completed(root)
        sigma = self._sigma(self.count[0])
        target, _ = raw_policy(logits.float() + sigma * qbar, legal)
        score = (self.score0 + sigma * qbar).gather(1, self.survivors)
        score = score.masked_fill(self.slots >= self.previous_m[:, None], -torch.inf)
        score = score.masked_fill(self.exact[0].gather(1, self.survivors), torch.inf)
        action = self.survivors.gather(1, score.argmax(-1)[:, None]).squeeze(1)
        value = torch.where(self.live, (target * qbar).sum(-1), -1.0)
        candidates = self.survivors.clone()
        candidate_q = qbar.gather(1, candidates)
        visited = self.count[0].gather(1, candidates) > 0
        return target, value, action, candidates, candidate_q, visited
