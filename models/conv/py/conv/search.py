"""Batched Gumbel search on the GPU (DESIGN.md items 13 to 19): one tree per
board in a static arena, sequential halving at the root, Triton descent and
backup kernels, exact tactics at every expanded node, one simulation captured
as a CUDA graph and replayed, and the played move's subtree kept for the next
search.

Every node keeps only its legal moves, in `SLOTS` slots holding the action
index of each; the root's policy target is scattered back over the 648
actions at the end. The actor is a callable `evaluate(board, since, ply,
clock) -> (logits, q, value, draw, legal, count)`, so the distillation phase can
drive the search with a network that reads different planes (DESIGN item 26);
`value` is what a leaf is worth before any visit. `draw` is `(B,)` from WDL or
`(B, 648)` from the Q-mass compatibility source (item 17).

Returns a `Root` per call: the legal moves of every root in their slots and,
over those slots, the policy target and the packed exact labels, plus the root
value and the move to play. The play-time search lives in the Rust `search`
crate; the two share their semantics, not their code."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from . import kernels
from .config import Search as SearchConfig
from .env import clock_reached, material_lead
from .planes import N_ACTIONS, N_SQUARES, initial_board

# A node holds its legal moves only: ten pieces with eight king steps each at most.
SLOTS = 80
BLOCK_S = 128
# A legal quiet move of the start position, played on rows that expand nothing.
SAFE_ACTION = 28
# Root ranks of a move that loses in two (below every other legal move, above the
# illegal ones at -inf) and of a move that wins in three (above every move that is
# not an outright win at once).
LOSING_RANK = -1e30
WINNING_RANK = 1e30


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
    """`(probabilities, log probabilities)` over the legal entries of the last
    dimension; rows without a legal entry get all their mass on entry 0 so
    nothing is NaN."""
    lanes = torch.arange(logits.shape[-1], device=legal.device)
    safe = legal | ((lanes == 0) & ~legal.any(-1)[:, None])
    logp = logits.float().masked_fill(~safe, -torch.inf).log_softmax(-1)
    return logp.exp().masked_fill(~legal, 0.0), logp


def compact(legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(actions, count)`: each row's legal actions in ascending order in the
    first `count` of its `SLOTS` slots; the remaining slots name illegal ones."""
    order = legal.to(torch.int8).argsort(dim=-1, descending=True, stable=True)
    return order[:, :SLOTS], legal.sum(-1)


def pack_tactics(win1: torch.Tensor, loss2: torch.Tensor, win3: torch.Tensor) -> torch.Tensor:
    """One uint8 per move: bit 0 wins at once, bit 1 loses in two, bit 2 wins in three."""
    return win1.to(torch.uint8) | (loss2.to(torch.uint8) << 1) | (win3.to(torch.uint8) << 2)


def unpack_tactics(packed: torch.Tensor) -> torch.Tensor:
    """`(..., 3)` bools from the packed labels, in the order they were packed."""
    bits = torch.arange(3, device=packed.device, dtype=torch.uint8)
    return ((packed[..., None] >> bits) & 1).bool()


def spread(moves: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """`(n, 648)` from a per-slot `(n, SLOTS)` tensor and the moves in the slots;
    actions in no slot read 0. Slots past a row's legal count name illegal
    actions and carry 0, so they leave the result untouched."""
    out = torch.zeros(x.shape[0], N_ACTIONS, dtype=x.dtype, device=x.device)
    return out.scatter_(1, moves.long(), x)


class Root(NamedTuple):
    """What one search returns for every board; `target` and `tactics` are laid
    out over the slots of `moves` (see `spread`)."""

    moves: torch.Tensor  # (n, SLOTS) int16 legal moves of the root in slot order
    target: torch.Tensor  # (n, SLOTS) float32 policy target
    value: torch.Tensor  # (n,) float32 root value
    action: torch.Tensor  # (n,) int64 move to play
    candidates: torch.Tensor  # (n, C) int64 root candidate actions
    candidate_q: torch.Tensor  # (n, C) float32 completed Q of the candidates
    visited: torch.Tensor  # (n, C) bool: the candidate was visited
    tactics: torch.Tensor  # (n, SLOTS) uint8 packed exact labels of the moves
    played_q: torch.Tensor  # (n,) float32 completed Q of the move to play
    rank: torch.Tensor  # (n,) float32 the search control's ranking score of the root


@triton.jit
def walk_kernel(
    PRIOR,
    Q,
    NLEGAL,
    WIN,
    LOSS,
    COUNT,
    TOTAL,
    CHILD,
    TERMINAL,
    VALUE,
    SLOT,
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
    """Descend one tree from its root along deficit selection until a move that
    wins or loses at once, an unexpanded edge or a terminal node, recording the
    path as node and slot pairs. A move that loses in two is skipped while the
    node has one that does not, and ends the descent at -1 for the mover. One
    program per board; `SLOT` holds the root move of this simulation."""
    row = tl.program_id(0)
    edges = tl.arange(0, BLOCK)
    node = tl.full((), 0, tl.int32)
    slot = tl.load(SLOT + row).to(tl.int32)
    active = tl.load(LIVE + row)
    depth = tl.full((), 0, tl.int32)
    expand = tl.full((), 0, tl.int1)
    value = tl.full((), 0.0, tl.float32)
    frontier_node = tl.full((), 0, tl.int32) + node
    frontier_slot = tl.full((), 0, tl.int32) + slot
    while active:
        base = (node * B + row) * A
        if depth > 0:
            n_legal = tl.load(NLEGAL + node * B + row)
            legal = (edges < n_legal) & (edges < A)
            n = tl.load(COUNT + base + edges, legal, 0).to(tl.int32)
            total = tl.load(TOTAL + base + edges, legal, 0.0)
            q = tl.load(Q + base + edges, legal, 0.0).to(tl.float32)
            win = tl.load(WIN + base + edges, legal, 0)
            loss = tl.load(LOSS + base + edges, legal, 0)
            prior = tl.load(PRIOR + base + edges, legal, -float("inf")).to(tl.float32)
            mean = tl.where(n > 0, total / tl.maximum(n, 1), q)
            completed = tl.where(win, 1.0, tl.where(loss, -1.0, mean))
            open_moves = tl.sum((legal & (loss == 0)).to(tl.int32), 0)
            usable = legal & ((loss == 0) | (open_moves == 0))
            z = tl.where(usable, prior + (CV + tl.max(n, 0)) * CS * completed, -float("inf"))
            p = tl.exp(z - tl.max(z, 0))
            p = p / tl.sum(p, 0)
            deficit = tl.where(usable, p - n / (1.0 + tl.sum(n, 0)), -float("inf"))
            slot = tl.min(tl.where(deficit == tl.max(deficit, 0), edges, 2147483647), 0)
        tl.store(PN + depth * B + row, node)
        tl.store(PA + depth * B + row, slot)
        depth += 1
        win_edge = tl.load(WIN + base + slot)
        loss_edge = tl.load(LOSS + base + slot)
        child = tl.load(CHILD + base + slot).to(tl.int32)
        frontier_node = node + tl.full((), 0, tl.int32)
        frontier_slot = slot + tl.full((), 0, tl.int32)
        if win_edge:
            value = -1.0
            active = False
        elif loss_edge:
            value = 1.0
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
    tl.store(FA + row, frontier_slot)
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
        slot = tl.load(PA + depth * B + row)
        address = (node * B + row) * A + slot
        value = -value
        tl.store(COUNT + address, tl.load(COUNT + address) + 1)
        tl.store(TOTAL + address, tl.load(TOTAL + address) + value)


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
        clock: torch.Tensor,
        clock_penalty: float,
        max_plies: int,
        seed: int,
        history: RepetitionHistory | None,
    ):
        """`clock` is the environment's per-board capture clock, shared as a
        tensor so a new draw on reset reaches the captured graph."""
        self.cfg, self.n = cfg, n
        if cfg.contempt_source not in ("wdl", "q"):
            raise ValueError(f"unknown contempt draw source {cfg.contempt_source!r}")
        self.device = torch.device(device)
        self.clock, self.clock_penalty, self.max_plies = clock, clock_penalty, max_plies
        self.history = history
        d = self.device
        sims, cands = cfg.sims, cfg.candidates
        assert cfg.cheap_sims <= sims and cfg.cheap_candidates <= cands and cands <= SLOTS
        self.capacity = cfg.reuse_nodes + sims + 1
        C, S = self.capacity, SLOTS
        self.rows = torch.arange(n, device=d)
        self.slots = torch.arange(cands, device=d)[None]
        self.lanes = torch.arange(S, device=d)[None]
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
        # Node arena: per node its position and, per slot, one legal move.
        self.boards = torch.zeros(C, n, N_SQUARES, dtype=torch.int8, device=d)
        self.since = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.plies = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.actions = torch.zeros(C, n, S, dtype=torch.int16, device=d)
        self.n_legal = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.prior = torch.zeros(C, n, S, dtype=torch.float16, device=d)
        self.net_q = torch.zeros(C, n, S, dtype=torch.bfloat16, device=d)
        self.wins = torch.zeros(C, n, S, dtype=torch.bool, device=d)
        self.losses = torch.zeros(C, n, S, dtype=torch.bool, device=d)
        self.count = torch.zeros(C, n, S, dtype=torch.int32, device=d)
        self.total = torch.zeros(C, n, S, dtype=torch.float32, device=d)
        self.child = torch.full((C, n, S), -1, dtype=torch.int16, device=d)
        self.terminal = torch.zeros(C, n, dtype=torch.bool, device=d)
        self.value = torch.zeros(C, n, dtype=torch.float32, device=d)
        # WDL draw probability of each node. Unlike the action-Q proxy, it is
        # attached to the state and survives tree reuse (DESIGN.md item 17).
        self.node_draw = torch.zeros(C, n, dtype=torch.float32, device=d)
        self.hash = torch.zeros(C, n, dtype=torch.int64, device=d)
        # The search control's ranking score and predicted regret of every node (DESIGN.md item 33).
        self.rank = torch.zeros(C, n, dtype=torch.float32, device=d)
        self.regret = torch.zeros(C, n, dtype=torch.float32, device=d)
        # The root's prior and Q at full precision, for the target and the completed Q.
        self.root_logp = torch.zeros(n, S, dtype=torch.float32, device=d)
        self.root_q = torch.zeros(n, S, dtype=torch.float32, device=d)
        self.root_draw = torch.zeros(n, S, dtype=torch.float32, device=d)
        # Per-simulation scratch.
        self.path_node = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.path_slot = torch.zeros(C, n, dtype=torch.int32, device=d)
        self.path_depth = torch.arange(C, device=d)[:, None]
        self.path_length = torch.zeros(n, dtype=torch.int32, device=d)
        self.frontier_node = torch.zeros(n, dtype=torch.int32, device=d)
        self.frontier_slot = torch.zeros(n, dtype=torch.int32, device=d)
        self.expand = torch.zeros(n, dtype=torch.bool, device=d)
        self.leaf_value = torch.zeros(n, dtype=torch.float32, device=d)
        self.node_count = torch.ones(n, dtype=torch.int32, device=d)
        self.exact_scratch = torch.zeros(n, N_SQUARES, dtype=torch.int32, device=d)
        self.loss_scratch = torch.zeros(n, kernels.LOSS_SCRATCH, dtype=torch.int32, device=d)
        self.win3_scratch = torch.zeros(n, kernels.WIN3_SCRATCH, dtype=torch.int32, device=d)
        self.flags = torch.zeros(3, n, dtype=torch.int8, device=d)
        self.safe_board = torch.tensor(initial_board(), dtype=torch.int8, device=d)
        # Tactics of the node being expanded, and the root's packed labels for the trainer.
        self.win_flags = torch.zeros(n, N_ACTIONS, dtype=torch.bool, device=d)
        self.loss_flags = torch.zeros(n, N_ACTIONS, dtype=torch.int8, device=d)
        self.win3_flags = torch.zeros(n, N_ACTIONS, dtype=torch.int8, device=d)
        self.root_tactics = torch.zeros(n, S, dtype=torch.uint8, device=d)
        # Root state, in slot space.
        self.live = torch.zeros(n, dtype=torch.bool, device=d)
        self.root_excluded = torch.zeros(n, S, dtype=torch.bool, device=d)
        self.root_win1 = torch.zeros(n, S, dtype=torch.bool, device=d)
        self.score0 = torch.zeros(n, S, dtype=torch.float32, device=d)
        self.survivors = torch.zeros(n, cands, dtype=torch.long, device=d)
        self.previous_m = torch.zeros(n, dtype=torch.long, device=d)
        self.schedule = torch.zeros(n, sims, 2, dtype=torch.long, device=d)
        self.sim_index = torch.zeros((), dtype=torch.long, device=d)
        # Reuse.
        self.kept = torch.zeros(n, dtype=torch.bool, device=d)
        self.played_slot = torch.zeros(n, dtype=torch.long, device=d)
        self.played_done = torch.ones(n, dtype=torch.bool, device=d)
        self.pending = False
        self.stats = torch.zeros(4, dtype=torch.int64, device=d)  # kept, calls, repetition draws, expansions
        self.graph = None
        self.evaluate = None

    # ------------------------------------------------------------------
    def advance(self, done: torch.Tensor) -> None:
        """The move returned by the last search was played; the next call
        re-roots on it unless the game ended."""
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
        """+1 for a move that wins, -1 for one that loses in two, else the mean
        backed-up value or the network's Q when unvisited."""
        n, total = self.count[node, self.rows], self.total[node, self.rows]
        network = torch.where((node == 0)[:, None], self.root_q, self.net_q[node, self.rows].float())
        q = torch.where(n > 0, total / n.clamp_min(1), network)
        q = torch.where(self.losses[node, self.rows], -1.0, q)
        return torch.where(self.wins[node, self.rows], 1.0, q)

    def _sigma(self, visits: torch.Tensor) -> torch.Tensor:
        return (self.cfg.c_visit + visits.amax(-1, keepdim=True)) * self.cfg.c_scale

    def _store(self, ids: torch.Tensor, mask: torch.Tensor, **fields) -> None:
        """Masked per-row write of node fields at `ids`."""
        rows = self.rows
        for name, value in fields.items():
            arena = getattr(self, name)
            old = arena[ids, rows]
            m = mask.view(-1, *([1] * (value.ndim - 1)))
            arena[ids, rows] = torch.where(m, value.to(arena.dtype), old)

    def _node_tactics(self, board, since, legal, value):
        """Wins at once and losses in two of freshly evaluated positions under
        their capture clock (`since` plies since the last capture, the boards'
        clocks), and their value: +1 with a winning move, -1 when every move
        loses in two or no move exists, else `value`."""
        kernels.exact_wins(board, legal, self.win_flags, self.exact_scratch)
        kernels.loses_in_two(board, legal, since, self.clock, self.loss_flags, self.loss_scratch)
        win1 = self.win_flags & legal
        loss2 = self.loss_flags.bool() & legal
        all_lose = loss2.sum(-1) == legal.sum(-1)
        value = torch.where(all_lose, -1.0, value.float())
        return win1, loss2, all_lose, torch.where(win1.any(-1), 1.0, value)

    def _compact(self, board, since, ply) -> None:
        """Re-root every tree on its played child. The child's subtree is kept when
        the child exists, is not terminal and holds the board now in play: its
        reachable nodes, at most `reuse_nodes` of them in id order, move to the
        lowest ids with their visits, values and child links; every other node
        is cleared. Parents always have smaller ids than their children, so one
        pass over the ids in order finds the whole subtree."""
        n, C, R, rows = self.n, self.capacity, self.cfg.reuse_nodes, self.rows
        d = self.device
        root = self.child[0, rows, self.played_slot].long()
        safe = root.clamp_min(0)
        valid = (root >= 0) & ~self.played_done & self.live & ~self.terminal[safe, rows]
        valid &= (self.boards[safe, rows] == board).all(-1)
        valid &= (self.since[safe, rows] == since) & (self.plies[safe, rows] == ply)
        # Marking: node i is reachable when a kept node listed it as a child. `marked`
        # is flat over (node, row) with one spare slot that absorbs unused writes.
        marked = torch.zeros(C * n + 1, dtype=torch.bool, device=d)
        marked[safe * n + rows] = valid
        dest = torch.full((C, n), -1, dtype=torch.long, device=d)
        size = torch.zeros(n, dtype=torch.long, device=d)
        flat_rows = rows[:, None].expand(n, SLOTS)
        spare = torch.full((n, SLOTS), C * n, dtype=torch.long, device=d)
        ones = torch.ones(n * SLOTS, dtype=torch.bool, device=d)
        for node in range(1, C):
            take = marked[node * n : (node + 1) * n] & (size < R)
            dest[node] = torch.where(take, size, -1)
            size += take
            kids = self.child[node].long()
            target = torch.where(take[:, None] & (kids >= 0), kids * n + flat_rows, spare)
            marked.scatter_(0, target.view(-1), ones)
        # Source node of every destination slot, -1 when the slot stays empty.
        source = torch.full((R * n + 1,), -1, dtype=torch.long, device=d)
        slot = torch.where(dest >= 0, dest * n + rows[None], R * n)
        source.scatter_(0, slot.view(-1), torch.arange(C, device=d)[:, None].expand(C, n).reshape(-1))
        source = source[: R * n].view(R, n)
        src = source.clamp_min(0)
        rows_r = rows[None].expand(R, n)
        fields = (
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
            "terminal",
            "value",
            "node_draw",
            "hash",
            "rank",
            "regret",
        )
        for name in fields:
            arena = getattr(self, name)
            arena[:R] = arena[src, rows_r]
        kids = self.child[src, rows_r].long()
        mapped = torch.where(kids >= 0, dest[kids.clamp_min(0), rows_r[..., None]], -1)
        self.child[:R] = mapped.to(self.child.dtype)
        dead = torch.arange(C, device=d)[:, None] >= size[None]
        self.count.masked_fill_(dead[..., None], 0)
        self.total.masked_fill_(dead[..., None], 0.0)
        self.child.masked_fill_(dead[..., None], -1)
        self.terminal.masked_fill_(dead, False)
        self.node_count.copy_(size.clamp_min(1).to(self.node_count.dtype))
        self.kept.copy_(valid)

    def _evaluate(self, board, since, ply):
        """The actor on a batch of positions, plus the search control's ranking
        score and predicted regret, zero for an actor without them."""
        out = self.evaluate(board, since, ply, self.clock)
        if len(out) == 6:
            zeros = torch.zeros_like(out[2])
            return (*out, zeros, zeros)
        return out

    @torch.no_grad()
    def sample_node(self, gumbels: torch.Tensor | None = None):
        """One non-root, nonterminal node per board sampled from softmax(rank)
        by Gumbel-max; boards without an eligible allocated node return -inf."""
        if gumbels is None:
            u = torch.rand(self.rank.shape, generator=self.rng, device=self.device).clamp_(1e-12, 1.0)
            gumbels = -(-u.log()).log()
        ids = torch.arange(self.capacity, device=self.device)[:, None]
        valid = (ids >= 1) & (ids < self.node_count[None]) & ~self.terminal
        score, node = (self.rank + gumbels).masked_fill(~valid, -torch.inf).max(0)
        rows = self.rows
        return (
            score,
            self.regret[node, rows],
            self.boards[node, rows],
            self.since[node, rows],
            self.plies[node, rows],
        )

    def _root_rank(self, score: torch.Tensor, excluded_rank: float = -torch.inf) -> torch.Tensor:
        """Rank legal root moves: immediate wins, longer exact wins, then search scores."""
        valid = self.lanes < self.n_legal[0][:, None]
        return (
            score.masked_fill(self.wins[0], WINNING_RANK)
            .masked_fill(self.root_win1, torch.inf)
            .masked_fill(self.root_excluded, excluded_rank)
            .masked_fill(~valid, -torch.inf)
        )

    def _start(self, board, since, ply, legal, logits, q, value, draw, rank, regret, gumbels, mode: str) -> None:
        """Seed the root: its exact tactics, prior, configured draw signal and
        candidate admission, which forces winning moves in and losing ones out."""
        self.live.copy_(legal.any(-1))
        if self.pending:
            self._compact(board, since, ply)
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
        win1, loss2, _, value = self._node_tactics(board, since, legal, value)
        kernels.wins_in_three(board, legal, since, self.clock, self.win3_flags, self.win3_scratch)
        win3 = self.win3_flags.bool() & legal
        actions, count = compact(legal)
        valid = self.lanes < count[:, None]
        self.root_tactics.copy_(pack_tactics(win1, loss2, win3).gather(1, actions))
        logp = raw_policy(logits, legal)[1].gather(1, actions)
        q = q.float().gather(1, actions)
        wins = (win1 | win3).gather(1, actions) & valid
        losses = loss2.gather(1, actions) & valid
        excluded = losses & (valid & ~losses).any(-1, keepdim=True)
        self.root_excluded.copy_(excluded)
        self.root_win1.copy_(win1.gather(1, actions) & valid)
        self.root_logp.copy_(logp)
        self.root_q.copy_(q)
        if self.cfg.contempt_source == "wdl":
            if draw.ndim != 1:
                raise ValueError(f"WDL contempt expects one draw probability per state, got {tuple(draw.shape)}")
            self.node_draw[0].copy_(draw.float())
        else:
            if draw.ndim != 2:
                raise ValueError(f"Q contempt expects one draw mass per action, got {tuple(draw.shape)}")
            self.root_draw.copy_(draw.float().gather(1, actions))
        self.boards[0].copy_(board)
        self.since[0].copy_(since)
        self.plies[0].copy_(ply)
        self.actions[0].copy_(actions)
        self.n_legal[0].copy_(count)
        self.prior[0].copy_(logp)
        self.net_q[0].copy_(q)
        self.wins[0].copy_(wins)
        self.losses[0].copy_(losses)
        self.terminal[0].zero_()
        self.value[0].copy_(torch.where(wins.any(-1), 1.0, value))
        self.hash[0].copy_(self.hasher(board, ply))
        self.rank[0].copy_(rank)
        self.regret[0].copy_(regret)
        temperature = torch.where(ply < self.cfg.temperature_plies, self.cfg.temperature, 1.0)
        self.score0.copy_((logp + temperature[:, None] * gumbels).masked_fill(~valid, -torch.inf))
        admission = self._root_rank(self.score0, LOSING_RANK)
        self.survivors.copy_(admission.topk(self.cfg.candidates, -1).indices)
        self.previous_m.copy_(count.clamp(max=self.cands[mode]))
        self.schedule.copy_(self.plans[mode][self.previous_m])
        self.sim_index.zero_()

    def _simulate(self) -> None:
        rows = self.rows
        root = torch.zeros_like(rows)
        sim = self.sim_index.expand_as(rows)
        m, slot = self.schedule[rows, sim].unbind(-1)
        score = self._root_rank(self.score0 + self._sigma(self.count[0]) * self._completed(root))
        rank = score.gather(1, self.survivors).masked_fill(self.slots >= self.previous_m[:, None], -torch.inf)
        order = rank.argsort(dim=-1, descending=True, stable=True)
        self.survivors.copy_(
            torch.where((m < self.previous_m)[:, None], self.survivors.gather(1, order), self.survivors)
        )
        self.previous_m.copy_(m)
        root_slot = self.survivors.gather(1, slot[:, None]).squeeze(1)
        walk_kernel[(self.n,)](
            self.prior,
            self.net_q,
            self.n_legal,
            self.wins,
            self.losses,
            self.count,
            self.total,
            self.child,
            self.terminal,
            self.value,
            root_slot,
            self.live,
            self.path_node,
            self.path_slot,
            self.path_length,
            self.frontier_node,
            self.frontier_slot,
            self.expand,
            self.leaf_value,
            self.n,
            SLOTS,
            self.cfg.c_visit,
            self.cfg.c_scale,
            BLOCK_S,
        )
        fn, fs = self.frontier_node.long(), self.frontier_slot.long()
        active = self.expand
        board = torch.where(active[:, None], self.boards[fn, rows], self.safe_board).contiguous()
        since = torch.where(active, self.since[fn, rows], 0).contiguous()
        ply = torch.where(active, self.plies[fn, rows], 0).contiguous()
        act = torch.where(active, self.actions[fn, rows, fs].int(), SAFE_ACTION).contiguous()
        kernels.apply(board, act, since, ply, self.flags[0], self.flags[1], self.flags[2])
        logits, q, value, draw_prob, legal, legal_count, rank, regret = self._evaluate(board, since, ply)
        win1, loss2, all_lose, value = self._node_tactics(board, since, legal, value)
        has_move = legal_count > 0
        clock = clock_reached(since, self.clock) & has_move & (ply < self.max_plies)
        value = torch.where(clock, -self.clock_penalty * material_lead(board), value)
        terminal = ~has_move | (ply >= self.max_plies) | clock | all_lose
        # Twofold repetition over the game history plus the path is a draw.
        leaf_hash = self.hasher(board, ply)
        on_path = self.path_depth < self.path_length[None]
        path_ids = torch.where(on_path, self.path_node, 0).long()
        matches = ((self.hash[path_ids, rows] == leaf_hash[None]) & on_path & (self.path_depth > 0)).sum(0)
        earlier = matches + (self.history.occurrences(leaf_hash) if self.history is not None else 0)
        repeated = self.cfg.repetition_draw & active & ~terminal & (earlier >= 1)
        self.stats[2].add_(repeated.sum())
        self.stats[3].add_(active.sum())
        terminal = terminal | repeated
        value = torch.where(repeated, 0.0, value)
        actions, count = compact(legal)
        valid = self.lanes < count[:, None]
        new_id = self.node_count.long()
        cached = active & (new_id < self.capacity)
        safe_id = new_id.clamp(max=self.capacity - 1)
        fields = dict(
            boards=board,
            since=since,
            plies=ply,
            actions=actions,
            n_legal=count,
            prior=raw_policy(logits, legal)[1].gather(1, actions),
            net_q=q.float().gather(1, actions),
            wins=win1.gather(1, actions) & valid,
            losses=loss2.gather(1, actions) & valid,
            terminal=terminal,
            value=value,
            hash=leaf_hash,
            rank=rank,
            regret=regret,
        )
        if self.cfg.contempt_source == "wdl":
            # A node the rules settle has an exact draw probability: 1 on a
            # clock or repetition draw, 0 with a winning move, no move at all
            # or every move losing; the head's estimate elsewhere.
            settled = win1.any(-1) | ~has_move | all_lose
            fields["node_draw"] = torch.where(clock | repeated, 1.0, torch.where(settled, 0.0, draw_prob.float()))
        self._store(safe_id, cached, **fields)
        self.node_count.add_(cached.int())
        link = self.child[fn, rows, fs]
        self.child[fn, rows, fs] = torch.where(cached, safe_id.to(self.child.dtype), link)
        self.leaf_value.copy_(torch.where(active, value, self.leaf_value))
        backup_kernel[(self.n,)](
            self.count,
            self.total,
            self.path_node,
            self.path_slot,
            self.path_length,
            self.leaf_value,
            self.n,
            SLOTS,
        )
        self.sim_index.add_(1)

    @torch.no_grad()
    def __call__(self, board, since_capture, ply, legal, evaluate, full: bool = True):
        """One search on every board with the actor `evaluate`.

        Returns a `Root`."""
        if evaluate is not self.evaluate:
            # A different actor invalidates the captured simulation.
            self.evaluate, self.graph = evaluate, None
        mode = "full" if full else "cheap"
        legal = legal.bool()
        root = self._evaluate(board, since_capture, ply)
        logits, q, value, draw, _, _, rank, regret = root
        u = torch.rand((self.n, SLOTS), generator=self.rng, device=self.device).clamp_(1e-12, 1.0)
        gumbels = -(-u.log()).log()
        use_graph = self.device.type == "cuda"
        if use_graph and self.graph is None:
            self.forget()
            self._start(board, since_capture, ply, legal, logits, q, value, draw, rank, regret, gumbels, mode)
            self._simulate()
            torch.cuda.synchronize()
            self._start(board, since_capture, ply, legal, logits, q, value, draw, rank, regret, gumbels, mode)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self._simulate()
        self._start(board, since_capture, ply, legal, logits, q, value, draw, rank, regret, gumbels, mode)
        for _ in range(self.sims[mode]):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._simulate()
        root = torch.zeros(self.n, dtype=torch.long, device=self.device)
        qbar = self._completed(root)
        if self.cfg.contempt_source == "wdl":
            # A root move charges its expanded child's state draw probability;
            # an unexpanded move falls back to the root state's probability.
            children = self.child[0].long()
            child_draw = self.node_draw[children.clamp_min(0), self.rows[:, None]]
            self.root_draw.copy_(torch.where(children >= 0, child_draw, self.node_draw[0, :, None]))
        # Contempt (DESIGN.md item 17): for the move choice and the policy
        # target every move loses `contempt` times its draw mass, as at play
        # time; the Q and value labels stay the zero-sum completed Q.
        qplay = qbar - self.cfg.contempt * self.root_draw
        sigma = self._sigma(self.count[0])
        valid = self.lanes < self.n_legal[0][:, None]
        # A move that loses in two is out of the improved policy, so the target gives it no mass.
        target, _ = raw_policy(self.root_logp + sigma * qplay, valid & ~self.root_excluded)
        actions = self.actions[0].long()
        score = self._root_rank(self.score0 + sigma * qplay).gather(1, self.survivors)
        score = score.masked_fill(self.slots >= self.previous_m[:, None], -torch.inf)
        self.played_slot.copy_(self.survivors.gather(1, score.argmax(-1)[:, None]).squeeze(1))
        action = actions.gather(1, self.played_slot[:, None]).squeeze(1)
        value = torch.where(self.live, (target * qbar).sum(-1), -1.0)
        candidates = actions.gather(1, self.survivors)
        candidate_q = qbar.gather(1, self.survivors)
        visited = self.count[0].gather(1, self.survivors) > 0
        played_q = qbar.gather(1, self.played_slot[:, None]).squeeze(1)
        return Root(
            self.actions[0], target, value, action, candidates, candidate_q, visited, self.root_tactics, played_q, rank
        )
