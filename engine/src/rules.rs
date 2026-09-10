//! Legal moves, move application and game end. Everything is from the mover's
//! point of view; `apply` returns the child already flipped for the next mover.

use crate::board::{
    flip, from_to, initial_board, step, Action, Board, Cell, Piece, DIRS, GOAL, N_ACTIONS,
    N_SQUARES,
};

/// A position: the canonical board plus the two counters the rules depend on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub board: Board,
    /// Plies since the last capture (the capture clock).
    pub since_capture: u32,
    /// Plies played from the start of the game; parity says which colour moves.
    pub ply: u32,
}

/// Rule parameters that can differ between the site and a training curriculum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    /// Plies without a capture that draw; `None` disables the clock.
    pub capture_clock: Option<u32>,
}

impl Rules {
    /// The site's rules: a 200-ply capture clock.
    pub const SITE: Rules = Rules {
        capture_clock: Some(200),
    };

    /// True when `state` has reached the capture clock.
    pub fn clock_expired(&self, state: &State) -> bool {
        self.capture_clock
            .is_some_and(|clock| clock > 0 && state.since_capture >= clock)
    }
}

/// Result of a move for the side that just moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ongoing,
    Win,
    Draw,
}

/// One flag per action index.
pub type ActionMask = [bool; N_ACTIONS];

/// Legal origins by direction, partitioned by whether the target is occupied.
pub struct LegalMoves {
    captures: [u128; 8],
    quiets: [u128; 8],
}

impl LegalMoves {
    pub fn len(&self) -> usize {
        self.captures
            .iter()
            .chain(&self.quiets)
            .map(|m| m.count_ones() as usize)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, action: Action) -> bool {
        let dir = action as usize / N_SQUARES;
        dir < 8
            && (self.captures[dir] | self.quiets[dir]) & (1 << (action as usize % N_SQUARES)) != 0
    }

    /// Append captures in ascending action order.
    pub fn captures_into(&self, out: &mut Vec<Action>) {
        append_origins(&self.captures, out);
    }

    /// Append quiet moves onto any square in `targets`, in ascending action order.
    pub fn quiets_into(&self, targets: u128, out: &mut Vec<Action>) {
        let origins = std::array::from_fn(|dir| self.quiets[dir] & target_origins(targets, dir));
        append_origins(&origins, out);
    }

    /// Append all legal moves in ascending action order.
    pub fn all_into(&self, out: &mut Vec<Action>) {
        let origins = std::array::from_fn(|dir| self.captures[dir] | self.quiets[dir]);
        append_origins(&origins, out);
    }
}

fn append_origins(origins: &[u128; 8], out: &mut Vec<Action>) {
    for (dir, &mask) in origins.iter().enumerate() {
        let mut remaining = mask;
        while remaining != 0 {
            out.push((dir * N_SQUARES + remaining.trailing_zeros() as usize) as Action);
            remaining &= remaining - 1;
        }
    }
}

fn target_origins(targets: u128, dir: usize) -> u128 {
    let (dr, df) = DIRS[dir];
    let delta = dr * 9 + df;
    if delta < 0 {
        targets << -delta
    } else {
        targets >> delta
    }
}

const ORIGIN_MASKS: [u128; 8] = {
    let mut masks = [0; 8];
    let mut dir = 0;
    while dir < 8 {
        let (dr, df) = DIRS[dir];
        let mut from = 0;
        while from < 81 {
            let r = from / 9 + dr;
            let f = from % 9 + df;
            if r >= 0 && r < 9 && f >= 0 && f < 9 {
                masks[dir] |= 1 << from;
            }
            from += 1;
        }
        dir += 1;
    }
    masks
};

/// `own` beats `enemy`.
pub fn beats(own: Piece, enemy: Piece) -> bool {
    own.beats(enemy)
}

/// A position with the same board and the same side to move as another one is
/// the same position for repetition purposes; the counters do not matter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PositionKey {
    pub board: Board,
    pub red_to_move: bool,
}

impl State {
    pub fn initial() -> State {
        State {
            board: initial_board(),
            since_capture: 0,
            ply: 0,
        }
    }

    pub fn key(&self) -> PositionKey {
        PositionKey {
            board: self.board,
            red_to_move: self.ply % 2 == 1,
        }
    }

    /// True when the mover's piece at `from` may move onto `target`.
    fn may_move(&self, piece: Piece, target: Cell) -> bool {
        match target {
            Cell::Empty => true,
            Cell::Enemy(other) => piece.beats(other),
            Cell::Own(_) => false,
        }
    }

    /// Mask of the mover's legal actions.
    pub fn legal_mask(&self) -> ActionMask {
        let mut mask = [false; N_ACTIONS];
        for from in 0..N_SQUARES {
            let Cell::Own(piece) = self.board[from] else {
                continue;
            };
            for dir in 0..DIRS.len() {
                if let Some(to) = step(from as u8, dir as u8) {
                    if self.may_move(piece, self.board[to as usize]) {
                        mask[dir * N_SQUARES + from] = true;
                    }
                }
            }
        }
        mask
    }

    /// Legal action indices in ascending order.
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut out = Vec::with_capacity(80);
        self.legal_actions_into(&mut out);
        out
    }

    /// Append the legal action indices to `out` in ascending order, without
    /// allocating: the form a search calling this millions of times uses.
    pub fn legal_actions_into(&self, out: &mut Vec<Action>) {
        self.legal_moves().all_into(out);
    }

    /// Build compact capture and quiet masks for staged move generation.
    pub fn legal_moves(&self) -> LegalMoves {
        let mut own = [0u128; 3];
        let mut enemy = [0u128; 3];
        for (square, &cell) in self.board.iter().enumerate() {
            match cell {
                Cell::Own(piece) => own[piece.code() as usize - 1] |= 1 << square,
                Cell::Enemy(piece) => enemy[piece.code() as usize - 1] |= 1 << square,
                Cell::Empty => {}
            }
        }
        let mover = own[0] | own[1] | own[2];
        let empty = !(mover | enemy[0] | enemy[1] | enemy[2]) & ((1u128 << 81) - 1);
        let mut moves = LegalMoves {
            captures: [0; 8],
            quiets: [0; 8],
        };
        for (dir, &edge) in ORIGIN_MASKS.iter().enumerate() {
            moves.quiets[dir] = mover & target_origins(empty, dir) & edge;
            for (piece, &origins) in own.iter().enumerate() {
                moves.captures[dir] |= origins & target_origins(enemy[(piece + 2) % 3], dir) & edge;
            }
        }
        moves
    }

    /// True when the mover has no legal move.
    pub fn is_stalemated(&self) -> bool {
        for from in 0..N_SQUARES as u8 {
            let Cell::Own(piece) = self.board[from as usize] else {
                continue;
            };
            for dir in 0..DIRS.len() as u8 {
                if let Some(to) = step(from, dir) {
                    if self.may_move(piece, self.board[to as usize]) {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn own_count(&self) -> u32 {
        self.board.iter().filter(|cell| cell.is_own()).count() as u32
    }

    pub fn enemy_count(&self) -> u32 {
        self.board.iter().filter(|cell| cell.is_enemy()).count() as u32
    }
}

/// Mask of legal actions for `state`.
pub fn legal_mask(state: &State) -> ActionMask {
    state.legal_mask()
}

/// Play `action` under `rules`: the child in the next mover's frame and the
/// outcome for the side that just moved. Precedence: goal entry, elimination,
/// opponent stalemate, capture clock.
pub fn apply(rules: &Rules, state: &State, action: Action) -> (State, Outcome) {
    let (from, to) = from_to(action);
    let mut moved = state.board;
    let capture = moved[to as usize] != Cell::Empty;
    moved[to as usize] = moved[from as usize];
    moved[from as usize] = Cell::Empty;
    let eliminated = !moved.iter().any(|cell| cell.is_enemy());
    let child = State {
        board: flip(&moved),
        since_capture: if capture { 0 } else { state.since_capture + 1 },
        ply: state.ply + 1,
    };
    let outcome = if to == GOAL || eliminated || child.is_stalemated() {
        Outcome::Win
    } else if rules.clock_expired(&child) {
        Outcome::Draw
    } else {
        Outcome::Ongoing
    };
    (child, outcome)
}
