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
        self.legal_mask()
            .iter()
            .enumerate()
            .filter_map(|(action, &legal)| legal.then_some(action as Action))
            .collect()
    }

    /// True when the mover has no legal move.
    pub fn is_stalemated(&self) -> bool {
        !self.legal_mask().iter().any(|&legal| legal)
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
