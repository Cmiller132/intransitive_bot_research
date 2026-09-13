//! The rules of Intransitive in the canonical (mover's) frame.

pub mod board;
pub mod notation;
pub mod rules;
pub mod tactics;

#[cfg(feature = "python")]
mod python;

pub use board::{
    action, codes, flip, from_codes, from_to, initial_board, mirror_anti, step, Action, Board,
    Cell, Dir, Piece, Square, DIRS, GOAL, N_ACTIONS, N_DIRS, N_SQUARES,
};
pub use rules::{apply, beats, legal_mask, ActionMask, Outcome, PositionKey, Rules, State};
