//! Games between two players: the runner, the paired eval, the site adapter
//! and the seat for external engines.

pub mod analysis;
pub mod client;
pub mod eval;
pub mod game;
pub mod player;
pub mod rpsi;

pub use analysis::{Analyser, Heads};
pub use client::RpsiPlayer;
pub use eval::{eval, EvalConfig, PlayerFactory, Report};
pub use game::{
    play, play_observed, play_with_random_moves, End, GameRecord, MoveStats, Opening, RandomMoves,
};
pub use player::{Clock, History, MoveInfo, Player};
pub use rpsi::Session;
