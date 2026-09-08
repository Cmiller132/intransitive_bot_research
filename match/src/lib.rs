//! Games between two players: the runner, the paired eval and the site adapter.

pub mod eval;
pub mod game;
pub mod player;
pub mod rpsi;

pub use eval::{eval, EvalConfig, PlayerFactory, Report};
pub use game::{play, End, GameRecord, Opening};
pub use player::{Clock, History, Player};
pub use rpsi::Session;
