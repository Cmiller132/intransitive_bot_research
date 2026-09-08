//! The interface between the runner and anything that plays.

use std::time::Duration;

use engine::{Action, State};

/// Positions already played in the current game with their occurrence counts,
/// including the position being asked about.
pub type History = search::History;

/// How much a player may spend on one move.
#[derive(Clone, Copy, Debug)]
pub enum Clock {
    /// Exactly this many search simulations; used by eval.
    Sims(u32),
    /// Wall time for the move; used by the site adapter.
    Time(Duration),
}

pub trait Player {
    /// Called before the first move of every game.
    fn new_game(&mut self);

    /// The move to play in `state`, which is always in the player's own frame
    /// and always has a legal move.
    fn choose(&mut self, state: &State, history: &History, clock: Clock) -> Action;

    /// Name shown in reports.
    fn name(&self) -> &str;
}

impl Player for Box<dyn Player> {
    fn new_game(&mut self) {
        (**self).new_game()
    }

    fn choose(&mut self, state: &State, history: &History, clock: Clock) -> Action {
        (**self).choose(state, history, clock)
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}
