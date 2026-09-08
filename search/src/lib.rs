//! Generic Gumbel MCTS for Intransitive. Rules come from `engine`; network
//! outputs come from an `Evaluator`; players in `match` drive `Gumbel`.

mod gumbel;
mod tree;

pub use gumbel::{Budget, Gumbel, Info, Params};

use engine::{Action, PositionKey, State};

/// What the search needs from a network for one position, over the legal
/// actions only, all from the mover's point of view.
#[derive(Clone, Debug)]
pub struct Eval {
    pub legal: Vec<Action>,
    /// Log probability of each legal action under the prior.
    pub log_prior: Vec<f32>,
    /// Action value in [-1, 1] for each legal action.
    pub q: Vec<f32>,
    /// State value in [-1, 1].
    pub value: f32,
    /// Expected plies to the end of the game, if the model predicts it.
    pub plies_left: Option<f32>,
}

/// A network behind a batch call. Implementations own caching and threading.
/// `signs[i]` is +1 when the searching side is to move in `states[i]` and -1
/// otherwise, for evaluators that value draws differently for the two sides.
pub trait Evaluator {
    fn evaluate(&mut self, states: &[&State], signs: &[i8]) -> Vec<Eval>;
}

/// Occurrence counts of positions already played in the game, including the
/// position being searched, for repetition handling inside the tree.
pub type History = std::collections::HashMap<PositionKey, u32>;
