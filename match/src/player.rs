//! The interface between the runner and anything that plays.

use std::time::Duration;

use engine::{Action, State};
use search::Info;

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

/// Summary of the search behind one move, from the mover's point of view.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoveInfo {
    pub sims: u32,
    /// Root value after the search.
    pub value: f32,
    pub plies_left: Option<f32>,
    /// Completed Q of the played move.
    pub q: f32,
    /// The played move's share of the root visits.
    pub pi: f32,
    pub exact_win: bool,
    /// Up to five root moves by visits: `(action, visits, q)`.
    pub top: Vec<(Action, u32, f32)>,
}

/// Optional RPSI statistics in the mover's canonical frame.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Telemetry {
    pub info: MoveInfo,
    pub search: Option<serde_json::Value>,
}

impl MoveInfo {
    pub fn from_search(info: &Info, played: Action) -> MoveInfo {
        let total: u32 = info.root.iter().map(|&(_, visits, _)| visits).sum();
        let (visits, q) = info
            .root
            .iter()
            .find(|&&(action, _, _)| action == played)
            .map_or((0, 0.0), |&(_, visits, q)| (visits, q as f32));
        let mut top: Vec<(Action, u32, f32)> = info
            .root
            .iter()
            .map(|&(action, visits, q)| (action, visits, q as f32))
            .collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));
        top.truncate(5);
        MoveInfo {
            sims: info.sims,
            value: info.root_value,
            plies_left: info.plies_left,
            q,
            pi: if total > 0 {
                visits as f32 / total as f32
            } else {
                0.0
            },
            exact_win: info.exact_win,
            top,
        }
    }
}

pub trait Player {
    /// Whether the player accepts this budget unit.
    fn supports_clock(&self, _clock: Clock) -> bool {
        true
    }

    /// Enable model-specific leaf collection; unsupported players reject it.
    fn set_leaves(&mut self, _path: &std::path::Path) -> anyhow::Result<()> {
        anyhow::bail!("this player cannot collect leaves")
    }

    /// Additional search telemetry for algorithms without Gumbel visits.
    fn search_details(&self) -> Option<serde_json::Value> {
        None
    }

    /// Called before the first move of every game.
    fn new_game(&mut self);

    /// Told every move of the game, opening plies included, in the mover's
    /// own frame. Players that keep their own state ignore it.
    fn observe(&mut self, _action: Action) {}

    /// The move to play in `state`, which is always in the player's own frame
    /// and always has a legal move.
    fn choose(&mut self, state: &State, history: &History, clock: Clock) -> Action;

    /// True after `choose` when the player could not produce a move; the game
    /// is then lost by forfeit.
    fn forfeited(&self) -> bool {
        false
    }

    /// The search behind the last `choose`, for players that expose it.
    fn info(&self) -> Option<MoveInfo> {
        None
    }

    /// Name shown in reports.
    fn name(&self) -> &str;
}

impl Player for Box<dyn Player> {
    fn supports_clock(&self, clock: Clock) -> bool {
        (**self).supports_clock(clock)
    }
    fn set_leaves(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        (**self).set_leaves(path)
    }
    fn search_details(&self) -> Option<serde_json::Value> {
        (**self).search_details()
    }

    fn new_game(&mut self) {
        (**self).new_game()
    }

    fn observe(&mut self, action: Action) {
        (**self).observe(action)
    }

    fn choose(&mut self, state: &State, history: &History, clock: Clock) -> Action {
        (**self).choose(state, history, clock)
    }

    fn forfeited(&self) -> bool {
        (**self).forfeited()
    }

    fn info(&self) -> Option<MoveInfo> {
        (**self).info()
    }

    fn name(&self) -> &str {
        (**self).name()
    }
}
