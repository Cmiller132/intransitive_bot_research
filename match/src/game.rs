//! One game between two players.

use engine::notation::action_name;
use engine::{apply, Action, Outcome, Rules, State, GOAL};
use rand::Rng;
use serde::Serialize;

use crate::player::{Clock, History, Player};

/// Random legal plies played before the players take over, so deterministic
/// players do not repeat one game.
#[derive(Clone, Debug, Default)]
pub struct Opening {
    pub plies: Vec<Action>,
}

impl Opening {
    /// `plies` random legal moves that leave the game ongoing.
    pub fn random<R: Rng>(rules: &Rules, plies: usize, rng: &mut R) -> Opening {
        loop {
            let mut state = State::initial();
            let mut chosen = Vec::with_capacity(plies);
            let mut ongoing = true;
            for _ in 0..plies {
                let legal = state.legal_actions();
                let action = legal[rng.random_range(0..legal.len())];
                let (child, outcome) = apply(rules, &state, action);
                if outcome != Outcome::Ongoing {
                    ongoing = false;
                    break;
                }
                chosen.push(action);
                state = child;
            }
            if ongoing {
                return Opening { plies: chosen };
            }
        }
    }

    /// The position after the opening plies.
    pub fn position(&self, rules: &Rules) -> State {
        let mut state = State::initial();
        for &action in &self.plies {
            state = apply(rules, &state, action).0;
        }
        state
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum End {
    Goal,
    Elimination,
    Stalemate,
    CaptureClock,
}

/// Result of one game; `winner` is 0 for the first player, 1 for the second,
/// `None` for a draw.
#[derive(Clone, Debug, Serialize)]
pub struct GameRecord {
    pub first: String,
    pub second: String,
    pub winner: Option<u8>,
    pub end: End,
    pub plies: u32,
    /// Every move in the canonical frame of its mover, opening included.
    pub moves: Vec<String>,
}

/// Play one game. `first` moves first from the opening's final position.
pub fn play(
    rules: &Rules,
    first: &mut dyn Player,
    second: &mut dyn Player,
    opening: &Opening,
    clock: Clock,
) -> GameRecord {
    first.new_game();
    second.new_game();
    let mut state = State::initial();
    let mut history = History::new();
    history.insert(state.key(), 1);
    let mut moves = Vec::new();
    for &action in &opening.plies {
        moves.push(action_name(action));
        state = apply(rules, &state, action).0;
        *history.entry(state.key()).or_insert(0) += 1;
    }
    let mut mover = 0u8;
    loop {
        let player: &mut dyn Player = if mover == 0 {
            &mut *first
        } else {
            &mut *second
        };
        let action = player.choose(&state, &history, clock);
        assert!(
            state.legal_mask()[action as usize],
            "{} played an illegal move",
            player.name()
        );
        moves.push(action_name(action));
        let target = engine::from_to(action).1;
        let (child, outcome) = apply(rules, &state, action);
        let end = match outcome {
            Outcome::Ongoing => None,
            Outcome::Draw => Some((None, End::CaptureClock)),
            Outcome::Win => Some((
                Some(mover),
                if target == GOAL {
                    End::Goal
                } else if child.own_count() == 0 {
                    End::Elimination
                } else {
                    End::Stalemate
                },
            )),
        };
        if let Some((winner, end)) = end {
            return GameRecord {
                first: first.name().to_owned(),
                second: second.name().to_owned(),
                winner,
                end,
                plies: child.ply,
                moves,
            };
        }
        state = child;
        *history.entry(state.key()).or_insert(0) += 1;
        mover = 1 - mover;
    }
}
