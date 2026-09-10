//! One game between two players. Records spell every move in the absolute
//! frame (Blue's home a1) and carry each mover's search summary.

use engine::{apply, Action, Outcome, Rules, State, GOAL};
use rand::Rng;
use serde::Serialize;

use crate::player::{Clock, History, MoveInfo, Player};
use crate::rpsi::Frame;

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

    /// The opening plies as absolute move tokens.
    pub fn tokens(&self) -> Vec<String> {
        let mut frame = Frame::with_home(0, false);
        self.plies
            .iter()
            .map(|&action| {
                let token = frame.spell(action);
                frame.red_to_move = !frame.red_to_move;
                token
            })
            .collect()
    }
}

/// A mover's search summary as recorded: absolute move tokens, values from
/// the mover's point of view.
#[derive(Clone, Debug, Serialize)]
pub struct MoveStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search: Option<serde_json::Value>,
    pub sims: u32,
    pub value: f32,
    pub plies_left: Option<f32>,
    pub q: f32,
    pub pi: f32,
    pub exact_win: bool,
    pub top: Vec<TopMove>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TopMove {
    #[serde(rename = "move")]
    pub action: String,
    pub visits: u32,
    pub q: f32,
}

impl MoveStats {
    /// `info` spelled in `frame`, the mover's frame.
    pub fn new(info: &MoveInfo, frame: &Frame) -> MoveStats {
        MoveStats {
            search: None,
            sims: info.sims,
            value: info.value,
            plies_left: info.plies_left,
            q: info.q,
            pi: info.pi,
            exact_win: info.exact_win,
            top: info
                .top
                .iter()
                .map(|&(action, visits, q)| TopMove {
                    action: frame.spell(action),
                    visits,
                    q,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum End {
    Goal,
    Elimination,
    Stalemate,
    CaptureClock,
    /// The loser produced no legal move (an external engine failed).
    Forfeit,
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
    /// Every move as an absolute token, opening included.
    pub moves: Vec<String>,
    /// The mover's search summary for each move, when it exposes one.
    pub stats: Vec<Option<MoveStats>>,
}

/// Play one game. `first` moves first from the opening's final position.
pub fn play(
    rules: &Rules,
    first: &mut dyn Player,
    second: &mut dyn Player,
    opening: &Opening,
    clock: Clock,
) -> GameRecord {
    play_observed(rules, first, second, opening, clock, &mut |_, _, _| {})
}

/// `play`, telling `observe` every played move as `(ply, token, stats)`
/// with the ply counted from the start of the game.
pub fn play_observed(
    rules: &Rules,
    first: &mut dyn Player,
    second: &mut dyn Player,
    opening: &Opening,
    clock: Clock,
    observe: &mut dyn FnMut(u32, &str, Option<&MoveStats>),
) -> GameRecord {
    play_observed_clocks(rules, first, second, opening, [clock; 2], observe)
}

/// Game runner with budgets attached to the first and second player's identity.
pub(crate) fn play_observed_clocks(
    rules: &Rules,
    first: &mut dyn Player,
    second: &mut dyn Player,
    opening: &Opening,
    clocks: [Clock; 2],
    observe: &mut dyn FnMut(u32, &str, Option<&MoveStats>),
) -> GameRecord {
    first.new_game();
    second.new_game();
    let mut state = State::initial();
    let mut history = History::new();
    history.insert(state.key(), 1);
    let mut frame = Frame::with_home(0, false);
    let mut moves = Vec::new();
    let mut stats = Vec::new();
    for &action in &opening.plies {
        moves.push(frame.spell(action));
        stats.push(None);
        frame.red_to_move = !frame.red_to_move;
        state = apply(rules, &state, action).0;
        *history.entry(state.key()).or_insert(0) += 1;
        first.observe(action);
        second.observe(action);
    }
    let mut mover = 0u8;
    loop {
        let player: &mut dyn Player = if mover == 0 {
            &mut *first
        } else {
            &mut *second
        };
        let action = player.choose(&state, &history, clocks[mover as usize]);
        if player.forfeited() {
            return GameRecord {
                first: first.name().to_owned(),
                second: second.name().to_owned(),
                winner: Some(1 - mover),
                end: End::Forfeit,
                plies: state.ply,
                moves,
                stats,
            };
        }
        assert!(
            state.legal_mask()[action as usize],
            "{} played an illegal move",
            player.name()
        );
        let token = frame.spell(action);
        let info = player.info().map(|info| {
            let mut stats = MoveStats::new(&info, &frame);
            stats.search = player.search_details();
            stats
        });
        observe(moves.len() as u32, &token, info.as_ref());
        moves.push(token);
        stats.push(info);
        frame.red_to_move = !frame.red_to_move;
        first.observe(action);
        second.observe(action);
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
                stats,
            };
        }
        state = child;
        *history.entry(state.key()).or_insert(0) += 1;
        mover = 1 - mover;
    }
}

#[cfg(test)]
mod tests {
    use engine::notation::parse_squares;
    use engine::Cell;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::rpsi::FirstMove;

    #[test]
    fn records_spell_moves_in_the_absolute_frame() {
        let opening = Opening::random(&Rules::SITE, 4, &mut StdRng::seed_from_u64(2));
        let mut seen = Vec::new();
        let record = play_observed(
            &Rules::SITE,
            &mut FirstMove,
            &mut FirstMove,
            &opening,
            Clock::Sims(1),
            &mut |ply, token, stats| seen.push((ply, token.to_owned(), stats.is_none())),
        );
        assert_eq!(record.moves[..4], opening.tokens()[..]);
        assert_eq!(record.stats.len(), record.moves.len());
        let initial = State::initial();
        let (from, _) = parse_squares(&record.moves[1]).unwrap();
        assert!(matches!(initial.board[from as usize], Cell::Enemy(_)));
        assert_eq!(seen[0].0, 4);
        assert_eq!(seen[0].1, record.moves[4]);
        assert!(seen.iter().all(|s| s.2));
        assert_eq!(seen.len() + 4, record.moves.len());
    }
}
