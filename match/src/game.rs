//! One game between two players. Records spell every move in the absolute
//! frame (Blue's home a1) and carry each mover's search summary.

use anyhow::{ensure, Result};
use engine::{apply, Action, Outcome, Rules, State, GOAL};
use rand::{rngs::StdRng, seq::index, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

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

/// Reproducible random injections at distinct zero-based game plies.
pub struct RandomMoves {
    from: u32,
    plies: Vec<u32>,
    rng: StdRng,
}

impl RandomMoves {
    pub fn new(count: u32, from: u32, to: u32, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        if count == 0 {
            return Ok(Self {
                from,
                plies: Vec::new(),
                rng,
            });
        }
        ensure!(from <= to, "random-from must not exceed random-to");
        let len = u64::from(to) - u64::from(from) + 1;
        ensure!(
            u64::from(count) <= len,
            "random-moves exceeds the ply interval"
        );
        let mut plies: Vec<u32> = index::sample(&mut rng, len as usize, count as usize)
            .iter()
            .map(|i| (u64::from(from) + i as u64) as u32)
            .collect();
        plies.sort_unstable();
        Ok(Self { from, plies, rng })
    }

    /// Planned plies; early termination or no safe action can leave slots unused.
    pub fn plies(&self) -> &[u32] {
        &self.plies
    }

    /// Uniform over legal actions that do not allow an immediately winning reply.
    pub fn choose(&mut self, rules: &Rules, state: &State) -> Option<Action> {
        if self.plies.binary_search(&state.ply).is_err() {
            return None;
        }
        self.choose_safe(rules, state)
    }

    /// Guarded uniform sampling, also used for self-play opening plies.
    pub fn choose_safe(&mut self, rules: &Rules, state: &State) -> Option<Action> {
        let legal = state.legal_actions();
        let losses = engine::tactics::loses_in_two(rules, state, &legal);
        let safe: Vec<Action> = legal
            .into_iter()
            .zip(losses)
            .filter_map(|(action, loses)| (!loses).then_some(action))
            .collect();
        if safe.is_empty() {
            None
        } else {
            Some(safe[self.rng.random_range(0..safe.len())])
        }
    }
}

/// A mover's search summary as recorded: absolute move tokens, values from
/// the mover's point of view.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum End {
    Goal,
    Elimination,
    Stalemate,
    CaptureClock,
    /// The loser produced no legal move (an external engine failed).
    Forfeit,
    PlyCap,
    Interrupted,
}

/// Result of one game; `winner` is 0 for the first player, 1 for the second,
/// `None` for a draw or a censored game, distinguished by `end`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameRecord {
    pub first: String,
    pub second: String,
    pub winner: Option<u8>,
    pub end: End,
    pub plies: u32,
    /// Every move as an absolute token, opening included.
    pub moves: Vec<String>,
    /// The mover's search summary for each move, when it exposes one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stats: Vec<Option<MoveStats>>,
    /// Zero-based indices into moves, only for injections actually played.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub random_plies: Vec<u32>,
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

/// `play` with guarded random moves; the opening precedes all injection slots.
pub fn play_with_random_moves(
    rules: &Rules,
    first: &mut dyn Player,
    second: &mut dyn Player,
    opening: &Opening,
    clock: Clock,
    random: &mut RandomMoves,
) -> Result<GameRecord> {
    ensure!(
        random.plies.is_empty() || random.from as usize >= opening.plies.len(),
        "random-from must be at or after the opening"
    );
    Ok(play_game(
        rules,
        first,
        second,
        opening,
        [clock; 2],
        Some(random),
        &mut |_, _, _| {},
    ))
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
    play_game(rules, first, second, opening, clocks, None, observe)
}

fn play_game(
    rules: &Rules,
    first: &mut dyn Player,
    second: &mut dyn Player,
    opening: &Opening,
    clocks: [Clock; 2],
    mut random: Option<&mut RandomMoves>,
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
    let mut random_plies = Vec::new();
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
        let injected = random.as_mut().and_then(|r| r.choose(rules, &state));
        let action =
            injected.unwrap_or_else(|| player.choose(&state, &history, clocks[mover as usize]));
        if injected.is_none() && player.forfeited() {
            return GameRecord {
                first: first.name().to_owned(),
                second: second.name().to_owned(),
                winner: Some(1 - mover),
                end: End::Forfeit,
                plies: state.ply,
                moves,
                stats,
                random_plies,
            };
        }
        assert!(
            state.legal_mask()[action as usize],
            "{} played an illegal move",
            player.name()
        );
        let token = frame.spell(action);
        let info = if injected.is_some() {
            random_plies.push(state.ply);
            None
        } else {
            player.info().map(|info| {
                let mut stats = MoveStats::new(&info, &frame);
                stats.search = player.search_details();
                stats
            })
        };
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
                random_plies,
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

    #[test]
    fn injection_plies_are_distinct_uniform_and_seeded() {
        let mut counts = [0usize; 5];
        for seed in 0..10_000 {
            let random = RandomMoves::new(2, 10, 14, seed).unwrap();
            assert_eq!(random.plies().len(), 2);
            assert!(random.plies()[0] < random.plies()[1]);
            assert_eq!(
                random.plies(),
                RandomMoves::new(2, 10, 14, seed).unwrap().plies()
            );
            for &ply in random.plies() {
                counts[(ply - 10) as usize] += 1;
            }
        }
        assert!(
            counts.iter().all(|&count| (3800..=4200).contains(&count)),
            "{counts:?}"
        );
        assert_eq!(
            RandomMoves::new(5, 10, 14, 0).unwrap().plies(),
            &[10, 11, 12, 13, 14]
        );
        assert!(RandomMoves::new(6, 10, 14, 0).is_err());
        assert!(RandomMoves::new(1, 14, 10, 0).is_err());
        assert_eq!(
            RandomMoves::new(1, u32::MAX, u32::MAX, 0).unwrap().plies(),
            &[u32::MAX]
        );
    }

    #[test]
    fn random_actions_exclude_immediately_losing_replies() {
        use engine::{from_to, Piece};
        let mut state = State::initial();
        state.board = [Cell::Empty; 81];
        state.board[10] = Cell::Enemy(Piece::Paper);
        state.board[20] = Cell::Own(Piece::Scissors);
        state.board[60] = Cell::Own(Piece::Rock);
        state.board[79] = Cell::Enemy(Piece::Rock);
        for seed in 0..64 {
            let action = RandomMoves::new(1, 0, 0, seed)
                .unwrap()
                .choose(&Rules::SITE, &state)
                .unwrap();
            assert_eq!(from_to(action), (20, 10));
        }
        state.board[20] = Cell::Empty;
        assert!(RandomMoves::new(1, 0, 0, 0)
            .unwrap()
            .choose(&Rules::SITE, &state)
            .is_none());
    }

    #[test]
    fn injected_moves_are_recorded_without_stale_search_stats() {
        struct Tracked {
            seen: usize,
            calls: usize,
        }
        impl Player for Tracked {
            fn new_game(&mut self) {
                self.seen = 0;
                self.calls = 0;
            }
            fn observe(&mut self, _: Action) {
                self.seen += 1;
            }
            fn choose(&mut self, state: &State, _: &History, _: Clock) -> Action {
                self.calls += 1;
                state.legal_actions()[0]
            }
            fn name(&self) -> &str {
                "tracked"
            }
            fn info(&self) -> Option<MoveInfo> {
                Some(MoveInfo {
                    sims: 1,
                    value: 0.,
                    plies_left: None,
                    q: 0.,
                    pi: 1.,
                    exact_win: false,
                    top: Vec::new(),
                })
            }
        }
        let mut a = Tracked { seen: 0, calls: 0 };
        let mut b = Tracked { seen: 0, calls: 0 };
        let mut random = RandomMoves::new(4, 0, 3, 47).unwrap();
        let record = play_with_random_moves(
            &Rules::SITE,
            &mut a,
            &mut b,
            &Opening::default(),
            Clock::Sims(1),
            &mut random,
        )
        .unwrap();
        assert_eq!(record.random_plies, [0, 1, 2, 3]);
        assert_eq!(a.seen, record.moves.len());
        assert_eq!(b.seen, record.moves.len());
        assert_eq!(a.calls + b.calls + 4, record.moves.len());
        for (i, stats) in record.stats.iter().enumerate() {
            assert_eq!(stats.is_none(), i < 4);
        }
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["random_plies"], serde_json::json!([0, 1, 2, 3]));
        let mut state = State::initial();
        let mut frame = Frame::with_home(0, false);
        for token in &record.moves {
            let action = frame.parse_move(token).unwrap();
            assert!(state.legal_mask()[action as usize]);
            state = apply(&Rules::SITE, &state, action).0;
            frame.red_to_move = !frame.red_to_move;
        }
    }

    #[test]
    fn zero_injection_preserves_the_record_and_opening_rng() {
        let opening = Opening::random(&Rules::SITE, 8, &mut StdRng::seed_from_u64(19));
        let ordinary = play(
            &Rules::SITE,
            &mut FirstMove,
            &mut FirstMove,
            &opening,
            Clock::Sims(1),
        );
        let injected = play_with_random_moves(
            &Rules::SITE,
            &mut FirstMove,
            &mut FirstMove,
            &opening,
            Clock::Sims(1),
            &mut RandomMoves::new(0, 0, 0, 123).unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(ordinary).unwrap(),
            serde_json::to_value(injected).unwrap()
        );
        assert!(play_with_random_moves(
            &Rules::SITE,
            &mut FirstMove,
            &mut FirstMove,
            &opening,
            Clock::Sims(1),
            &mut RandomMoves::new(1, 0, 20, 0).unwrap()
        )
        .is_err());
    }

    #[test]
    fn injections_stop_at_game_end_without_rescheduling_unused_plies() {
        let rules = Rules {
            capture_clock: Some(1),
        };
        let record = play_with_random_moves(
            &rules,
            &mut FirstMove,
            &mut FirstMove,
            &Opening::default(),
            Clock::Sims(1),
            &mut RandomMoves::new(51, 0, 50, 7).unwrap(),
        )
        .unwrap();
        assert_eq!(record.end, End::CaptureClock);
        assert!(record.moves.len() < 51);
        assert_eq!(record.random_plies.len(), record.moves.len());
        assert!(record.random_plies.iter().all(|&p| p < record.plies));
    }
}
