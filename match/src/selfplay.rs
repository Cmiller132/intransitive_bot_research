//! Compact search-labelled extensions of GameRecord, generation and engine replay.
use anyhow::{ensure, Result};
use engine::{apply, codes, from_codes, Action, Outcome, Rules, State};
use serde::{Deserialize, Serialize};

use crate::{rpsi::Frame, End, GameRecord, RandomMoves};

pub const SCHEMA: u32 = 3;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    /// Canonical mover-relative cell codes; moves use the absolute blue-home frame.
    pub board: Vec<u8>,
    pub ply: u32,
    pub mover: u8,
    pub since_capture: u32,
    pub blue_home: u8,
}
impl Position {
    pub fn from_state(state: &State) -> Self {
        Self {
            board: codes(&state.board).to_vec(),
            ply: state.ply,
            mover: (state.ply % 2) as u8,
            since_capture: state.since_capture,
            blue_home: 0,
        }
    }
    pub fn state(&self) -> Result<State> {
        ensure!(
            self.mover == (self.ply % 2) as u8 && self.blue_home == 0,
            "invalid initial frame"
        );
        Ok(State {
            board: from_codes(&self.board).ok_or_else(|| anyhow::anyhow!("invalid board"))?,
            ply: self.ply,
            since_capture: self.since_capture,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreKind {
    Search,
    EngineProof,
    Unlabelled,
}

#[derive(Clone, Copy, Debug)]
pub struct Label {
    pub action: Action,
    pub score: i32,
    pub depth: u8,
    pub kind: ScoreKind,
}

pub struct Decision {
    pub action: Action,
    pub label: Option<Label>,
    /// Principal variation of the labelled search, starting at the labelled move.
    pub pv: Vec<Action>,
    /// The second line's move and score, when a multipv search looked for one.
    pub alternative: Option<(Action, i32)>,
    pub nodes: u64,
    pub elapsed_ns: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Root {
    pub ply: u32,
    pub mover: u8,
    pub since_capture: u32,
    pub capture_clock: u32,
    pub board_fingerprint: u64,
    /// Canonical action at this root, not necessarily the action actually played.
    pub searched_best: Option<Action>,
    pub played_action: Option<Action>,
    pub root_score: Option<i32>,
    pub score_kind: ScoreKind,
    pub completed_depth: u8,
    pub nodes: u64,
    pub elapsed_ns: u64,
    /// Labelled line from this root; empty unless --pv-labels asked for one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pv: Vec<Action>,
    /// The best move other than `searched_best` and its score, when --multipv 2 searched
    /// for one; `played_action` is this move when the sampler chose it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alternative: Option<(Action, i32)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    #[serde(flatten)]
    pub game: GameRecord,
    pub schema: u32,
    pub game_id: u64,
    pub seed: u64,
    pub initial: Position,
    pub final_position: Position,
    pub capture_clock: u32,
    pub roots: Vec<Root>,
    pub opening_plies: u32,
    pub random_plan: Vec<u32>,
    pub random_skipped: Vec<u32>,
    pub censored: bool,
    /// Absolute player-zero result; null for censoring, zero only for a real draw.
    pub outcome: Option<i8>,
    /// First ply eligible for outcome supervision; null for censoring.
    pub outcome_after_ply: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub ply: u32,
    pub action: Option<Action>,
    pub random: bool,
    pub skipped: bool,
    pub root: Option<Root>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub opening_plies: u32,
    pub random_moves: u32,
    pub random_from: u32,
    pub random_to: u32,
    pub max_plies: u32,
}

/// SplitMix64 domain separation: game scheduling never consumes a shared RNG.
pub fn derive_seed(seed: u64, id: u64) -> u64 {
    let mut n = seed.wrapping_add(id.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15));
    n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    n = (n ^ (n >> 27)).wrapping_mul(0x94d049bb133111eb);
    n ^ (n >> 31)
}

fn fingerprint(state: &State) -> u64 {
    codes(&state.board)
        .iter()
        .fold(0xcbf29ce484222325u64, |hash, &code| {
            (hash ^ u64::from(code)).wrapping_mul(0x100000001b3)
        })
}

fn end_reason(reason: engine::rules::EndReason) -> End {
    use engine::rules::EndReason as E;
    match reason {
        E::Goal => End::Goal,
        E::Elimination => End::Elimination,
        E::Stalemate => End::Stalemate,
        E::CaptureClock => End::CaptureClock,
    }
}

impl Record {
    pub fn new(config: &Config, player: &str, game_id: u64, seed: u64) -> Result<Self> {
        ensure!(
            config.max_plies > 0 && config.opening_plies <= config.max_plies,
            "invalid opening/ply cap"
        );
        ensure!(
            config.random_moves == 0 || config.random_from >= config.opening_plies,
            "random-from must be at or after the opening"
        );
        let random = RandomMoves::new(
            config.random_moves,
            config.random_from,
            config.random_to,
            derive_seed(seed, 1),
        )?;
        let initial = Position::from_state(&State::initial());
        Ok(Self {
            game: GameRecord {
                first: player.into(),
                second: player.into(),
                winner: None,
                end: End::Interrupted,
                plies: 0,
                moves: Vec::new(),
                stats: Vec::new(),
                random_plies: Vec::new(),
            },
            schema: SCHEMA,
            game_id,
            seed,
            final_position: initial.clone(),
            initial,
            capture_clock: Rules::SITE.capture_clock.expect("site clock"),
            roots: Vec::new(),
            opening_plies: config.opening_plies,
            random_plan: random.plies().to_vec(),
            random_skipped: Vec::new(),
            censored: true,
            outcome: None,
            outcome_after_ply: None,
        })
    }

    /// Append one journalled decision and replay its played action through the engine.
    pub fn append(&mut self, state: &mut State, event: Event) -> Result<()> {
        ensure!(
            event.ply == state.ply && state.terminal(&Rules::SITE).is_none(),
            "event after game end or wrong ply"
        );
        if event.skipped {
            self.random_skipped.push(state.ply);
        }
        if let Some(root) = event.root {
            self.roots.push(root);
        }
        if let Some(action) = event.action {
            ensure!(
                state.legal_moves().contains(action),
                "illegal played action"
            );
            self.game
                .moves
                .push(Frame::with_home(0, state.ply % 2 == 1).spell(action));
            if event.random {
                self.game.random_plies.push(state.ply);
            }
            *state = apply(&Rules::SITE, state, action).0;
        }
        self.game.plies = state.ply;
        self.final_position = Position::from_state(state);
        Ok(())
    }

    pub fn finish(&mut self, state: &State, cutoff: End) {
        if let Some(end) = state.terminal(&Rules::SITE) {
            self.game.winner = end.winner;
            self.game.end = end_reason(end.reason);
            self.censored = false;
            self.outcome = Some(match end.winner {
                Some(0) => 1,
                Some(_) => -1,
                None => 0,
            });
            self.outcome_after_ply = Some(
                self.game
                    .random_plies
                    .last()
                    .map_or(self.initial.ply, |p| p + 1),
            );
        } else {
            self.game.end = cutoff;
            self.game.winner = None;
            self.censored = true;
            self.outcome = None;
            self.outcome_after_ply = None;
        }
    }

    pub fn eligible_roots(&self) -> usize {
        self.roots
            .iter()
            .filter(|r| r.ply >= 16 && r.root_score.is_some())
            .count()
    }

    /// Audit canonical boards, frame, labels, moves and real versus censored outcomes.
    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA && self.capture_clock == 200,
            "unsupported generation schema/rules"
        );
        ensure!(
            self.game.stats.is_empty(),
            "production records must use compact roots"
        );
        let mut state = self.initial.state()?;
        let mut roots = self.roots.iter().peekable();
        for token in self
            .game
            .moves
            .iter()
            .map(Some)
            .chain(std::iter::once(None))
        {
            let action = token
                .map(|t| Frame::with_home(0, state.ply % 2 == 1).parse_move(t))
                .transpose()?;
            if action.is_some() || roots.peek().is_some_and(|root| root.ply == state.ply) {
                let random = self.game.random_plies.contains(&state.ply);
                let skipped = self.random_skipped.contains(&state.ply);
                ensure!(
                    (state.ply < self.opening_plies || self.random_plan.contains(&state.ply))
                        == (random || skipped),
                    "missing random-slot provenance"
                );
                if random || skipped {
                    let legal = state.legal_actions();
                    let losses = engine::tactics::loses_in_two(&Rules::SITE, &state, &legal);
                    ensure!(
                        !random
                            || (!skipped
                                && legal
                                    .iter()
                                    .zip(&losses)
                                    .any(|(&a, &loss)| Some(a) == action && !loss)),
                        "unsafe random action"
                    );
                    ensure!(
                        !skipped || losses.iter().all(|&loss| loss),
                        "skipped a safe injection"
                    );
                    ensure!(
                        !random || roots.peek().is_none_or(|r| r.ply != state.ply),
                        "random move has a search label"
                    );
                }
            }
            if let Some(root) = roots.peek().filter(|r| r.ply == state.ply) {
                ensure!(
                    root.mover == (state.ply % 2) as u8
                        && root.since_capture == state.since_capture
                        && root.capture_clock == self.capture_clock
                        && root.board_fingerprint == fingerprint(&state),
                    "root position mismatch"
                );
                ensure!(root.played_action == action, "played/root action mismatch");
                match root.score_kind {
                    ScoreKind::Unlabelled => ensure!(
                        root.root_score.is_none()
                            && root.searched_best.is_none()
                            && root.completed_depth == 0,
                        "uncompleted label"
                    ),
                    ScoreKind::Search | ScoreKind::EngineProof => {
                        let best = root
                            .searched_best
                            .ok_or_else(|| anyhow::anyhow!("label has no best move"))?;
                        ensure!(
                            state.legal_moves().contains(best) && root.root_score.is_some(),
                            "invalid label"
                        );
                        if root.score_kind == ScoreKind::EngineProof {
                            ensure!(
                                apply(&Rules::SITE, &state, best).1 == Outcome::Win
                                    && root.root_score == Some(29_999)
                                    && root.completed_depth == 0,
                                "unverified proof"
                            );
                        } else {
                            ensure!(
                                root.completed_depth > 0,
                                "search label has no completed depth"
                            );
                        }
                    }
                }
                if !root.pv.is_empty() {
                    ensure!(
                        root.pv.first().copied() == root.searched_best,
                        "pv does not start at the labelled move"
                    );
                    let mut line = state.clone();
                    for (index, &step) in root.pv.iter().enumerate() {
                        ensure!(line.legal_moves().contains(step), "illegal pv action");
                        let (child, outcome) = apply(&Rules::SITE, &line, step);
                        ensure!(
                            outcome == Outcome::Ongoing || index + 1 == root.pv.len(),
                            "pv continues past a terminal position"
                        );
                        line = child;
                    }
                }
                // Multipv sampling plays one of the two searched lines and records the second.
                if let Some((alternative, _)) = root.alternative {
                    let best = root
                        .searched_best
                        .ok_or_else(|| anyhow::anyhow!("alternative without a label"))?;
                    ensure!(
                        alternative != best && state.legal_moves().contains(alternative),
                        "invalid alternative"
                    );
                    ensure!(
                        action.is_none_or(|played| played == alternative || played == best),
                        "played neither the searched best nor its alternative"
                    );
                } else if let (Some(best), Some(played)) = (root.searched_best, action) {
                    // A bare substitution is the search's own late choice, never a sampled move;
                    // the generation settings decide whether it is allowed (see the CLI's verify).
                    ensure!(
                        played == best || root.score_kind == ScoreKind::Search,
                        "proved move replaced without an alternative"
                    );
                }
                if action.is_none() {
                    ensure!(
                        self.game.end == End::Interrupted,
                        "unplayed root without interruption"
                    );
                }
                roots.next();
            } else if action.is_some() {
                ensure!(
                    self.game.random_plies.contains(&state.ply),
                    "move without search or random provenance"
                );
            }
            if let Some(action) = action {
                ensure!(
                    state.terminal(&Rules::SITE).is_none() && state.legal_moves().contains(action),
                    "move after terminal or illegal move"
                );
                state = apply(&Rules::SITE, &state, action).0;
            }
        }
        ensure!(roots.next().is_none(), "unordered or excess roots");
        ensure!(
            Position::from_state(&state) == self.final_position && state.ply == self.game.plies,
            "final board mismatch"
        );
        let mut expected = self.clone();
        expected.finish(&state, self.game.end);
        ensure!(
            matches!(self.game.end, End::PlyCap | End::Interrupted) || !self.censored,
            "terminal marked censored"
        );
        ensure!(
            expected.game.end == self.game.end
                && expected.game.winner == self.game.winner
                && expected.outcome == self.outcome
                && expected.censored == self.censored
                && expected.outcome_after_ply == self.outcome_after_ply,
            "outcome mismatch"
        );
        for plies in [&self.game.random_plies, &self.random_skipped] {
            ensure!(
                plies.windows(2).all(|p| p[0] < p[1]),
                "unordered random plies"
            );
            for &ply in plies {
                ensure!(
                    ply < self.opening_plies || self.random_plan.contains(&ply),
                    "unplanned random ply"
                );
                ensure!(
                    ply < self.game.plies
                        || self
                            .roots
                            .iter()
                            .any(|r| r.ply == ply && r.played_action.is_none()),
                    "random ply outside record"
                );
            }
        }
        Ok(())
    }
}

pub fn generate(
    mut record: Record,
    config: &Config,
    search: &mut impl FnMut(&State) -> Result<Decision>,
    stopped: &impl Fn() -> bool,
    emit: &mut impl FnMut(&Event) -> Result<()>,
) -> Result<Record> {
    let mut state = record.initial.state()?;
    let mut opening = RandomMoves::new(0, 0, 0, derive_seed(record.seed, 0))?;
    let mut random = RandomMoves::new(
        config.random_moves,
        config.random_from,
        config.random_to,
        derive_seed(record.seed, 1),
    )?;
    loop {
        if state.terminal(&Rules::SITE).is_some() || state.ply >= config.max_plies || stopped() {
            break;
        }
        let requested = state.ply < config.opening_plies || random.plies().contains(&state.ply);
        let injected = if state.ply < config.opening_plies {
            opening.choose_safe(&Rules::SITE, &state)
        } else {
            random.choose(&Rules::SITE, &state)
        };
        let mut event = Event {
            ply: state.ply,
            action: injected,
            random: injected.is_some(),
            skipped: requested && injected.is_none(),
            root: None,
        };
        if injected.is_none() {
            let decision = search(&state)?;
            let played = (!stopped()).then_some(decision.action);
            event.action = played;
            event.root = Some(Root {
                ply: state.ply,
                mover: (state.ply % 2) as u8,
                since_capture: state.since_capture,
                capture_clock: record.capture_clock,
                board_fingerprint: fingerprint(&state),
                searched_best: decision.label.map(|l| l.action),
                played_action: played,
                root_score: decision.label.map(|l| l.score),
                score_kind: decision.label.map_or(ScoreKind::Unlabelled, |l| l.kind),
                completed_depth: decision.label.map_or(0, |l| l.depth),
                nodes: decision.nodes,
                elapsed_ns: decision.elapsed_ns,
                pv: decision.pv,
                alternative: decision.alternative,
            });
        }
        emit(&event)?;
        record.append(&mut state, event)?;
        if stopped() {
            break;
        }
    }
    record.finish(
        &state,
        if stopped() {
            End::Interrupted
        } else {
            End::PlyCap
        },
    );
    record.verify()?;
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{Cell, Piece};
    use std::sync::atomic::{AtomicBool, Ordering};

    fn config() -> Config {
        Config {
            opening_plies: 8,
            random_moves: 2,
            random_from: 8,
            random_to: 15,
            max_plies: 24,
        }
    }
    fn first(state: &State) -> Result<Decision> {
        let action = state.legal_actions()[0];
        Ok(Decision {
            action,
            label: Some(Label {
                action,
                score: 42,
                depth: 1,
                kind: ScoreKind::Search,
            }),
            pv: Vec::new(),
            alternative: None,
            nodes: 10,
            elapsed_ns: 123,
        })
    }
    /// The first-legal-action line from `state`, stopping at a terminal position.
    fn line(state: &State, length: usize) -> Vec<Action> {
        let mut pv = Vec::new();
        let mut state = state.clone();
        for _ in 0..length {
            let action = state.legal_actions()[0];
            pv.push(action);
            let (child, outcome) = apply(&Rules::SITE, &state, action);
            if outcome != Outcome::Ongoing {
                break;
            }
            state = child;
        }
        pv
    }
    fn custom(state: &State, config: &Config) -> Record {
        let mut record = Record::new(config, "test", 0, 7).unwrap();
        record.initial = Position::from_state(state);
        record.final_position = record.initial.clone();
        record.game.plies = state.ply;
        record
    }

    #[test]
    fn seeded_games_replay_and_detect_corrupted_root_frames() {
        let config = config();
        for id in 0..8 {
            let start = Record::new(&config, "test", id, derive_seed(93, id)).unwrap();
            let record = generate(start.clone(), &config, &mut first, &|| false, &mut |_| {
                Ok(())
            })
            .unwrap();
            record.verify().unwrap();
            let again = generate(start, &config, &mut first, &|| false, &mut |_| Ok(())).unwrap();
            assert_eq!(
                serde_json::to_vec(&record).unwrap(),
                serde_json::to_vec(&again).unwrap()
            );
            assert_eq!(record.game.end, End::PlyCap);
            assert_eq!(record.outcome, None);
            assert!(record.censored && record.game.winner.is_none());
            let mut bad = record.clone();
            bad.roots[0].mover ^= 1;
            assert!(bad.verify().is_err());
            let mut bad = record.clone();
            bad.final_position.board[0] ^= 1;
            assert!(bad.verify().is_err());
            let mut bad = record;
            bad.outcome = Some(0);
            assert!(bad.verify().is_err());
        }
    }

    #[test]
    fn engine_proofs_keep_mover_sign_and_goal_precedes_clock() {
        let config = Config {
            opening_plies: 0,
            random_moves: 0,
            max_plies: 4,
            ..config()
        };
        for ply in [0, 1] {
            let state = State {
                board: {
                    let mut b = [Cell::Empty; 81];
                    b[79] = Cell::Own(Piece::Rock);
                    b[20] = Cell::Enemy(Piece::Paper);
                    b
                },
                ply,
                since_capture: 199,
            };
            let action = engine::notation::action_between(79, 80).unwrap();
            let record = generate(
                custom(&state, &config),
                &config,
                &mut |_| {
                    Ok(Decision {
                        action,
                        label: Some(Label {
                            action,
                            score: 29_999,
                            depth: 0,
                            kind: ScoreKind::EngineProof,
                        }),
                        pv: Vec::new(),
                        alternative: None,
                        nodes: 0,
                        elapsed_ns: 5,
                    })
                },
                &|| false,
                &mut |_| Ok(()),
            )
            .unwrap();
            assert_eq!(record.game.end, End::Goal);
            assert_eq!(record.game.winner, Some(ply as u8));
            assert_eq!(record.outcome, Some(if ply == 0 { 1 } else { -1 }));
            assert_eq!(record.roots[0].root_score, Some(29_999));
            assert_eq!(record.roots[0].mover, ply as u8);
            record.verify().unwrap();
        }
    }

    #[test]
    fn terminal_roots_are_not_searched_and_draws_are_not_censoring() {
        let config = config();
        let mut state = State::initial();
        state.since_capture = 200;
        let record = generate(
            custom(&state, &config),
            &config,
            &mut |_| panic!("searched terminal"),
            &|| false,
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(record.game.end, End::CaptureClock);
        assert_eq!(record.outcome, Some(0));
        assert!(!record.censored);
        record.verify().unwrap();
    }

    #[test]
    fn interrupted_completed_label_is_preserved_without_playing_a_move() {
        let config = Config {
            opening_plies: 0,
            random_moves: 0,
            ..config()
        };
        let stop = AtomicBool::new(false);
        let record = generate(
            Record::new(&config, "test", 0, 1).unwrap(),
            &config,
            &mut |s| {
                stop.store(true, Ordering::Relaxed);
                first(s)
            },
            &|| stop.load(Ordering::Relaxed),
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(record.game.end, End::Interrupted);
        assert_eq!(record.game.moves.len(), 0);
        assert_eq!(record.roots.len(), 1);
        assert_eq!(record.roots[0].root_score, Some(42));
        assert_eq!(record.roots[0].played_action, None);
        assert!(record.outcome.is_none());
        record.verify().unwrap();
    }

    #[test]
    fn uncompleted_search_has_no_label_and_unsafe_injection_is_recorded() {
        let config = Config {
            opening_plies: 0,
            random_moves: 1,
            random_from: 0,
            random_to: 0,
            max_plies: 1,
        };
        let mut state = State::initial();
        state.board.fill(Cell::Empty);
        state.board[60] = Cell::Own(Piece::Rock);
        state.board[10] = Cell::Enemy(Piece::Paper);
        let record = generate(
            custom(&state, &config),
            &config,
            &mut |s| {
                let mut d = first(s)?;
                d.label = None;
                Ok(d)
            },
            &|| false,
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(record.random_skipped, [0]);
        assert!(record.game.random_plies.is_empty());
        assert_eq!(record.roots[0].root_score, None);
        assert_eq!(record.roots[0].score_kind, ScoreKind::Unlabelled);
        record.verify().unwrap();
        // An unlabelled root has no best move, so it may not carry a line either.
        let mut bad = record.clone();
        bad.roots[0].pv = vec![state.legal_actions()[0]];
        assert!(bad.verify().is_err());
        // ... nor an alternative to a move it never chose.
        let mut bad = record;
        bad.roots[0].alternative = Some((state.legal_actions()[0], 8));
        assert!(bad.verify().is_err());
    }

    #[test]
    fn principal_variations_replay_and_broken_lines_are_rejected() {
        let config = Config {
            opening_plies: 0,
            random_moves: 0,
            max_plies: 2,
            ..config()
        };
        let record = generate(
            Record::new(&config, "test", 0, 5).unwrap(),
            &config,
            &mut |s| {
                let mut decision = first(s)?;
                decision.pv = line(s, 3);
                Ok(decision)
            },
            &|| false,
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(record.roots[0].pv.len(), 3);
        record.verify().unwrap();
        // The line must start at the labelled move, even when the substitute is legal.
        let root = record.initial.state().unwrap();
        let mut bad = record.clone();
        bad.roots[0].pv[0] = root.legal_actions()[1];
        assert!(bad.verify().is_err());
        // Every continuation must be legal where it is played.
        let child = apply(&Rules::SITE, &root, record.roots[0].pv[0]).0;
        let illegal = (0..648u16)
            .find(|&a| !child.legal_moves().contains(a))
            .unwrap();
        let mut bad = record.clone();
        bad.roots[0].pv[1] = illegal;
        assert!(bad.verify().is_err());
        // Recording no line at all stays legal.
        let mut empty = record.clone();
        empty.roots[0].pv.clear();
        empty.verify().unwrap();
    }

    #[test]
    fn a_principal_variation_may_not_continue_past_a_terminal_position() {
        let config = Config {
            opening_plies: 0,
            random_moves: 0,
            max_plies: 4,
            ..config()
        };
        let state = State {
            board: {
                let mut b = [Cell::Empty; 81];
                b[79] = Cell::Own(Piece::Rock);
                b[20] = Cell::Enemy(Piece::Paper);
                b
            },
            ply: 0,
            since_capture: 0,
        };
        let win = engine::notation::action_between(79, 80).unwrap();
        let record = generate(
            custom(&state, &config),
            &config,
            &mut |_| {
                Ok(Decision {
                    action: win,
                    label: Some(Label {
                        action: win,
                        score: 29_999,
                        depth: 1,
                        kind: ScoreKind::Search,
                    }),
                    pv: vec![win],
                    alternative: None,
                    nodes: 7,
                    elapsed_ns: 5,
                })
            },
            &|| false,
            &mut |_| Ok(()),
        )
        .unwrap();
        assert_eq!(record.game.end, End::Goal);
        assert_eq!(record.roots[0].pv, vec![win]);
        record.verify().unwrap();
        let mut bad = record;
        bad.roots[0].pv.push(win);
        assert!(bad.verify().is_err());
    }

    #[test]
    fn a_sampled_alternative_is_recorded_as_the_move_it_played() {
        let config = Config {
            opening_plies: 0,
            random_moves: 0,
            max_plies: 4,
            ..config()
        };
        // The label stays the main search's first move while the second line is played.
        let record = generate(
            Record::new(&config, "test", 0, 11).unwrap(),
            &config,
            &mut |state| {
                let mut decision = first(state)?;
                let second = state.legal_actions()[1];
                decision.action = second;
                decision.alternative = Some((second, 40));
                Ok(decision)
            },
            &|| false,
            &mut |_| Ok(()),
        )
        .unwrap();
        assert!(!record.roots.is_empty());
        assert!(record
            .roots
            .iter()
            .all(|r| r.alternative.is_some() && r.played_action != r.searched_best));
        record.verify().unwrap();
        let root = record.initial.state().unwrap();
        // The second line is a different move from the label's.
        let mut bad = record.clone();
        bad.roots[0].alternative = Some((record.roots[0].searched_best.unwrap(), 40));
        assert!(bad.verify().is_err());
        // It must be legal at the root it was searched from.
        let illegal = (0..648u16)
            .find(|&a| !root.legal_moves().contains(a))
            .unwrap();
        let mut bad = record.clone();
        bad.roots[0].alternative = Some((illegal, 40));
        assert!(bad.verify().is_err());
        // The move played is one of the two lines, never a third.
        let mut bad = record.clone();
        bad.roots[0].alternative = Some((root.legal_actions()[2], 40));
        assert!(bad.verify().is_err());
        // Recording no alternative at all is the ordinary single-line record.
        let mut plain = record;
        for r in &mut plain.roots {
            r.alternative = None;
        }
        plain.verify().unwrap();
    }
}
