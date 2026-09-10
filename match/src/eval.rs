//! Paired games at simulation or per-player wall-time budgets.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{ensure, Context, Result};
use engine::Rules;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

use crate::game::{play_observed_clocks, GameRecord, Opening};
use crate::player::{Clock, Player};

pub type PlayerFactory = dyn Fn() -> Result<Box<dyn Player>> + Sync;

/// Both games of one opening: candidate first, then reference first.
type Pair = (GameRecord, GameRecord);

#[derive(Clone, Debug)]
pub struct EvalConfig {
    pub rules: Rules,
    /// Game pairs; each pair plays one opening with colours swapped.
    pub pairs: usize,
    pub sims: u32,
    pub move_ms: Option<u64>,
    pub reference_move_ms: Option<u64>,
    pub reference_sims: Option<u32>,
    pub opening_plies: usize,
    pub seed: u64,
    /// Pairs run concurrently on this many threads; players are built per thread.
    pub threads: usize,
    /// Print every game's start, moves and end as JSON lines on stdout.
    pub stream: bool,
}

impl Default for EvalConfig {
    fn default() -> EvalConfig {
        EvalConfig {
            rules: Rules::SITE,
            pairs: 32,
            sims: 32,
            move_ms: None,
            reference_move_ms: None,
            reference_sims: None,
            opening_plies: 8,
            seed: 0,
            threads: 4,
            stream: false,
        }
    }
}

/// One JSON line on stdout, whole even when threads interleave.
fn emit(event: serde_json::Value) {
    let line = event.to_string();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// Play both games of `opening`, streaming them when asked.
fn play_pair(
    config: &EvalConfig,
    a: &mut dyn Player,
    b: &mut dyn Player,
    pair: usize,
    opening: &Opening,
) -> Pair {
    let clocks = [clock(config), reference_clock(config)];
    let mut games = Vec::with_capacity(2);
    for game in 0..2u8 {
        let (first, second): (&mut dyn Player, &mut dyn Player) = if game == 0 {
            (&mut *a, &mut *b)
        } else {
            (&mut *b, &mut *a)
        };
        if config.stream {
            emit(serde_json::json!({
                "event": "start", "pair": pair, "game": game,
                "first": first.name(), "second": second.name(), "opening": opening.tokens(),
            }));
        }
        let stream = config.stream;
        let record = play_observed_clocks(
            &config.rules,
            first,
            second,
            opening,
            if game == 0 {
                clocks
            } else {
                [clocks[1], clocks[0]]
            },
            &mut |ply, token, stats| {
                if stream {
                    emit(serde_json::json!({
                        "event": "move", "pair": pair, "game": game,
                        "ply": ply, "move": token, "info": stats,
                    }));
                }
            },
        );
        if config.stream {
            emit(serde_json::json!({
                "event": "end", "pair": pair, "game": game,
                "winner": record.winner, "end": record.end, "plies": record.plies,
            }));
        }
        games.push(record);
    }
    let second = games.pop().expect("two games");
    let first = games.pop().expect("two games");
    (first, second)
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub candidate: String,
    pub reference: String,
    pub pairs: usize,
    pub complete_pairs: usize,
    pub move_ms: Option<u64>,
    pub reference_move_ms: Option<u64>,
    pub reference_sims: Option<u32>,
    pub sims: u32,
    pub forfeits: u32,
    pub bootstrap_interval: (f64, f64),
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    /// Mean over pairs of (candidate points - reference points) / 2, in [-1, 1].
    pub margin: f64,
    /// 95 % interval of the margin over pairs.
    pub interval: (f64, f64),
    pub mean_plies: f64,
}

/// Evaluate `candidate` against `reference`. `records` optionally receives
/// every game as JSON lines.
pub fn eval(
    config: &EvalConfig,
    candidate: &PlayerFactory,
    reference: &PlayerFactory,
    records: Option<&Path>,
) -> Result<Report> {
    ensure!(
        config.pairs > 0 && config.threads > 0,
        "pairs and threads must be positive"
    );
    ensure!(
        config.move_ms.is_none_or(|ms| ms > 0),
        "move-ms must be positive"
    );
    ensure!(
        config.move_ms.is_some() || config.sims > 0,
        "sims must be positive"
    );
    ensure!(
        config
            .reference_move_ms
            .is_none_or(|ms| ms > 0 && config.move_ms.is_some()),
        "reference-move-ms must be positive and requires move-ms"
    );
    ensure!(
        config
            .reference_sims
            .is_none_or(|sims| sims > 0 && config.move_ms.is_some()),
        "reference-sims must be positive and requires move-ms"
    );
    ensure!(
        config.reference_sims.is_none() || config.reference_move_ms.is_none(),
        "reference-sims conflicts with reference-move-ms"
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    let openings: Vec<Opening> = (0..config.pairs)
        .map(|_| Opening::random(&config.rules, config.opening_plies, &mut rng))
        .collect();
    let next = Arc::new(Mutex::new(0usize));
    let games: Arc<Mutex<Vec<Option<Pair>>>> = Arc::new(Mutex::new(vec![None; config.pairs]));
    let threads = config.threads.max(1).min(config.pairs.max(1));
    let mut names: Option<(String, String)> = None;
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for _ in 0..threads {
            let (next, games, openings) = (Arc::clone(&next), Arc::clone(&games), &openings);
            handles.push(scope.spawn(move || -> Result<(String, String)> {
                let mut a = candidate()?;
                let mut b = reference()?;
                ensure!(
                    a.supports_clock(clock(config)) && b.supports_clock(reference_clock(config)),
                    "player rejects simulation budgets; use --move-ms"
                );
                let names = (a.name().to_owned(), b.name().to_owned());
                loop {
                    let pair = {
                        let mut guard = next.lock().expect("pair counter");
                        let pair = *guard;
                        *guard += 1;
                        pair
                    };
                    if pair >= openings.len() {
                        return Ok(names);
                    }
                    let played = play_pair(config, &mut *a, &mut *b, pair, &openings[pair]);
                    games.lock().expect("games")[pair] = Some(played);
                }
            }));
        }
        for handle in handles {
            let pair = handle.join().expect("eval thread")?;
            names.get_or_insert(pair);
        }
        Ok(())
    })?;
    let (candidate, reference) = names.unwrap_or_default();
    let games = Arc::try_unwrap(games)
        .expect("threads joined")
        .into_inner()
        .expect("games");
    let (mut wins, mut draws, mut losses) = (0u32, 0u32, 0u32);
    let mut pair_scores = Vec::with_capacity(config.pairs);
    let mut plies = 0u64;
    let mut forfeits = 0u32;
    let mut complete_pairs = 0usize;
    let mut writer = match records {
        Some(path) => Some(BufWriter::new(
            File::create(path).with_context(|| format!("creating {}", path.display()))?,
        )),
        None => None,
    };
    for pair in games.into_iter().flatten() {
        if pair.0.end != crate::game::End::Forfeit && pair.1.end != crate::game::End::Forfeit {
            complete_pairs += 1;
        }
        let mut score = 0.0;
        // In the first game the candidate moved first; in the second, the reference did.
        for (game, candidate_index) in [(&pair.0, 0u8), (&pair.1, 1u8)] {
            forfeits += u32::from(game.end == crate::game::End::Forfeit);
            match game.winner {
                Some(w) if w == candidate_index => {
                    wins += 1;
                    score += 1.0;
                }
                Some(_) => {
                    losses += 1;
                    score -= 1.0;
                }
                None => draws += 1,
            }
            plies += game.plies as u64;
            if let Some(writer) = writer.as_mut() {
                serde_json::to_writer(&mut *writer, game)?;
                writer.write_all(b"\n")?;
            }
        }
        pair_scores.push(score / 2.0);
    }
    let n = pair_scores.len().max(1) as f64;
    let margin = pair_scores.iter().sum::<f64>() / n;
    let variance = pair_scores
        .iter()
        .map(|s| (s - margin).powi(2))
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    let half_width = 1.96 * (variance / n).sqrt();
    Ok(Report {
        candidate,
        reference,
        pairs: pair_scores.len(),
        sims: if config.move_ms.is_some() {
            0
        } else {
            config.sims
        },
        move_ms: config.move_ms,
        reference_move_ms: match reference_clock(config) {
            Clock::Time(duration) => Some(duration.as_millis() as u64),
            Clock::Sims(_) => None,
        },
        reference_sims: match reference_clock(config) {
            Clock::Sims(sims) => Some(sims),
            Clock::Time(_) => None,
        },
        complete_pairs,
        forfeits,
        bootstrap_interval: bootstrap(&pair_scores, config.seed),
        wins,
        draws,
        losses,
        margin,
        interval: (margin - half_width, margin + half_width),
        mean_plies: plies as f64 / (2.0 * n),
    })
}

/// Reference budget, defaulting to the candidate budget.
fn reference_clock(config: &EvalConfig) -> Clock {
    if let Some(sims) = config.reference_sims {
        return Clock::Sims(sims);
    }
    config.reference_move_ms.map_or_else(
        || clock(config),
        |ms| Clock::Time(std::time::Duration::from_millis(ms)),
    )
}
/// Candidate budget.
fn clock(config: &EvalConfig) -> Clock {
    config.move_ms.map_or(Clock::Sims(config.sims), |ms| {
        Clock::Time(std::time::Duration::from_millis(ms))
    })
}
/// Percentile bootstrap of opening-pair margins; no independent-game assumption.
fn bootstrap(scores: &[f64], seed: u64) -> (f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x6e6e7565);
    let mut means = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        means.push(
            (0..scores.len())
                .map(|_| scores[rng.random_range(0..scores.len())])
                .sum::<f64>()
                / scores.len() as f64,
        );
    }
    means.sort_by(f64::total_cmp);
    (means[249], means[9749])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{player::Clock, History};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct BudgetPlayer {
        calls: Arc<AtomicUsize>,
        expected: Clock,
    }
    impl Player for BudgetPlayer {
        fn new_game(&mut self) {}
        fn choose(&mut self, state: &engine::State, _: &History, clock: Clock) -> engine::Action {
            match (clock, self.expected) {
                (Clock::Time(actual), Clock::Time(expected)) => assert_eq!(actual, expected),
                (Clock::Sims(actual), Clock::Sims(expected)) => assert_eq!(actual, expected),
                _ => panic!("wrong budget unit: {clock:?}, expected {:?}", self.expected),
            }
            self.calls.fetch_add(1, Ordering::Relaxed);
            state.legal_actions()[0]
        }
        fn name(&self) -> &str {
            "budget-test"
        }
    }
    #[test]
    fn timed_eval_delivers_equal_budgets_and_counts_both_colours() {
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        let factory = move || {
            Ok(Box::new(BudgetPlayer {
                calls: Arc::clone(&factory_calls),
                expected: Clock::Time(std::time::Duration::from_millis(17)),
            }) as Box<dyn Player>)
        };
        let report = eval(
            &EvalConfig {
                move_ms: Some(17),
                pairs: 2,
                threads: 1,
                ..EvalConfig::default()
            },
            &factory,
            &factory,
            None,
        )
        .unwrap();
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert_eq!(report.wins, report.losses);
        assert_eq!(report.wins + report.draws + report.losses, 4);
        assert_eq!(report.complete_pairs, 2);
        assert_eq!(report.forfeits, 0);
        assert_eq!(report.sims, 0);
        assert_eq!(report.move_ms, Some(17));
        assert_eq!(report.bootstrap_interval, (0., 0.));
    }
    #[test]
    fn bootstrap_resamples_pairs_reproducibly() {
        assert_eq!(bootstrap(&[1.; 8], 1), (1., 1.));
        let scores = [-1., 0., 0.5, 1.];
        let interval = bootstrap(&scores, 712);
        assert_eq!(interval, bootstrap(&scores, 712));
        assert!(interval.0 < 0. && interval.1 > 0.);
    }
    #[test]
    fn asymmetric_budgets_follow_players_across_swapped_seats() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let a = Arc::clone(&calls_a);
        let b = Arc::clone(&calls_b);
        let report = eval(
            &EvalConfig {
                move_ms: Some(34),
                reference_move_ms: Some(17),
                pairs: 2,
                threads: 1,
                ..EvalConfig::default()
            },
            &move || {
                Ok(Box::new(BudgetPlayer {
                    calls: Arc::clone(&a),
                    expected: Clock::Time(std::time::Duration::from_millis(34)),
                }))
            },
            &move || {
                Ok(Box::new(BudgetPlayer {
                    calls: Arc::clone(&b),
                    expected: Clock::Time(std::time::Duration::from_millis(17)),
                }))
            },
            None,
        )
        .unwrap();
        assert!(calls_a.load(Ordering::Relaxed) > 0 && calls_b.load(Ordering::Relaxed) > 0);
        assert_eq!(report.complete_pairs, 2);
        assert_eq!(report.forfeits, 0);
        assert_eq!(report.move_ms, Some(34));
        assert_eq!(report.reference_move_ms, Some(17));
        assert_eq!(report.wins, report.losses);
    }

    #[test]
    fn mixed_budgets_follow_players_across_swapped_seats() {
        let calls_a = Arc::new(AtomicUsize::new(0));
        let calls_b = Arc::new(AtomicUsize::new(0));
        let a = Arc::clone(&calls_a);
        let b = Arc::clone(&calls_b);
        let report = eval(
            &EvalConfig {
                move_ms: Some(34),
                reference_sims: Some(32),
                pairs: 2,
                threads: 1,
                ..EvalConfig::default()
            },
            &move || {
                Ok(Box::new(BudgetPlayer {
                    calls: Arc::clone(&a),
                    expected: Clock::Time(std::time::Duration::from_millis(34)),
                }))
            },
            &move || {
                Ok(Box::new(BudgetPlayer {
                    calls: Arc::clone(&b),
                    expected: Clock::Sims(32),
                }))
            },
            None,
        )
        .unwrap();
        assert!(calls_a.load(Ordering::Relaxed) > 0 && calls_b.load(Ordering::Relaxed) > 0);
        assert_eq!(report.complete_pairs, 2);
        assert_eq!(report.forfeits, 0);
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["move_ms"], 34);
        assert_eq!(json["sims"], 0);
        assert_eq!(json["reference_sims"], 32);
        assert!(json["reference_move_ms"].is_null());
    }

    #[test]
    fn invalid_reference_simulations_are_rejected_before_constructing_players() {
        for (candidate, sims, reference_time) in [
            (None, 32, None),
            (Some(17), 0, None),
            (Some(17), 32, Some(17)),
        ] {
            let result = eval(
                &EvalConfig {
                    move_ms: candidate,
                    reference_sims: Some(sims),
                    reference_move_ms: reference_time,
                    ..EvalConfig::default()
                },
                &|| panic!("invalid budget must be rejected first"),
                &|| panic!("invalid budget must be rejected first"),
                None,
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn invalid_reference_budgets_are_rejected_before_constructing_players() {
        for (candidate, reference) in [(None, 17), (Some(17), 0)] {
            let result = eval(
                &EvalConfig {
                    move_ms: candidate,
                    reference_move_ms: Some(reference),
                    ..EvalConfig::default()
                },
                &|| panic!("invalid budget must be rejected first"),
                &|| panic!("invalid budget must be rejected first"),
                None,
            );
            assert!(result.is_err());
        }
    }
}
