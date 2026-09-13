//! Paired games at simulation or per-player wall-time budgets.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{ensure, Context, Result};
use engine::Rules;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

use crate::game::{play_observed_clocks, GameRecord, Opening};
use crate::player::{Clock, Player};
use crate::sprt::{self, Stop};

mod journal;

pub type PlayerFactory = dyn Fn() -> Result<Box<dyn Player>> + Sync;

/// Both games of one opening: candidate first, then reference first.
type Pair = (GameRecord, GameRecord);

#[derive(Clone, Debug)]
pub struct Sequential {
    pub journal: PathBuf,
    pub resume: bool,
    /// Model-agnostic artifact identities supplied and hashed by the CLI.
    pub provenance: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct SequentialReport {
    pub protocol: serde_json::Value,
    pub protocol_digest: String,
    #[serde(flatten)]
    pub state: sprt::State,
    pub intervals_descriptive_only: bool,
}

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
    pub sequential: Option<Sequential>,
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
            sequential: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequential: Option<SequentialReport>,
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
    ensure!(
        config.sequential.is_none()
            || (config.pairs == sprt::CAP && config.threads <= sprt::BATCH && records.is_none()),
        "sequential eval requires 3008 pairs, at most 16 workers and its own journal"
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    let openings: Vec<Opening> = (0..config.pairs)
        .map(|_| Opening::random(&config.rules, config.opening_plies, &mut rng))
        .collect();
    let batch = if config.sequential.is_some() {
        sprt::BATCH
    } else {
        config.threads.max(sprt::BATCH)
    };
    let mut summary = Summary::new(config.sequential.is_some());
    let protocol = serde_json::json!({
        "schema": 1,
        "test": {"method":"pentanomial_expectation_mle", "scores":sprt::SCORES,
            "s0":sprt::S0,"s1":sprt::S1,"alpha":0.05,"beta":0.05,
            "lower_bound":-sprt::BOUND,"upper_bound":sprt::BOUND,
            "zero_cell":0.001,"batch":sprt::BATCH,"first_check":sprt::FIRST_CHECK,"cap":sprt::CAP},
        "settings": {"capture_clock":config.rules.capture_clock,"pairs":config.pairs,
            "sims":config.sims,"move_ms":config.move_ms,"reference_move_ms":config.reference_move_ms,
            "reference_sims":config.reference_sims,"workers":config.threads,
            "seed":config.seed,"opening_plies":config.opening_plies,"stream":config.stream},
        "openings_sha256":journal::digest(&serde_json::json!(openings.iter().map(|o| &o.plies).collect::<Vec<_>>()))?,
        "provenance":config.sequential.as_ref().map(|s| &s.provenance),
    });
    let mut journal = if let Some(sequential) = &config.sequential {
        Some(journal::Journal::open(
            &sequential.journal,
            sequential.resume,
            &protocol,
            |id, games| {
                ensure!(
                    id == summary.scores.len() && id < openings.len(),
                    "noncontiguous journal pair id"
                );
                ensure!(
                    !id.is_multiple_of(sprt::BATCH) || summary.running(),
                    "journal continues after a stopping batch"
                );
                let tokens = openings[id].tokens();
                ensure!(
                    games.0.moves.starts_with(&tokens) && games.1.moves.starts_with(&tokens),
                    "journal opening differs"
                );
                summary.retire(games)
            },
        )?)
    } else {
        None
    };
    let mut writer = records
        .map(|path| {
            File::create(path)
                .map(BufWriter::new)
                .with_context(|| format!("creating {}", path.display()))
        })
        .transpose()?;
    if summary.needs_pairs(batch) && summary.scores.len() < config.pairs {
        let workers = config.threads.min(config.pairs).min(batch);
        std::thread::scope(|scope| -> Result<()> {
            let (results_tx, results_rx) = mpsc::channel();
            let (ready_tx, ready_rx) = mpsc::channel();
            let mut senders = Vec::with_capacity(workers);
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let (jobs_tx, jobs_rx) = mpsc::channel::<(usize, Opening)>();
                senders.push(jobs_tx);
                let results = results_tx.clone();
                let ready = ready_tx.clone();
                handles.push(scope.spawn(move || {
                    let initialized =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
                            let a = candidate()?;
                            let b = reference()?;
                            ensure!(
                                a.supports_clock(clock(config))
                                    && b.supports_clock(reference_clock(config)),
                                "player rejects simulation budgets; use --move-ms"
                            );
                            Ok((a, b))
                        }))
                        .unwrap_or_else(|_| Err(anyhow::anyhow!("player factory panicked")));
                    let (mut a, mut b) = match initialized {
                        Ok(players) => players,
                        Err(error) => {
                            let _ = ready.send(Err(error));
                            return;
                        }
                    };
                    if ready
                        .send(Ok((a.name().to_owned(), b.name().to_owned())))
                        .is_err()
                    {
                        return;
                    }
                    for (id, opening) in jobs_rx {
                        let played = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            play_pair(config, &mut *a, &mut *b, id, &opening)
                        }))
                        .map_err(|_| anyhow::anyhow!("eval player panicked at pair {id}"));
                        if results.send((id, played)).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(results_tx);
            drop(ready_tx);
            let result = (|| -> Result<()> {
                for _ in 0..workers {
                    let (a, b) = ready_rx.recv().context("player factory disconnected")??;
                    summary.names(&a, &b)?;
                }
                while summary.needs_pairs(batch) && summary.scores.len() < config.pairs {
                    let start = summary.scores.len();
                    let end = ((start / batch + 1) * batch).min(config.pairs);
                    for id in start..end {
                        senders[id % workers].send((id, openings[id].clone()))?;
                    }
                    let mut pending: Vec<Option<Pair>> = vec![None; end - start];
                    for _ in start..end {
                        let (id, played) =
                            results_rx.recv().context("eval workers disconnected")?;
                        ensure!(
                            (start..end).contains(&id) && pending[id - start].is_none(),
                            "invalid worker pair id"
                        );
                        pending[id - start] = Some(played?);
                    }
                    for (offset, pair) in pending.into_iter().enumerate() {
                        let pair = pair.expect("all batch results received");
                        if let Some(journal) = &mut journal {
                            journal.append(start + offset, &pair)?;
                        }
                        if let Some(writer) = &mut writer {
                            for game in [&pair.0, &pair.1] {
                                serde_json::to_writer(&mut *writer, game)?;
                                writer.write_all(b"\n")?;
                            }
                        }
                        summary.retire(pair)?;
                    }
                    if let Some(journal) = &journal {
                        journal.sync()?;
                    }
                    if let Some(writer) = &mut writer {
                        writer.flush()?;
                    }
                }
                Ok(())
            })();
            drop(senders);
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("eval worker panicked"))?;
            }
            result
        })?;
    }
    let n = summary.scores.len().max(1) as f64;
    let margin = summary.scores.iter().sum::<f64>() / n;
    let variance = summary
        .scores
        .iter()
        .map(|s| (s - margin).powi(2))
        .sum::<f64>()
        / (n - 1.).max(1.);
    let half_width = 1.96 * (variance / n).sqrt();
    let sequential = summary
        .sequential
        .map(|state| -> Result<SequentialReport> {
            Ok(SequentialReport {
                protocol_digest: journal::digest(&protocol)?,
                protocol,
                state,
                intervals_descriptive_only: true,
            })
        })
        .transpose()?;
    Ok(Report {
        candidate: summary.candidate,
        reference: summary.reference,
        pairs: summary.scores.len(),
        complete_pairs: summary.complete,
        sims: if config.move_ms.is_some() {
            0
        } else {
            config.sims
        },
        move_ms: config.move_ms,
        reference_move_ms: match reference_clock(config) {
            Clock::Time(time) => Some(time.as_millis() as u64),
            Clock::Sims(_) => None,
        },
        reference_sims: match reference_clock(config) {
            Clock::Sims(n) => Some(n),
            Clock::Time(_) => None,
        },
        forfeits: summary.forfeits,
        wins: summary.wins,
        draws: summary.draws,
        losses: summary.losses,
        bootstrap_interval: bootstrap(&summary.scores, config.seed),
        margin,
        interval: (margin - half_width, margin + half_width),
        mean_plies: summary.plies as f64 / (2. * n),
        sequential,
    })
}

/// Compact statistics; full game records are retired to the journal each batch.
#[derive(Default)]
struct Summary {
    candidate: String,
    reference: String,
    scores: Vec<f64>,
    complete: usize,
    wins: u32,
    draws: u32,
    losses: u32,
    forfeits: u32,
    plies: u64,
    sequential: Option<sprt::State>,
}

impl Summary {
    fn new(sequential: bool) -> Self {
        Self {
            sequential: sequential.then(sprt::State::default),
            ..Self::default()
        }
    }
    fn running(&self) -> bool {
        self.sequential
            .as_ref()
            .is_none_or(|s| s.stop_reason == Stop::Running)
    }
    fn needs_pairs(&self, batch: usize) -> bool {
        self.running() || !self.scores.len().is_multiple_of(batch)
    }
    fn names(&mut self, candidate: &str, reference: &str) -> Result<()> {
        if self.candidate.is_empty() && self.reference.is_empty() {
            self.candidate = candidate.into();
            self.reference = reference.into();
        }
        ensure!(
            self.candidate == candidate && self.reference == reference,
            "player identities differ between pairs"
        );
        Ok(())
    }
    fn retire(&mut self, pair: Pair) -> Result<()> {
        self.names(&pair.0.first, &pair.0.second)?;
        self.names(&pair.1.second, &pair.1.first)?;
        let mut cell = 0;
        let mut valid = true;
        for (game, candidate) in [(&pair.0, 0u8), (&pair.1, 1u8)] {
            ensure!(
                game.winner.is_none_or(|w| w <= 1),
                "invalid winner in pair record"
            );
            let forfeit = game.end == crate::game::End::Forfeit;
            self.forfeits += u32::from(forfeit);
            valid &= !forfeit
                && !matches!(
                    game.end,
                    crate::game::End::PlyCap | crate::game::End::Interrupted
                );
            match game.winner {
                Some(w) if w == candidate => {
                    self.wins += 1;
                    cell += 2;
                }
                Some(_) => self.losses += 1,
                None => {
                    self.draws += 1;
                    cell += 1;
                }
            }
            self.plies += game.plies as u64;
        }
        self.complete += usize::from(valid);
        self.scores.push(cell as f64 / 2. - 1.);
        if let Some(state) = &mut self.sequential {
            if valid {
                state.counts[cell] += 1;
            } else {
                state.invalidate("forfeit or censored game in opening pair");
            }
            state.check(self.scores.len());
        }
        Ok(())
    }
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
    use std::sync::Arc;

    struct TempJournal(PathBuf);
    impl TempJournal {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "paired-sprt-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&dir).unwrap();
            Self(dir.join("pairs.jsonl"))
        }
    }
    impl Drop for TempJournal {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            std::fs::remove_dir(self.0.parent().unwrap()).unwrap();
        }
    }

    fn sequential_config(path: &Path) -> EvalConfig {
        EvalConfig {
            pairs: sprt::CAP,
            threads: 2,
            opening_plies: 2,
            sequential: Some(Sequential {
                journal: path.into(),
                resume: false,
                provenance: serde_json::json!({"test":"fixture"}),
            }),
            ..EvalConfig::default()
        }
    }

    #[test]
    fn sequential_batches_stop_and_resume_a_torn_partial_batch() {
        let path = TempJournal::new();
        let mut config = sequential_config(&path.0);
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        let factory = move || {
            Ok(Box::new(BudgetPlayer {
                calls: Arc::clone(&factory_calls),
                expected: Clock::Sims(32),
            }) as Box<dyn Player>)
        };
        let report = eval(&config, &factory, &factory, None).unwrap();
        assert_eq!(report.pairs, sprt::FIRST_CHECK);
        assert_eq!(
            report.sequential.as_ref().unwrap().state.stop_reason,
            Stop::Reject
        );
        assert_eq!(
            report.sequential.as_ref().unwrap().state.counts,
            [0, 0, 128, 0, 0]
        );
        let original = std::fs::read_to_string(&path.0).unwrap();
        let lines: Vec<&str> = original.lines().collect();
        for (id, line) in lines[1..].iter().enumerate() {
            let entry: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(entry["pair"], id);
            assert_eq!(entry["games"].as_array().unwrap().len(), 2);
        }
        // Retain 117 complete pairs and a torn next record.
        let mut partial = lines[..118].join("\n");
        partial.push_str("\n{\"pair\":117,\"games\":[");
        std::fs::write(&path.0, partial).unwrap();
        config.sequential.as_mut().unwrap().resume = true;
        let resumed = eval(&config, &factory, &factory, None).unwrap();
        assert_eq!(
            serde_json::to_value(&resumed).unwrap(),
            serde_json::to_value(&report).unwrap()
        );
        assert_eq!(std::fs::read_to_string(&path.0).unwrap(), original);
        // A stopped journal requires no player construction.
        let stopped = eval(
            &config,
            &|| panic!("already stopped"),
            &|| panic!("already stopped"),
            None,
        )
        .unwrap();
        assert_eq!(stopped.pairs, 128);
        // Refuse a changed protocol without trimming or appending anything.
        config.seed += 1;
        assert!(eval(&config, &factory, &factory, None).is_err());
        assert_eq!(std::fs::read_to_string(&path.0).unwrap(), original);
    }

    struct ForfeitPlayer;
    impl Player for ForfeitPlayer {
        fn new_game(&mut self) {}
        fn choose(&mut self, _: &engine::State, _: &History, _: Clock) -> engine::Action {
            0
        }
        fn forfeited(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "forfeit-test"
        }
    }

    #[test]
    fn forfeit_finishes_both_colours_and_invalidates_the_first_batch() {
        let path = TempJournal::new();
        let mut config = sequential_config(&path.0);
        let factory = || Ok(Box::new(ForfeitPlayer) as Box<dyn Player>);
        let report = eval(&config, &factory, &factory, None).unwrap();
        assert_eq!(report.pairs, 16);
        assert_eq!(report.forfeits, 32);
        assert_eq!(report.complete_pairs, 0);
        let sequential = report.sequential.unwrap();
        assert_eq!(sequential.state.stop_reason, Stop::Invalid);
        assert_eq!(sequential.state.counts, [0; 5]);
        config.sequential.as_mut().unwrap().resume = true;
        let resumed = eval(
            &config,
            &|| panic!("invalid journal"),
            &|| panic!("invalid journal"),
            None,
        )
        .unwrap();
        assert_eq!(resumed.sequential.unwrap().state.stop_reason, Stop::Invalid);
        // A crash after journalling a forfeit must still finish that batch on resume.
        let original = std::fs::read_to_string(&path.0).unwrap();
        let partial = original.lines().take(4).collect::<Vec<_>>().join("\n") + "\n";
        std::fs::write(&path.0, partial).unwrap();
        let resumed = eval(&config, &factory, &factory, None).unwrap();
        assert_eq!(resumed.pairs, 16);
        assert_eq!(resumed.forfeits, 32);
        assert_eq!(resumed.sequential.unwrap().state.stop_reason, Stop::Invalid);
        assert_eq!(std::fs::read_to_string(&path.0).unwrap(), original);
    }

    #[test]
    fn factory_failure_closes_waiting_workers() {
        let config = EvalConfig {
            pairs: 2,
            threads: 2,
            ..EvalConfig::default()
        };
        assert!(eval(
            &config,
            &|| anyhow::bail!("factory failure"),
            &|| panic!("unused"),
            None
        )
        .is_err());
    }

    #[test]
    fn journal_has_one_writer_and_rejects_a_complete_corrupt_line() {
        let path = TempJournal::new();
        let protocol = serde_json::json!({"schema":1});
        let first = journal::Journal::open(&path.0, false, &protocol, |_, _| Ok(())).unwrap();
        assert!(journal::Journal::open(&path.0, true, &protocol, |_, _| Ok(())).is_err());
        drop(first);
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path.0)
            .unwrap();
        file.write_all(b"not-json\n").unwrap();
        drop(file);
        let bytes = std::fs::read(&path.0).unwrap();
        assert!(journal::Journal::open(&path.0, true, &protocol, |_, _| Ok(())).is_err());
        assert_eq!(std::fs::read(&path.0).unwrap(), bytes);
    }

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
