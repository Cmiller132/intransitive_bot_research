//! The one evaluation: paired games at equal simulations, no clock.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use engine::Rules;
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;

use crate::game::{play, GameRecord, Opening};
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
    pub opening_plies: usize,
    pub seed: u64,
    /// Pairs run concurrently on this many threads; players are built per thread.
    pub threads: usize,
}

impl Default for EvalConfig {
    fn default() -> EvalConfig {
        EvalConfig {
            rules: Rules::SITE,
            pairs: 32,
            sims: 32,
            opening_plies: 8,
            seed: 0,
            threads: 4,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub candidate: String,
    pub reference: String,
    pub pairs: usize,
    pub sims: u32,
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
    let mut rng = StdRng::seed_from_u64(config.seed);
    let openings: Vec<Opening> = (0..config.pairs)
        .map(|_| Opening::random(&config.rules, config.opening_plies, &mut rng))
        .collect();
    let next = Arc::new(Mutex::new(0usize));
    let games: Arc<Mutex<Vec<Option<Pair>>>> = Arc::new(Mutex::new(vec![None; config.pairs]));
    let threads = config.threads.max(1).min(config.pairs.max(1));
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::new();
        for _ in 0..threads {
            let (next, games, openings) = (Arc::clone(&next), Arc::clone(&games), &openings);
            handles.push(scope.spawn(move || -> Result<()> {
                let mut a = candidate()?;
                let mut b = reference()?;
                loop {
                    let pair = {
                        let mut guard = next.lock().expect("pair counter");
                        let pair = *guard;
                        *guard += 1;
                        pair
                    };
                    if pair >= openings.len() {
                        return Ok(());
                    }
                    let clock = Clock::Sims(config.sims);
                    let first = play(&config.rules, &mut *a, &mut *b, &openings[pair], clock);
                    let second = play(&config.rules, &mut *b, &mut *a, &openings[pair], clock);
                    games.lock().expect("games")[pair] = Some((first, second));
                }
            }));
        }
        for handle in handles {
            handle.join().expect("eval thread")?;
        }
        Ok(())
    })?;
    let games = Arc::try_unwrap(games)
        .expect("threads joined")
        .into_inner()
        .expect("games");
    let (mut wins, mut draws, mut losses) = (0u32, 0u32, 0u32);
    let mut pair_scores = Vec::with_capacity(config.pairs);
    let mut plies = 0u64;
    let mut writer = match records {
        Some(path) => Some(BufWriter::new(
            File::create(path).with_context(|| format!("creating {}", path.display()))?,
        )),
        None => None,
    };
    for pair in games.into_iter().flatten() {
        let mut score = 0.0;
        // In the first game the candidate moved first; in the second, the reference did.
        for (game, candidate_index) in [(&pair.0, 0u8), (&pair.1, 1u8)] {
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
        candidate: candidate()?.name().to_owned(),
        reference: reference()?.name().to_owned(),
        pairs: pair_scores.len(),
        sims: config.sims,
        wins,
        draws,
        losses,
        margin,
        interval: (margin - half_width, margin + half_width),
        mean_plies: plies as f64 / (2.0 * n),
    })
}
