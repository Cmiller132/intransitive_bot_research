//! In-process generation with per-game journals and atomic, ordered gzip shards.
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc,
};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use clap::Args as ClapArgs;
use flate2::{read::GzDecoder, Compression, GzBuilder};
use nnue::search::{Limits, Searcher};
use r#match::selfplay::{self, Config, Decision, Event, Label, Record, ScoreKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long, required_unless_present = "verify")]
    player: Option<String>,
    #[arg(long, default_value_t = 250_000)]
    nodes: u64,
    #[arg(long, required_unless_present = "verify")]
    games: Option<u64>,
    /// Independent one-thread games; no search SMP.
    #[arg(long, default_value_t = 16)]
    threads: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 8)]
    opening_plies: u32,
    #[arg(long, default_value_t = 0)]
    random_moves: u32,
    #[arg(long, default_value_t = 8)]
    random_from: u32,
    #[arg(long, default_value_t = 40)]
    random_to: u32,
    #[arg(long, default_value_t = 1)]
    multipv: u32,
    #[arg(long, default_value_t = 60)]
    multipv_margin: i32,
    #[arg(long, default_value_t = 1024)]
    max_plies: u32,
    #[arg(long)]
    records: PathBuf,
    #[arg(long, default_value_t = 256)]
    shard_games: u64,
    #[arg(long, conflicts_with = "verify")]
    resume: bool,
    /// Verify all published shard hashes, game replay, labels and outcomes.
    #[arg(long)]
    verify: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Settings {
    player: String,
    nodes: u64,
    games: u64,
    threads: usize,
    hash_mib: usize,
    seed: u64,
    #[serde(flatten)]
    game: Config,
    multipv: u32,
    multipv_margin: i32,
    shard_games: u64,
    compression_level: u32,
    max_depth: u8,
    logical_cpus: Vec<usize>,
    tt_policy: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Identity {
    schema: u32,
    model_sha256: String,
    binary_sha256: String,
    rules_sha256: String,
    capture_clock: u32,
    repetition_draw: bool,
    score_adjudication: bool,
    score_scale: u32,
    seed_algorithm: String,
    settings: Settings,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Counts {
    games: u64,
    roots: u64,
    labelled_roots: u64,
    eligible_roots: u64,
    censored: u64,
    nodes: u64,
    search_ns: u64,
}
impl Counts {
    fn add(&mut self, r: &Record) {
        self.games += 1;
        self.roots += r.roots.len() as u64;
        self.labelled_roots += r.roots.iter().filter(|r| r.root_score.is_some()).count() as u64;
        self.eligible_roots += r.eligible_roots() as u64;
        self.censored += u64::from(r.censored);
        self.nodes += r.roots.iter().map(|r| r.nodes).sum::<u64>();
        self.search_ns += r.roots.iter().map(|r| r.elapsed_ns).sum::<u64>();
    }
    fn merge(&mut self, other: &Self) {
        self.games += other.games;
        self.roots += other.roots;
        self.labelled_roots += other.labelled_roots;
        self.eligible_roots += other.eligible_roots;
        self.censored += other.censored;
        self.nodes += other.nodes;
        self.search_ns += other.search_ns;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Shard {
    file: String,
    first_game: u64,
    sha256: String,
    bytes: u64,
    counts: Counts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Manifest {
    #[serde(flatten)]
    identity: Identity,
    shards: Vec<Shard>,
    complete: bool,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(crate) fn file_hash(path: &Path) -> Result<String> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn rules_hash() -> String {
    let mut hash = Sha256::new();
    for source in [
        include_bytes!("../../engine/src/rules.rs").as_slice(),
        include_bytes!("../../engine/src/board.rs"),
        include_bytes!("../../engine/src/tactics.rs"),
        include_bytes!("../../engine/src/notation.rs"),
    ] {
        let source = String::from_utf8_lossy(source).replace("\r\n", "\n");
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temp = path.with_extension("tmp");
    let mut file = BufWriter::new(File::create(&temp)?);
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.get_ref().sync_all()?;
    drop(file);
    fs::rename(temp, path)?;
    Ok(())
}

fn pending(dir: &Path, id: u64) -> PathBuf {
    dir.join("pending").join(format!("{id:012}.json"))
}
fn active(dir: &Path, id: u64) -> PathBuf {
    dir.join("active").join(format!("{id:012}.jsonl"))
}

struct Journal {
    writer: BufWriter<File>,
}
impl Journal {
    fn new(path: &Path, record: &Record) -> Result<Self> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut this = Self {
            writer: BufWriter::new(file),
        };
        this.line(record)?;
        this.writer.get_ref().sync_all()?;
        Ok(this)
    }
    fn line(&mut self, value: &impl Serialize) -> Result<()> {
        serde_json::to_writer(&mut self.writer, value)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
    fn finish(mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }
}

/// Only complete journal lines are committed decisions; a torn final write is ignored.
fn recover_journal(path: &Path, initial: &Record) -> Result<Record> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = Vec::new();
    reader.read_until(b'\n', &mut line)?;
    if line.last() != Some(&b'\n') {
        return Ok(initial.clone());
    }
    let mut record: Record = serde_json::from_slice(&line)?;
    ensure!(
        serde_json::to_value(&record)? == serde_json::to_value(initial)?,
        "journal header differs from seeded initial game"
    );
    let mut state = record.initial.state()?;
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 || line.last() != Some(&b'\n') {
            break;
        }
        let event: Event = serde_json::from_slice(&line)?;
        record.append(&mut state, event)?;
    }
    record.finish(&state, r#match::End::Interrupted);
    record.verify()?;
    Ok(record)
}

fn decision(result: nnue::search::SearchResult, state: &engine::State) -> Result<Decision> {
    let action = result.action.context("NNUE returned no legal action")?;
    let label = result
        .completed_root()
        .map(|(action, score, depth)| Label {
            action,
            score,
            depth,
            kind: ScoreKind::Search,
        })
        .or_else(|| {
            (engine::apply(&engine::Rules::SITE, state, action).1 == engine::Outcome::Win)
                .then_some(Label {
                    action,
                    score: 29_999,
                    depth: 0,
                    kind: ScoreKind::EngineProof,
                })
        });
    Ok(Decision {
        action,
        label,
        nodes: result.nodes,
        elapsed_ns: result
            .elapsed
            .as_nanos()
            .try_into()
            .context("search timer overflow")?,
    })
}

fn load_manifest(dir: &Path) -> Result<Manifest> {
    serde_json::from_reader(BufReader::new(File::open(dir.join("manifest.json"))?))
        .map_err(Into::into)
}

fn verify_record(record: &Record, settings: &Settings) -> Result<()> {
    let expected = Record::new(
        &settings.game,
        &settings.player,
        record.game_id,
        selfplay::derive_seed(settings.seed, record.game_id),
    )?;
    ensure!(
        record.game_id < settings.games
            && record.seed == expected.seed
            && record.initial == expected.initial
            && record.opening_plies == expected.opening_plies
            && record.random_plan == expected.random_plan
            && record.game.first == expected.game.first
            && record.game.second == expected.game.second
            && record.game.plies <= settings.game.max_plies
            && (record.game.end != r#match::End::PlyCap
                || record.game.plies == settings.game.max_plies)
            && record
                .roots
                .iter()
                .all(|root| root.nodes <= settings.nodes
                    && root.completed_depth <= settings.max_depth),
        "record differs from effective settings"
    );
    record.verify()
}

/// Verify every published byte and replay every game; usable without loading a model.
fn verify(dir: &Path, manifest: &Manifest) -> Result<Counts> {
    ensure!(
        manifest.identity.schema == selfplay::SCHEMA
            && manifest.identity.rules_sha256 == rules_hash(),
        "manifest schema/rules do not match this engine"
    );
    let mut total = Counts::default();
    for (index, shard) in manifest.shards.iter().enumerate() {
        ensure!(
            shard.file == format!("shard-{index:06}.jsonl.gz") && shard.first_game == total.games,
            "noncontiguous shard manifest"
        );
        let path = dir.join(&shard.file);
        ensure!(
            fs::metadata(&path)?.len() == shard.bytes && file_hash(&path)? == shard.sha256,
            "shard hash mismatch: {}",
            shard.file
        );
        let mut counts = Counts::default();
        for line in BufReader::new(GzDecoder::new(File::open(path)?)).lines() {
            let record: Record = serde_json::from_str(&line?)?;
            ensure!(
                record.game_id == shard.first_game + counts.games
                    && record.seed
                        == selfplay::derive_seed(manifest.identity.settings.seed, record.game_id),
                "game identity mismatch"
            );
            verify_record(&record, &manifest.identity.settings)?;
            counts.add(&record);
        }
        ensure!(
            counts == shard.counts && counts.games > 0,
            "shard counts mismatch"
        );
        total.merge(&counts);
    }
    ensure!(
        total.games <= manifest.identity.settings.games
            && manifest.complete == (total.games == manifest.identity.settings.games),
        "manifest completion mismatch"
    );
    Ok(total)
}

/// Publish only fixed game-ID ranges so resuming never changes shard boundaries.
fn publish(dir: &Path, manifest: &mut Manifest) -> Result<()> {
    loop {
        let first = manifest.shards.iter().map(|s| s.counts.games).sum::<u64>();
        let count = manifest
            .identity
            .settings
            .shard_games
            .min(manifest.identity.settings.games - first);
        if count == 0 {
            break;
        }
        if !(first..first + count).all(|id| pending(dir, id).exists()) {
            break;
        }
        let name = format!("shard-{:06}.jsonl.gz", manifest.shards.len());
        let temp = dir.join(format!("{name}.tmp"));
        let mut gz = GzBuilder::new().mtime(0).operating_system(255).write(
            BufWriter::new(File::create(&temp)?),
            Compression::new(manifest.identity.settings.compression_level),
        );
        let mut counts = Counts::default();
        for id in first..first + count {
            let bytes = fs::read(pending(dir, id))?;
            let record: Record = serde_json::from_slice(&bytes)?;
            ensure!(
                record.game_id == id
                    && record.seed == selfplay::derive_seed(manifest.identity.settings.seed, id),
                "pending game identity mismatch"
            );
            verify_record(&record, &manifest.identity.settings)?;
            counts.add(&record);
            gz.write_all(&bytes)?;
        }
        let mut writer = gz.finish()?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        let sha256 = file_hash(&temp)?;
        let bytes = fs::metadata(&temp)?.len();
        fs::rename(&temp, dir.join(&name))?;
        manifest.shards.push(Shard {
            file: name,
            first_game: first,
            sha256,
            bytes,
            counts,
        });
        manifest.complete = first + count == manifest.identity.settings.games;
        atomic_json(&dir.join("manifest.json"), manifest)?;
        for id in first..first + count {
            fs::remove_file(pending(dir, id))?;
        }
    }
    Ok(())
}

fn recover(dir: &Path, manifest: &mut Manifest) -> Result<Vec<u64>> {
    let completed = verify(dir, manifest)?.games;
    // A published shard may have committed immediately before journal cleanup.
    for entry in fs::read_dir(dir.join("pending"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|s| s == "json") {
            let r: Record = serde_json::from_reader(BufReader::new(File::open(&path)?))?;
            ensure!(
                path == pending(dir, r.game_id) && r.game_id < manifest.identity.settings.games,
                "unexpected pending record"
            );
            verify_record(&r, &manifest.identity.settings)?;
            if r.game_id < completed {
                fs::remove_file(path)?;
            }
        } else {
            ensure!(
                path.extension().is_some_and(|s| s == "tmp"),
                "unexpected pending file"
            );
            fs::remove_file(path)?;
        }
    }
    for entry in fs::read_dir(dir.join("active"))? {
        let path = entry?.path();
        let id: u64 = path
            .file_stem()
            .context("invalid journal name")?
            .to_str()
            .context("invalid journal name")?
            .parse()?;
        ensure!(
            path == active(dir, id) && id < manifest.identity.settings.games,
            "unexpected active record"
        );
        if id >= completed && !pending(dir, id).exists() {
            let initial = Record::new(
                &manifest.identity.settings.game,
                &manifest.identity.settings.player,
                id,
                selfplay::derive_seed(manifest.identity.settings.seed, id),
            )?;
            let r = recover_journal(&path, &initial)?;
            ensure!(
                r.game_id == id
                    && r.seed == selfplay::derive_seed(manifest.identity.settings.seed, id),
                "journal identity mismatch"
            );
            verify_record(&r, &manifest.identity.settings)?;
            atomic_json(&pending(dir, id), &r)?;
        }
        fs::remove_file(path)?;
    }
    publish(dir, manifest)?;
    let completed = manifest.shards.iter().map(|s| s.counts.games).sum::<u64>();
    Ok((completed..manifest.identity.settings.games)
        .filter(|&id| !pending(dir, id).exists())
        .collect())
}

pub fn run(args: Args) -> Result<()> {
    if args.verify {
        let manifest = load_manifest(&args.records)?;
        println!(
            "{}",
            serde_json::json!({"event":"verified", "counts":verify(&args.records, &manifest)?, "complete":manifest.complete})
        );
        return Ok(());
    }
    ensure!(
        args.multipv == 1,
        "this generator supports only --multipv 1"
    );
    ensure!(
        args.multipv_margin >= 0 && args.nodes > 0 && args.threads > 0 && args.shard_games > 0,
        "invalid search/worker/shard setting"
    );
    let games = args.games.context("--games required")?;
    ensure!(games > 0, "games must be positive");
    let spec = args.player.context("--player required")?;
    let spec = spec
        .strip_prefix("nnue:")
        .context("selfplay currently requires a CPU nnue:<model> player")?;
    let (path, hash_mib) = nnue::player_options(spec)?;
    let path = fs::canonicalize(path)?;
    let bytes = fs::read(&path)?;
    let model = nnue::Model::from_bytes(&bytes).map_err(anyhow::Error::msg)?;
    let settings = Settings {
        player: format!("nnue:{}?hash={hash_mib}", path.display()),
        nodes: args.nodes,
        games,
        threads: args.threads,
        hash_mib,
        seed: args.seed,
        game: Config {
            opening_plies: args.opening_plies,
            random_moves: args.random_moves,
            random_from: args.random_from,
            random_to: args.random_to,
            max_plies: args.max_plies,
        },
        multipv: args.multipv,
        multipv_margin: args.multipv_margin,
        shard_games: args.shard_games,
        compression_level: 1,
        max_depth: 64,
        logical_cpus: core_affinity::get_core_ids()
            .context("cannot read CPU affinity")?
            .iter()
            .map(|c| c.id)
            .collect(),
        tt_policy: "clear_per_game_retain_between_moves".into(),
    };
    Record::new(
        &settings.game,
        &settings.player,
        0,
        selfplay::derive_seed(settings.seed, 0),
    )?;
    let identity = Identity {
        schema: selfplay::SCHEMA,
        model_sha256: digest(&bytes),
        binary_sha256: file_hash(&std::env::current_exe()?)?,
        rules_sha256: rules_hash(),
        capture_clock: 200,
        repetition_draw: false,
        score_adjudication: false,
        score_scale: 600,
        seed_algorithm:
            "splitmix64-v1; game=derive(seed,id); opening=derive(game,0); injection=derive(game,1)"
                .into(),
        settings,
    };
    fs::create_dir_all(&args.records)?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(args.records.join("writer.lock"))?;
    lock.try_lock()
        .context("another generator owns this records directory")?;
    let mut manifest = if args.resume {
        let existing = load_manifest(&args.records)?;
        ensure!(
            existing.identity == identity,
            "resume rejected: model, binary, rules or effective settings differ"
        );
        existing
    } else {
        ensure!(
            fs::read_dir(&args.records)?.all(|e| e.is_ok_and(|e| e.file_name() == "writer.lock")),
            "records directory is not empty; use --resume with identical settings"
        );
        let new = Manifest {
            identity,
            shards: Vec::new(),
            complete: false,
        };
        atomic_json(&args.records.join("manifest.json"), &new)?;
        new
    };
    fs::create_dir_all(args.records.join("pending"))?;
    fs::create_dir_all(args.records.join("active"))?;
    let jobs = recover(&args.records, &mut manifest)?;
    let stop = Arc::new(AtomicU64::new(0));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        signal.store(1, Ordering::Relaxed);
    })?;
    let start = Instant::now();
    let mut new_counts = Counts::default();
    let mut worker_time = Duration::ZERO;
    let mut publication_time = Duration::ZERO;
    let worker_count = args.threads.min(jobs.len());
    let next = AtomicU64::new(0);
    let settings = &manifest.identity.settings.clone();
    let dir = &args.records;
    let outcome: Result<()> = std::thread::scope(|scope| {
        let (sender, receiver) =
            mpsc::sync_channel::<Result<(Counts, Duration)>>(worker_count.max(1));
        for _ in 0..worker_count {
            let sender = sender.clone();
            let jobs = &jobs;
            let next = &next;
            let model = &model;
            let stop = Arc::clone(&stop);
            scope.spawn(move || {
                let mut searcher = Searcher::new(settings.hash_mib);
                searcher.set_stop_signal(Arc::clone(&stop), 0);
                while stop.load(Ordering::Relaxed) == 0 {
                    let index = next.fetch_add(1, Ordering::Relaxed) as usize;
                    let Some(&id) = jobs.get(index) else {
                        break;
                    };
                    let result = (|| -> Result<(Counts, Duration)> {
                        let game_start = Instant::now();
                        searcher.clear();
                        let record = Record::new(
                            &settings.game,
                            &settings.player,
                            id,
                            selfplay::derive_seed(settings.seed, id),
                        )?;
                        let mut journal = Journal::new(&active(dir, id), &record)?;
                        let record = selfplay::generate(
                            record,
                            &settings.game,
                            &mut |state| {
                                decision(
                                    searcher.search(
                                        model,
                                        state,
                                        None,
                                        Limits {
                                            nodes: settings.nodes,
                                            time: Duration::MAX,
                                            depth: settings.max_depth,
                                        },
                                    ),
                                    state,
                                )
                            },
                            &|| stop.load(Ordering::Relaxed) != 0,
                            &mut |event| journal.line(event),
                        )?;
                        journal.finish()?;
                        atomic_json(&pending(dir, id), &record)?;
                        fs::remove_file(active(dir, id))?;
                        let mut counts = Counts::default();
                        counts.add(&record);
                        Ok((counts, game_start.elapsed()))
                    })();
                    if result.is_err() {
                        stop.store(1, Ordering::Relaxed);
                    }
                    if sender.send(result).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        let mut failure = None;
        for result in receiver {
            match result {
                Ok((counts, elapsed)) => {
                    new_counts.merge(&counts);
                    worker_time += elapsed;
                    if failure.is_none() {
                        let publication_start = Instant::now();
                        if let Err(error) = publish(dir, &mut manifest) {
                            stop.store(1, Ordering::Relaxed);
                            failure = Some(error);
                        }
                        publication_time += publication_start.elapsed();
                    }
                    if new_counts.games.is_multiple_of(16) {
                        eprintln!(
                            "{}",
                            serde_json::json!({"event":"progress", "new_games":new_counts.games,
                            "eligible_roots":new_counts.eligible_roots, "wall_seconds":start.elapsed().as_secs_f64()})
                        );
                    }
                }
                Err(error) => {
                    stop.store(1, Ordering::Relaxed);
                    failure.get_or_insert(error);
                }
            }
        }
        if let Some(error) = failure {
            Err(error)
        } else {
            Ok(())
        }
    });
    outcome?;
    let publication_start = Instant::now();
    publish(dir, &mut manifest)?;
    publication_time += publication_start.elapsed();
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"event":"selfplay", "complete":manifest.complete,
        "interrupted":stop.load(Ordering::Relaxed) != 0, "new":new_counts,
        "published_games":manifest.shards.iter().map(|s| s.counts.games).sum::<u64>(),
        "wall_seconds":seconds, "games_per_hour":new_counts.games as f64*3600./seconds,
        "worker_game_seconds":worker_time.as_secs_f64(),
        "worker_outside_search_seconds":worker_time.saturating_sub(Duration::from_nanos(new_counts.search_ns)).as_secs_f64(),
        "publication_seconds":publication_time.as_secs_f64(),
        "eligible_roots_per_hour":new_counts.eligible_roots as f64*3600./seconds})
    );
    if stop.load(Ordering::Relaxed) != 0 {
        bail!("generation interrupted; durable games preserved, resume with the same settings");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Manifest {
        Manifest {
            identity: Identity {
                schema: selfplay::SCHEMA,
                model_sha256: "model".into(),
                binary_sha256: "binary".into(),
                rules_sha256: rules_hash(),
                capture_clock: 200,
                repetition_draw: false,
                score_adjudication: false,
                score_scale: 600,
                seed_algorithm: "test".into(),
                settings: Settings {
                    player: "nnue:test?hash=1".into(),
                    nodes: 10,
                    games: 4,
                    threads: 2,
                    hash_mib: 1,
                    seed: 19,
                    game: Config {
                        opening_plies: 0,
                        random_moves: 0,
                        random_from: 0,
                        random_to: 10,
                        max_plies: 4,
                    },
                    multipv: 1,
                    multipv_margin: 60,
                    shard_games: 2,
                    compression_level: 1,
                    max_depth: 64,
                    logical_cpus: vec![16, 17],
                    tt_policy: "clear_per_game_retain_between_moves".into(),
                },
            },
            shards: vec![],
            complete: false,
        }
    }

    #[test]
    fn durable_records_republish_byte_identical_shards_and_recover_torn_events() {
        let dir = std::env::temp_dir().join(format!(
            "bot-selfplay-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(dir.join("active")).unwrap();
        fs::create_dir_all(dir.join("pending")).unwrap();
        let mut manifest = fixture();
        let settings = manifest.identity.settings.clone();
        let mut bytes = Vec::new();
        for id in 0..4 {
            let record = Record::new(
                &settings.game,
                &settings.player,
                id,
                selfplay::derive_seed(settings.seed, id),
            )
            .unwrap();
            let mut journal = Journal::new(&active(&dir, id), &record).unwrap();
            let record = selfplay::generate(
                record,
                &settings.game,
                &mut |state| {
                    let action = state.legal_actions()[0];
                    Ok(Decision {
                        action,
                        label: Some(Label {
                            action,
                            score: -77,
                            depth: 1,
                            kind: ScoreKind::Search,
                        }),
                        nodes: 10,
                        elapsed_ns: 12345,
                    })
                },
                &|| false,
                &mut |event| journal.line(event),
            )
            .unwrap();
            journal.finish().unwrap();
            atomic_json(&pending(&dir, id), &record).unwrap();
            bytes.push(fs::read(pending(&dir, id)).unwrap());
        }
        publish(&dir, &mut manifest).unwrap();
        assert_eq!(verify(&dir, &manifest).unwrap().games, 4);
        let expected: Vec<_> = manifest
            .shards
            .iter()
            .map(|s| fs::read(dir.join(&s.file)).unwrap())
            .collect();
        // Model a crash after shard rename but before the manifest commit.
        for (id, data) in bytes.iter().enumerate() {
            fs::write(pending(&dir, id as u64), data).unwrap();
        }
        manifest.shards.clear();
        manifest.complete = false;
        assert!(recover(&dir, &mut manifest).unwrap().is_empty());
        for (shard, expected) in manifest.shards.iter().zip(expected) {
            assert_eq!(fs::read(dir.join(&shard.file)).unwrap(), expected);
        }
        assert_eq!(verify(&dir, &manifest).unwrap().games, 4);
        assert!(recover(&dir, &mut manifest).unwrap().is_empty());
        let mut wrong = settings.clone();
        wrong.nodes = 9;
        assert!(verify_record(
            &serde_json::from_slice::<Record>(&bytes[0]).unwrap(),
            &wrong
        )
        .is_err());

        let record = Record::new(
            &settings.game,
            &settings.player,
            0,
            selfplay::derive_seed(settings.seed, 0),
        )
        .unwrap();
        let path = active(&dir, 0);
        let mut journal = Journal::new(&path, &record).unwrap();
        let state = engine::State::initial();
        journal
            .line(&Event {
                ply: 0,
                action: Some(state.legal_actions()[0]),
                random: true,
                skipped: false,
                root: None,
            })
            .unwrap();
        journal.finish().unwrap();
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"ply\":1")
            .unwrap();
        // The fixture header has no random opening: the replay catches the mismatch.
        assert!(recover_journal(&path, &record).is_err());
        fs::remove_file(&path).unwrap();
        let mut config = settings.game.clone();
        config.opening_plies = 1;
        let record = Record::new(
            &config,
            &settings.player,
            0,
            selfplay::derive_seed(settings.seed, 0),
        )
        .unwrap();
        let header = record.clone();
        let mut journal = Journal::new(&path, &record).unwrap();
        let mut one_ply = config.clone();
        one_ply.max_plies = 1;
        let record = selfplay::generate(
            record,
            &one_ply,
            &mut |_| unreachable!(),
            &|| false,
            &mut |event| journal.line(event),
        )
        .unwrap();
        journal.finish().unwrap();
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"ply\":1")
            .unwrap();
        let recovered = recover_journal(&path, &header).unwrap();
        assert_eq!(recovered.game.moves, record.game.moves);
        assert_eq!(recovered.game.end, r#match::End::Interrupted);
        assert!(recovered.censored && recovered.outcome.is_none());
        fs::write(&path, b"{\"schema\":").unwrap();
        let empty = recover_journal(&path, &header).unwrap();
        empty.verify().unwrap();
        assert!(empty.game.moves.is_empty() && empty.censored && empty.outcome.is_none());
        fs::remove_dir_all(dir).unwrap();
    }
}
