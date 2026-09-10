//! `bot`: eval, play, rpsi and analyse over every model crate.

use std::io::{self, BufReader, Write};
use std::path::PathBuf;

use anyhow::{bail, ensure, Result};
use clap::{Parser, Subcommand};
use engine::Rules;
use r#match::{
    analysis, eval, play, Analyser, Clock, EvalConfig, Opening, Player, PlayerFactory, RpsiPlayer,
    Session,
};
use rand::{rngs::StdRng, SeedableRng};

/// Crate version plus the git short hash and dirty marker captured by build.rs.
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), env!("BOT_GIT"));

#[derive(Parser)]
#[command(name = "bot", version = VERSION)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// NNUE migration conversion, parity validation and fixed-node diagnostics.
    Nnue {
        #[command(subcommand)]
        command: NnueCommand,
    },
    /// Paired evaluation at simulation or per-player wall-time budgets.
    Eval {
        #[arg(long)]
        candidate: String,
        /// The named eval reference, relative to the workspace root.
        #[arg(long, default_value = "sq:weights/sq_g128.onnx")]
        reference: String,
        #[arg(long, default_value_t = 32)]
        pairs: usize,
        #[arg(long, conflicts_with = "move_ms")]
        sims: Option<u32>,
        #[arg(long)]
        move_ms: Option<u64>,
        /// Reference wall time; defaults to move-ms, follows the player across seats.
        #[arg(long, requires = "move_ms")]
        reference_move_ms: Option<u64>,
        /// Fixed simulations for the reference while the candidate uses move-ms.
        #[arg(long, requires = "move_ms", conflicts_with = "reference_move_ms")]
        reference_sims: Option<u32>,
        #[arg(long, default_value_t = 1)]
        player_threads: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Random plies before the players take over.
        #[arg(long, default_value_t = 8)]
        opening_plies: usize,
        /// Write every game as JSON lines here.
        #[arg(long)]
        records: Option<PathBuf>,
        /// Print every game's start, moves and end as JSON lines while playing;
        /// the report follows as an `event: report` line.
        #[arg(long)]
        stream: bool,
    },
    /// One game between two players, printed as JSON.
    Play {
        /// Collect evaluated leaves from the first player only.
        #[arg(long)]
        leaves: Option<PathBuf>,
        #[arg(long)]
        first: String,
        #[arg(long)]
        second: String,
        #[arg(long, conflicts_with = "move_ms")]
        sims: Option<u32>,
        #[arg(long)]
        move_ms: Option<u64>,
        #[arg(long, default_value_t = 1)]
        player_threads: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 8)]
        opening_plies: usize,
    },
    /// Serve a player to the site client over RPSI on stdin and stdout.
    Rpsi {
        #[arg(long)]
        player: String,
        /// Wall time per move without a clock, and the floor with one.
        #[arg(long, default_value_t = 250)]
        move_ms: i64,
        /// Cap on the per-move thinking time under a clock.
        #[arg(long, default_value_t = 250)]
        max_move_ms: i64,
        /// Simulations per move when the host gives no time.
        #[arg(long, default_value_t = 32)]
        sims: u32,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        #[arg(long, default_value = "intransitive_bot")]
        name: String,
    },
    /// Analyse positions for an engine: JSON requests on stdin, one response
    /// line each (match::analysis).
    Analyse {
        #[arg(long)]
        engine: String,
        #[arg(long, default_value_t = 1)]
        threads: usize,
    },
}

#[derive(Subcommand)]
enum NnueCommand {
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Validate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
    },
    Diagnose {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        nodes: u64,
        #[arg(long, default_value_t = 1)]
        threads: usize,
    },
}

fn move_clock(sims: Option<u32>, ms: Option<u64>) -> Result<Clock> {
    ensure!(
        ms.is_none_or(|v| v > 0) && sims.is_none_or(|v| v > 0),
        "move-ms and sims must be positive"
    );
    Ok(ms.map_or(Clock::Sims(sims.unwrap_or(32)), |v| {
        Clock::Time(std::time::Duration::from_millis(v))
    }))
}
fn nnue_command(command: NnueCommand) -> Result<()> {
    match command {
        NnueCommand::Convert { input, output } => {
            let bytes =
                nnue::net::convert_v3(&std::fs::read(input)?).map_err(anyhow::Error::msg)?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            println!("{}", serde_json::json!({"format":6,"bytes":bytes.len()}));
        }
        NnueCommand::Validate { model, input } => {
            let model = nnue::Model::load(model).map_err(anyhow::Error::msg)?;
            nnue::diagnostic::run(
                &model,
                BufReader::new(std::fs::File::open(input)?),
                io::stdout().lock(),
                None,
                1,
            )?;
        }
        NnueCommand::Diagnose {
            model,
            input,
            nodes,
            threads,
        } => {
            let model = nnue::Model::load(model).map_err(anyhow::Error::msg)?;
            nnue::diagnostic::run(
                &model,
                BufReader::new(std::fs::File::open(input)?),
                io::stdout().lock(),
                Some(nodes),
                threads,
            )?;
        }
    }
    Ok(())
}

/// `<model>:<path>` -> a player, one arm per model crate; `rpsi:<command>`
/// seats an external engine speaking RPSI.
fn player_from_spec(spec: &str, threads: usize) -> Result<Box<dyn Player>> {
    let Some((model, path)) = spec.split_once(':') else {
        bail!("player spec must be <model>:<path>, got {spec}");
    };
    match model {
        "nnue" => Ok(Box::new(nnue::NnuePlayer::load(path, threads)?)),
        "sq" => Ok(Box::new(sq::SqPlayer::load(path, threads)?)),
        "conv" => Ok(Box::new(conv::ConvPlayer::load(path, threads)?)),
        "rpsi" => {
            let command: Vec<String> = path.split_whitespace().map(str::to_owned).collect();
            Ok(Box::new(RpsiPlayer::spawn(&command)?))
        }
        other => bail!("unknown model {other}"),
    }
}

/// `<model>:<path>` -> an analyser: the model crates and the NNUE.
fn analyser_from_spec(spec: &str, threads: usize) -> Result<Box<dyn Analyser>> {
    let Some((model, path)) = spec.split_once(':') else {
        bail!("engine spec must be <model>:<path>, got {spec}");
    };
    match model {
        "sq" => Ok(Box::new(sq::SqPlayer::load(path, threads)?)),
        "conv" => Ok(Box::new(conv::ConvPlayer::load(path, threads)?)),
        "nnue" => Ok(Box::new(nnue::NnuePlayer::load(path, threads)?)),
        other => bail!("{other} engines cannot be analysed"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Nnue { command } => nnue_command(command)?,
        Command::Eval {
            candidate,
            reference,
            pairs,
            sims,
            move_ms,
            reference_move_ms,
            reference_sims,
            player_threads,
            threads,
            seed,
            opening_plies,
            records,
            stream,
        } => {
            move_clock(sims, move_ms)?;
            ensure!(player_threads > 0, "player-threads must be positive");
            let config = EvalConfig {
                rules: Rules::SITE,
                pairs,
                sims: sims.unwrap_or(32),
                move_ms,
                reference_move_ms,
                reference_sims,
                opening_plies,
                seed,
                threads,
                stream,
            };
            let per_thread = player_threads;
            let candidate_factory: Box<PlayerFactory> =
                Box::new(move || player_from_spec(&candidate, per_thread));
            let reference_factory: Box<PlayerFactory> =
                Box::new(move || player_from_spec(&reference, per_thread));
            let report = eval(
                &config,
                &*candidate_factory,
                &*reference_factory,
                records.as_deref(),
            )?;
            if stream {
                let mut event = serde_json::to_value(&report)?;
                event["event"] = "report".into();
                println!("{event}");
            } else {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        }
        Command::Play {
            first,
            second,
            sims,
            move_ms,
            player_threads,
            leaves,
            seed,
            opening_plies,
        } => {
            ensure!(player_threads > 0, "player-threads must be positive");
            let clock = move_clock(sims, move_ms)?;
            let mut a = player_from_spec(&first, player_threads)?;
            let mut b = player_from_spec(&second, player_threads)?;
            ensure!(
                a.supports_clock(clock) && b.supports_clock(clock),
                "player rejects simulation budgets; use --move-ms"
            );
            if let Some(path) = leaves {
                a.set_leaves(&path)?;
            }
            let mut rng = StdRng::seed_from_u64(seed);
            let opening = Opening::random(&Rules::SITE, opening_plies, &mut rng);
            let record = play(&Rules::SITE, &mut *a, &mut *b, &opening, clock);
            println!("{}", serde_json::to_string_pretty(&record)?);
        }
        Command::Rpsi {
            player,
            move_ms,
            max_move_ms,
            sims,
            threads,
            name,
        } => {
            let player = player_from_spec(&player, threads)?;
            let mut session = Session::new(player, Rules::SITE, &name, move_ms, max_move_ms, sims);
            let stdin = io::stdin();
            let mut input = BufReader::new(stdin.lock());
            let stdout = io::stdout();
            let mut out = stdout.lock();
            session.run(&mut input, &mut out)?;
        }
        Command::Analyse { engine, threads } => {
            let mut analyser = analyser_from_spec(&engine, threads)?;
            let stdin = io::stdin();
            let mut input = BufReader::new(stdin.lock());
            let stdout = io::stdout();
            let mut out = stdout.lock();
            analysis::serve(&mut *analyser, &mut input, &mut out, &Rules::SITE)?;
        }
    }
    Ok(())
}
