//! `bot`: eval, play and rpsi over every model crate.

use std::io::{self, BufReader};
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use engine::Rules;
use r#match::{eval, play, Clock, EvalConfig, Opening, Player, PlayerFactory, Session};
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
    /// Paired evaluation of a candidate against a reference at equal simulations.
    Eval {
        #[arg(long)]
        candidate: String,
        /// The named eval reference, relative to the workspace root.
        #[arg(long, default_value = "sq:weights/sq_g128.onnx")]
        reference: String,
        #[arg(long, default_value_t = 32)]
        pairs: usize,
        #[arg(long, default_value_t = 32)]
        sims: u32,
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
    },
    /// One game between two players, printed as JSON.
    Play {
        #[arg(long)]
        first: String,
        #[arg(long)]
        second: String,
        #[arg(long, default_value_t = 32)]
        sims: u32,
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
}

/// `<model>:<path>` -> a player. One arm per model crate.
fn player_from_spec(spec: &str, threads: usize) -> Result<Box<dyn Player>> {
    let Some((model, path)) = spec.split_once(':') else {
        bail!("player spec must be <model>:<path>, got {spec}");
    };
    match model {
        "sq" => Ok(Box::new(sq::SqPlayer::load(path, threads)?)),
        other => bail!("unknown model {other}"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Eval {
            candidate,
            reference,
            pairs,
            sims,
            threads,
            seed,
            opening_plies,
            records,
        } => {
            let config = EvalConfig {
                rules: Rules::SITE,
                pairs,
                sims,
                opening_plies,
                seed,
                threads,
            };
            let per_thread = 1;
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
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Play {
            first,
            second,
            sims,
            seed,
            opening_plies,
        } => {
            let mut a = player_from_spec(&first, 4)?;
            let mut b = player_from_spec(&second, 4)?;
            let mut rng = StdRng::seed_from_u64(seed);
            let opening = Opening::random(&Rules::SITE, opening_plies, &mut rng);
            let record = play(&Rules::SITE, &mut *a, &mut *b, &opening, Clock::Sims(sims));
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
    }
    Ok(())
}
