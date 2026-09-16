//! Sequential eval options and identities of the coordinator, engine and network files.
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use clap::Args as ClapArgs;
use r#match::eval::Sequential;
use r#match::sprt;
use serde_json::{json, Value};

#[derive(ClapArgs)]
pub struct Args {
    /// Pentanomial SPRT journal: 16-pair batches, first check 128, and the
    /// hypotheses and cap of --sprt-target and --sprt-cap.
    #[arg(long, conflicts_with_all = ["pairs", "records"])]
    pub sprt: Option<PathBuf>,
    /// Upper score hypothesis of the sequential test; the null stays 0.50.
    #[arg(long, requires = "sprt", default_value_t = sprt::S1)]
    sprt_target: f64,
    /// Pairs after which an undecided sequential test stops inconclusive;
    /// a multiple of 16, at least 128.
    #[arg(long, requires = "sprt", default_value_t = sprt::CAP)]
    sprt_cap: usize,
    /// Resume the journal; bounds, settings, openings and artifact hashes must match.
    #[arg(long, requires = "sprt")]
    resume: bool,
    /// Actual weights used by an external RPSI candidate (native specs infer this).
    #[arg(long, requires = "sprt")]
    candidate_network: Option<PathBuf>,
    /// Actual weights used by an external RPSI reference (native specs infer this).
    #[arg(long, requires = "sprt")]
    reference_network: Option<PathBuf>,
}

impl Args {
    /// The hypotheses and cap this run tests under; the journal records them.
    fn bounds(&self) -> Result<sprt::Bounds> {
        let bounds = sprt::Bounds {
            s0: sprt::S0,
            s1: self.sprt_target,
            cap: self.sprt_cap,
        };
        bounds
            .validate()
            .context("--sprt-target and --sprt-cap are the bounds of the test")?;
        Ok(bounds)
    }

    pub fn config(
        &self,
        candidate: &str,
        reference: &str,
        player_threads: usize,
    ) -> Result<Option<Sequential>> {
        let Some(journal) = &self.sprt else {
            return Ok(None);
        };
        let bounds = self.bounds()?;
        let executable = std::env::current_exe()?;
        let coordinator = artifact(&executable)?;
        Ok(Some(Sequential {
            journal: journal.clone(),
            resume: self.resume,
            bounds,
            provenance: json!({
                "coordinator":coordinator,
                "version":super::VERSION,
                "player_threads":player_threads,
                "logical_cpus":core_affinity::get_core_ids().context("cannot read process affinity")?.iter().map(|c|c.id).collect::<Vec<_>>(),
                "candidate":identity(candidate, self.candidate_network.as_deref(), &executable)?,
                "reference":identity(reference, self.reference_network.as_deref(), &executable)?,
            }),
        }))
    }
}

fn artifact(path: &Path) -> Result<Value> {
    let path = path
        .canonicalize()
        .with_context(|| format!("artifact must be an existing file: {}", path.display()))?;
    ensure!(path.is_file(), "artifact is not a file: {}", path.display());
    Ok(json!({"path":path,"sha256":super::selfplay::file_hash(&path)?}))
}

fn identity(spec: &str, external_network: Option<&Path>, coordinator: &Path) -> Result<Value> {
    let (kind, value) = spec
        .split_once(':')
        .context("player spec must be model:path")?;
    let (binary, network) = match kind {
        "nnue" | "sq" | "conv" => {
            ensure!(
                external_network.is_none(),
                "network overrides are only for external RPSI players"
            );
            let path = if kind == "nnue" {
                value.split('?').next().unwrap()
            } else {
                value
            };
            (coordinator, Path::new(path))
        }
        "rpsi" => (
            Path::new(
                value
                    .split_whitespace()
                    .next()
                    .context("empty RPSI command")?,
            ),
            external_network.context(
                "SPRT external engines require --candidate-network / --reference-network",
            )?,
        ),
        _ => bail!("unknown model {kind}"),
    };
    Ok(json!({"spec":spec,"binary":artifact(binary)?,"network":artifact(network)?}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The eval arguments of a parsed command line, sequential options included.
    fn eval_args(extra: &[&str]) -> Result<super::super::Command> {
        let base = ["bot", "eval", "--candidate", "nnue:fake"];
        let cli = super::super::Cli::try_parse_from(base.into_iter().chain(extra.iter().copied()))?;
        Ok(cli.command)
    }

    #[test]
    fn sequential_options_cannot_change_cap_or_omit_resume_journal() {
        for extra in [
            vec!["--resume"],
            vec!["--sprt", "x", "--pairs", "128"],
            vec!["--sprt", "x", "--records", "y"],
            // The cap and the target are settable through their own options only.
            vec!["--sprt-cap", "128"],
            vec!["--sprt-target", "0.515"],
        ] {
            assert!(eval_args(&extra).is_err(), "{extra:?}");
        }
        assert!(eval_args(&["--sprt", "x"]).is_ok());
        let command = eval_args(&["--sprt", "x", "--sprt-cap", "128", "--sprt-target", "0.515"])
            .expect("the cap and the target are options of the journal");
        let super::super::Command::Eval { sequential, .. } = command else {
            panic!("eval command");
        };
        let bounds = sequential.bounds().unwrap();
        assert_eq!(
            (bounds.s0, bounds.s1, bounds.cap),
            (sprt::S0, 0.515, 128usize)
        );
    }

    #[test]
    fn default_bounds_are_the_protocol_constants_and_invalid_ones_are_refused() {
        let default = eval_args(&["--sprt", "x"]).unwrap();
        let super::super::Command::Eval { sequential, .. } = default else {
            panic!("eval command");
        };
        assert_eq!(sequential.bounds().unwrap(), sprt::Bounds::default());
        for extra in [
            vec!["--sprt", "x", "--sprt-cap", "120"],
            vec!["--sprt", "x", "--sprt-cap", "64"],
            vec!["--sprt", "x", "--sprt-target", "0.5"],
            vec!["--sprt", "x", "--sprt-target", "1.5"],
        ] {
            let command = eval_args(&extra).unwrap();
            let super::super::Command::Eval { sequential, .. } = command else {
                panic!("eval command");
            };
            let error = format!("{:#}", sequential.bounds().unwrap_err());
            assert!(error.contains("--sprt-target and --sprt-cap"), "{error}");
        }
    }

    #[test]
    fn external_network_identity_is_required_and_native_overrides_fail() {
        assert!(identity("rpsi:missing.exe", None, Path::new("unused")).is_err());
        assert!(identity(
            "nnue:missing",
            Some(Path::new("other")),
            Path::new("unused")
        )
        .is_err());
    }
}
