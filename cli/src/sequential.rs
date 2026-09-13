//! Sequential eval options and identities of the coordinator, engine and network files.
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use clap::Args as ClapArgs;
use r#match::eval::Sequential;
use serde_json::{json, Value};

#[derive(ClapArgs)]
pub struct Args {
    /// Pentanomial SPRT journal: 16-pair batches, first check 128, cap 3008.
    #[arg(long, conflicts_with_all = ["pairs", "records"])]
    pub sprt: Option<PathBuf>,
    /// Resume the journal; settings, openings and artifact hashes must match.
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
    pub fn config(
        &self,
        candidate: &str,
        reference: &str,
        player_threads: usize,
    ) -> Result<Option<Sequential>> {
        let Some(journal) = &self.sprt else {
            return Ok(None);
        };
        let executable = std::env::current_exe()?;
        let coordinator = artifact(&executable)?;
        Ok(Some(Sequential {
            journal: journal.clone(),
            resume: self.resume,
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

    #[test]
    fn sequential_options_cannot_change_cap_or_omit_resume_journal() {
        let base = ["bot", "eval", "--candidate", "nnue:fake"];
        for extra in [
            vec!["--resume"],
            vec!["--sprt", "x", "--pairs", "128"],
            vec!["--sprt", "x", "--records", "y"],
        ] {
            assert!(super::super::Cli::try_parse_from(base.into_iter().chain(extra)).is_err());
        }
        assert!(super::super::Cli::try_parse_from(base.into_iter().chain(["--sprt", "x"])).is_ok());
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
