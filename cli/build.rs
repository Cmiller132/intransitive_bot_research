//! Captures the git short hash and dirty state for `bot --version`.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let suffix = match git(&["rev-parse", "--short", "HEAD"]) {
        Some(hash) => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|status| !status.is_empty());
            format!(" {hash}{}", if dirty { "-dirty" } else { "" })
        }
        None => String::new(),
    };
    println!("cargo:rustc-env=BOT_GIT={suffix}");
    for path in ["../.git/HEAD", "../.git/index"] {
        if std::path::Path::new(path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
