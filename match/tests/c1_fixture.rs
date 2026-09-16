//! Scripted games exercise the shared journal/report fixture through production C1.
use std::path::{Path, PathBuf};

use engine::{Action, State};
use r#match::eval::{eval, EvalConfig, Sequential};
use r#match::player::{Clock, History, Player};
use r#match::sprt::{self, Stop};
use serde_json::Value;
use sha2::{Digest, Sha256};

struct Scripted {
    candidate: bool,
    game: usize,
}

impl Player for Scripted {
    fn new_game(&mut self) {
        self.game += 1;
    }

    fn choose(&mut self, state: &State, _: &History, clock: Clock) -> Action {
        assert!(matches!(clock, Clock::Time(time) if time.as_millis() == 50));
        let legal = state.legal_actions();
        if self.candidate && self.game.is_multiple_of(16) {
            *legal.last().unwrap()
        } else {
            legal[0]
        }
    }

    fn name(&self) -> &str {
        "intransitive_bot"
    }
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        for name in ["generated.jsonl", "resume.jsonl"] {
            let _ = std::fs::remove_file(self.0.join(name));
        }
        std::fs::remove_dir(&self.0).unwrap();
    }
}

fn sha256(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

#[test]
fn scripted_journal_and_report_agree_with_c1() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let fixtures = root.join("models/nnue/tests/fixtures");
    let scratch = Scratch(std::env::temp_dir().join(format!("c1-fixture-{}", std::process::id())));
    std::fs::create_dir(&scratch.0).unwrap();
    let expected: Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("c1_report.json")).unwrap()).unwrap();
    for (role, net) in [
        ("candidate", "format8_h32.nnue"),
        ("reference", "format6_h32.nnue"),
    ] {
        assert_eq!(
            expected["sequential"]["protocol"]["provenance"][role]["network"]["sha256"],
            sha256(&fixtures.join(net))
        );
    }
    let mut config = EvalConfig {
        pairs: sprt::Bounds::default().cap,
        move_ms: Some(50),
        opening_plies: 2,
        seed: 2026091225,
        threads: 1,
        sequential: Some(Sequential {
            journal: scratch.0.join("generated.jsonl"),
            resume: false,
            bounds: sprt::Bounds::default(),
            provenance: expected["sequential"]["protocol"]["provenance"].clone(),
        }),
        ..EvalConfig::default()
    };
    let report = eval(
        &config,
        &|| {
            Ok(Box::new(Scripted {
                candidate: true,
                game: 0,
            }))
        },
        &|| {
            Ok(Box::new(Scripted {
                candidate: false,
                game: 0,
            }))
        },
        None,
    )
    .unwrap();
    let state = &report.sequential.as_ref().unwrap().state;
    assert!(report.pairs >= 32 && report.pairs.is_multiple_of(sprt::BATCH));
    assert!(matches!(
        state.stop_reason,
        Stop::Accept | Stop::Reject | Stop::Inconclusive
    ));
    assert!(state.counts.iter().filter(|&&n| n > 0).count() >= 2);
    assert_eq!(report.complete_pairs, report.pairs);
    assert_eq!(report.forfeits, 0);
    assert_eq!(report.candidate, report.reference);
    let bytes = std::fs::read(scratch.0.join("generated.jsonl")).unwrap();
    let rendered = serde_json::to_string_pretty(&report).unwrap() + "\n";
    assert_eq!(
        bytes,
        std::fs::read(fixtures.join("c1_journal.jsonl")).unwrap()
    );
    assert_eq!(serde_json::from_str::<Value>(&rendered).unwrap(), expected);
    std::fs::write(scratch.0.join("resume.jsonl"), &bytes).unwrap();
    let sequential = config.sequential.as_mut().unwrap();
    sequential.journal = scratch.0.join("resume.jsonl");
    sequential.resume = true;
    let resumed = eval(
        &config,
        &|| panic!("terminal journal"),
        &|| panic!("terminal journal"),
        None,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_string_pretty(&resumed).unwrap() + "\n",
        rendered
    );
    assert_eq!(
        std::fs::read(scratch.0.join("resume.jsonl")).unwrap(),
        bytes
    );
}
