//! File validation and fixed-node position diagnostics for the bot CLI.
use std::io::{BufRead, Write};
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use engine::{from_codes, Outcome, Rules, State};
use serde::Deserialize;

use crate::net::Accumulator;
use crate::search::{Limits, SearchPool};
use crate::Model;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub board: Vec<u8>,
    pub since_capture: u32,
    pub ply: u32,
    pub clock: u32,
    pub raw: Option<f64>,
    pub raw6: Option<f64>,
    pub context: Option<[usize; 2]>,
    pub ids: Option<[Vec<usize>; 2]>,
}

impl Position {
    pub fn state(&self) -> Result<State> {
        ensure!(
            (50..=200).contains(&self.clock),
            "capture clock must be 50..=200"
        );
        ensure!(self.since_capture < self.clock, "expired position");
        let board = from_codes(&self.board).context("board must have 81 cells coded 0..6")?;
        let state = State {
            board,
            since_capture: self.since_capture,
            ply: self.ply,
        };
        ensure!(
            state.own_count() > 0 && state.enemy_count() > 0,
            "terminal position"
        );
        ensure!(
            state.own_count() <= 10 && state.enemy_count() <= 10,
            "too many pieces"
        );
        ensure!(state.ply < u32::MAX - 256, "ply overflows search counters");
        ensure!(
            !state.board[0].is_enemy() && !state.board[80].is_own(),
            "goal already entered"
        );
        ensure!(!state.is_stalemated(), "stalemated root");
        Ok(state)
    }
}

pub fn run(
    model: &Model,
    input: impl BufRead,
    mut output: impl Write,
    nodes: Option<u64>,
    threads: usize,
) -> Result<()> {
    ensure!((1..=4).contains(&threads), "threads must be in 1..=4");
    ensure!(nodes != Some(0), "nodes must be positive");
    #[cfg(feature = "profile")]
    ensure!(threads == 1, "profiling requires one search thread");
    #[cfg(feature = "profile")]
    let profile_timer_ns = crate::profile::calibrate();
    let mut search = SearchPool::new(64, threads);
    let mut count = 0;
    for (row, line) in input.lines().enumerate() {
        let request: Position =
            serde_json::from_str(&line?).with_context(|| format!("row {row}"))?;
        let state = request.state().with_context(|| format!("row {row}"))?;
        let acc = model.refresh(&state.board, state.since_capture, request.clock);
        let raw = model.raw(&acc);
        let contexts = acc.contexts();
        let ids = model.feature_ids(&state.board, state.since_capture, request.clock);
        if let Some(expected) = request.context {
            ensure!(expected == contexts, "context mismatch at row {row}");
        }
        if let Some(expected) = &request.ids {
            for side in 0..2 {
                ensure!(
                    expected[side].len() == 42,
                    "expected 42 feature slots at row {row}"
                );
                ensure!(
                    expected[side].iter().all(|&id| id <= 13640),
                    "invalid feature id at row {row}"
                );
                let mut expected: Vec<_> = expected[side]
                    .iter()
                    .copied()
                    .filter(|&id| id != 13640)
                    .collect();
                expected.sort_unstable();
                ensure!(
                    expected.windows(2).all(|v| v[0] != v[1]),
                    "duplicate feature at row {row}"
                );
                let mut actual: Vec<_> = ids[side]
                    .iter()
                    .copied()
                    .filter(|&id| id != model.features)
                    .map(|id| {
                        if model.features == 1004 {
                            if id < 486 {
                                id + 486 * contexts[side]
                            } else {
                                id + 12636
                            }
                        } else {
                            id
                        }
                    })
                    .collect();
                actual.sort_unstable();
                ensure!(
                    actual == expected,
                    "feature ids mismatch at row {row}, half {side}"
                );
            }
        }
        let expected = if model.features == 1004 {
            request.raw6.or(request.raw)
        } else {
            request.raw
        };
        if let Some(expected) = expected {
            ensure!(
                expected.is_finite() && (raw - expected).abs() <= 1e-12,
                "reference raw mismatch at row {row}: {raw} != {expected}"
            );
        }
        ensure!(
            raw == model.raw_scalar(&acc),
            "scalar/SIMD mismatch at row {row}"
        );
        let mut child_acc = Accumulator::empty(model.hidden);
        let mut children = 0;
        for a in state.legal_actions() {
            let (child, outcome) = engine::apply(
                &Rules {
                    capture_clock: Some(request.clock),
                },
                &state,
                a,
            );
            model.update(&acc, &state.board, a, child.since_capture, &mut child_acc);
            ensure!(
                child_acc == model.refresh(&child.board, child.since_capture, request.clock),
                "incremental mismatch at row {row}, action {a}"
            );
            if outcome == Outcome::Ongoing {
                children += 1;
            }
        }
        let mut response = serde_json::json!({"row":row,"raw":raw,"score":model.evaluate(&acc),"context":contexts,"ids":ids.map(|ids| ids.to_vec()),"incremental_children":children});
        if let Some(nodes) = nodes {
            ensure!(
                request.clock == 200,
                "search diagnostics require site clock=200"
            );
            search.clear();
            #[cfg(feature = "profile")]
            crate::profile::reset();
            let r = search.search(
                model,
                &state,
                None,
                Limits {
                    time: Duration::from_secs(60),
                    nodes,
                    ..Limits::default()
                },
            );
            #[cfg(feature = "profile")]
            {
                response["profile"] = crate::profile::snapshot();
                response["profile_timer_ns"] = serde_json::json!(profile_timer_ns);
            }
            response["action"] = serde_json::json!(r.action);
            response["score"] = serde_json::json!(r.score);
            response["depth"] = serde_json::json!(r.depth);
            response["nodes"] = serde_json::json!(r.nodes);
            response["qnodes"] = serde_json::json!(r.qnodes);
            response["elapsed_ms"] = serde_json::json!(r.elapsed.as_secs_f64() * 1000.);
            response["aborted"] = serde_json::json!(r.aborted);
            response["partial"] = serde_json::json!(r.partial);
            response["pv"] = serde_json::json!(r.pv);
        }
        serde_json::to_writer(&mut output, &response)?;
        writeln!(output)?;
        count += 1;
    }
    if count == 0 {
        bail!("position input is empty");
    }
    Ok(())
}
