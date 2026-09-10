//! The NNUE seat in the shared match runner.
use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, Result};
use engine::{Action, Rules, State};
use r#match::{Analyser, Clock, Heads, History, MoveInfo, Player};
use search::Info;

use crate::search::{Limits, SearchPool, SearchResult};
use crate::Model;

pub struct NnuePlayer {
    name: String,
    model: Model,
    search: SearchPool,
    last: Option<SearchResult>,
    failed: bool,
    sims: u32,
    hash_mib: usize,
}

impl NnuePlayer {
    pub fn load(spec: &str, threads: usize) -> Result<Self> {
        let (path, hash_mib) = player_options(spec)?;
        if !(1..=4).contains(&threads) {
            return Err(anyhow!("NNUE threads must be in 1..=4"));
        }
        Ok(Self {
            name: format!("nnue:{spec}"),
            model: Model::load(path).map_err(anyhow::Error::msg)?,
            search: SearchPool::new(hash_mib, threads),
            last: None,
            failed: false,
            sims: 0,
            hash_mib,
        })
    }
    fn choose_with_reserve(&mut self, state: &State, clock: Clock, reserve: Duration) -> Action {
        let result = match clock {
            Clock::Time(time) => {
                self.sims = 0;
                self.search.search(
                    &self.model,
                    state,
                    None,
                    Limits {
                        time: time.saturating_sub(reserve),
                        ..Limits::default()
                    },
                )
            }
            Clock::Sims(sims) => {
                self.sims = sims;
                self.search.search_one(
                    &self.model,
                    state,
                    Limits {
                        nodes: u64::from(sims.max(1)) * 2_500,
                        time: Duration::from_secs(120),
                        ..Limits::default()
                    },
                )
            }
        };
        self.failed = self.search.leaf_error().is_some();
        let action = result.action.unwrap_or_else(|| {
            self.failed = true;
            state.legal_actions()[0]
        });
        self.last = Some(result);
        action
    }
}

impl Player for NnuePlayer {
    fn new_game(&mut self) {
        self.search.clear();
        self.last = None;
        self.failed = false;
        self.sims = 0;
    }

    fn set_leaves(&mut self, path: &Path) -> Result<()> {
        self.search.set_leaves(path)
    }

    fn choose(&mut self, state: &State, _: &History, clock: Clock) -> Action {
        self.choose_with_reserve(state, clock, Duration::from_millis(20))
    }

    fn choose_self_timed(&mut self, state: &State, _: &History, time: Duration) -> Action {
        self.choose_with_reserve(state, Clock::Time(time), Duration::ZERO)
    }

    fn info(&self) -> Option<MoveInfo> {
        self.last.as_ref().map(|s| MoveInfo {
            sims: self.sims,
            value: (s.score as f64 / 600.).tanh() as f32,
            plies_left: None,
            q: (s.score as f64 / 600.).tanh() as f32,
            pi: 0.,
            exact_win: s.score > 29_000,
            top: Vec::new(),
        })
    }

    fn search_details(&self) -> Option<serde_json::Value> {
        self.last.as_ref().map(|s| {
            serde_json::json!({
                "nodes": s.nodes, "qnodes": s.qnodes, "depth": s.depth,
                "score": s.score, "pv": s.pv, "elapsed_ms": s.elapsed.as_secs_f64()*1000.,
                "aborted": s.aborted, "partial": s.partial, "hash_mib": self.hash_mib, "leaf_error": self.search.leaf_error(),
                "root_moves": s.root_moves, "iterations": s.iterations
            })
        })
    }

    fn forfeited(&self) -> bool {
        self.failed
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// The clock context the search refreshes with (site rules).
const SITE_CLOCK: u32 = 200;

/// Search scores map to values through the export's scale (DESIGN item 6).
fn value_of(score: i32) -> f32 {
    (f64::from(score) / 600.).tanh() as f32
}

/// `bot analyse --engine nnue:<file>`: the static evaluation as the network
/// head (no policy; every legal action carries the child's static value as
/// its Q) and a node-budget search (`sims` x 2,500 nodes, one thread) whose
/// root value labels positions in `nnue.label`, so a network can rescore
/// its own games (DESIGN item 30).
impl Analyser for NnuePlayer {
    fn heads(&mut self, state: &State) -> Heads {
        let legal = state.legal_actions();
        let q: Vec<f32> = legal
            .iter()
            .map(|&action| {
                let (child, outcome) = engine::apply(&Rules::SITE, state, action);
                match outcome {
                    engine::Outcome::Ongoing => {
                        -value_of(self.model.evaluate(&self.model.refresh(
                            &child.board,
                            child.since_capture,
                            SITE_CLOCK,
                        )))
                    }
                    engine::Outcome::Draw => 0.,
                    _ => 1.,
                }
            })
            .collect();
        let acc = self
            .model
            .refresh(&state.board, state.since_capture, SITE_CLOCK);
        Heads {
            policy: vec![1. / legal.len().max(1) as f32; legal.len()],
            draw: vec![0.; legal.len()],
            q,
            legal,
            value: value_of(self.model.evaluate(&acc)),
            value_source: "static_eval",
            plies_left: 0.,
            plies_to_end: Vec::new(),
            tte_centers: Vec::new(),
            wdl: None,
        }
    }

    fn search(&mut self, state: &State, _: &History, sims: u32) -> Info {
        self.search.clear();
        let result = self.search.search_one(
            &self.model,
            state,
            Limits {
                nodes: u64::from(sims.max(1)) * 2_500,
                time: Duration::from_secs(120),
                ..Limits::default()
            },
        );
        let legal = state.legal_actions();
        let root_value = value_of(result.score);
        let chosen = result.action;
        let root = legal
            .iter()
            .map(|&action| {
                if Some(action) == chosen {
                    (
                        action,
                        result.nodes.min(u64::from(u32::MAX)) as u32,
                        f64::from(root_value),
                    )
                } else {
                    (action, 0, 0.)
                }
            })
            .collect();
        let pvs = legal
            .iter()
            .map(|&action| {
                if Some(action) == chosen {
                    result.pv.iter().skip(1).copied().collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        let mut order: Vec<Action> = chosen.into_iter().collect();
        order.extend(legal.iter().copied().filter(|&a| Some(a) != chosen));
        Info {
            sims,
            nodes: result.nodes.min(u64::from(u32::MAX)) as u32,
            evaluations: result.nodes + result.qnodes,
            root_value,
            plies_left: None,
            exact_win: result.score > 29_000,
            root,
            pvs,
            order,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// NNUE paths optionally carry one table-budget parameter, in MiB.
fn player_options(spec: &str) -> Result<(&str, usize)> {
    let (path, hash_mib) = if let Some((path, query)) = spec.split_once('?') {
        let value = query
            .strip_prefix("hash=")
            .ok_or_else(|| anyhow!("NNUE option must be ?hash=<MiB>"))?;
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            return Err(anyhow!("NNUE hash must be an integer in 1..=2048 MiB"));
        }
        (
            path,
            value
                .parse::<usize>()
                .map_err(|_| anyhow!("NNUE hash must be in 1..=2048 MiB"))?,
        )
    } else {
        (spec, 64)
    };
    if path.is_empty() || !(1..=2048).contains(&hash_mib) {
        return Err(anyhow!("NNUE needs a path and hash in 1..=2048 MiB"));
    }
    Ok((path, hash_mib))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyser_reports_static_heads_and_a_budgeted_search() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/h768_dense.nnue"
        );
        let mut player = NnuePlayer::load(path, 1).unwrap();
        let state = State::initial();
        let heads = player.heads(&state);
        assert_eq!(heads.legal, state.legal_actions());
        assert_eq!(heads.q.len(), heads.legal.len());
        assert!(heads.value.abs() <= 1. && heads.q.iter().all(|q| q.abs() <= 1.));
        let info = player.search(&state, &History::new(), 1);
        assert_eq!(info.root.len(), heads.legal.len());
        assert!(info.nodes > 0 && info.root_value.abs() <= 1.);
        assert_eq!(info.order.len(), heads.legal.len());
        let (chosen, visits, q) = info.root.iter().copied().max_by_key(|r| r.1).unwrap();
        assert_eq!(info.order[0], chosen);
        assert_eq!(visits, info.nodes);
        assert!((q as f32 - info.root_value).abs() < 1e-6);
    }

    #[test]
    fn table_option_is_strict_and_preserves_windows_paths() {
        assert_eq!(
            player_options("D:/weights/net.nnue").unwrap(),
            ("D:/weights/net.nnue", 64)
        );
        assert_eq!(
            player_options("D:/weights/net.nnue?hash=128").unwrap(),
            ("D:/weights/net.nnue", 128)
        );
        assert_eq!(player_options("net?hash=1").unwrap(), ("net", 1));
        assert_eq!(player_options("net?hash=2048").unwrap(), ("net", 2048));
        for spec in [
            "",
            "?hash=64",
            "net?hash=0",
            "net?hash=2049",
            "net?hash=-1",
            "net?hash=+1",
            "net?hash=1.5",
            "net?hash=",
            "net?hash=999999999999999999999999",
            "net?hash=64&hash=32",
            "net?foo=1",
        ] {
            assert!(player_options(spec).is_err(), "{spec}");
        }
    }
}
