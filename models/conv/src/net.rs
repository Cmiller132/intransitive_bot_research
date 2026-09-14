//! ONNX Runtime evaluator for exported conv checkpoints, with a position cache.
//! Implements `search::Evaluator`; the graph and its sidecar come from `conv.export`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, ensure, Context, Result};
use engine::{Rules, State, N_ACTIONS};
use ort::session::{builder::GraphOptimizationLevel, Session};
use r#match::Heads;
use search::{Eval, Evaluator};
use serde::Deserialize;

use crate::planes::{encode, N_PLANES, PLANE_LEN};

/// `<model>.onnx.json`, written by `conv.export`.
#[derive(Clone, Debug, Deserialize)]
pub struct Meta {
    pub name: String,
    pub checkpoint_sha256: String,
    pub planes: usize,
    pub atoms: usize,
    /// Temperatures of the improved policy `softmax((q + beta log pi) / (alpha + beta))`.
    pub alpha: f32,
    pub beta: f32,
    /// Expected plies for each plies-to-end class.
    pub tte_centers: Vec<f32>,
}

/// Network outputs for one position over its legal actions, before any
/// play-time adjustment.
#[derive(Clone)]
struct Cached {
    legal: Vec<u16>,
    log_policy: Vec<f32>,
    q: Vec<f32>,
    /// Mass of each action's value distribution near zero.
    draw: Vec<f32>,
    /// The state value head (DESIGN.md item 10).
    value: f32,
    /// Probability of each plies-to-end class.
    tte: Vec<f32>,
    plies_left: f32,
    /// Win, draw and loss probabilities (DESIGN.md item 31) when exported.
    wdl: Option<[f32; 3]>,
}

pub struct OrtNet {
    pub meta: Meta,
    session: Session,
    /// Whether the graph has the `wdl` output.
    has_wdl: bool,
    cache: HashMap<State, Cached>,
    cache_limit: usize,
    /// The capture clock plane 15 counts down comes from these rules.
    rules: Rules,
    /// A rule draw is worth `-contempt` to the searching side, through the draw mass.
    pub contempt: f32,
    /// Positions evaluated by the network; cache hits are not counted.
    pub evaluated: u64,
    pub path: PathBuf,
    planes: Vec<f32>,
}

impl OrtNet {
    pub fn load(model: &Path, threads: usize, cache_limit: usize, rules: Rules) -> Result<OrtNet> {
        let sidecar = PathBuf::from(format!("{}.json", model.display()));
        let text = std::fs::read_to_string(&sidecar)
            .with_context(|| format!("reading {}", sidecar.display()))?;
        let meta: Meta = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", sidecar.display()))?;
        ensure!(
            meta.planes == N_PLANES,
            "model expects {} planes, this crate encodes {N_PLANES}",
            meta.planes
        );
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!(e.to_string()))?
            .with_intra_threads(threads.max(1))
            .map_err(|e| anyhow!(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| anyhow!(e.to_string()))?;
        let session = builder
            .commit_from_file(model)
            .with_context(|| format!("loading {}", model.display()))?;
        for required in ["logits", "q", "value", "plies_to_end", "draw"] {
            ensure!(
                session.outputs().iter().any(|o| o.name() == required),
                "model is missing output {required}"
            );
        }
        let has_wdl = session.outputs().iter().any(|o| o.name() == "wdl");
        Ok(OrtNet {
            meta,
            session,
            has_wdl,
            cache: HashMap::new(),
            cache_limit,
            rules,
            contempt: 0.0,
            evaluated: 0,
            path: model.to_owned(),
            planes: Vec::new(),
        })
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// The raw heads for `state`, without contempt, from the mover's view.
    pub fn heads(&mut self, state: &State) -> Heads {
        if !self.cache.contains_key(state) {
            let computed = self
                .forward(&[state])
                .expect("ONNX Runtime inference failed");
            if self.cache.len() >= self.cache_limit {
                self.cache.clear();
            }
            self.cache
                .insert(state.clone(), computed.into_iter().next().expect("one row"));
        }
        let cached = &self.cache[state];
        Heads {
            legal: cached.legal.clone(),
            policy: cached.log_policy.iter().map(|lp| lp.exp()).collect(),
            q: cached.q.clone(),
            value: cached.value,
            value_source: "state_head",
            plies_left: cached.plies_left,
            plies_to_end: cached.tte.clone(),
            tte_centers: self.meta.tte_centers.clone(),
            draw: cached.draw.clone(),
            wdl: cached.wdl,
        }
    }

    /// One session run on `states`; the raw outputs restricted to legal actions.
    fn forward(&mut self, states: &[&State]) -> Result<Vec<Cached>> {
        let batch = states.len();
        let clock = self.rules.capture_clock.unwrap_or(0);
        self.planes.resize(batch * PLANE_LEN, 0.0);
        for (row, state) in states.iter().enumerate() {
            encode(
                state,
                clock,
                &mut self.planes[row * PLANE_LEN..(row + 1) * PLANE_LEN],
            );
        }
        let input =
            ort::value::Tensor::from_array((vec![batch, N_PLANES, 81], self.planes.clone()))?;
        let outputs = self.session.run(ort::inputs!["planes" => input])?;
        let logits = outputs["logits"].try_extract_tensor::<f32>()?.1;
        let q = outputs["q"].try_extract_tensor::<f32>()?.1;
        let value = outputs["value"].try_extract_tensor::<f32>()?.1;
        let tte = outputs["plies_to_end"].try_extract_tensor::<f32>()?.1;
        let draw = outputs["draw"].try_extract_tensor::<f32>()?.1;
        let wdl = if self.has_wdl {
            let probs = outputs["wdl"].try_extract_tensor::<f32>()?.1;
            ensure!(probs.len() == batch * 3, "bad wdl size");
            Some(probs)
        } else {
            None
        };
        let classes = self.meta.tte_centers.len();
        ensure!(
            logits.len() == batch * N_ACTIONS && q.len() == batch * N_ACTIONS,
            "bad output size"
        );
        ensure!(value.len() == batch, "bad value size");
        ensure!(tte.len() == batch * classes, "bad plies_to_end size");
        self.evaluated += batch as u64;
        Ok(states
            .iter()
            .enumerate()
            .map(|(row, state)| {
                let legal = state.legal_actions();
                let at = |values: &[f32], a: u16| values[row * N_ACTIONS + a as usize];
                Cached {
                    log_policy: log_softmax(
                        &legal.iter().map(|&a| at(logits, a)).collect::<Vec<_>>(),
                    ),
                    q: legal.iter().map(|&a| at(q, a)).collect(),
                    draw: legal.iter().map(|&a| at(draw, a)).collect(),
                    value: value[row],
                    tte: softmax(&tte[row * classes..(row + 1) * classes]),
                    plies_left: expectation(
                        &tte[row * classes..(row + 1) * classes],
                        &self.meta.tte_centers,
                    ),
                    wdl: wdl.map(|w| [w[row * 3], w[row * 3 + 1], w[row * 3 + 2]]),
                    legal,
                }
            })
            .collect())
    }

    /// The improved policy and the state value with contempt applied for
    /// `sign`: every action loses `c` times its draw mass, and the value loses
    /// `c` times the draw mass the improved policy expects.
    fn finish(&self, cached: &Cached, sign: i8) -> Eval {
        let c = self.contempt * sign as f32;
        let q: Vec<f32> = cached
            .q
            .iter()
            .zip(&cached.draw)
            .map(|(&value, &draw)| {
                if c != 0.0 {
                    (value - c * draw).clamp(-1.0, 1.0)
                } else {
                    value
                }
            })
            .collect();
        let log_prior = improved_policy(&cached.log_policy, &q, self.meta.alpha, self.meta.beta);
        let draw_mass: f32 = log_prior
            .iter()
            .zip(&cached.draw)
            .map(|(lp, draw)| lp.exp() * draw)
            .sum();
        Eval {
            legal: cached.legal.clone(),
            log_prior,
            q,
            value: (cached.value - c * draw_mass).clamp(-1.0, 1.0),
            plies_left: Some(cached.plies_left),
        }
    }
}

impl Evaluator for OrtNet {
    fn evaluate(&mut self, states: &[&State], signs: &[i8]) -> Vec<Eval> {
        let mut missing: Vec<usize> = Vec::new();
        for (i, state) in states.iter().enumerate() {
            if !self.cache.contains_key(*state) {
                missing.push(i);
            }
        }
        if !missing.is_empty() {
            let fresh: Vec<&State> = missing.iter().map(|&i| states[i]).collect();
            let computed = self.forward(&fresh).expect("ONNX Runtime inference failed");
            if self.cache.len() + computed.len() > self.cache_limit {
                self.cache.clear();
            }
            for (state, cached) in fresh.into_iter().zip(computed) {
                self.cache.insert(state.clone(), cached);
            }
        }
        states
            .iter()
            .zip(signs)
            .map(|(state, &sign)| self.finish(&self.cache[*state], sign))
            .collect()
    }
}

fn softmax(values: &[f32]) -> Vec<f32> {
    log_softmax(values).iter().map(|lp| lp.exp()).collect()
}

fn log_softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = max + values.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    values.iter().map(|v| v - log_sum).collect()
}

/// `log pi'` of `pi' = softmax((q + beta log pi) / (alpha + beta))`.
pub fn improved_policy(log_policy: &[f32], q: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    let z: Vec<f32> = log_policy
        .iter()
        .zip(q)
        .map(|(&lp, &qv)| (qv + beta * lp) / (alpha + beta))
        .collect();
    log_softmax(&z).iter().map(|lp| lp.max(-27.63)).collect()
}

fn expectation(logits: &[f32], centers: &[f32]) -> f32 {
    let probabilities = log_softmax(logits);
    probabilities
        .iter()
        .zip(centers)
        .map(|(lp, c)| lp.exp() * c)
        .sum()
}
