//! Position analysis for the arena: an engine's raw heads and a search on any
//! position, served as JSON lines (one request per line in, one response per
//! line out) by `bot analyse`. Boards and moves are in the absolute frame.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use engine::{apply, Action, Outcome, Rules, State, GOAL};
use search::Info;
use serde::{Deserialize, Serialize};

use crate::player::History;
use crate::rpsi::Frame;

/// The network's outputs for one position, from the side to move's view,
/// over the legal actions in order.
pub struct Heads {
    pub legal: Vec<Action>,
    /// Network policy probabilities.
    pub policy: Vec<f32>,
    pub q: Vec<f32>,
    pub value: f32,
    /// `state_head` or `policy_weighted_q`.
    pub value_source: &'static str,
    pub plies_left: f32,
    /// Probability of each plies-to-end class, centred on `tte_centers`.
    pub plies_to_end: Vec<f32>,
    pub tte_centers: Vec<f32>,
    /// Draw mass of each action.
    pub draw: Vec<f32>,
    /// Win, draw and loss probabilities from the mover's view, for a model
    /// with the outcome head.
    pub wdl: Option<[f32; 3]>,
}

/// An engine that exposes its network and its search.
pub trait Analyser {
    fn heads(&mut self, state: &State) -> Heads;

    /// A fresh search of `sims` simulations.
    fn search(&mut self, state: &State, history: &History, sims: u32) -> Info;

    fn name(&self) -> &str;
}

#[derive(Deserialize)]
pub struct Request {
    pub id: serde_json::Value,
    #[serde(default)]
    pub setup: Option<Vec<u8>>,
    #[serde(default = "blue")]
    pub to_move: String,
    #[serde(default)]
    pub psc: u32,
    #[serde(default)]
    pub ply: u32,
    #[serde(default)]
    pub moves: Vec<String>,
    #[serde(default)]
    pub sims: u32,
}

fn blue() -> String {
    "blue".to_owned()
}

#[derive(Serialize, Default)]
pub struct Response {
    pub id: serde_json::Value,
    pub to_move: String,
    pub ply: u32,
    pub psc: u32,
    pub legal: Vec<String>,
    pub terminal: Option<Terminal>,
    pub network: Option<Network>,
    pub search: Option<Search>,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct Terminal {
    pub winner: Option<String>,
    pub reason: String,
}

#[derive(Serialize)]
pub struct Network {
    pub policy: BTreeMap<String, f32>,
    pub q: BTreeMap<String, f32>,
    pub value: f32,
    pub value_source: String,
    pub plies_left: f32,
    pub plies_to_end: Distribution,
    pub draw: BTreeMap<String, f32>,
    pub wdl: Option<[f32; 3]>,
}

#[derive(Serialize)]
pub struct Distribution {
    pub centers: Vec<f32>,
    pub p: Vec<f32>,
}

#[derive(Serialize)]
pub struct Search {
    pub sims: u32,
    pub nodes: u32,
    pub evaluations: u64,
    pub ms: u64,
    pub root_value: f32,
    pub plies_left: Option<f32>,
    pub lines: Vec<Line>,
}

#[derive(Serialize)]
pub struct Line {
    #[serde(rename = "move")]
    pub action: String,
    pub visits: u32,
    pub q: f64,
    pub pi: f32,
    pub pv: Vec<String>,
}

/// The position a request describes: the canonical state for the side to
/// move, its frame, the repetition history, and how the game ended if it did.
struct Position {
    state: State,
    frame: Frame,
    history: History,
    terminal: Option<Terminal>,
}

fn side(red: bool) -> String {
    if red { "red" } else { "blue" }.to_owned()
}

/// Resolve a request's setup and moves under `rules`.
fn position(request: &Request, rules: &Rules) -> Result<Position> {
    let red = match request.to_move.as_str() {
        "blue" => false,
        "red" => true,
        other => bail!("to_move must be blue or red, got {other:?}"),
    };
    let (mut state, mut frame) = match &request.setup {
        None => {
            if red {
                bail!("the standard setup has Blue to move");
            }
            (State::initial(), Frame::with_home(0, false))
        }
        Some(codes) => {
            if codes.len() != 81 {
                bail!("setup must hold 81 cells");
            }
            let mut board = [0u8; 81];
            board.copy_from_slice(codes);
            let frame = Frame::with_home(0, red);
            let mut state = frame.canonical(&board, request.ply)?;
            state.since_capture = request.psc;
            (state, frame)
        }
    };
    let mut history = History::new();
    history.insert(state.key(), 1);
    let mut terminal = None;
    for (i, token) in request.moves.iter().enumerate() {
        if terminal.is_some() {
            bail!("move {i} ({token}) comes after the game ended");
        }
        let action = frame.parse_move(token)?;
        if !state.legal_mask()[action as usize] {
            bail!("move {i} ({token}) is not legal");
        }
        let target = engine::from_to(action).1;
        let (child, outcome) = apply(rules, &state, action);
        terminal = match outcome {
            Outcome::Ongoing => None,
            Outcome::Draw => Some(Terminal {
                winner: None,
                reason: "stagnation".to_owned(),
            }),
            Outcome::Win => Some(Terminal {
                winner: Some(side(frame.red_to_move)),
                reason: if target == GOAL {
                    "corner"
                } else if child.own_count() == 0 {
                    "no_pieces"
                } else {
                    "no_moves"
                }
                .to_owned(),
            }),
        };
        state = child;
        frame.red_to_move = !frame.red_to_move;
        *history.entry(state.key()).or_insert(0) += 1;
    }
    if terminal.is_none() {
        if state.is_stalemated() {
            terminal = Some(Terminal {
                winner: Some(side(!frame.red_to_move)),
                reason: "no_moves".to_owned(),
            });
        } else if rules.clock_expired(&state) {
            terminal = Some(Terminal {
                winner: None,
                reason: "stagnation".to_owned(),
            });
        }
    }
    Ok(Position {
        state,
        frame,
        history,
        terminal,
    })
}

/// Spell a line of actions that starts with the side to move in `frame`.
fn spell_line(frame: &Frame, actions: &[Action]) -> Vec<String> {
    let mut frame = frame.clone();
    actions
        .iter()
        .map(|&action| {
            let token = frame.spell(action);
            frame.red_to_move = !frame.red_to_move;
            token
        })
        .collect()
}

/// Root lines of a search ordered by visits, then by the search's own order.
pub fn lines(info: &Info, frame: &Frame) -> Vec<Line> {
    let total: u32 = info.root.iter().map(|&(_, visits, _)| visits).sum();
    let rank = |action: Action| {
        info.order
            .iter()
            .position(|&a| a == action)
            .unwrap_or(usize::MAX)
    };
    let mut indices: Vec<usize> = (0..info.root.len()).collect();
    indices.sort_by(|&a, &b| {
        info.root[b]
            .1
            .cmp(&info.root[a].1)
            .then(rank(info.root[a].0).cmp(&rank(info.root[b].0)))
    });
    indices
        .into_iter()
        .map(|i| {
            let (action, visits, q) = info.root[i];
            let mut line = vec![action];
            line.extend(info.pvs.get(i).into_iter().flatten().copied());
            Line {
                action: frame.spell(action),
                visits,
                q,
                pi: if total > 0 {
                    visits as f32 / total as f32
                } else {
                    0.0
                },
                pv: spell_line(frame, &line),
            }
        })
        .collect()
}

/// Answer one request.
pub fn answer(analyser: &mut dyn Analyser, request: &Request, rules: &Rules) -> Response {
    let mut response = Response {
        id: request.id.clone(),
        ..Response::default()
    };
    let position = match position(request, rules) {
        Ok(position) => position,
        Err(error) => {
            response.error = Some(error.to_string());
            return response;
        }
    };
    let Position {
        state,
        frame,
        history,
        terminal,
    } = position;
    response.to_move = side(frame.red_to_move);
    response.ply = state.ply;
    response.psc = state.since_capture;
    if terminal.is_some() {
        response.terminal = terminal;
        return response;
    }
    let heads = analyser.heads(&state);
    let spelled: Vec<String> = heads.legal.iter().map(|&a| frame.spell(a)).collect();
    let map = |values: &[f32]| -> BTreeMap<String, f32> {
        spelled
            .iter()
            .cloned()
            .zip(values.iter().copied())
            .collect()
    };
    response.legal = spelled.clone();
    response.network = Some(Network {
        policy: map(&heads.policy),
        q: map(&heads.q),
        value: heads.value,
        value_source: heads.value_source.to_owned(),
        plies_left: heads.plies_left,
        plies_to_end: Distribution {
            centers: heads.tte_centers,
            p: heads.plies_to_end,
        },
        draw: map(&heads.draw),
        wdl: heads.wdl,
    });
    if request.sims > 0 {
        let started = Instant::now();
        let info = analyser.search(&state, &history, request.sims);
        response.search = Some(Search {
            sims: info.sims,
            nodes: info.nodes,
            evaluations: info.evaluations,
            ms: started.elapsed().as_millis() as u64,
            root_value: info.root_value,
            plies_left: info.plies_left,
            lines: lines(&info, &frame),
        });
    }
    response
}

/// Serve requests from `input` until it ends.
pub fn serve(
    analyser: &mut dyn Analyser,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    rules: &Rules,
) -> Result<()> {
    let mut line = String::new();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            return Ok(());
        }
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => answer(analyser, &request, rules),
            Err(error) => Response {
                id: serde_json::from_str::<serde_json::Value>(&line)
                    .ok()
                    .and_then(|v| v.get("id").cloned())
                    .unwrap_or(serde_json::Value::Null),
                error: Some(anyhow!("bad request: {error}").to_string()),
                ..Response::default()
            },
        };
        serde_json::to_writer(&mut *out, &response)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform heads; the search visits the first legal action most.
    struct Flat;

    impl Analyser for Flat {
        fn heads(&mut self, state: &State) -> Heads {
            let legal = state.legal_actions();
            let n = legal.len();
            Heads {
                policy: vec![1.0 / n as f32; n],
                q: vec![0.0; n],
                value: 0.25,
                value_source: "state_head",
                plies_left: 40.0,
                plies_to_end: vec![0.5, 0.5],
                tte_centers: vec![10.0, 70.0],
                draw: vec![0.1; n],
                wdl: None,
                legal,
            }
        }

        fn search(&mut self, state: &State, _: &History, sims: u32) -> Info {
            let legal = state.legal_actions();
            let root: Vec<(Action, u32, f64)> = legal
                .iter()
                .enumerate()
                .map(|(i, &a)| (a, if i == 0 { sims } else { 0 }, 0.5 - i as f64 * 0.01))
                .collect();
            let reply = apply(&Rules::SITE, state, legal[0]).0.legal_actions()[0];
            Info {
                sims,
                nodes: 3,
                evaluations: 2,
                root_value: 0.3,
                plies_left: Some(39.0),
                root,
                pvs: vec![vec![reply]],
                order: legal,
                ..Info::default()
            }
        }

        fn name(&self) -> &str {
            "flat"
        }
    }

    fn request(json: &str) -> Request {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn the_standard_setup_with_a_line_is_analysed_in_absolute_tokens() {
        let mut flat = Flat;
        let r = answer(
            &mut flat,
            &request(r#"{"id":1,"moves":["b5-a6","h5-i4"],"sims":16}"#),
            &Rules::SITE,
        );
        assert_eq!(r.error, None);
        assert_eq!((r.to_move.as_str(), r.ply, r.psc), ("blue", 2, 2));
        assert!(r.terminal.is_none());
        let network = r.network.unwrap();
        assert_eq!(network.policy.len(), r.legal.len());
        assert_eq!(network.value_source, "state_head");
        let search = r.search.unwrap();
        assert_eq!(search.lines[0].visits, 16);
        assert_eq!(search.lines[0].pi, 1.0);
        assert_eq!(search.lines[0].pv.len(), 2);
        assert!(r.legal.contains(&search.lines[0].pv[0]));
        // Blue's reply line and Red's answer are both spelled from their own pieces.
        let (from, _) = engine::notation::parse_squares(&search.lines[0].pv[1]).unwrap();
        assert!(
            from >= 40,
            "Red's move in the pv starts in Red's half: {from}"
        );
    }

    #[test]
    fn red_to_move_from_a_setup_and_errors_are_reported() {
        let mut flat = Flat;
        let setup: Vec<u8> = crate::rpsi::site_start(0).to_vec();
        let json =
            format!(r#"{{"id":"a","setup":{setup:?},"to_move":"red","psc":7,"ply":3,"moves":[]}}"#);
        let r = answer(&mut flat, &request(&json), &Rules::SITE);
        assert_eq!(r.error, None);
        assert_eq!((r.to_move.as_str(), r.ply, r.psc), ("red", 3, 7));
        let (from, _) = engine::notation::parse_squares(&r.legal[0]).unwrap();
        assert!(from >= 40, "Red moves a red piece: {}", r.legal[0]);
        assert!(r.search.is_none());
        let bad = answer(
            &mut flat,
            &request(r#"{"id":2,"moves":["a1-a2"]}"#),
            &Rules::SITE,
        );
        assert!(bad.error.unwrap().contains("not legal"));
        assert!(bad.network.is_none());
    }

    #[test]
    fn serve_answers_every_line_and_survives_garbage() {
        let mut flat = Flat;
        let input = b"{\"id\":1}\nnot json\n\n{\"id\":3,\"sims\":2}\n";
        let mut out = Vec::new();
        serve(&mut flat, &mut &input[..], &mut out, &Rules::SITE).unwrap();
        let lines: Vec<serde_json::Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0]["error"].is_null());
        assert!(lines[1]["error"]
            .as_str()
            .unwrap()
            .starts_with("bad request"));
        assert_eq!(lines[2]["search"]["sims"], 2);
    }
}
