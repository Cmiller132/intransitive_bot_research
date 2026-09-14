//! `SqPlayer`: the generic Gumbel search over an `OrtNet` (DESIGN.md item 16).

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use engine::{Action, Rules, State};
use r#match::{Analyser, Clock, Heads, History, MoveInfo, Player};
use rand::{rngs::StdRng, SeedableRng};
use search::{Budget, Gumbel, Info, Params};

pub struct SqPlayer {
    name: String,
    net: OrtNet,
    search: Gumbel,
    rng: StdRng,
    /// The last move chosen and the search behind it.
    last: Option<(Action, Info)>,
}

use crate::net::OrtNet;

/// Play-time settings of the sq line.
pub struct PlaySettings {
    pub contempt: f32,
    pub repetition_penalty: f32,
    pub moves_left: f32,
    pub batch: usize,
    pub cache: usize,
    pub seed: u64,
}

impl Default for PlaySettings {
    fn default() -> PlaySettings {
        PlaySettings {
            contempt: 0.15,
            repetition_penalty: 0.1,
            moves_left: 0.04,
            batch: 8,
            cache: 200_000,
            seed: 0,
        }
    }
}

impl SqPlayer {
    /// Load `<path>` and `<path>.json`; `threads` sets ONNX Runtime's intra-op threads.
    pub fn load(path: &str, threads: usize) -> Result<SqPlayer> {
        SqPlayer::with_settings(path, threads, PlaySettings::default())
    }

    pub fn with_settings(path: &str, threads: usize, settings: PlaySettings) -> Result<SqPlayer> {
        let mut net = OrtNet::load(Path::new(path), threads, settings.cache)?;
        net.contempt = settings.contempt;
        let params = Params {
            rules: Rules::SITE,
            batch: settings.batch,
            moves_left: settings.moves_left,
            contempt: settings.contempt,
            repetition_penalty: settings.repetition_penalty,
            ..Params::default()
        };
        Ok(SqPlayer {
            name: net.meta.name.clone(),
            net,
            search: Gumbel::new(params),
            rng: StdRng::seed_from_u64(settings.seed),
            last: None,
        })
    }
}

impl Player for SqPlayer {
    fn new_game(&mut self) {
        self.search.reset();
        self.net.clear_cache();
    }

    fn choose(&mut self, state: &State, history: &History, clock: Clock) -> Action {
        let budget = match clock {
            Clock::Sims(sims) => Budget::Sims(sims),
            Clock::Time(duration) => Budget::Deadline(Instant::now() + duration),
        };
        let (action, info) =
            self.search
                .choose(&mut self.net, state, history, budget, &mut self.rng);
        self.last = Some((action, info));
        action
    }

    fn info(&self) -> Option<MoveInfo> {
        self.last
            .as_ref()
            .map(|(action, info)| MoveInfo::from_search(info, *action))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Analyser for SqPlayer {
    fn heads(&mut self, state: &State) -> Heads {
        self.net.heads(state)
    }

    fn search(&mut self, state: &State, history: &History, sims: u32) -> Info {
        self.search.reset();
        self.search
            .choose(
                &mut self.net,
                state,
                history,
                Budget::Sims(sims),
                &mut self.rng,
            )
            .1
    }

    fn name(&self) -> &str {
        &self.name
    }
}
