//! Iterative-deepening PVS with tactical quiescence and a clock-aware TT.
use crate::net::{Accumulator, Model};
use engine::tactics::{goal_move as own_goal_move, goal_threat as enemy_goal_threat};
use engine::{Board, Cell, Outcome, Rules, State};
fn action_from_to(a: u16) -> (usize, usize) {
    let (f, t) = engine::from_to(a);
    (f as usize, t as usize)
}

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

pub const MATE: i32 = 30_000;
pub const INF: i32 = 32_000;
pub const MAX_PLY: usize = 120;
const NO_MOVE: u16 = u16::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub time: Duration,
    pub depth: u8,
    pub nodes: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            time: Duration::from_millis(750),
            depth: 64,
            nodes: u64::MAX,
        }
    }
}

/// One root iteration, including aspiration retries, from the selected worker.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Iteration {
    pub depth: u8,
    pub nodes: u64,
    pub elapsed_ms: f64,
    pub action: Option<u16>,
    pub score: i32,
    pub completed: bool,
    pub changed: bool,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub action: Option<u16>,
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
    pub qnodes: u64,
    pub elapsed: Duration,
    pub aborted: bool,
    /// The chosen move improved in an unfinished iteration; depth is the last completed depth.
    pub partial: bool,
    pub pv: Vec<u16>,
    pub root_moves: usize,
    pub iterations: Vec<Iteration>,
}

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    score: i32,
    static_eval: i32,
    static_valid: bool,
    action: u16,
    depth: i16,
    bound: u8, // 1 exact, 2 lower, 3 upper; zero is empty.
    generation: u8,
}

enum Table {
    Local(Vec<Entry>),
    Shared(Arc<Vec<Mutex<Entry>>>),
}

impl Table {
    fn count(bytes: usize, entry_bytes: usize) -> usize {
        let desired = (bytes / entry_bytes).max(1);
        1usize << (usize::BITS - 1 - desired.leading_zeros())
    }
    fn local(bytes: usize) -> Self {
        Self::Local(vec![
            Entry::default();
            Self::count(bytes, std::mem::size_of::<Entry>())
        ])
    }
    fn shared(bytes: usize) -> Self {
        let count = Self::count(bytes, std::mem::size_of::<Mutex<Entry>>());
        Self::Shared(Arc::new(
            (0..count).map(|_| Mutex::new(Entry::default())).collect(),
        ))
    }
    #[inline]
    fn get(&self, key: u64) -> Entry {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Table);
        match self {
            Self::Local(slots) => slots[key as usize & (slots.len() - 1)],
            // Never wait for another worker: a contended entry is a cache miss.
            // Locking the whole entry keeps key/score/bound coherent on all CPUs.
            Self::Shared(slots) => slots[key as usize & (slots.len() - 1)]
                .try_lock()
                .map(|entry| *entry)
                .unwrap_or_default(),
        }
    }
    #[inline]
    fn store(&mut self, entry: Entry) {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Table);
        let replace = |slot: &mut Entry, shared: bool| {
            if shared
                && slot.key == entry.key
                && slot.generation == entry.generation
                && slot.depth > entry.depth
                && slot.bound != 0
            {
                return;
            }
            if slot.bound == 0
                || slot.key == entry.key
                || slot.generation != entry.generation
                || entry.depth >= slot.depth - 2
            {
                *slot = entry;
            }
        };
        match self {
            Self::Local(slots) => {
                let index = entry.key as usize & (slots.len() - 1);
                replace(&mut slots[index], false);
            }
            Self::Shared(slots) => {
                if let Ok(mut slot) = slots[entry.key as usize & (slots.len() - 1)].try_lock() {
                    replace(&mut slot, true);
                }
            }
        }
    }
    fn clear(&mut self) {
        match self {
            Self::Local(slots) => slots.fill(Entry::default()),
            // Called only between searches, after every worker has joined.
            Self::Shared(slots) => {
                for slot in slots.iter() {
                    if let Ok(mut entry) = slot.try_lock() {
                        *entry = Entry::default();
                    }
                }
            }
        }
    }
}

pub struct Searcher {
    tt: Table,
    generation: u8,
    killers: [[u16; 2]; MAX_PLY + 2],
    history: [[i32; 648]; 2],
    acc: Vec<Accumulator>,
    hashes: [[u64; 2]; MAX_PLY + 2],
    move_buffers: Vec<Vec<u16>>,
    quiet_buffers: Vec<Vec<u16>>,
    move_scores: [i32; 648],
    start: Instant,
    deadline: Instant,
    limits: Limits,
    nodes: u64,
    qnodes: u64,
    stopped: bool,

    stop_signal: Option<(Arc<AtomicU64>, u64)>,
    forced_deadline: Option<Instant>,
    worker_id: u64,
    leaf_sink: Option<Arc<Mutex<LeafSink>>>,
    leaf_root: State,
    leaf_search: u64,
    leaf_path: [u16; MAX_PLY + 2],
}

/// Optional Lazy SMP. Each worker completes its own legal iterative search;
/// workers share a hard deadline, model, stop signal and transposition table.
pub struct SearchPool {
    workers: Vec<Searcher>,
    generation: u8,
}

impl SearchPool {
    pub fn set_leaves(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let sink = Arc::new(Mutex::new(LeafSink {
            writer: std::io::BufWriter::new(file),
            error: None,
            next_id: 0,
        }));
        for worker in &mut self.workers {
            worker.leaf_sink = Some(Arc::clone(&sink));
        }
        Ok(())
    }
    pub fn leaf_error(&self) -> Option<String> {
        self.workers[0]
            .leaf_sink
            .as_ref()
            .and_then(|s| s.lock().expect("leaf sink").error.clone())
    }

    pub fn new(hash_mb: usize, threads: usize) -> Self {
        let threads = threads.clamp(1, 8);
        let bytes = hash_mb.clamp(1, 2048) * 1024 * 1024;
        let shared = if threads > 1 {
            Some(Table::shared(bytes))
        } else {
            None
        };
        Self {
            workers: (0..threads)
                .map(|i| {
                    let mut worker = if let Some(Table::Shared(slots)) = &shared {
                        Searcher::with_table(Table::Shared(Arc::clone(slots)))
                    } else {
                        Searcher::with_bytes(bytes)
                    };
                    worker.worker_id = i as u64;
                    worker
                })
                .collect(),
            generation: 0,
        }
    }
    pub fn threads(&self) -> usize {
        self.workers.len()
    }
    pub fn clear(&mut self) {
        self.workers[0].tt.clear();
        for worker in &mut self.workers {
            worker.clear_heuristics();
        }
    }
    pub fn set_stop_signal(&mut self, signal: Arc<AtomicU64>, epoch: u64) {
        for worker in &mut self.workers {
            worker.set_stop_signal(Arc::clone(&signal), epoch);
        }
    }
    /// Node-budget play uses one worker regardless of the timed-search configuration.
    pub fn search_one(&mut self, model: &Model, state: &State, limits: Limits) -> SearchResult {
        self.workers[0].forced_deadline = None;
        self.workers[0].search(model, state, None, limits)
    }
    pub fn search(
        &mut self,
        model: &Model,
        state: &State,
        allowed: Option<&[u16]>,
        limits: Limits,
    ) -> SearchResult {
        if self.workers.len() == 1 {
            self.workers[0].forced_deadline = None;
            return self.workers[0].search(model, state, allowed, limits);
        }
        let start = Instant::now();
        let deadline = start
            .checked_add(limits.time)
            .unwrap_or(start + Duration::from_secs(86400));
        let active = limits.nodes.max(1).min(self.workers.len() as u64) as usize;
        self.generation = self.generation.wrapping_add(1);
        for worker in &mut self.workers[..active] {
            worker.forced_deadline = Some(deadline);
            worker.generation = self.generation.wrapping_sub(1);
        }
        let mut results = std::thread::scope(|scope| {
            let handles = self.workers[..active]
                .iter_mut()
                .enumerate()
                .map(|(i, worker)| {
                    let mut local = limits;
                    local.nodes = limits.nodes / active as u64
                        + u64::from((i as u64) < limits.nodes % active as u64);
                    scope.spawn(move || worker.search(model, state, allowed, local))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().expect("NNUE search worker failed"))
                .collect::<Vec<_>>()
        });
        let mut best = 0;
        let priority = |r: &SearchResult| {
            let won = r.score > MATE - MAX_PLY as i32;
            (won, if won { r.score } else { 0 }, r.depth, r.partial)
        };
        for i in 1..results.len() {
            if priority(&results[i]) > priority(&results[best]) {
                best = i;
            }
        }
        let nodes = results.iter().map(|r| r.nodes).sum();
        let qnodes = results.iter().map(|r| r.qnodes).sum();
        let aborted = results.iter().any(|r| r.aborted);
        let mut result = results.swap_remove(best);
        result.nodes = nodes;
        result.qnodes = qnodes;
        result.aborted = aborted;
        result.elapsed = start.elapsed();
        result
    }
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Self {
        Self::with_bytes(hash_mb.clamp(1, 2048) * 1024 * 1024)
    }
    fn with_bytes(bytes: usize) -> Self {
        Self::with_table(Table::local(bytes))
    }
    fn with_table(tt: Table) -> Self {
        let now = Instant::now();
        Self {
            tt,
            generation: 0,
            killers: [[NO_MOVE; 2]; MAX_PLY + 2],
            history: [[0; 648]; 2],
            acc: Vec::new(),
            hashes: [[0; 2]; MAX_PLY + 2],
            move_buffers: (0..MAX_PLY + 2).map(|_| Vec::with_capacity(80)).collect(),
            quiet_buffers: (0..MAX_PLY + 2).map(|_| Vec::with_capacity(80)).collect(),
            move_scores: [0; 648],
            start: now,
            deadline: now,
            limits: Limits::default(),
            nodes: 0,
            qnodes: 0,
            stopped: false,

            stop_signal: None,
            forced_deadline: None,
            worker_id: 0,
            leaf_sink: None,
            leaf_root: State::initial(),
            leaf_search: 0,
            leaf_path: [0; MAX_PLY + 2],
        }
    }
    pub fn clear(&mut self) {
        self.tt.clear();
        self.clear_heuristics();
    }
    fn clear_heuristics(&mut self) {
        self.history = [[0; 648]; 2];
        self.killers.fill([NO_MOVE; 2]);
    }
    pub fn set_stop_signal(&mut self, signal: Arc<AtomicU64>, epoch: u64) {
        self.stop_signal = Some((signal, epoch));
    }

    pub fn search(
        &mut self,
        model: &Model,
        state: &State,
        allowed: Option<&[u16]>,
        limits: Limits,
    ) -> SearchResult {
        self.start = Instant::now();
        if let Some(sink) = &self.leaf_sink {
            let mut sink = sink.lock().expect("leaf sink");
            sink.next_id += 1;
            self.leaf_search = sink.next_id;
            self.leaf_root = state.clone();
        }
        self.deadline = self.forced_deadline.unwrap_or_else(|| {
            self.start
                .checked_add(limits.time)
                .unwrap_or(self.start + Duration::from_secs(86400))
        });
        self.limits = limits;
        self.nodes = 0;
        self.qnodes = 0;
        self.stopped = false;
        self.generation = self.generation.wrapping_add(1);
        self.killers.fill([NO_MOVE; 2]);
        for side in &mut self.history {
            for value in side {
                *value /= 2;
            }
        }
        if self.acc.len() != MAX_PLY + 2
            || self.acc.first().is_none_or(|a| a.own.len() != model.hidden)
        {
            self.acc = (0..MAX_PLY + 2)
                .map(|_| Accumulator::empty(model.hidden))
                .collect();
        }
        self.acc[0] = model.refresh(&state.board, state.since_capture, 200);
        self.hashes[0] = [
            board_hash(&state.board),
            board_hash(&engine::flip(&state.board)),
        ];
        let legal = {
            #[cfg(feature = "profile")]
            let _probe = crate::profile::Probe::new(crate::profile::Zone::Movegen);
            state.legal_actions()
        };
        let root_moves: Vec<u16> = legal
            .iter()
            .copied()
            .filter(|m| allowed.is_none_or(|a| a.contains(m)))
            .collect();
        let mut result = SearchResult {
            action: root_moves.first().copied(),
            score: model.evaluate(&self.acc[0]),
            depth: 0,
            nodes: 0,
            qnodes: 0,
            elapsed: Duration::ZERO,
            aborted: false,
            partial: false,
            pv: Vec::new(),
            root_moves: root_moves.len(),
            iterations: Vec::new(),
        };
        if root_moves.is_empty() {
            result.score = -MATE;
            return result;
        }
        if draw_clock(state) {
            result.score = 0;
            return result;
        }
        // Establish a strong, legal emergency choice before the first timed iteration.
        let mut ordered = root_moves;
        self.order(state, &mut ordered, NO_MOVE, 0);
        let mut safe_fallback = None;
        for &action in &ordered {
            let (child, terminal) = apply_position(state, action);
            if terminal == Outcome::Win {
                result.action = Some(action);
                result.score = MATE - 1;
                result.depth = 1;
                result.pv = vec![action];
                result.elapsed = self.start.elapsed();
                return result;
            }
            if safe_fallback.is_none()
                && (terminal == Outcome::Draw || own_goal_move(&child.board).is_none())
            {
                safe_fallback = Some(action);
            }
        }
        result.action = safe_fallback.or_else(|| ordered.first().copied());
        for depth in 1..=limits.depth.clamp(1, (MAX_PLY - 2) as u8) {
            if self.should_stop(true) {
                break;
            }
            let iteration_start = Instant::now();
            let iteration_nodes = self.nodes;
            let previous_action = result.action;
            if let Some(best) = result.action {
                if let Some(i) = ordered.iter().position(|&m| m == best) {
                    ordered.swap(0, i);
                }
            }
            let mut width = if depth >= 4 { 80 } else { INF };
            let mut alpha = (result.score - width).max(-INF);
            let mut beta = (result.score + width).min(INF);
            loop {
                let (score, best) = self.root(model, state, &ordered, depth as i32, alpha, beta);
                if self.stopped {
                    if best.is_some() && best != result.action && score > alpha && score < beta {
                        result.action = best;
                        result.score = score;
                        result.partial = true;
                    }
                    break;
                }
                if score <= alpha && alpha > -INF {
                    width = (width * 3).min(INF);
                    alpha = (score - width).max(-INF);
                    continue;
                }
                if score >= beta && beta < INF {
                    width = (width * 3).min(INF);
                    beta = (score + width).min(INF);
                    continue;
                }
                result.score = score;
                result.action = best;
                result.depth = depth;
                result.pv = best.into_iter().collect();
                break;
            }
            result.iterations.push(Iteration {
                depth,
                nodes: self.nodes - iteration_nodes,
                elapsed_ms: iteration_start.elapsed().as_secs_f64() * 1000.,
                action: result.action,
                score: result.score,
                completed: !self.stopped,
                changed: result.action != previous_action,
            });
            if self.stopped || result.score.abs() >= MATE - MAX_PLY as i32 {
                break;
            }
            // Start another depth only while a useful fraction of the budget remains.
            if self.start.elapsed() >= limits.time.mul_f64(0.72) {
                break;
            }
        }
        result.pv = self.extract_pv(state, result.action, result.depth as usize);
        result.nodes = self.nodes;
        result.qnodes = self.qnodes;
        result.elapsed = self.start.elapsed();
        result.aborted = self.stopped;
        result
    }

    /// Reconstruct legal table moves from the current search generation.
    fn extract_pv(&mut self, root: &State, first: Option<u16>, depth: usize) -> Vec<u16> {
        let mut pv = Vec::with_capacity(depth.max(1));
        let mut state = root.clone();
        let mut next = first;
        for index in 0..depth.max(1) {
            let Some(action) = next else {
                break;
            };
            if action >= 648 || !state.legal_mask()[action as usize] {
                break;
            }
            pv.push(action);
            let (child, terminal) = apply_position(&state, action);
            if terminal != Outcome::Ongoing {
                break;
            }

            state = child;
            let key = position_key(board_hash(&state.board), state.since_capture);
            let entry = self.tt.get(key);
            next = if entry.bound != 0
                && entry.key == key
                && entry.generation == self.generation
                && entry.depth as usize >= depth.saturating_sub(index + 1)
            {
                Some(entry.action)
            } else {
                None
            };
        }
        pv
    }

    fn root(
        &mut self,
        model: &Model,
        state: &State,
        actions: &[u16],
        depth: i32,
        mut alpha: i32,
        beta: i32,
    ) -> (i32, Option<u16>) {
        let mut best = None;
        let mut score_best = -INF;
        for (index, &action) in actions.iter().enumerate() {
            if self.should_stop(true) {
                return (score_best, best);
            }
            let (child, terminal) = apply_position(state, action);
            let score = if terminal != Outcome::Ongoing {
                terminal_score(terminal, 1)
            } else {
                self.update_acc(model, state, action, child.since_capture, 0);

                let mut score = if index == 0 {
                    -self.negamax(model, &child, depth - 1, -beta, -alpha, 1, true)
                } else {
                    -self.negamax(model, &child, depth - 1, -alpha - 1, -alpha, 1, false)
                };
                if index != 0 && score > alpha && score < beta && !self.stopped {
                    score = -self.negamax(model, &child, depth - 1, -beta, -alpha, 1, true);
                }

                score
            };
            if self.stopped {
                return (score_best, best);
            }
            if score > score_best {
                score_best = score;
                best = Some(action);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        (score_best, best)
    }

    #[allow(clippy::too_many_arguments)] // Explicit window/ply arguments stay allocation-free.
    fn negamax(
        &mut self,
        model: &Model,
        state: &State,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        pv: bool,
    ) -> i32 {
        if depth <= 0 {
            return self.quiescence(model, state, alpha, beta, ply, 6);
        }
        self.nodes += 1;
        if self.should_stop(false) {
            return 0;
        }
        if draw_clock(state) {
            return 0;
        }
        if ply >= MAX_PLY {
            return self.frontier(model, state, ply);
        }
        let alpha_orig = alpha;
        let key = position_key(self.hashes[ply][0], state.since_capture);
        let entry = self.tt.get(key);
        let tt_move = if entry.bound != 0 && entry.key == key {
            entry.action
        } else {
            NO_MOVE
        };
        if entry.bound != 0 && entry.key == key && entry.depth >= depth as i16 && !pv {
            let score = from_tt(entry.score, ply);
            if entry.bound == 1
                || (entry.bound == 2 && score >= beta)
                || (entry.bound == 3 && score <= alpha)
            {
                return score;
            }
        }
        let mut actions = std::mem::take(&mut self.move_buffers[ply]);
        actions.clear();
        {
            #[cfg(feature = "profile")]
            let _probe = crate::profile::Probe::new(crate::profile::Zone::Movegen);
            state.legal_actions_into(&mut actions);
        }
        if actions.is_empty() {
            self.move_buffers[ply] = actions;
            return -MATE + ply as i32;
        }
        if !pv && depth <= 2 && beta.abs() < 28_000 && actions.len() > 1 {
            let value = self.static_eval(model, state, ply);
            let margin = FUTILITY_MARGIN * depth;
            if value - margin >= beta && pruning_safe(state) {
                self.move_buffers[ply] = actions;
                return value - margin;
            }
        }
        let prune_quiets = !pv
            && depth <= 2
            && alpha.abs() < 28_000
            && actions.len() > 1
            && self.static_eval(model, state, ply) + FUTILITY_MARGIN * depth <= alpha
            && pruning_safe(state);
        let threatened = enemy_goal_threat(&state.board);
        self.order(state, &mut actions, tt_move, ply);
        let mut best_score = -INF;
        let mut best_action = actions[0];
        let mut quiets = std::mem::take(&mut self.quiet_buffers[ply]);
        quiets.clear();
        for (index, &action) in actions.iter().enumerate() {
            let (from, target) = action_from_to(action);
            let quiet = state.board[target] == Cell::Empty;
            let (child, terminal) = apply_position(state, action);
            if prune_quiets
                && quiet
                && best_score > -INF
                && terminal == Outcome::Ongoing
                && !near_goal(from)
                && !near_goal(target)
                && !enemy_goal_threat(&child.board)
            {
                continue;
            }
            let score = if terminal != Outcome::Ongoing {
                terminal_score(terminal, ply + 1)
            } else if own_goal_move(&child.board).is_some() {
                // The opponent can end the game immediately after this move.
                -MATE + ply as i32 + 2
            } else {
                self.update_acc(model, state, action, child.since_capture, ply);

                let child_threat = enemy_goal_threat(&child.board);
                let reduction = if depth >= 3
                    && index >= 4
                    && quiet
                    && !pv
                    && !threatened
                    && !child_threat
                    && !near_goal(target)
                    && self.killers[ply].iter().all(|&m| m != action)
                {
                    history_reduction(
                        depth,
                        index,
                        self.history[state.ply as usize & 1][action as usize],
                    )
                } else {
                    0
                };
                let mut score;
                if index == 0 {
                    score = -self.negamax(model, &child, depth - 1, -beta, -alpha, ply + 1, pv);
                } else {
                    score = -self.negamax(
                        model,
                        &child,
                        (depth - 1 - reduction).max(0),
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        false,
                    );
                    if reduction > 0 && score > alpha && !self.stopped {
                        score = -self.negamax(
                            model,
                            &child,
                            depth - 1,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            false,
                        );
                    }
                    if score > alpha && score < beta && !self.stopped {
                        score = -self.negamax(model, &child, depth - 1, -beta, -alpha, ply + 1, pv);
                    }
                }

                score
            };
            if self.stopped {
                self.move_buffers[ply] = actions;
                self.quiet_buffers[ply] = quiets;
                return 0;
            }
            if score > best_score {
                best_score = score;
                best_action = action;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                if quiet {
                    if self.killers[ply][0] != action {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = action;
                    }
                    let side = state.ply as usize & 1;
                    let bonus = (depth * depth * 32).min(1600);
                    history_update(&mut self.history[side][action as usize], bonus);
                    for &quiet_action in &quiets {
                        history_update(&mut self.history[side][quiet_action as usize], -bonus);
                    }
                }
                break;
            }
            if quiet {
                quiets.push(action);
            }
        }
        self.move_buffers[ply] = actions;
        self.quiet_buffers[ply] = quiets;
        let bound = if best_score >= beta {
            2
        } else if best_score <= alpha_orig {
            3
        } else {
            1
        };
        self.tt.store(Entry {
            key,
            score: to_tt(best_score, ply),
            static_eval: entry.static_eval,
            static_valid: entry.key == key && entry.static_valid,
            action: best_action,
            depth: depth as i16,
            bound,
            generation: self.generation,
        });
        best_score
    }

    fn quiescence(
        &mut self,
        model: &Model,
        state: &State,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        remaining: i32,
    ) -> i32 {
        self.nodes += 1;
        self.qnodes += 1;
        if self.should_stop(false) {
            return 0;
        }
        if draw_clock(state) {
            return 0;
        }
        if own_goal_move(&state.board).is_some() {
            return MATE - ply as i32 - 1;
        }
        if ply >= MAX_PLY {
            return self.frontier(model, state, ply);
        }
        let stalemated = {
            #[cfg(feature = "profile")]
            let _probe = crate::profile::Probe::new(crate::profile::Zone::Movegen);
            state.is_stalemated()
        };
        if stalemated {
            return -MATE + ply as i32;
        }
        let threatened = enemy_goal_threat(&state.board);
        let stand = self.static_eval(model, state, ply);
        let mut best = if threatened { -INF } else { stand };
        if !threatened {
            if stand >= beta {
                return stand;
            }
            alpha = alpha.max(stand);
            if remaining <= 0 {
                return stand;
            }
        }
        let mut all = std::mem::take(&mut self.move_buffers[ply]);
        all.clear();
        {
            #[cfg(feature = "profile")]
            let _probe = crate::profile::Probe::new(crate::profile::Zone::Movegen);
            state.legal_actions_into(&mut all);
        }
        all.retain(|&action| {
            let (from, to) = action_from_to(action);
            threatened
                || state.board[to] != Cell::Empty
                || (remaining >= 5
                    && [70, 71, 79].contains(&to)
                    && (state.board[80] == Cell::Empty
                        || (state.board[80].is_enemy()
                            && engine::tactics::can_capture(state.board[from], state.board[80]))))
        });
        self.order(state, &mut all, NO_MOVE, ply);
        for &action in &all {
            let (child, terminal) = apply_position(state, action);
            let score = if terminal != Outcome::Ongoing {
                terminal_score(terminal, ply + 1)
            } else if own_goal_move(&child.board).is_some() {
                -MATE + ply as i32 + 2
            } else {
                self.update_acc(model, state, action, child.since_capture, ply);

                -self.quiescence(model, &child, -beta, -alpha, ply + 1, remaining - 1)
            };
            if self.stopped {
                self.move_buffers[ply] = all;
                return 0;
            }
            best = best.max(score);
            alpha = alpha.max(score);
            if alpha >= beta {
                self.move_buffers[ply] = all;
                return best;
            }
        }
        self.move_buffers[ply] = all;
        best
    }

    // At the recursion safety limit, still prove immediate wins and forced losses;
    // threatened positions are evaluated after a legal evasion, never by stand-pat.
    fn frontier(&mut self, model: &Model, state: &State, ply: usize) -> i32 {
        if own_goal_move(&state.board).is_some() {
            return MATE - ply as i32 - 1;
        }
        let actions = {
            #[cfg(feature = "profile")]
            let _probe = crate::profile::Probe::new(crate::profile::Zone::Movegen);
            state.legal_actions()
        };
        if actions.is_empty() {
            return -MATE + ply as i32;
        }
        if !enemy_goal_threat(&state.board) {
            return self.static_eval(model, state, ply);
        }
        let mut best = -MATE + ply as i32 + 2;
        for action in actions {
            let (child, terminal) = apply_position(state, action);
            if terminal != Outcome::Ongoing {
                best = best.max(terminal_score(terminal, ply + 1));
            } else if own_goal_move(&child.board).is_none() {
                self.update_acc(model, state, action, child.since_capture, ply);
                best = best.max(-self.static_eval(model, &child, ply + 1));
            }
        }
        best
    }

    fn static_eval(&mut self, model: &Model, state: &State, ply: usize) -> i32 {
        let key = position_key(self.hashes[ply][0], state.since_capture);
        let mut entry = self.tt.get(key);
        let score = if entry.key == key && entry.static_valid {
            entry.static_eval
        } else {
            model.resolve_race(&state.board, &mut self.acc[ply]);
            let score = model.evaluate(&self.acc[ply]);
            if entry.key != key {
                entry = Entry::default();
            }
            entry.key = key;
            entry.static_eval = score;
            entry.static_valid = true;
            entry.generation = self.generation;
            self.tt.store(entry);
            score
        };
        if let Some(sink) = &self.leaf_sink {
            model.resolve_race(&state.board, &mut self.acc[ply]);
            let record = serde_json::json!({
                "schema":1,"search_id":self.leaf_search,"worker":self.worker_id,
                "root":{"board":engine::codes(&self.leaf_root.board).as_slice(),"since_capture":self.leaf_root.since_capture,"ply":self.leaf_root.ply,"clock":200},
                "board":engine::codes(&state.board).as_slice(),"since_capture":state.since_capture,"ply":state.ply,"clock":200,
                "raw":model.raw(&self.acc[ply]),"score":score,"path":&self.leaf_path[..ply]
            });
            let mut sink = sink.lock().expect("leaf sink");
            if sink.error.is_none() {
                use std::io::Write;
                let result = serde_json::to_writer(&mut sink.writer, &record)
                    .map_err(std::io::Error::other)
                    .and_then(|()| sink.writer.write_all(b"\n"))
                    .and_then(|()| sink.writer.flush());
                if let Err(e) = result {
                    sink.error = Some(e.to_string());
                    self.stopped = true;
                }
            } else {
                self.stopped = true;
            }
        }
        score
    }
    fn update_acc(
        &mut self,
        model: &Model,
        state: &State,
        action: u16,
        child_since_capture: u32,
        ply: usize,
    ) {
        self.leaf_path[ply] = action;
        self.hashes[ply + 1] = child_hashes(self.hashes[ply], &state.board, action);
        let (parents, children) = self.acc.split_at_mut(ply + 1);
        model.update_deferred(
            &parents[ply],
            &state.board,
            action,
            child_since_capture,
            &mut children[0],
        );
    }

    fn order(&mut self, state: &State, actions: &mut [u16], tt_move: u16, ply: usize) {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Ordering);
        for &action in actions.iter() {
            let (from, to) = action_from_to(action);
            let mut score = if action == tt_move { 2_000_000 } else { 0 };
            if to == 80 {
                score += 1_000_000;
            }
            if state.board[to] != Cell::Empty {
                score += 100_000 + (8 - (to / 9).max(to % 9)) as i32 * 200;
            }
            if self.killers[ply][0] == action {
                score += 60_000;
            }
            if self.killers[ply][1] == action {
                score += 50_000;
            }
            score += self.history[state.ply as usize & 1][action as usize];
            if self.worker_id != 0 {
                score += (mix(action as u64 ^ self.worker_id.wrapping_mul(0x9e3779b97f4a7c15)) % 97)
                    as i32
                    - 48;
            }
            let old_distance = (8 - from / 9).max(8 - from % 9);
            let distance = (8 - to / 9).max(8 - to % 9);
            score += (old_distance as i32 - distance as i32) * 64;
            if distance <= 2 {
                score += (3 - distance) as i32 * 80;
            }
            self.move_scores[action as usize] = score;
        }
        actions.sort_unstable_by(|a, b| {
            self.move_scores[*b as usize]
                .cmp(&self.move_scores[*a as usize])
                .then(a.cmp(b))
        });
    }

    #[inline]
    fn should_stop(&mut self, force: bool) -> bool {
        if self.stopped {
            return true;
        }
        if self.nodes >= self.limits.nodes
            || ((force || self.nodes & 63 == 0)
                && (Instant::now() >= self.deadline
                    || self
                        .stop_signal
                        .as_ref()
                        .is_some_and(|(signal, epoch)| signal.load(Ordering::Relaxed) != *epoch)))
        {
            self.stopped = true;
        }
        self.stopped
    }
}

fn history_update(value: &mut i32, bonus: i32) {
    *value += bonus - *value * bonus.abs() / 16384;
}
#[inline]
fn to_tt(score: i32, ply: usize) -> i32 {
    if score > 29000 {
        score + ply as i32
    } else if score < -29000 {
        score - ply as i32
    } else {
        score
    }
}
#[inline]
fn from_tt(score: i32, ply: usize) -> i32 {
    if score > 29000 {
        score - ply as i32
    } else if score < -29000 {
        score + ply as i32
    } else {
        score
    }
}
#[inline]
fn terminal_score(value: Outcome, child_ply: usize) -> i32 {
    if value == Outcome::Draw {
        0
    } else {
        MATE - child_ply as i32
    }
}
#[inline]
fn draw_clock(state: &State) -> bool {
    Rules::SITE.clock_expired(state)
}
#[inline]
fn near_goal(square: usize) -> bool {
    (square / 9 >= 6 && square % 9 >= 6) || (square / 9 <= 2 && square % 9 <= 2)
}

#[inline]
const fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn board_hash(board: &Board) -> u64 {
    let mut hash = 0x72cbb37ce9a493a1;
    for (sq, &piece) in board.iter().enumerate() {
        if piece != Cell::Empty {
            hash ^= mix((piece.code() as u64 * 81 + sq as u64).wrapping_add(0x9e3779b97f4a7c15));
        }
    }
    hash
}
#[inline]
fn apply_position(state: &State, action: u16) -> (State, Outcome) {
    engine::apply(&Rules::SITE, state, action)
}

struct LeafSink {
    writer: std::io::BufWriter<std::fs::File>,
    error: Option<String>,
    next_id: u64,
}

const ZOBRIST: [[u64; 81]; 7] = {
    let mut keys = [[0; 81]; 7];
    let mut piece = 1;
    while piece < 7 {
        let mut sq = 0;
        while sq < 81 {
            keys[piece][sq] = mix((piece as u64 * 81 + sq as u64).wrapping_add(0x9e3779b97f4a7c15));
            sq += 1;
        }
        piece += 1;
    }
    keys
};
fn position_key(board: u64, since_capture: u32) -> u64 {
    board ^ mix(since_capture as u64 ^ 0x61eeddbb) ^ mix(200u64 ^ 0x8794368a)
}
fn child_hashes(parent: [u64; 2], board: &Board, action: u16) -> [u64; 2] {
    let (from, to) = action_from_to(action);
    let piece = board[from];
    let captured = board[to];
    let own = parent[0]
        ^ ZOBRIST[piece.code() as usize][from]
        ^ ZOBRIST[piece.code() as usize][to]
        ^ ZOBRIST[captured.code() as usize][to];
    let from = engine::mirror_anti(from as u8) as usize;
    let to = engine::mirror_anti(to as u8) as usize;
    let other = parent[1]
        ^ ZOBRIST[piece.swap_side().code() as usize][from]
        ^ ZOBRIST[piece.swap_side().code() as usize][to]
        ^ ZOBRIST[captured.swap_side().code() as usize][to];
    [other, own]
}

const FUTILITY_MARGIN: i32 = 288;

fn pruning_safe(state: &State) -> bool {
    if state.since_capture >= 192 || state.own_count() < 3 || state.enemy_count() < 3 {
        return false;
    }
    if state.board.iter().enumerate().any(|(sq, cell)| {
        (cell.is_own() && sq / 9 >= 6 && sq % 9 >= 6)
            || (cell.is_enemy() && sq / 9 <= 2 && sq % 9 <= 2)
    }) {
        return false;
    }
    if engine::tactics::any_win_at_once(&Rules::SITE, state) {
        return false;
    }
    let enemy = State {
        board: engine::flip(&state.board),
        since_capture: state.since_capture,
        ply: state.ply + 1,
    };
    !engine::tactics::any_win_at_once(&Rules::SITE, &enemy)
}

fn history_reduction(depth: i32, index: usize, history: i32) -> i32 {
    let base = 1 + i32::from(depth >= 6 && index >= 12);
    (base - (history / 1024).clamp(-1, 1)).clamp(0, depth - 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn iteration_records_account_for_the_selected_search() {
        let model = Model::from_bytes(include_bytes!("../tests/fixtures/h768_dense.nnue")).unwrap();
        let state = State::initial();
        let result = Searcher::new(1).search(
            &model,
            &state,
            None,
            Limits {
                nodes: 10_000,
                time: Duration::from_secs(30),
                ..Limits::default()
            },
        );
        assert_eq!(result.root_moves, state.legal_actions().len());
        assert_eq!(
            result.iterations.iter().map(|i| i.nodes).sum::<u64>(),
            result.nodes
        );
        assert_eq!(
            result.iterations.iter().filter(|i| i.completed).count(),
            result.depth as usize
        );
        for (index, iteration) in result.iterations.iter().enumerate() {
            assert_eq!(iteration.depth as usize, index + 1);
            assert!(iteration.elapsed_ms >= 0.);
            if index > 0 {
                assert_eq!(
                    iteration.changed,
                    iteration.action != result.iterations[index - 1].action
                );
            }
        }
        let last = result.iterations.last().unwrap();
        assert_eq!(last.action, result.action);
        assert_eq!(last.score, result.score);
        assert_eq!(!last.completed, result.aborted);
    }

    #[test]
    fn quiescence_proves_stalemate_before_stand_pat() {
        let model = Model::from_bytes(include_bytes!("../tests/fixtures/h768_dense.nnue")).unwrap();
        let mut state = State::initial();
        state.board.fill(Cell::Empty);
        state.board[40] = Cell::Own(engine::Piece::Rock);
        for square in [30, 31, 32, 39, 41, 48, 49, 50] {
            state.board[square] = Cell::Enemy(engine::Piece::Paper);
        }
        assert!(state.legal_actions().is_empty());
        let mut search = Searcher::new(1);
        search.deadline = Instant::now() + Duration::from_secs(10);
        search.acc.push(model.refresh(&state.board, 0, 200));
        assert_eq!(search.quiescence(&model, &state, -INF, -29000, 0, 0), -MATE);
        state.board[30] = Cell::Empty;
        assert!(!state.legal_actions().is_empty());
        search.acc[0] = model.refresh(&state.board, 0, 200);
        search.hashes[0][0] = board_hash(&state.board);
        let stand = model.evaluate(&search.acc[0]);
        assert_eq!(search.quiescence(&model, &state, -INF, INF, 0, 0), stand);
    }

    #[test]
    fn canonical_hash_updates_match_full_boards() {
        let mut state = State::initial();
        let mut hashes = [
            board_hash(&state.board),
            board_hash(&engine::flip(&state.board)),
        ];
        for i in 0..2000 {
            let moves = state.legal_actions();
            let a = moves[(i * 713 + i / 7) % moves.len()];
            let (child, end) = apply_position(&state, a);
            hashes = child_hashes(hashes, &state.board, a);
            assert_eq!(
                hashes,
                [
                    board_hash(&child.board),
                    board_hash(&engine::flip(&child.board))
                ]
            );
            state = child;
            if end != Outcome::Ongoing {
                state = State::initial();
                hashes = [
                    board_hash(&state.board),
                    board_hash(&engine::flip(&state.board)),
                ];
            }
        }
        assert_ne!(position_key(hashes[0], 10), position_key(hashes[0], 11));
    }
    #[test]
    fn pruning_is_disabled_near_goals_draws_and_sparse_endings() {
        let mut state = State::initial();
        assert!(pruning_safe(&state));
        state.since_capture = 192;
        assert!(!pruning_safe(&state));
        state.since_capture = 0;
        state.board[70] = Cell::Own(engine::Piece::Rock);
        assert!(!pruning_safe(&state));
        state = State::initial();
        state.board[10] = Cell::Enemy(engine::Piece::Rock);
        assert!(!pruning_safe(&state));
        state.board.fill(Cell::Empty);
        state.board[40] = Cell::Own(engine::Piece::Rock);
        state.board[50] = Cell::Enemy(engine::Piece::Rock);
        assert!(!pruning_safe(&state));
    }
    #[test]
    fn history_reductions_remain_bounded_and_monotonic() {
        for depth in 3..30 {
            for index in [4, 12, 40] {
                let mut previous = depth;
                for history in (-16000..=16000).step_by(500) {
                    let r = history_reduction(depth, index, history);
                    assert!(r >= 0 && r <= depth - 2);
                    assert!(r <= previous);
                    previous = r;
                }
            }
        }
    }
}
