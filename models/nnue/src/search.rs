//! Iterative-deepening PVS with tactical quiescence and a clock-aware TT.
use crate::net::{Accumulator, FeatureState, Model};
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
const CAPTURE_HISTORY_SIZE: usize = 2 * 3 * 81 * 3;

#[cfg(test)]
mod accumulator_tests;
#[cfg(test)]
mod capture_history_tests;

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

impl SearchResult {
    /// The completed root label, independent of a later partial playing choice.
    pub fn completed_root(&self) -> Option<(u16, i32, u8)> {
        self.iterations
            .iter()
            .rev()
            .find(|i| i.completed)
            .and_then(|i| i.action.map(|action| (action, i.score, i.depth)))
    }
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
    fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        if let Self::Local(slots) = self {
            let entry = &slots[key as usize & (slots.len() - 1)];
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    (entry as *const Entry).cast(),
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
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

#[derive(Clone, Copy)]
struct Pending {
    board: Board,
    action: u16,
    before: FeatureState,
    after: FeatureState,
    ready: u8,
}

/// Diagnostic work counts summed over search workers; times sum elapsed spans on those workers.
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct AccumulatorMetrics {
    pub prepared_edges: u64,
    pub exact_count_transitions: u64,
    pub materializations: u64,
    pub capped_context_changes: u64,
    pub full_refreshes: u64,
    pub half_refreshes: u64,
    pub incremental_updates: u64,
    pub row_additions: u64,
    pub row_subtractions: u64,
    pub refresh_ns: u64,
    pub update_ns: u64,
    pub pending_nodes_discarded: u64,
    pub pending_halves_discarded: u64,
}

pub struct Searcher {
    tt: Table,
    generation: u8,
    killers: [[u16; 2]; MAX_PLY + 2],
    history: [[i32; 648]; 2],
    capture_history: [i32; CAPTURE_HISTORY_SIZE],
    acc: Vec<Accumulator>,
    pending: Vec<Option<Pending>>,
    metrics: Option<AccumulatorMetrics>,
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
}

/// Optional Lazy SMP. Each worker completes its own legal iterative search;
/// workers share a hard deadline, model, stop signal and transposition table.
pub struct SearchPool {
    workers: Vec<Searcher>,
    generation: u8,
}

impl SearchPool {
    pub fn new(hash_mb: usize, threads: usize) -> Self {
        let threads = threads.clamp(1, 6);
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
    pub fn enable_accumulator_metrics(&mut self) {
        for worker in &mut self.workers {
            worker.metrics = Some(AccumulatorMetrics::default());
        }
    }
    pub fn accumulator_metrics(&self) -> AccumulatorMetrics {
        let mut total = AccumulatorMetrics::default();
        for worker in &self.workers {
            if let Some(m) = worker.metrics {
                total.prepared_edges += m.prepared_edges;
                total.exact_count_transitions += m.exact_count_transitions;
                total.materializations += m.materializations;
                total.capped_context_changes += m.capped_context_changes;
                total.full_refreshes += m.full_refreshes;
                total.half_refreshes += m.half_refreshes;
                total.incremental_updates += m.incremental_updates;
                total.row_additions += m.row_additions;
                total.row_subtractions += m.row_subtractions;
                total.refresh_ns += m.refresh_ns;
                total.update_ns += m.update_ns;
                total.pending_nodes_discarded += m.pending_nodes_discarded;
                total.pending_halves_discarded += m.pending_halves_discarded;
            }
        }
        total
    }
    fn reset_metrics(&mut self) {
        for worker in &mut self.workers {
            if let Some(m) = &mut worker.metrics {
                *m = AccumulatorMetrics::default();
            }
        }
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
        self.reset_metrics();
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
        self.reset_metrics();
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
            capture_history: [0; CAPTURE_HISTORY_SIZE],
            acc: Vec::new(),
            pending: vec![None; MAX_PLY + 2],
            metrics: None,
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
        }
    }
    pub fn clear(&mut self) {
        self.tt.clear();
        self.clear_heuristics();
    }
    fn clear_heuristics(&mut self) {
        self.generation = 0;
        self.history = [[0; 648]; 2];
        self.capture_history.fill(0);
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
        for value in &mut self.capture_history {
            *value /= 2;
        }
        if self.acc.len() != MAX_PLY + 2
            || self.acc.first().is_none_or(|a| a.own.len() != model.hidden)
        {
            self.acc = (0..MAX_PLY + 2)
                .map(|_| Accumulator::empty(model.hidden))
                .collect();
        }
        self.pending.fill(None);
        let timer = self.metrics.map(|_| Instant::now());
        let rows = model.refresh_into(&state.board, state.since_capture, 200, &mut self.acc[0]);
        if let Some(m) = &mut self.metrics {
            *m = AccumulatorMetrics {
                full_refreshes: 1,
                materializations: 2,
                row_additions: rows,
                refresh_ns: timer.unwrap().elapsed().as_nanos() as u64,
                ..AccumulatorMetrics::default()
            };
        }
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
            if limits.time != Duration::MAX && self.start.elapsed() >= limits.time.mul_f64(0.72) {
                break;
            }
        }
        result.pv = self.extract_pv(state, result.action, result.depth as usize);
        result.nodes = self.nodes;
        result.qnodes = self.qnodes;
        result.elapsed = self.start.elapsed();
        result.aborted = self.stopped;
        for ply in 1..self.pending.len() {
            self.discard_pending(ply);
        }
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
                self.prepare_child(state, action, child.since_capture, 0);

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

    /// Searches a child whose terminal outcome has already been resolved.
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
        debug_assert!(!draw_clock(state));
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
        let moves = {
            #[cfg(feature = "profile")]
            let _probe = crate::profile::Probe::new(crate::profile::Zone::Movegen);
            state.legal_moves()
        };
        if moves.is_empty() {
            self.move_buffers[ply] = actions;
            return -MATE + ply as i32;
        }
        if !pv && depth <= 2 && beta.abs() < 28_000 && moves.len() > 1 {
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
            && moves.len() > 1
            && self.static_eval(model, state, ply) + FUTILITY_MARGIN * depth <= alpha
            && pruning_safe(state);
        let threatened = enemy_goal_threat(&state.board);
        let history = self.history[state.ply as usize & 1];
        let killers = self.killers[ply];
        {
            #[cfg(feature = "profile")]
            let _probe = crate::profile::Probe::new(crate::profile::Zone::Movegen);
            moves.captures_into(&mut actions);
            moves.quiets_into(1 << 80, &mut actions);
            for action in [tt_move, killers[0], killers[1]] {
                if moves.contains(action) && !actions.contains(&action) {
                    actions.push(action);
                }
            }
        }
        self.order(state, &mut actions, tt_move, ply);
        let mut best_score = -INF;
        let mut best_action = NO_MOVE;
        let mut quiets = std::mem::take(&mut self.quiet_buffers[ply]);
        quiets.clear();
        let mut index = 0;
        let mut quiets_generated = false;
        loop {
            if index == actions.len() && !quiets_generated {
                let start = actions.len();
                {
                    #[cfg(feature = "profile")]
                    let _probe = crate::profile::Probe::new(crate::profile::Zone::Movegen);
                    moves.quiets_into(!(1 << 80), &mut actions);
                    let mut end = start;
                    for next in start..actions.len() {
                        let action = actions[next];
                        if action != tt_move && !killers.contains(&action) {
                            actions[end] = action;
                            end += 1;
                        }
                    }
                    actions.truncate(end);
                }
                self.order_using(state, &mut actions[start..], tt_move, ply, Some(&history));
                quiets_generated = true;
            }
            let Some(&action) = actions.get(index) else {
                break;
            };
            let move_index = index;
            index += 1;
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
                self.prepare_child(state, action, child.since_capture, ply);

                let child_threat = enemy_goal_threat(&child.board);
                let reduction = if depth >= 3
                    && move_index >= 4
                    && quiet
                    && !pv
                    && !threatened
                    && !child_threat
                    && !near_goal(target)
                    && self.killers[ply].iter().all(|&m| m != action)
                {
                    history_reduction(
                        depth,
                        move_index,
                        self.history[state.ply as usize & 1][action as usize],
                    )
                } else {
                    0
                };
                let mut score;
                if move_index == 0 {
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
                self.update_captures(state, action, &actions[..move_index], depth);
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

    /// Searches an ongoing child; below ply one its immediate goal win is excluded.
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
        debug_assert!(!draw_clock(state) && !state.is_stalemated());
        debug_assert!(ply <= 1 || own_goal_move(&state.board).is_none());
        if ply <= 1 && own_goal_move(&state.board).is_some() {
            return MATE - ply as i32 - 1;
        }
        if ply >= MAX_PLY {
            return self.frontier(model, state, ply);
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
                self.prepare_child(state, action, child.since_capture, ply);

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
                self.prepare_child(state, action, child.since_capture, ply);
                best = best.max(-self.static_eval(model, &child, ply + 1));
            }
        }
        best
    }

    fn static_eval(&mut self, model: &Model, state: &State, ply: usize) -> i32 {
        let key = position_key(self.hashes[ply][0], state.since_capture);
        let mut entry = self.tt.get(key);
        if entry.key == key && entry.static_valid {
            entry.static_eval
        } else {
            self.resolve_acc(model, ply);
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
        }
    }
    fn prepare_child(&mut self, state: &State, action: u16, child_since_capture: u32, ply: usize) {
        self.hashes[ply + 1] = child_hashes(self.hashes[ply], &state.board, action);
        self.tt
            .prefetch(position_key(self.hashes[ply + 1][0], child_since_capture));
        self.discard_pending(ply + 1);
        let before = self.pending[ply].map_or(self.acc[ply].state, |p| p.after);
        let after = before.after(&state.board, action, child_since_capture);
        self.pending[ply + 1] = Some(Pending {
            board: state.board,
            action,
            before,
            after,
            ready: 0,
        });
        if let Some(m) = &mut self.metrics {
            m.prepared_edges += 1;
            m.exact_count_transitions +=
                u64::from(state.board[action_from_to(action).1] != Cell::Empty);
        }
    }

    fn discard_pending(&mut self, ply: usize) {
        if let Some(p) = self.pending[ply].take() {
            if let Some(m) = &mut self.metrics {
                m.pending_nodes_discarded += 1;
                m.pending_halves_discarded += 2 - p.ready.count_ones() as u64;
            }
        }
    }

    /// Resolve each half through its ancestors only as far as its last context change.
    fn resolve_acc(&mut self, model: &Model, ply: usize) {
        self.resolve_half(model, ply, 0);
        self.resolve_half(model, ply, 1);
    }

    fn resolve_half(&mut self, model: &Model, ply: usize, side: usize) {
        let Some(p) = self.pending[ply] else {
            return;
        };
        if p.ready & (1 << side) != 0 {
            return;
        }
        let refresh = model.requires_refresh(p.before, p.after, side);
        if !refresh {
            self.resolve_half(model, ply - 1, side ^ 1);
        }
        let timer = self.metrics.map(|_| Instant::now());
        let (parents, children) = self.acc.split_at_mut(ply);
        let (src, dst) = if side == 0 {
            (&parents[ply - 1].opponent, &mut children[0].own)
        } else {
            (&parents[ply - 1].own, &mut children[0].opponent)
        };
        let (added, removed) =
            model.update_perspective(src, &p.board, p.action, p.before, p.after, side, dst);
        if let Some(m) = &mut self.metrics {
            m.materializations += 1;
            m.capped_context_changes +=
                u64::from(p.before.contexts()[side ^ 1] != p.after.contexts()[side]);
            m.row_additions += added;
            m.row_subtractions += removed;
            let elapsed = timer.unwrap().elapsed().as_nanos() as u64;
            if refresh {
                m.half_refreshes += 1;
                m.refresh_ns += elapsed;
            } else {
                m.incremental_updates += 1;
                m.update_ns += elapsed;
            }
        }
        let pending = self.pending[ply].as_mut().unwrap();
        pending.ready |= 1 << side;
        if pending.ready == 3 {
            self.acc[ply].state = p.after;
            self.pending[ply] = None;
        }
    }

    /// Train captures only from completed main-search cutoffs and searched alternatives.
    fn update_captures(&mut self, state: &State, winner: u16, searched: &[u16], depth: i32) {
        let bonus = (depth * depth * 32).min(1600);
        if let Some(index) = capture_history_index(state, winner) {
            history_update(&mut self.capture_history[index], bonus);
        }
        for &action in searched {
            if let Some(index) = capture_history_index(state, action) {
                history_update(&mut self.capture_history[index], -bonus);
            }
        }
    }

    fn order(&mut self, state: &State, actions: &mut [u16], tt_move: u16, ply: usize) {
        self.order_using(state, actions, tt_move, ply, None);
    }

    fn order_using(
        &mut self,
        state: &State,
        actions: &mut [u16],
        tt_move: u16,
        ply: usize,
        history: Option<&[i32; 648]>,
    ) {
        #[cfg(feature = "profile")]
        let _probe = crate::profile::Probe::new(crate::profile::Zone::Ordering);
        let history = history.unwrap_or(&self.history[state.ply as usize & 1]);
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
            score += capture_history_index(state, action)
                .map_or(history[action as usize], |index| {
                    self.capture_history[index]
                });
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
                && ((self.limits.time != Duration::MAX && Instant::now() >= self.deadline)
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

/// Piece types and destination use the mover's canonical frame; side is absolute.
fn capture_history_index(state: &State, action: u16) -> Option<usize> {
    let (from, to) = action_from_to(action);
    match (state.board[from], state.board[to]) {
        (Cell::Own(attacker), Cell::Enemy(victim)) => Some(
            (((state.ply as usize & 1) * 3 + attacker.code() as usize - 1) * 81 + to) * 3
                + victim.code() as usize
                - 1,
        ),
        _ => None,
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
    fn deferred_accumulators_resolve_ancestors_and_replaced_branches() {
        for features in [1004, 13640] {
            let mut model =
                Model::from_bytes(include_bytes!("../tests/fixtures/h768_dense.nnue")).unwrap();
            model.features = features;
            model.weights.resize(features * model.hidden, 0);
            let root = State::initial();
            let mut search = Searcher::new(1);
            search.acc = (0..MAX_PLY + 2)
                .map(|_| Accumulator::empty(model.hidden))
                .collect();
            search.acc[0] = model.refresh(&root.board, 0, 200);
            search.hashes[0] = [
                board_hash(&root.board),
                board_hash(&engine::flip(&root.board)),
            ];
            let mut state = root.clone();
            let mut depth = 0;
            for ply in 0..16 {
                let action = state.legal_actions()[0];
                let (child, outcome) = apply_position(&state, action);
                if outcome != Outcome::Ongoing {
                    break;
                }
                search.prepare_child(&state, action, child.since_capture, ply);
                state = child;
                depth += 1;
            }
            assert!(depth >= 4);
            assert!(search.pending[1..=depth].iter().all(Option::is_some));
            search.resolve_acc(&model, depth);
            assert_eq!(
                search.acc[depth],
                model.refresh(&state.board, state.since_capture, 200)
            );

            let action = root.legal_actions()[1];
            let (child, _) = apply_position(&root, action);
            search.prepare_child(&root, action, child.since_capture, 0);
            search.resolve_acc(&model, 1);
            assert_eq!(
                search.acc[1],
                model.refresh(&child.board, child.since_capture, 200)
            );
        }
    }

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
    fn stalemate_is_resolved_before_searching_a_child() {
        let model = Model::from_bytes(include_bytes!("../tests/fixtures/h768_dense.nnue")).unwrap();
        let mut state = State::initial();
        state.board.fill(Cell::Empty);
        state.board[40] = Cell::Own(engine::Piece::Rock);
        for square in [30, 31, 32, 39, 41, 48, 49, 50] {
            state.board[square] = Cell::Enemy(engine::Piece::Paper);
        }
        assert!(state.legal_actions().is_empty());
        let mut search = Searcher::new(1);
        let result = search.search(&model, &state, None, Limits::default());
        assert_eq!((result.action, result.score), (None, -MATE));
        let mut before = state.clone();
        before.board[30] = Cell::Empty;
        before.board[20] = Cell::Enemy(engine::Piece::Paper);
        before.board = engine::flip(&before.board);
        let result = search.search(&model, &before, None, Limits::default());
        let (child, outcome) = apply_position(&before, result.action.unwrap());
        assert_eq!(outcome, Outcome::Win);
        assert!(child.is_stalemated());
        assert_eq!(result.score, MATE - 1);
        search.deadline = Instant::now() + Duration::from_secs(10);
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

#[cfg(test)]
mod generation_tests {
    use super::*;
    #[test]
    fn unlimited_wall_time_still_obeys_node_and_interrupt_limits() {
        let mut searcher = Searcher::new(1);
        searcher.limits.time = Duration::MAX;
        searcher.limits.nodes = 10;
        searcher.deadline = Instant::now();
        assert!(!searcher.should_stop(true));
        searcher.nodes = 10;
        assert!(searcher.should_stop(true));
        searcher.nodes = 0;
        searcher.stopped = false;
        searcher.set_stop_signal(Arc::new(AtomicU64::new(1)), 0);
        assert!(searcher.should_stop(true));
    }

    #[test]
    fn decisive_root_scores_follow_the_mover_for_both_absolute_players() {
        let model = Model::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/h768_dense.nnue"
        ))
        .unwrap();
        for ply in [0, 1, 20, 21] {
            let mut searcher = Searcher::new(1);
            let mut state = State {
                board: [Cell::Empty; 81],
                since_capture: 0,
                ply,
            };
            state.board[60] = Cell::Own(engine::Piece::Rock);
            state.board[10] = Cell::Enemy(engine::Piece::Paper);
            let result = searcher.search(
                &model,
                &state,
                None,
                Limits {
                    nodes: 10_000,
                    time: Duration::MAX,
                    ..Limits::default()
                },
            );
            let (action, score, _) = result.completed_root().unwrap();
            assert_eq!(score, -MATE + 2);
            let child = engine::apply(&Rules::SITE, &state, action).0;
            assert!(child
                .legal_actions()
                .iter()
                .any(|&a| engine::apply(&Rules::SITE, &child, a).1 == Outcome::Win));
            state.board[60] = Cell::Empty;
            state.board[79] = Cell::Own(engine::Piece::Rock);
            let result = searcher.search(&model, &state, None, Limits::default());
            assert_eq!(result.score, MATE - 1);
            assert!(result.completed_root().is_none());
            assert_eq!(
                engine::apply(&Rules::SITE, &state, result.action.unwrap()).1,
                Outcome::Win
            );
        }
    }
    #[test]
    fn completed_label_does_not_follow_a_partial_playing_choice() {
        let mut result = SearchResult {
            action: Some(2),
            score: 123,
            depth: 1,
            nodes: 17,
            qnodes: 8,
            elapsed: Duration::ZERO,
            aborted: true,
            partial: true,
            pv: vec![2],
            root_moves: 4,
            iterations: vec![
                Iteration {
                    depth: 1,
                    nodes: 6,
                    elapsed_ms: 1.,
                    action: Some(1),
                    score: -45,
                    completed: true,
                    changed: true,
                },
                Iteration {
                    depth: 2,
                    nodes: 11,
                    elapsed_ms: 2.,
                    action: Some(2),
                    score: 123,
                    completed: false,
                    changed: true,
                },
            ],
        };
        assert_eq!(result.completed_root(), Some((1, -45, 1)));
        result.iterations.clear();
        result.score = MATE - 1;
        assert_eq!(result.completed_root(), None);
    }

    #[test]
    fn game_reset_removes_previous_worker_assignment_history() {
        let model = Model::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/h768_dense.nnue"
        ))
        .unwrap();
        let mut searcher = Searcher::new(1);
        let limits = Limits {
            nodes: 2000,
            time: Duration::MAX,
            ..Limits::default()
        };
        let signature = |r: SearchResult| {
            (
                r.action,
                r.score,
                r.depth,
                r.nodes,
                r.iterations
                    .iter()
                    .map(|i| (i.action, i.score, i.depth, i.nodes, i.completed))
                    .collect::<Vec<_>>(),
            )
        };
        let expected = signature(searcher.search(&model, &State::initial(), None, limits));
        let child = engine::apply(
            &Rules::SITE,
            &State::initial(),
            State::initial().legal_actions()[0],
        )
        .0;
        searcher.search(&model, &child, None, limits);
        searcher.clear();
        assert_eq!(searcher.generation, 0);
        assert_eq!(
            signature(searcher.search(&model, &State::initial(), None, limits)),
            expected
        );
    }
    #[test]
    fn fixture_fixed_node_signatures_are_repeatable() {
        const DENSE: &[u8] = include_bytes!("../tests/fixtures/h768_dense.nnue");
        let golden = [vec![
            (Some(604), 29, 2, 4000, 3624, Some((516, 36, 2))),
            (Some(576), -39, 3, 4000, 3543, Some((576, -39, 3))),
            (Some(348), -41, 2, 4000, 3835, Some((348, -41, 2))),
            (Some(588), 13, 2, 4000, 3660, Some((507, -42, 2))),
        ]];
        for (bytes, golden) in [DENSE].into_iter().zip(golden) {
            let model = Model::from_bytes(bytes).unwrap();
            let mut signatures = Vec::new();
            for line in include_str!("../tests/fixtures/h768_positions.jsonl")
                .lines()
                .take(4)
            {
                let position: crate::diagnostic::Position = serde_json::from_str(line).unwrap();
                let state = position.state().unwrap();
                let mut search = Searcher::new(1);
                let limits = Limits {
                    nodes: 4000,
                    time: Duration::MAX,
                    ..Limits::default()
                };
                let signature = |r: SearchResult| {
                    (
                        r.action,
                        r.score,
                        r.depth,
                        r.nodes,
                        r.qnodes,
                        r.completed_root(),
                    )
                };
                let expected = signature(search.search(&model, &state, None, limits));
                search.clear();
                assert_eq!(
                    signature(search.search(&model, &state, None, limits)),
                    expected
                );
                signatures.push(expected);
            }
            assert_eq!(signatures, golden);
        }
    }
}
