//! Gumbel MCTS: root candidates by prior plus Gumbel noise, sequential halving,
//! deficit selection below the root, batched leaf evaluation with virtual
//! visits, immediate-win detection, tree reuse and budget planning.

use std::time::Instant;

use engine::{apply, Action, Outcome, Rules, State};
use rand::Rng;

use crate::tree::{Node, NodeId, Tree, NONE};
use crate::{Evaluator, History};

/// Seconds per simulation assumed before anything is measured.
const DEFAULT_SIM_COST: f64 = 0.002;
/// Weight of the previous move's per-simulation cost when carrying it forward.
const COST_EMA: f64 = 0.7;
/// Time kept back for final selection and reply.
const RETURN_RESERVE_SECONDS: f64 = 0.003;
/// Safety factor on the measured cost of a fresh leaf evaluation.
const CALL_COST_MARGIN: f64 = 1.5;

#[derive(Clone, Debug)]
pub struct Params {
    pub rules: Rules,
    /// Root candidates considered by sequential halving.
    pub candidates: usize,
    pub c_visit: f32,
    pub c_scale: f32,
    /// Descents per evaluator call.
    pub batch: usize,
    /// Simulations per move under `Budget::Sims`, and the floor under a deadline.
    pub sims: u32,
    /// Cap on the simulations planned under a deadline.
    pub max_sims: u32,
    /// Keep the played child's subtree for the next move.
    pub reuse: bool,
    /// Moves-left utility cap in value units; 0 disables it.
    pub moves_left: f32,
    pub moves_left_slope: f32,
    /// |Q| below which the moves-left utility is zero.
    pub moves_left_threshold: f32,
    /// A rule draw is worth `-contempt` to the searching side; 0 disables it.
    pub contempt: f32,
    /// Value taken off a leaf that repeats a game or path position; 0 disables it.
    pub repetition_penalty: f32,
    /// Score a third occurrence of a position as a draw inside the tree.
    pub repetition_draw: bool,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            rules: Rules::SITE,
            candidates: 16,
            c_visit: 50.0,
            c_scale: 1.0,
            batch: 8,
            sims: 32,
            max_sims: 4096,
            reuse: true,
            moves_left: 0.0,
            moves_left_slope: 0.002,
            moves_left_threshold: 0.5,
            contempt: 0.0,
            repetition_penalty: 0.0,
            repetition_draw: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Budget {
    Sims(u32),
    Deadline(Instant),
}

#[derive(Clone, Debug, Default)]
pub struct Info {
    pub sims: u32,
    /// Nodes in the tree after the search.
    pub nodes: u32,
    /// Leaf evaluations requested from the evaluator.
    pub evaluations: u64,
    pub root_value: f32,
    pub plies_left: Option<f32>,
    /// The move wins at once; no search was run.
    pub exact_win: bool,
    /// `(action, visits, completed Q)` for every legal root action, ascending by action.
    pub root: Vec<(Action, u32, f64)>,
    /// Candidates by root score, then the remaining legal actions by prior.
    pub order: Vec<Action>,
}

pub struct Gumbel {
    pub params: Params,
    tree: Tree,
    root: Option<NodeId>,
    /// Seconds per simulation, carried between moves for deadline planning.
    sim_cost: Option<f64>,
    /// High-water seconds per uncached leaf evaluation.
    call_cost: Option<f64>,
    evaluations: u64,
    path: Vec<(NodeId, usize)>,
    virt: Vec<VirtualVisit>,
    descents: Vec<Descent>,
    pending: Vec<Pending>,
    seen: Vec<(engine::PositionKey, u32)>,
}

#[derive(Clone, Copy)]
struct VirtualVisit {
    node: NodeId,
    edge: usize,
    visits: u32,
    total: f64,
}

struct Descent {
    path_start: usize,
    path_len: usize,
    value: Option<f32>,
    pending: Option<usize>,
}

struct Pending {
    node: NodeId,
    edge: usize,
    state: State,
    /// Plies below the root; odd means the opponent is to move at the leaf.
    depth: usize,
    /// Earlier occurrences of the leaf position in the game plus on the path.
    repeats: u32,
}

/// Legal actions of `state` that win at once.
pub fn immediate_wins(rules: &Rules, state: &State, legal: &[Action]) -> Vec<bool> {
    legal
        .iter()
        .map(|&action| apply(rules, state, action).1 == Outcome::Win)
        .collect()
}

impl Gumbel {
    pub fn new(params: Params) -> Gumbel {
        let mut params = params;
        params.batch = params.batch.max(1);
        params.candidates = params.candidates.max(1);
        params.sims = params.sims.max(1);
        params.max_sims = params.max_sims.max(params.sims);
        params.moves_left_threshold = params.moves_left_threshold.clamp(0.0, 0.999);
        Gumbel {
            params,
            tree: Tree::new(),
            root: None,
            sim_cost: None,
            call_cost: None,
            evaluations: 0,
            path: Vec::new(),
            virt: Vec::new(),
            descents: Vec::new(),
            pending: Vec::new(),
            seen: Vec::new(),
        }
    }

    /// Forget the kept tree: a new game or a changed evaluator.
    pub fn reset(&mut self) {
        self.tree.clear();
        self.root = None;
    }

    /// Pick a move for `state`. `history` counts positions already played in
    /// the game, including `state` itself. Panics without a legal move.
    pub fn choose<E: Evaluator, R: Rng>(
        &mut self,
        evaluator: &mut E,
        state: &State,
        history: &History,
        budget: Budget,
        rng: &mut R,
    ) -> (Action, Info) {
        let started = Instant::now();
        let evaluations0 = self.evaluations;
        let root = self.root_for(evaluator, state, history);
        self.observe_call(
            started.elapsed().as_secs_f64(),
            self.evaluations - evaluations0,
        );
        let k = self.tree.get(root).legal.len();
        assert!(k > 0, "no legal move to choose from");

        if let Some(edge) = self.best_immediate_win(root) {
            let node = self.tree.get(root);
            let action = node.legal[edge];
            let mut info = self.info(root, &[edge], 0, evaluations0);
            info.exact_win = true;
            return (action, self.finish(info));
        }

        let noise: Vec<f64> = (0..k)
            .map(|_| {
                let u: f64 = rng.random::<f64>().clamp(1e-12, 1.0 - 1e-12);
                -(-u.ln()).ln()
            })
            .collect();
        let mut candidates: Vec<usize> = (0..k).collect();
        self.sort_by_score(root, &noise, &mut candidates);
        candidates.truncate(self.params.candidates.min(k));

        let search_started = Instant::now();
        let previous_cost = self.sim_cost;
        let mut used = 0u32;
        let mut budget_sims = self.plan(budget, used);
        let mut out_of_time = false;
        let mut schedule: Vec<usize> = Vec::new();
        // Sequential halving: each phase spreads the remaining budget evenly over the
        // survivors, then keeps the better half by root score.
        while candidates.len() > 1 && used < budget_sims && !out_of_time {
            let phases = ceil_log2(candidates.len()).max(1);
            let per = (((budget_sims - used) as usize) / (phases * candidates.len())).max(1);
            schedule.clear();
            for _ in 0..per {
                schedule.extend(candidates.iter().copied());
            }
            let mut at = 0;
            while at < schedule.len() && used < budget_sims {
                let wanted = self
                    .params
                    .batch
                    .min(schedule.len() - at)
                    .min((budget_sims - used) as usize);
                let take = self.take(wanted, budget);
                if take == 0 {
                    out_of_time = true;
                    break;
                }
                self.run_batch(evaluator, root, &schedule[at..at + take], history);
                at += take;
                used += take as u32;
                self.sim_cost = Some(search_started.elapsed().as_secs_f64() / used as f64);
                budget_sims = self.plan(budget, used);
            }
            self.sort_by_score(root, &noise, &mut candidates);
            candidates.truncate((candidates.len() / 2).max(1));
        }
        // One survivor: spend whatever is left deepening its line.
        while used < budget_sims && candidates.len() == 1 && !out_of_time {
            let wanted = self.params.batch.min((budget_sims - used) as usize);
            let take = self.take(wanted, budget);
            if take == 0 {
                break;
            }
            schedule.clear();
            schedule.resize(take, candidates[0]);
            self.run_batch(evaluator, root, &schedule, history);
            used += take as u32;
            self.sim_cost = Some(search_started.elapsed().as_secs_f64() / used as f64);
            budget_sims = self.plan(budget, used);
        }
        self.remember_cost(search_started.elapsed().as_secs_f64(), used, previous_cost);

        self.sort_by_score(root, &noise, &mut candidates);
        let best = candidates[0];
        let action = self.tree.get(root).legal[best];
        let info = self.info(root, &candidates, used, evaluations0);
        (action, self.finish(info))
    }

    /// Drop the tree unless it is kept for the next move.
    fn finish(&mut self, info: Info) -> Info {
        if !self.params.reuse {
            self.reset();
        }
        info
    }

    fn info(&self, root: NodeId, candidates: &[usize], sims: u32, evaluations0: u64) -> Info {
        let node = self.tree.get(root);
        let k = node.legal.len();
        let mut rest: Vec<usize> = (0..k).filter(|i| !candidates.contains(i)).collect();
        rest.sort_by(|&a, &b| node.log_prior[b].total_cmp(&node.log_prior[a]));
        let mut order: Vec<Action> = candidates.iter().map(|&i| node.legal[i]).collect();
        order.extend(rest.iter().map(|&i| node.legal[i]));
        Info {
            sims,
            nodes: self.tree.nodes.len() as u32,
            evaluations: self.evaluations - evaluations0,
            root_value: node.value,
            plies_left: node.plies_left,
            exact_win: false,
            root: (0..k)
                .map(|i| (node.legal[i], node.visits[i], node.completed_q(i)))
                .collect(),
            order,
        }
    }

    /// The winning root move with the highest prior, if any wins at once.
    fn best_immediate_win(&self, root: NodeId) -> Option<usize> {
        let node = self.tree.get(root);
        (0..node.legal.len())
            .filter(|&i| node.wins[i])
            .max_by(|&a, &b| node.log_prior[a].total_cmp(&node.log_prior[b]))
    }

    /// Reuse the kept subtree if it contains `state`, else evaluate a fresh root.
    fn root_for<E: Evaluator>(
        &mut self,
        evaluator: &mut E,
        state: &State,
        history: &History,
    ) -> NodeId {
        if let Some(root) = self.reroot(state, history) {
            return root;
        }
        self.reset();
        let eval = self.evaluate_one(evaluator, state, 1);
        let wins = immediate_wins(&self.params.rules, state, &eval.legal);
        let root = self.tree.push(Node::expanded(state.clone(), eval, wins));
        self.root = self.params.reuse.then_some(root);
        root
    }

    fn evaluate_one<E: Evaluator>(
        &mut self,
        evaluator: &mut E,
        state: &State,
        sign: i8,
    ) -> crate::Eval {
        self.evaluations += 1;
        evaluator
            .evaluate(&[state], &[sign])
            .pop()
            .expect("one state in, one eval out")
    }

    /// The kept root, its child or grandchild that is `state`, made the new root.
    fn reroot(&mut self, state: &State, history: &History) -> Option<NodeId> {
        let old_root = self.root?;
        let mut found = None;
        let mut frontier = vec![old_root];
        for _ in 0..3 {
            let mut next = Vec::new();
            for id in frontier {
                let node = self.tree.get(id);
                if node.terminal.is_some() {
                    continue;
                }
                if node.state == *state {
                    found = Some(id);
                    break;
                }
                next.extend(node.children.iter().copied().filter(|&c| c != NONE));
            }
            if found.is_some() {
                break;
            }
            frontier = next;
        }
        let found = found?;
        if self.tree.get(found).legal.is_empty()
            || (self.params.repetition_draw && !self.repetition_ok(found, history))
        {
            return None;
        }
        self.tree.reroot(found);
        self.root = Some(0);
        Some(0)
    }

    /// True when no non-terminal node below `root` is a third occurrence of its
    /// position given the game history.
    fn repetition_ok(&self, root: NodeId, history: &History) -> bool {
        let mut seen: History = History::new();
        let mut stack: Vec<(NodeId, usize, bool)> = vec![(root, 0, false)];
        while let Some(&(id, next, release)) = stack.last() {
            let node = self.tree.get(id);
            if next >= node.children.len() {
                stack.pop();
                if release {
                    *seen.get_mut(&node.key).expect("released key") -= 1;
                }
                continue;
            }
            stack.last_mut().expect("frame").1 = next + 1;
            let child_id = node.children[next];
            if child_id == NONE {
                continue;
            }
            let child = self.tree.get(child_id);
            if child.terminal.is_some() {
                continue;
            }
            let total = history.get(&child.key).copied().unwrap_or(0)
                + seen.get(&child.key).copied().unwrap_or(0)
                + 1;
            if total >= 3 {
                return false;
            }
            *seen.entry(child.key).or_insert(0) += 1;
            stack.push((child_id, 0, true));
        }
        true
    }

    /// Descend once per root edge, evaluate the new leaves in one call, back up.
    fn run_batch<E: Evaluator>(
        &mut self,
        evaluator: &mut E,
        root: NodeId,
        root_edges: &[usize],
        history: &History,
    ) {
        self.path.clear();
        self.virt.clear();
        self.descents.clear();
        self.pending.clear();
        for &edge in root_edges {
            self.descend(root, edge, history);
        }
        for visit in self.virt.iter().rev() {
            let node = self.tree.get_mut(visit.node);
            node.visits[visit.edge] = visit.visits;
            node.total[visit.edge] = visit.total;
        }
        let mut leaf_values = Vec::with_capacity(self.pending.len());
        if !self.pending.is_empty() {
            let started = Instant::now();
            let states: Vec<&State> = self.pending.iter().map(|p| &p.state).collect();
            let signs: Vec<i8> = self
                .pending
                .iter()
                .map(|p| if p.depth % 2 == 0 { 1 } else { -1 })
                .collect();
            let evals = evaluator.evaluate(&states, &signs);
            self.evaluations += states.len() as u64;
            self.observe_call(started.elapsed().as_secs_f64(), states.len() as u64);
            for (pending, eval) in self.pending.iter().zip(evals) {
                let wins = immediate_wins(&self.params.rules, &pending.state, &eval.legal);
                let mut node = Node::expanded(pending.state.clone(), eval, wins);
                if self.params.repetition_penalty != 0.0 && pending.repeats > 0 {
                    // The opponent (odd depth) likes a repeat from its own view.
                    let shift = if pending.depth % 2 == 1 {
                        self.params.repetition_penalty
                    } else {
                        -self.params.repetition_penalty
                    };
                    node.value = (node.value + shift).clamp(-1.0, 1.0);
                }
                leaf_values.push(node.value);
                let child = self.tree.push(node);
                self.tree.get_mut(pending.node).children[pending.edge] = child;
            }
        }
        for descent in &self.descents {
            let mut value = descent
                .value
                .unwrap_or_else(|| leaf_values[descent.pending.expect("leaf slot")]);
            let end = descent.path_start + descent.path_len;
            for &(id, edge) in self.path[descent.path_start..end].iter().rev() {
                value = -value;
                let node = self.tree.get_mut(id);
                node.visits[edge] += 1;
                node.total[edge] += value as f64;
            }
        }
    }

    fn descend(&mut self, root: NodeId, root_edge: usize, history: &History) {
        let path_start = self.path.len();
        self.seen.clear();
        let mut id = root;
        let mut edge = root_edge;
        loop {
            self.path.push((id, edge));
            let node = self.tree.get_mut(id);
            self.virt.push(VirtualVisit {
                node: id,
                edge,
                visits: node.visits[edge],
                total: node.total[edge],
            });
            let virtual_q = node.completed_q(edge);
            node.visits[edge] += 1;
            node.total[edge] += virtual_q;
            let child = node.children[edge];
            let depth = self.path.len() - path_start;
            if child == NONE {
                let (child_state, mut outcome) =
                    apply(&self.params.rules, &node.state, node.legal[edge]);
                let key = child_state.key();
                let repeats = history.get(&key).copied().unwrap_or(0)
                    + self
                        .seen
                        .iter()
                        .find(|(k, _)| *k == key)
                        .map_or(0, |(_, n)| *n);
                if outcome == Outcome::Ongoing && self.params.repetition_draw && repeats + 1 >= 3 {
                    outcome = Outcome::Draw;
                }
                let terminal_value = match outcome {
                    Outcome::Ongoing => None,
                    // From the child's mover's view: it lost.
                    Outcome::Win => Some(-1.0),
                    Outcome::Draw => Some(if depth % 2 == 1 {
                        self.params.contempt
                    } else {
                        -self.params.contempt
                    }),
                };
                if let Some(value) = terminal_value {
                    let terminal = self.tree.push(Node::terminal(child_state, value));
                    self.tree.get_mut(id).children[edge] = terminal;
                    self.descents.push(Descent {
                        path_start,
                        path_len: depth,
                        value: Some(value),
                        pending: None,
                    });
                    return;
                }
                let slot = self
                    .pending
                    .iter()
                    .position(|p| p.node == id && p.edge == edge)
                    .unwrap_or_else(|| {
                        self.pending.push(Pending {
                            node: id,
                            edge,
                            state: child_state,
                            depth,
                            repeats,
                        });
                        self.pending.len() - 1
                    });
                self.descents.push(Descent {
                    path_start,
                    path_len: depth,
                    value: None,
                    pending: Some(slot),
                });
                return;
            }
            let child_node = self.tree.get(child);
            if let Some(value) = child_node.terminal {
                self.descents.push(Descent {
                    path_start,
                    path_len: depth,
                    value: Some(value),
                    pending: None,
                });
                return;
            }
            let key = child_node.key;
            match self.seen.iter_mut().find(|(k, _)| *k == key) {
                Some((_, n)) => *n += 1,
                None => self.seen.push((key, 1)),
            }
            id = child;
            edge = self.select(id);
        }
    }

    /// Non-root selection: argmax of softmax(prior + sigma * utility) - visits / (1 + total).
    fn select(&self, id: NodeId) -> usize {
        let node = self.tree.get(id);
        let scale = self.sigma(node);
        let z: Vec<f64> = (0..node.legal.len())
            .map(|i| node.log_prior[i] as f64 + scale * self.utility(node, i))
            .collect();
        let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let norm: f64 = z.iter().map(|v| (v - top).exp()).sum();
        let total: u32 = node.visits.iter().sum();
        let mut best = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (i, &zi) in z.iter().enumerate() {
            let value = (zi - top).exp() / norm - node.visits[i] as f64 / (1.0 + total as f64);
            if value > best_value {
                best = i;
                best_value = value;
            }
        }
        best
    }

    fn sigma(&self, node: &Node) -> f64 {
        (self.params.c_visit as f64 + node.max_visits() as f64) * self.params.c_scale as f64
    }

    fn utility(&self, node: &Node, edge: usize) -> f64 {
        let q = node.completed_q(edge);
        q + self.moves_left_bonus(node, edge, q)
    }

    /// Prefers shorter wins and longer losses once |Q| passes the threshold.
    fn moves_left_bonus(&self, node: &Node, edge: usize, q: f64) -> f64 {
        if self.params.moves_left <= 0.0 || node.children[edge] == NONE {
            return 0.0;
        }
        let (Some(parent_m), Some(child_m)) = (
            node.plies_left,
            self.tree.get(node.children[edge]).plies_left,
        ) else {
            return 0.0;
        };
        let threshold = self.params.moves_left_threshold as f64;
        let magnitude = q.abs();
        if magnitude <= threshold {
            return 0.0;
        }
        let factor = ((magnitude - threshold) / (1.0 - threshold)).powi(2);
        let cap = self.params.moves_left as f64;
        let delta = (self.params.moves_left_slope as f64
            * (child_m as f64 - (parent_m as f64 - 1.0)))
            .clamp(-cap, cap);
        if q > 0.0 {
            -factor * delta
        } else {
            factor * delta
        }
    }

    fn root_score(&self, root: NodeId, noise: &[f64], edge: usize) -> f64 {
        let node = self.tree.get(root);
        noise[edge] + node.log_prior[edge] as f64 + self.sigma(node) * self.utility(node, edge)
    }

    fn sort_by_score(&self, root: NodeId, noise: &[f64], candidates: &mut [usize]) {
        candidates.sort_by(|&a, &b| {
            self.root_score(root, noise, b)
                .total_cmp(&self.root_score(root, noise, a))
        });
    }

    /// Simulations to plan for under `budget` given `used` so far.
    fn plan(&self, budget: Budget, used: u32) -> u32 {
        match budget {
            Budget::Sims(sims) => sims.max(1),
            Budget::Deadline(deadline) => {
                let left = deadline
                    .saturating_duration_since(Instant::now())
                    .as_secs_f64();
                let usable = (left - RETURN_RESERVE_SECONDS).max(0.0);
                let planned = used.saturating_add((usable / self.cost()).floor() as u32);
                planned.clamp(self.params.sims, self.params.max_sims)
            }
        }
    }

    /// How many of `cap` descents fit before the deadline.
    fn take(&self, cap: usize, budget: Budget) -> usize {
        let Budget::Deadline(deadline) = budget else {
            return cap;
        };
        let left = deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        let usable = (left - RETURN_RESERVE_SECONDS).max(0.0);
        let fresh_leaf = self.call_cost.unwrap_or(DEFAULT_SIM_COST).max(1e-6);
        let bound = fresh_leaf.max(self.cost()) * CALL_COST_MARGIN;
        cap.min((usable / bound).floor() as usize)
    }

    fn cost(&self) -> f64 {
        self.sim_cost.unwrap_or(DEFAULT_SIM_COST)
    }

    fn observe_call(&mut self, elapsed: f64, evaluated: u64) {
        if evaluated != 0 {
            let per_leaf = elapsed / evaluated as f64;
            self.call_cost = Some(self.call_cost.map_or(per_leaf, |old| old.max(per_leaf)));
        }
    }

    fn remember_cost(&mut self, elapsed: f64, used: u32, previous: Option<f64>) {
        self.sim_cost = if used == 0 {
            previous
        } else {
            let measured = elapsed / used as f64;
            Some(previous.map_or(measured, |old| COST_EMA * old + (1.0 - COST_EMA) * measured))
        };
    }
}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Eval;
    use engine::{from_to, Cell, Piece, N_SQUARES};
    use rand::{rngs::StdRng, SeedableRng};

    /// Uniform prior, zero Q, so only the rules and the search decide.
    struct Flat;

    impl Evaluator for Flat {
        fn evaluate(&mut self, states: &[&State], _signs: &[i8]) -> Vec<Eval> {
            states
                .iter()
                .map(|state| {
                    let legal = state.legal_actions();
                    let n = legal.len().max(1) as f32;
                    Eval {
                        log_prior: vec![-(n.ln()); legal.len()],
                        q: vec![0.0; legal.len()],
                        legal,
                        value: 0.0,
                        plies_left: None,
                    }
                })
                .collect()
        }
    }

    fn place(cells: &[(usize, Cell)]) -> State {
        let mut state = State::initial();
        state.board = [Cell::Empty; N_SQUARES];
        for &(square, cell) in cells {
            state.board[square] = cell;
        }
        state
    }

    #[test]
    fn immediate_win_is_played_without_search() {
        let state = place(&[
            (71, Cell::Own(Piece::Rock)),
            (0, Cell::Enemy(Piece::Paper)),
            (40, Cell::Enemy(Piece::Rock)),
        ]);
        let mut search = Gumbel::new(Params::default());
        let mut rng = StdRng::seed_from_u64(1);
        let (action, info) = search.choose(
            &mut Flat,
            &state,
            &History::new(),
            Budget::Sims(32),
            &mut rng,
        );
        assert!(info.exact_win);
        assert_eq!(from_to(action).1, 80);
        assert_eq!(info.sims, 0);
    }

    #[test]
    fn search_avoids_a_move_that_allows_an_immediate_win() {
        // Own rock e5 and own paper a2 guard nothing; enemy paper on h8 wins by entering
        // our home a1... in the enemy's frame its goal is our a1 = its i9. The enemy paper
        // sits at b2 (10): it reaches a1 next move unless captured. Our scissors on c3 (20)
        // can capture it (scissors beats paper); every other move loses at once.
        let state = place(&[
            (10, Cell::Enemy(Piece::Paper)),
            (20, Cell::Own(Piece::Scissors)),
            (60, Cell::Own(Piece::Rock)),
            (80 - 1, Cell::Enemy(Piece::Rock)),
        ]);
        let mut search = Gumbel::new(Params {
            reuse: false,
            ..Params::default()
        });
        let mut rng = StdRng::seed_from_u64(3);
        let (action, info) = search.choose(
            &mut Flat,
            &state,
            &History::new(),
            Budget::Sims(64),
            &mut rng,
        );
        assert_eq!(from_to(action), (20, 10), "capture the paper: {info:?}");
        assert!(info.sims > 0);
    }

    #[test]
    fn reuse_keeps_the_played_subtree() {
        let state = State::initial();
        let mut search = Gumbel::new(Params::default());
        let mut rng = StdRng::seed_from_u64(5);
        let mut history = History::new();
        history.insert(state.key(), 1);
        let (action, _) = search.choose(&mut Flat, &state, &history, Budget::Sims(32), &mut rng);
        let (child, _) = apply(&Rules::SITE, &state, action);
        let reply = child.legal_actions()[0];
        let (grandchild, _) = apply(&Rules::SITE, &child, reply);
        let before = search.tree.nodes.len();
        assert!(before > 1);
        let root = search.root_for(&mut Flat, &grandchild, &history);
        assert_eq!(root, 0);
        assert!(search.tree.nodes.len() <= before);
        assert_eq!(search.tree.get(0).state, grandchild);
    }
}
