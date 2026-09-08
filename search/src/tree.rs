//! Node arena for one search. Values are stored from each node's mover's view
//! and negated on every backup step.

use engine::{Action, PositionKey, State};

use crate::Eval;

pub type NodeId = u32;
pub const NONE: NodeId = u32::MAX;

pub struct Node {
    pub state: State,
    pub key: PositionKey,
    pub legal: Vec<Action>,
    pub log_prior: Vec<f32>,
    pub q: Vec<f32>,
    pub value: f32,
    pub plies_left: Option<f32>,
    pub visits: Vec<u32>,
    pub total: Vec<f64>,
    pub children: Vec<NodeId>,
    /// Legal actions that win at once; their completed Q is +1 without visits.
    pub wins: Vec<bool>,
    /// Exact game value from this node's mover's view when the game is over here.
    pub terminal: Option<f32>,
}

impl Node {
    pub fn expanded(state: State, eval: Eval, wins: Vec<bool>) -> Node {
        let edges = eval.legal.len();
        let value = if wins.iter().any(|&win| win) {
            1.0
        } else {
            eval.value
        };
        Node {
            key: state.key(),
            state,
            legal: eval.legal,
            log_prior: eval.log_prior,
            q: eval.q,
            value,
            plies_left: eval.plies_left,
            visits: vec![0; edges],
            total: vec![0.0; edges],
            children: vec![NONE; edges],
            wins,
            terminal: None,
        }
    }

    pub fn terminal(state: State, value: f32) -> Node {
        Node {
            key: state.key(),
            state,
            legal: Vec::new(),
            log_prior: Vec::new(),
            q: Vec::new(),
            value,
            plies_left: Some(0.0),
            visits: Vec::new(),
            total: Vec::new(),
            children: Vec::new(),
            wins: Vec::new(),
            terminal: Some(value),
        }
    }

    /// Mean backed-up value of an edge, +1 for an immediate win, or the
    /// network's q when unvisited.
    pub fn completed_q(&self, edge: usize) -> f64 {
        if self.wins[edge] {
            1.0
        } else if self.visits[edge] > 0 {
            self.total[edge] / self.visits[edge] as f64
        } else {
            self.q[edge] as f64
        }
    }

    pub fn max_visits(&self) -> u32 {
        self.visits.iter().copied().max().unwrap_or(0)
    }
}

pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    pub fn new() -> Tree {
        Tree { nodes: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn push(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        (self.nodes.len() - 1) as NodeId
    }

    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    pub fn get_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    /// Keep only the subtree under `root`, renumbering nodes so that `root`
    /// becomes 0. Iterative: a kept line can be thousands of nodes deep.
    pub fn reroot(&mut self, root: NodeId) {
        let old = std::mem::take(&mut self.nodes);
        let mut work = vec![(root, NONE, 0usize)];
        while let Some((id, parent, edge)) = work.pop() {
            let new_id = self.nodes.len() as NodeId;
            let node = &old[id as usize];
            for (e, &child) in node.children.iter().enumerate().rev() {
                if child != NONE {
                    work.push((child, new_id, e));
                }
            }
            self.nodes.push(Node {
                state: node.state.clone(),
                key: node.key,
                legal: node.legal.clone(),
                log_prior: node.log_prior.clone(),
                q: node.q.clone(),
                value: node.value,
                plies_left: node.plies_left,
                visits: node.visits.clone(),
                total: node.total.clone(),
                children: vec![NONE; node.children.len()],
                wins: node.wins.clone(),
                terminal: node.terminal,
            });
            if parent != NONE {
                self.nodes[parent as usize].children[edge] = new_id;
            }
        }
    }
}
